# ============================================================
# SafeGuard AI - FastAPI Backend
# MC Dropout uncertainty + Grad-CAM + Bangla triage report
# ============================================================
import os, io, base64
import numpy as np
import torch, torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
UNCERTAINTY_THRESHOLD = 0.08

app = FastAPI(title="SafeGuard AI - Medical Scan Triage")

# ---------------- model loading ----------------
def load_model(path):
    m = models.mobilenet_v3_small(weights=None)
    ckpt = torch.load(path, map_location=device)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, len(ckpt['class_names']))
    m.load_state_dict(ckpt['state_dict'])
    m.eval()
    return m.to(device), ckpt['class_names']

print("Loading models...")
model_pneu, cls_pneu = load_model("models/model_pneumonia.pth")
model_tb, cls_tb     = load_model("models/model_tb.pth")
print("Models ready.")

# ---------------- MC dropout inference ----------------
def enable_dropout(m):
    for module in m.modules():
        if isinstance(module, nn.Dropout):
            module.train()

@torch.no_grad()
def mc_predict(model, img_tensor, T=20):
    model.eval()
    enable_dropout(model)
    probs = torch.stack([torch.softmax(model(img_tensor.to(device)), 1) for _ in range(T)])
    mean_p = probs.mean(0).squeeze(0)
    std_p  = probs.std(0).squeeze(0)
    pred = mean_p.argmax().item()
    return mean_p.cpu().numpy(), std_p[pred].item(), pred

# ---------------- Grad-CAM ----------------
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRADCAM_OK = True
except ImportError:
    GRADCAM_OK = False
    print("WARNING: Grad-CAM not available")

def make_gradcam(model, img_tensor, pil_rgb):
    if not GRADCAM_OK:
        return None
    cam = GradCAM(model=model, target_layers=[model.features[-1]])
    gcam = cam(input_tensor=img_tensor.to(device), targets=None)[0]
    vis = show_cam_on_image(np.array(pil_rgb) / 255.0, gcam, use_rgb=True)
    return vis

def img_to_b64(img_array):
    im = Image.fromarray(img_array)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

# ---------------- transforms ----------------
IMG = 224
val_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG, IMG)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ---------------- Bangla report ----------------
def bangla_report(disease_name_bn, disease_prob, uncertain):
    if uncertain:
        return (f"Warning: The AI is not confident about this X-ray. "
                f"A doctor must review this case. Please submit the report at the counter.")
    if disease_prob > 0.5:
        return (f"This X-ray shows a high probability of {disease_name_bn} "
                f"({int(disease_prob*100)} percent). Please consult a doctor urgently.")
    return ("No significant problem was found in this X-ray. However, consult a doctor if needed.")

# ---------------- routes ----------------
@app.get("/", response_class=HTMLResponse)
async def home():
    with open("static/index.html", encoding="utf-8") as f:
        return f.read()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    pil = Image.open(io.BytesIO(await file.read())).convert("RGB")
    small = pil.resize((IMG, IMG))
    tensor = val_tf(small).unsqueeze(0)

    results = {}
    for key, model, names, bn_name in [
        ("pneumonia", model_pneu, cls_pneu, "Pneumonia"),
        ("tb",        model_tb,   cls_tb,   "Tuberculosis (TB)")
    ]:
        mean_p, unc, pred = mc_predict(model, tensor)
        disease_prob = float(mean_p[1])
        uncertain = unc > UNCERTAINTY_THRESHOLD

        if unc > 0.08:
            status = "YELLOW"       # high uncertainty -> doctor review
        elif pred == 1 and disease_prob > 0.7:
            status = "RED"          # confident disease -> high risk
        elif pred == 1:
            status = "YELLOW"       # borderline disease -> doctor review
        else:
            status = "GREEN"        # confident normal

        heatmap_b64 = None
        if pred == 1:
            hm = make_gradcam(model, tensor, small)
            if hm is not None:
                heatmap_b64 = img_to_b64(hm)

        results[key] = {
            "label": names[pred],
            "disease_probability": round(disease_prob, 3),
            "uncertainty": round(unc, 4),
            "status": status,
            "uncertain": uncertain,
            "bangla_report": bangla_report(bn_name, disease_prob, uncertain),
            "heatmap": heatmap_b64
        }

    order = {"RED": 2, "YELLOW": 1, "GREEN": 0}
    overall = max([r["status"] for r in results.values()], key=lambda s: order[s])
    msg = {
        "RED":    "HIGH RISK - Consult a doctor urgently",
        "YELLOW": "UNCERTAIN - Human doctor review needed",
        "GREEN":  "NORMAL - No significant issue detected"
    }[overall]

    return JSONResponse({
        "overall_status": overall,
        "overall_message": msg,
        "results": results
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)