<div align="center">

# SafeGuard AI

### Uncertainty-Aware Medical X-Ray Triage for Rural India

[![Live Demo](https://img.shields.io/badge/Live_Demo-3.7.68.92:8000-success?style=for-the-badge)](http://3.7.68.92:8000)
[![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14-orange?style=for-the-badge&logo=pytorch)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.141-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker)](https://docker.com)
[![AWS](https://img.shields.io/badge/AWS-EC2_Deployed-FF9900?style=for-the-badge&logo=amazonaws)](https://aws.amazon.com)

**Built for [First Commit Hackathon](https://wemakedevs.org) | Bharat Builds Tour | September 2026**

</div>

---

## The Problem

In rural India, there is roughly one radiologist for every 100,000+ people. When a patient gets a chest X-ray:

- The X-ray is taken within hours
- The report takes 1 to 2 weeks to come back
- By then, pneumonia or TB may have become critical

Existing AI models produce outputs like "Pneumonia: 92%" but they do not know when they are wrong. In rural clinics with no doctor on-site, a confidently wrong prediction is dangerous.

The real question is not whether AI can detect disease. The real question is whether AI knows when it does not know.

---

## The Solution

SafeGuard AI is a dual-disease chest X-ray triage system that:

1. Detects Pneumonia and Tuberculosis from X-ray images
2. Quantifies its own uncertainty using Monte Carlo Dropout
3. Escalates to a human doctor when uncertain, instead of guessing
4. Explains its decisions using Grad-CAM heatmaps

### The Hero Feature: Uncertainty Quantification

**Case 1 - Confident Detection**
- Pneumonia probability: 100%
- Uncertainty: 0.00
- Output: HIGH RISK - Refer to doctor immediately

**Case 2 - Uncertain (the key differentiator)**
- Pneumonia probability: 56.5%
- Uncertainty: 0.0498
- Output: DOCTOR REVIEW NEEDED
- The model knows it does not know.

**Case 3 - Confident Normal**
- Pneumonia probability: 0%
- Uncertainty: 0.00
- Output: NORMAL

---

## Features

| Feature | Description |
|---------|-------------|
| Pneumonia Detection | MobileNetV3, 98.7% validation accuracy |
| TB Detection | MobileNetV3, 100% validation accuracy |
| MC Dropout Uncertainty | 20 stochastic forward passes, standard deviation as uncertainty |
| Grad-CAM Heatmaps | Visual explanation of where the model is looking |
| Smart Triage | Auto-escalate uncertain cases to human review |
| Live on AWS | Production deployment, not a local demo |
| Fast Inference | CPU-optimized, runs on free-tier EC2 |
| Zero Monthly Cost | Fully within AWS Free Tier |

---

## Architecture

```
Client (Browser, Tailwind CSS, Mobile-first UI)
    |
    | HTTP POST /predict
    v
AWS EC2 t3.micro (Mumbai Region)
    |
    | Docker Container (safeguard-ai)
    v
FastAPI + Uvicorn
    |
    | PyTorch Inference Pipeline
    v
MobileNetV3 (Pneumonia) + MobileNetV3 (TB)
    |
    | MC Dropout (20 forward passes)
    v
Grad-CAM Heatmap + Triage Decision
    |
    | JSON Response
    v
Client renders prediction, uncertainty bars, heatmap, and report
```

Cost breakdown: $0 per month (AWS Free Tier, t3.micro 750 hours)

---

## Model Performance

| Model | Training Images | Validation Accuracy | CPU Inference Time |
|-------|----------------|---------------------|---------------------|
| Pneumonia | 5,216 | 98.7% | 1.2 seconds |
| Tuberculosis | 4,200 | 100.0% | 1.2 seconds |

**Training strategy:**
- Transfer learning from ImageNet (MobileNetV3-Small)
- 4 epochs, AdamW optimizer, cosine learning rate schedule
- Trained on Kaggle Free GPU (T4 x2)
- MC Dropout: 20 stochastic passes, mean and standard deviation computed

### Honest Note on Metrics

The 100% TB validation accuracy is suspicious and likely reflects dataset artifacts such as similar patients appearing in both training and validation splits. This is exactly why uncertainty quantification was added. Accuracy alone is not trustworthy in medical AI.

---

## Live Demo

**URL:** [http://3.7.68.92:8000](http://3.7.68.92:8000)

### How to try it:

1. Download a chest X-ray from the [Kaggle Chest X-Ray dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
2. Upload it to the live URL
3. View the prediction, uncertainty score, and Grad-CAM heatmap

**Note:** This is a triage tool, not a diagnostic system. Always consult a qualified doctor.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Framework | PyTorch 2.14 (CPU) |
| Model | MobileNetV3-Small (transfer learning) |
| Uncertainty | Monte Carlo Dropout |
| Explainability | Grad-CAM |
| Backend | FastAPI + Uvicorn |
| Containerization | Docker |
| Deployment | AWS EC2 t3.micro (Mumbai) |
| Training | Kaggle Free GPU (T4 x2) |
| Frontend | Vanilla JS + Tailwind CSS |

---

## Run Locally

### Prerequisites
- Python 3.11 or higher
- Approximately 500 MB of disk space

### Setup

```bash
git clone https://github.com/ArpanC6/safeguard-ai.git
cd safeguard-ai
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Then open http://localhost:8000 in a browser.

### Docker

```bash
docker build -t safeguard-ai .
docker run -d -p 8000:8000 --name safeguard-app safeguard-ai
```

---

## Project Structure

```
safeguard-ai/
├── main.py                    FastAPI backend
├── requirements.txt           Python dependencies
├── Dockerfile                 Container definition
├── .dockerignore              Docker ignore rules
├── .gitignore                 Git ignore rules
├── README.md                  This file
├── models/
│   ├── model_pneumonia.pth    Pneumonia weights (6 MB)
│   └── model_tb.pth           TB weights (6 MB)
└── static/
    └── index.html             Frontend UI
```

---

## Key Design Decisions

### Why MC Dropout instead of a single forward pass?

A standard neural network outputs a single probability and hides its own uncertainty. MC Dropout runs the network 20 times with dropout active at inference, giving a distribution of predictions. The standard deviation of that distribution is the uncertainty estimate.

### Why MobileNetV3-Small?

- Small (approximately 6 MB weights), fits inside a Docker image
- Fast on CPU (about 1.2 seconds per image), works on free-tier EC2 without a GPU
- Accurate, achieving 98.7% validation accuracy on pneumonia

### Why EC2 t3.micro with Docker instead of AWS Lambda?

- PyTorch and OpenCV exceed Lambda's 250 MB unzipped deployment limit
- EC2 t3.micro is free for 12 months
- Docker makes the deployment reproducible and portable

### Why not use AWS Bedrock or SageMaker?

Bedrock access was not guaranteed for all students during the hackathon, and SageMaker endpoints cost money beyond the free tier. A self-contained EC2 with Docker deployment is zero cost per month and fully reproducible.

---

## Limitations and Future Work

This is a triage tool, not a diagnostic device. Limitations:

- No clinical validation, trained on public datasets only
- Two diseases only (pneumonia and TB), not general-purpose
- Dataset artifacts suggest possible leakage in the TB dataset
- No multi-class disease detection

**Future roadmap:**

- Add more diseases such as COVID-19, lung cancer, and pleural effusion
- Integrate with ASHA worker apps via a WhatsApp bot interface
- Replace MC Dropout with full Bayesian inference using variational layers
- Add federated learning for privacy-preserving updates from clinics
- Partner with rural health NGOs for real-world validation

---

## What I Learned Building This

1. **Uncertainty is a feature, not a bug.** A model that knows what it does not know is more useful than one that is confidently wrong.

2. **Deployment is most of the work.** The model was the easy part. Docker, EC2, environment paths, kernel updates, and disk space management took longer.

3. **Cost-aware architecture matters.** Using CPU-only PyTorch (about 200 MB) instead of GPU-enabled PyTorch (about 4 GB) saved both time and money.

4. **Grad-CAM is underrated.** Showing where the model looks makes the AI immediately more trustworthy.

---

## License

MIT License. Free to use, modify, and distribute.

---

## Author

**Arpan Chakraborty**
- GitHub: [@ArpanC6](https://github.com/ArpanC6)
- Email: chakrabortyarpan224@gmail.com

Built for the First Commit Hackathon, Bharat Builds Tour 2026.

---

<div align="center">

**If this project helped you, give it a star.**

*The goal is not to replace doctors. The goal is to make sure no patient waits two weeks for an answer that AI could have flagged in two seconds.*

</div>
