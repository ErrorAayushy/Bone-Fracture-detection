---
title: Bone Fracture Detection AI
emoji: "??"
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
---

# Bone Fracture Detection AI

## Dataset Source:
Bone Fracture Multi-Region X-ray Dataset (Kaggle)
Link: https://www.kaggle.com/search?q=Bone+Fracture+Multi-Region+X-ray+Data

## Overview
This project presents a deep learning-based system for detecting bone fractures from X-ray images using an EfficientNetB3 + DenseNet121 ensemble.

## Features
- Binary classification (Fractured / Not Fractured)
- Explainability with Grad-CAM, Grad-CAM++, and Score-CAM
- Web-based interface using Gradio
- Real-time prediction with confidence score
- Batch prediction support
- Downloadable PDF clinical-style report

## Dataset
- Bone Fracture Multi-Region X-ray Dataset (Kaggle)
- ~9000+ images used

## Model
- EfficientNetB3 (Transfer Learning)
- DenseNet121 integration for ensemble-ready workflows

## Explainability Outputs
- Grad-CAM heatmap and overlay
- Grad-CAM++ heatmap and overlay
- Score-CAM heatmap and overlay

## Improvements Implemented
- Upgraded from a single-model setup to an EfficientNetB3 + DenseNet121 ensemble
- Added model-specific preprocessing (`300x300` for EfficientNetB3, `224x224` for DenseNet121)
- Added advanced explainability methods beyond basic Grad-CAM: Grad-CAM++, Score-CAM
- Added batch prediction workflow for faster multi-image screening
- Added downloadable PDF reports with prediction summary and full explainability visuals
- Improved UI with a clean medical-style layout and a user-friendly explanation panel

## Why This Is Better Than Normal Implementations
- Higher robustness: Ensemble averaging reduces single-model bias and improves prediction stability.
- Stronger explainability: Three CAM variants provide richer visual evidence than one heatmap.
- Workflow-ready UX: Batch mode + report export make the system practical for real use and presentation.
- Better interpretability: The explanation panel helps non-technical users understand predictions.
- Deployment-ready structure: Modular backend + Gradio app design is easier to maintain and extend.

## Model Performance

- Accuracy: 98.81%
- Precision: 98.74%
- Recall: 98.74%
- F1 Score: 98.74%

Confusion Matrix shows only 6 misclassifications out of 506 test samples.

## Demo
Temporary live demo:

## How to Run
```bash
pip install -r requirements.txt
python app.py
```

## Tech Stack
- Python
- PyTorch
- OpenCV
- Gradio

## Note
This project is for educational purposes only and not a substitute for professional medical diagnosis.
