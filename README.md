# Bone Fracture Detection AI

## Dataset Source:
Bone Fracture Multi-Region X-ray Dataset (Kaggle)
Link: https://www.kaggle.com/search?q=Bone+Fracture+Multi-Region+X-ray+Data

## Overview
This project presents a deep learning-based system for detecting bone fractures from X-ray images using EfficientNetB3.

## Features
- Binary classification (Fractured / Not Fractured)
- Grad-CAM visualization for explainability
- Web-based interface using Gradio
- Real-time prediction with confidence score

## Dataset
- Bone Fracture Multi-Region X-ray Dataset (Kaggle)
- ~9000+ images used

## Model
- EfficientNetB3 (Transfer Learning)
- Accuracy: ~97%

## Results
- High accuracy with strong generalization
- Grad-CAM highlights fracture regions clearly

## Demo
Temporary live demo:
[Paste your gradio.live link here]

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
