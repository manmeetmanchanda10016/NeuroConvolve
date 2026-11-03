
---
title: NeuroConvolve
emoji: "〰️"
colorFrom: purple
colorTo: blue
sdk: gradio
app_file: app.py
python_version: "3.11"
---

# 〰️ NeuroConvolve

### Where Mathematical Convolution Meets Neural Networks

NeuroConvolve is an interactive application that combines numerical convolution with deep learning to predict convolution outputs.

## Features

- Generate Gaussian, Sine, Square and Triangle signals.
- Configure signal amplitude, frequency, phase, center and width.
- Calculate mathematical convolution using numerical integration.
- Predict convolution outputs using a trained PyTorch MLP.
- Compare actual and AI-predicted convolution.
- Evaluate performance using MAE, MSE and RMSE.
- Visualize prediction errors.
- Download numerical results as CSV.

## Technology Stack

- Python
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Gradio

## Model Architecture

The neural network uses a Multi-Layer Perceptron (MLP).

- Input layer: 128 features
- Hidden layer 1: 256 neurons
- Hidden layer 2: 256 neurons
- Output layer: 64 values
- Activation: ReLU

The model accepts two 64-point signals and predicts their 64-point convolution output.

## Project Structure

```text
NeuroConvolve/
│
├── app.py
├── model.py
├── convolution.py
├── evaluate.py
├── requirements.txt
├── README.md
│
└── models/
    └── neuroconvolve_mlp.pth
```

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the application:

```bash
python app.py
```

## Performance Evaluation

The model can be evaluated across all 16 ordered combinations of the four signal types using `evaluate.py`.

## Author

Developed as a deep learning and mathematical computing project.

**NeuroConvolve | Mathematical Intelligence in Motion**
