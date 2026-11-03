# 〰️ NeuroConvolve
### Where Mathematical Convolution Meets Neural Networks

**NeuroConvolve** is an interactive deep learning application that combines mathematical convolution with neural network prediction. It generates different types of signals, computes their actual convolution using numerical integration, and compares the result with predictions from a trained PyTorch neural network.

🌐 **Live Demo:** [Try NeuroConvolve](https://neuroconvolve-r9ipjwho4ympdk2mnvwffc.streamlit.app/)

---

## 🚀 Project Overview

Mathematical convolution is an important operation in signal processing, mathematics, and engineering.

NeuroConvolve explores how a neural network can learn the relationship between two input signals and predict their convolution output.

The application provides an interactive environment to:
- Generate and customize mathematical signals.
- Compute actual convolution using numerical integration.
- Predict convolution outputs using a trained neural network.
- Compare actual and predicted results visually.
- Evaluate prediction performance using error metrics.

## ✨ Features

- **Multiple Signal Types:** Gaussian, Sine, Square, and Triangle.
- **Customizable Parameters:** Amplitude, frequency, phase, center, and width.
- **Numerical Convolution:** Computes convolution using numerical integration with the trapezoidal rule.
- **Neural Network Prediction:** Uses a trained PyTorch Multi-Layer Perceptron (MLP).
- **Actual vs Predicted Comparison:** Visualizes mathematical and neural network outputs.
- **Error Visualization:** Displays the difference between actual and predicted convolution.
- **Performance Evaluation:** Calculates MAE, MSE, and RMSE.
- **CSV Export:** Allows users to download numerical prediction results.

## 🧠 Neural Network Architecture

NeuroConvolve uses a **Multi-Layer Perceptron (MLP)** implemented using PyTorch.

| Layer | Configuration |
|---|---|
| Input Layer | 128 features |
| Hidden Layer 1 | 256 neurons |
| Activation | ReLU |
| Hidden Layer 2 | 256 neurons |
| Activation | ReLU |
| Output Layer | 64 values |

**Input:** Two signals, each containing 64 data points, concatenated into a 128-dimensional vector.

**Output:** A predicted convolution containing 64 values.

The trained model learns to approximate the convolution operation from examples generated using numerical computation.

## 📊 Signal Types

NeuroConvolve supports four different signal types:

| Signal | Description |
|---|---|
| Gaussian | Bell-shaped continuous signal |
| Sine | Periodic sinusoidal signal |
| Square | Periodic square waveform |
| Triangle | Periodic triangular waveform |

The application supports all **16 ordered combinations** of these four signal types.

## 📈 Model Evaluation

The trained model was evaluated across all 16 ordered signal combinations using randomized test samples.

### Performance by Signal Combination

| Signal Combination | MAE | RMSE |
|---|---:|---:|
| Gaussian + Gaussian | 0.1064 | 0.1302 |
| Gaussian + Sine | 0.1538 | 0.1869 |
| Gaussian + Square | 0.2456 | 0.2985 |
| Gaussian + Triangle | 0.1404 | 0.1712 |
| Sine + Gaussian | 0.1470 | 0.1814 |
| Sine + Sine | 0.2450 | 0.2958 |
| Sine + Square | 0.3915 | 0.4712 |
| Sine + Triangle | 0.2148 | 0.2608 |
| Square + Gaussian | 0.2207 | 0.2682 |
| Square + Sine | 0.3848 | 0.4652 |
| Square + Square | 0.6542 | 0.8131 |
| Square + Triangle | 0.3443 | 0.4170 |
| Triangle + Gaussian | 0.1289 | 0.1568 |
| Triangle + Sine | 0.2148 | 0.2608 |
| Triangle + Square | 0.3582 | 0.4368 |
| Triangle + Triangle | 0.1761 | 0.2174 |

### Overall Evaluation

| Metric | Value |
|---|---:|
| Overall MAE | 0.2589 |
| Overall RMSE | 0.3156 |

These overall values summarize the randomized evaluation across the signal combinations.

### Example Application Prediction

The following metrics were observed for one application input:

| Metric | Value |
|---|---:|
| MAE | 0.29438 |
| MSE | 0.14099 |
| RMSE | 0.37549 |

*These are example-input metrics, not the overall evaluation results.*

## 🛠️ Technology Stack

- **Python** – Core programming language
- **PyTorch** – Neural network implementation and inference
- **NumPy** – Numerical computations and signal generation
- **Pandas** – Numerical result handling and CSV export
- **Matplotlib** – Data visualization
- **Streamlit** – Interactive web application
- **SciPy** – Scientific computing support

## 📂 Project Structure

```text
NeuroConvolve/
│
├── app.py
├── gradio_app.py
├── model.py
├── convolution.py
├── evaluate.py
├── test_convolution.py
├── test_model.py
│
├── models/
│   └── neuroconvolve_mlp.pth
│
├── screenshots/
│
├── requirements.txt
└── README.md
```

## ⚙️ Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/manmeetmanchanda10016/NeuroConvolve.git
cd NeuroConvolve
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## 🔬 Evaluation

The `evaluate.py` script evaluates the trained neural network across all 16 ordered combinations of the four supported signal types.

The evaluation uses MAE and RMSE to measure the difference between mathematically calculated convolution outputs and neural network predictions.

## 🎯 Project Objective

The objective of NeuroConvolve is to explore the application of deep learning to mathematical operations and investigate how neural networks can approximate computationally defined signal transformations.

It combines concepts from:
- Mathematical convolution
- Signal processing
- Numerical integration
- Neural networks
- Model evaluation
- Interactive machine learning applications

## 👨‍💻 Author

**Manmeet Singh Manchanda**

Computer Science and Engineering

---

**NeuroConvolve | Mathematical Intelligence in Motion**

[Live Application](https://neuroconvolve-r9ipjwho4ympdk2mnvwffc.streamlit.app/) · [GitHub Repository](https://github.com/manmeetmanchanda10016/NeuroConvolve)