# 〰️ NeuroConvolve — Mathematical Convolution Meets Deep Learning

**NeuroConvolve** is an interactive mathematical convolution simulator that combines numerical computation with deep learning. It allows users to generate different types of signals, calculate their mathematical convolution, and compare the actual output with predictions from a trained PyTorch neural network.

The project explores how neural networks can learn mathematical operations through synthetic data generation and model training.

## 🚀 Features

- **Interactive Signal Generation:** Generate Gaussian, Sine, Square, and Triangle signals.
- **Customizable Parameters:** Adjust amplitude, frequency, phase, center, and width.
- **Numerical Convolution:** Calculate convolution using numerical integration and the trapezoidal rule.
- **Deep Learning Prediction:** Use a trained PyTorch MLP to predict convolution outputs.
- **Model Comparison:** Experiment with different neural network architectures, including MLPs and CNNs.
- **Interactive Dashboard:** Visualize input signals, actual convolution, AI predictions, and prediction errors.
- **Performance Evaluation:** Calculate MAE, MSE, and RMSE.
- **Data Export:** Download numerical results as CSV files.

## 🧠 How It Works

NeuroConvolve follows three main stages:

### 1. Signal Generation

Two input signals are generated using configurable parameters.

Supported signals:

| Signal | Description |
|---|---|
| Gaussian | Smooth bell-shaped signal |
| Sine | Periodic sinusoidal signal |
| Square | Periodic square waveform |
| Triangle | Periodic triangular waveform |

### 2. Mathematical Convolution

The actual convolution is calculated numerically using the convolution integral:

\[
(f*g)(t)=\int_{-\infty}^{\infty}f(\tau)g(t-\tau)d\tau
\]

The implementation approximates this integral over a finite sampled interval using NumPy interpolation and trapezoidal integration.

### 3. Neural Network Prediction

A trained PyTorch Multi-Layer Perceptron (MLP) learns the mapping between two input signals and their convolution output.

**MLP Architecture:**

| Layer | Configuration |
|---|---|
| Input | 128 neurons |
| Hidden Layer 1 | 256 neurons + ReLU |
| Hidden Layer 2 | 256 neurons + ReLU |
| Output | 64 neurons |

The network receives two 64-point signals concatenated into a 128-dimensional input and predicts a 64-point output.

## 📊 Model Experimentation

Multiple architectures were explored to understand how different neural networks learn convolution across different signal distributions.

Models experimented with:

- Gaussian-only MLP
- Mixed-signal MLP
- Original CNN
- Improved CNN

**Selected model:** Mixed-signal MLP, trained on Gaussian, Sine, Square, and Triangle signals.

### Model Evaluation

The following metrics were used to evaluate prediction performance:

- **MAE (Mean Absolute Error):** Average absolute difference between actual and predicted values.
- **MSE (Mean Squared Error):** Average squared prediction error.
- **RMSE (Root Mean Squared Error):** Square root of the mean squared error.

The selected mixed-signal MLP achieved the following recorded evaluation results:

| Metric | Value |
|---|---:|
| MAE | 0.495320 |
| MSE | 0.763817 |
| RMSE | 0.873966 |

*These results are from the recorded Colab evaluation. The Streamlit dashboard calculates metrics dynamically for the selected input signals.*

## 🖥️ Application Dashboard

The Streamlit interface provides:

- Interactive signal parameter controls.
- Side-by-side input signal visualizations.
- Actual convolution vs. neural network prediction.
- Prediction error visualization.
- Numerical results table.
- CSV export functionality.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| NumPy | Numerical computations and signal generation |
| SciPy | Scientific computing and experimentation |
| PyTorch | Neural network development and training |
| Pandas | Data handling and result tables |
| Plotly | Interactive visualizations |
| Streamlit | Web application interface |
| Scikit-learn | Model evaluation and experimentation |
| Google Colab | Model development and experimentation |
| Git & GitHub | Version control |

## 📁 Project Structure

```text
NeuroConvolve/
│
├── models/
│   └── neuroconvolve_mlp.pth
│
├── app.py
├── model.py
├── convolution.py
├── test_model.py
├── requirements.txt
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd MKL
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

**Windows:**

```powershell
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## 📈 Future Improvements

- Experiment with additional neural network architectures.
- Improve model generalization across different signal parameters.
- Explore deeper CNN architectures and alternative learning approaches.
- Add real-time signal animation.
- Expand the dataset with more diverse signal combinations.
- Optimize prediction accuracy and inference performance.

## 🎯 Project Objective

The primary objective of NeuroConvolve is to explore the relationship between mathematical operations and deep learning by training neural networks to approximate convolution outputs.

It provides an interactive environment to understand signal processing, numerical methods, neural networks, and model evaluation.

---

**Built with Python, PyTorch, NumPy, and Streamlit.**

*NeuroConvolve — Mathematical Intelligence in Motion.*