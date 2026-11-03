
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model import load_model, predict
from convolution import generate_signal, calculate_convolution


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="NeuroConvolve",
    page_icon="〰️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

.stApp {
    background:
        radial-gradient(ellipse at 5% 5%, #35205D 0%, transparent 35%),
        radial-gradient(ellipse at 95% 10%, #123B5B 0%, transparent 35%),
        #101225;
    color: #F5F3FF;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #201B40, #151A34);
    border-right: 1px solid #514078;
}

[data-testid="stSidebar"] * {
    color: #F5F3FF;
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: white !important;
}

.hero {
    padding: 35px 30px;
    border-radius: 25px;
    background: linear-gradient(
        115deg,
        rgba(124,58,237,0.4),
        rgba(219,39,119,0.3),
        rgba(14,165,233,0.3)
    );
    border: 1px solid rgba(196,181,253,0.35);
    box-shadow: 0 15px 50px rgba(0,0,0,0.25);
    margin-bottom: 28px;
}

.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 43px;
    font-weight: 800;
    background: linear-gradient(90deg,#C4B5FD,#F9A8D4,#7DD3FC);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    color: #DDD6FE;
    font-size: 16px;
    margin-top: 8px;
    line-height: 1.7;
}

.section-heading {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 23px;
    font-weight: 700;
    margin-top: 30px;
    margin-bottom: 15px;
    color: #E9D5FF;
}

.metric-card {
    padding: 20px;
    border-radius: 18px;
    background: linear-gradient(145deg,#272448,#202D4C);
    border: 1px solid rgba(167,139,250,0.3);
    box-shadow: 0 8px 25px rgba(0,0,0,0.2);
    min-height: 115px;
}

.metric-label {
    color: #C4B5FD;
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.metric-value {
    color: white;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 27px;
    font-weight: 800;
    margin-top: 8px;
}

.info-card {
    background: linear-gradient(135deg,#252044,#192C45);
    padding: 24px;
    border-radius: 18px;
    border: 1px solid rgba(56,189,248,0.3);
    line-height: 1.8;
    color: #F5F3FF;
}

.info-card h2 {
    color: #E9D5FF !important;
    margin-top: 0;
}

.footer {
    text-align: center;
    color: #A5A3C7;
    padding: 25px 0 10px 0;
    font-size: 13px;
}

div[data-testid="stDownloadButton"] button {
    background: linear-gradient(90deg,#0891B2,#2563EB);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 700;
}

hr {
    border-color: rgba(196,181,253,0.2);
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# HERO
# =====================================================

st.html("""
<div class="hero">
    <div class="hero-title">〰️ NeuroConvolve</div>
    <div class="hero-subtitle">
        Where Mathematical Convolution Meets Neural Networks.
        <br>
        Explore signals, calculate convolution and compare it
        with AI predictions.
    </div>
</div>
""")


# =====================================================
# CHART STYLING
# =====================================================

def style_axis(ax, title, xlabel="Time", ylabel="Amplitude"):

    ax.set_title(title, color="white", fontsize=13, pad=12)
    ax.set_xlabel(xlabel, color="white")
    ax.set_ylabel(ylabel, color="white")

    ax.tick_params(colors="white")
    ax.grid(alpha=0.2, linestyle="--")

    for spine in ax.spines.values():
        spine.set_color("#514078")


def show_figure(fig):

    fig.patch.set_facecolor("#20203D")
    fig.tight_layout(pad=2)

    st.pyplot(fig, width="stretch")

    plt.close(fig)


# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def get_model():
    return load_model("models/neuroconvolve_mlp.pth")


try:
    model = get_model()

except FileNotFoundError:
    st.error("Model not found: models/neuroconvolve_mlp.pth")
    st.stop()

except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.markdown("## 🎛️ Signal Studio")
st.sidebar.caption("Configure your input signals")

signal_options = ["Gaussian", "Sine", "Square", "Triangle"]

st.sidebar.markdown("### 🟣 First Signal")

signal1_type = st.sidebar.selectbox(
    "Choose Signal 1",
    signal_options,
    key="signal1_type"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔵 Second Signal")

signal2_type = st.sidebar.selectbox(
    "Choose Signal 2",
    signal_options,
    key="signal2_type"
)


# =====================================================
# SIGNAL CONTROLS
# =====================================================

def signal_controls(name, signal_type):

    st.sidebar.markdown(f"#### ⚙️ {name} Parameters")

    amplitude = st.sidebar.slider(
        f"{name} Amplitude",
        0.1, 3.0, 1.0, 0.1,
        key=f"{name}_amp"
    )

    if signal_type == "Gaussian":

        center = st.sidebar.slider(
            f"{name} Center",
            -3.0, 3.0, 0.0, 0.1,
            key=f"{name}_center"
        )

        width = st.sidebar.slider(
            f"{name} Width",
            0.2, 3.0, 1.0, 0.1,
            key=f"{name}_width"
        )

        return amplitude, 1.0, 0.0, center, width

    frequency = st.sidebar.slider(
        f"{name} Frequency",
        0.1, 3.0, 1.0, 0.1,
        key=f"{name}_freq"
    )

    phase = st.sidebar.slider(
        f"{name} Phase",
        0.0, 6.28, 0.0, 0.1,
        key=f"{name}_phase"
    )

    return amplitude, frequency, phase, 0.0, 1.0


amp1, freq1, phase1, center1, width1 = signal_controls(
    "Signal 1", signal1_type
)

st.sidebar.markdown("---")

amp2, freq2, phase2, center2, width2 = signal_controls(
    "Signal 2", signal2_type
)


# =====================================================
# GENERATE SIGNALS
# =====================================================

t = np.linspace(-5, 5, 64)

signal1 = generate_signal(
    t,
    signal1_type,
    amplitude=amp1,
    frequency=freq1,
    phase=phase1,
    center=center1,
    width=width1
)

signal2 = generate_signal(
    t,
    signal2_type,
    amplitude=amp2,
    frequency=freq2,
    phase=phase2,
    center=center2,
    width=width2
)


# =====================================================
# MATHEMATICAL CONVOLUTION
# =====================================================

conv_t, actual = calculate_convolution(signal1, signal2)


# =====================================================
# AI PREDICTION
# =====================================================

try:
    predicted = predict(model, signal1, signal2)

except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()


if len(actual) != len(predicted):
    st.error("Actual and predicted dimensions do not match.")
    st.stop()


# =====================================================
# METRICS
# =====================================================

error = actual - predicted

mae = np.mean(np.abs(error))
mse = np.mean(error ** 2)
rmse = np.sqrt(mse)


# =====================================================
# MODEL PERFORMANCE
# =====================================================

st.html("""
<div class="section-heading">📊 Model Performance</div>
""")

c1, c2, c3 = st.columns(3)

with c1:
    st.html(f"""
    <div class="metric-card">
        <div class="metric-label">💜 Mean Absolute Error</div>
        <div class="metric-value">{mae:.5f}</div>
    </div>
    """)

with c2:
    st.html(f"""
    <div class="metric-card">
        <div class="metric-label">💗 Mean Squared Error</div>
        <div class="metric-value">{mse:.5f}</div>
    </div>
    """)

with c3:
    st.html(f"""
    <div class="metric-card">
        <div class="metric-label">💙 Root Mean Squared Error</div>
        <div class="metric-value">{rmse:.5f}</div>
    </div>
    """)


# =====================================================
# INPUT SIGNAL VISUALIZATION
# =====================================================

st.html("""
<div class="section-heading">〰️ Input Signal Visualization</div>
""")

# Both signals are drawn in ONE figure.
# This avoids the missing second-column chart issue.

fig, axes = plt.subplots(
    1, 2,
    figsize=(12, 4)
)

fig.patch.set_facecolor("#20203D")

axes[0].set_facecolor("#20203D")
axes[1].set_facecolor("#20203D")

axes[0].plot(
    t,
    signal1,
    color="#C084FC",
    linewidth=2.5,
    label="Signal 1"
)

style_axis(
    axes[0],
    f"Signal 1 — {signal1_type}"
)

axes[0].legend(
    facecolor="#292747",
    labelcolor="white"
)

axes[1].plot(
    t,
    signal2,
    color="#38BDF8",
    linewidth=2.5,
    label="Signal 2"
)

style_axis(
    axes[1],
    f"Signal 2 — {signal2_type}"
)

axes[1].legend(
    facecolor="#292747",
    labelcolor="white"
)

show_figure(fig)


# =====================================================
# CONVOLUTION COMPARISON
# =====================================================

st.html("""
<div class="section-heading">
    🧠 Mathematical Convolution vs AI Prediction
</div>
""")

fig, ax = plt.subplots(figsize=(11, 5))

fig.patch.set_facecolor("#20203D")
ax.set_facecolor("#20203D")

ax.plot(
    conv_t,
    actual,
    color="#34D399",
    linewidth=2.5,
    label="Actual Convolution"
)

ax.plot(
    conv_t,
    predicted,
    color="#F472B6",
    linewidth=2.5,
    linestyle="--",
    label="AI Prediction"
)

style_axis(
    ax,
    "Actual vs AI Predicted Convolution",
    "Time Shift",
    "Convolution"
)

ax.legend(
    facecolor="#292747",
    edgecolor="#514078",
    labelcolor="white"
)

show_figure(fig)


# =====================================================
# ERROR ANALYSIS
# =====================================================

st.html("""
<div class="section-heading">
    📉 Prediction Error Analysis
</div>
""")

fig, ax = plt.subplots(figsize=(11, 4))

fig.patch.set_facecolor("#20203D")
ax.set_facecolor("#20203D")

ax.plot(
    conv_t,
    error,
    color="#FBBF24",
    linewidth=2
)

ax.axhline(
    0,
    color="white",
    linestyle="--",
    alpha=0.5
)

style_axis(
    ax,
    "Prediction Error",
    "Time Shift",
    "Actual - Predicted"
)

show_figure(fig)


# =====================================================
# NUMERICAL RESULTS
# =====================================================

st.html("""
<div class="section-heading">📋 Numerical Results</div>
""")

results = pd.DataFrame({
    "Time Shift": conv_t,
    "Actual Convolution": actual,
    "AI Prediction": predicted,
    "Error": error
})

st.dataframe(
    results.style.format({
        "Time Shift": "{:.3f}",
        "Actual Convolution": "{:.5f}",
        "AI Prediction": "{:.5f}",
        "Error": "{:.5f}"
    }),
    width="stretch",
    height=300
)


# =====================================================
# CSV DOWNLOAD
# =====================================================

csv = results.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇️ Download Results as CSV",
    data=csv,
    file_name="neuroconvolve_results.csv",
    mime="text/csv"
)


# =====================================================
# ABOUT
# =====================================================

st.markdown("---")

st.html("""
<div class="info-card">

    <h2>✨ About NeuroConvolve</h2>

    <p>
        NeuroConvolve combines numerical integration and
        deep learning to explore mathematical convolution.
    </p>

    <p>
        <b style="color:#C4B5FD;">Mathematical Engine</b>
        — Numerical convolution using trapezoidal integration.
    </p>

    <p>
        <b style="color:#F9A8D4;">AI Engine</b>
        — A trained PyTorch MLP predicts convolution outputs.
    </p>

    <p>
        <b style="color:#7DD3FC;">Visualization</b>
        — Matplotlib charts compare actual outputs with
        neural network predictions.
    </p>

</div>
""")


# =====================================================
# FOOTER
# =====================================================

st.html("""
<div class="footer">
    Built with Python • PyTorch • NumPy • Pandas • Matplotlib • Streamlit
    <br><br>
    NeuroConvolve | Mathematical Intelligence in Motion ✨
</div>
""")
