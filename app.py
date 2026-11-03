
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from model import load_model, predict
from convolution import generate_signal, calculate_convolution


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="NeuroConvolve",
    page_icon="〰️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    --bg: #101225;
    --panel: #1B1E38;
    --purple: #A78BFA;
    --pink: #F472B6;
    --cyan: #38BDF8;
    --green: #34D399;
    --orange: #FBBF24;
    --text: #F5F3FF;
    --muted: #B6B5D0;
}

.stApp {
    background:
        radial-gradient(ellipse at 5% 5%, #35205D 0%, transparent 35%),
        radial-gradient(ellipse at 95% 10%, #123B5B 0%, transparent 35%),
        radial-gradient(ellipse at 50% 100%, #3D1D50 0%, transparent 40%),
        var(--bg);
    color: var(--text);
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
    color: #FFFFFF !important;
}

p, label, span {
    color: #D5D3EA;
}

.hero {
    padding: 35px 30px;
    border-radius: 25px;
    background: linear-gradient(
        115deg,
        rgba(124, 58, 237, 0.32),
        rgba(219, 39, 119, 0.22),
        rgba(14, 165, 233, 0.25)
    );
    border: 1px solid rgba(196, 181, 253, 0.3);
    box-shadow: 0 15px 50px rgba(0,0,0,0.25);
    margin-bottom: 28px;
}

.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 43px;
    font-weight: 800;
    background: linear-gradient(90deg, #C4B5FD, #F9A8D4, #7DD3FC);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
}

.hero-subtitle {
    color: #DDD6FE;
    font-size: 16px;
}

.section-heading {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 23px;
    font-weight: 700;
    margin-top: 25px;
    margin-bottom: 15px;
    color: #E9D5FF;
}

.metric-card {
    padding: 20px;
    border-radius: 18px;
    background: linear-gradient(145deg, #272448, #202D4C);
    border: 1px solid rgba(167, 139, 250, 0.3);
    box-shadow: 0 8px 25px rgba(0,0,0,0.2);
    min-height: 115px;
}

.metric-label {
    color: #C4B5FD;
    font-size: 13px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.metric-value {
    color: #FFFFFF;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 27px;
    font-weight: 800;
    margin-top: 8px;
}

.info-card {
    background: linear-gradient(135deg, #252044, #192C45);
    padding: 18px;
    border-radius: 16px;
    border: 1px solid rgba(56, 189, 248, 0.25);
    margin-bottom: 12px;
}

div[data-testid="stButton"] button {
    background: linear-gradient(90deg, #8B5CF6, #EC4899);
    color: white;
    border: none;
    border-radius: 12px;
    font-weight: 700;
    padding: 10px 20px;
    transition: 0.2s;
}

div[data-testid="stButton"] button:hover {
    background: linear-gradient(90deg, #A78BFA, #F472B6);
    box-shadow: 0 0 20px rgba(236, 72, 153, 0.35);
    transform: translateY(-2px);
}

div[data-testid="stDownloadButton"] button {
    background: linear-gradient(90deg, #0891B2, #2563EB);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
}

.stSelectbox, .stSlider {
    margin-bottom: 12px;
}

hr {
    border-color: rgba(196, 181, 253, 0.2);
}

.footer {
    text-align: center;
    color: #A5A3C7;
    padding: 25px 0 10px 0;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# HERO SECTION
# --------------------------------------------------

st.markdown("""
<div class="hero">
    <div class="hero-title">〰️ NeuroConvolve</div>
    <div class="hero-subtitle">
        Where Mathematical Convolution Meets Neural Networks.
        <br>
        Explore signals, calculate convolution and compare it with AI predictions.
    </div>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# MODEL LOADING
# --------------------------------------------------

@st.cache_resource
def get_model():
    return load_model()


try:
    model = get_model()
except FileNotFoundError:
    st.error(
        "Model file not found. Make sure "
        "`models/neuroconvolve_mlp.pth` exists."
    )
    st.stop()
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

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


# --------------------------------------------------
# SIGNAL CONTROLS
# --------------------------------------------------

def signal_controls(signal_name, signal_type):

    st.sidebar.markdown(f"#### ⚙️ {signal_name} Parameters")

    amplitude = st.sidebar.slider(
        f"{signal_name} Amplitude",
        0.1, 3.0, 1.0, 0.1,
        key=f"{signal_name}_amp"
    )

    if signal_type == "Gaussian":

        center = st.sidebar.slider(
            f"{signal_name} Center",
            -3.0, 3.0, 0.0, 0.1,
            key=f"{signal_name}_center"
        )

        width = st.sidebar.slider(
            f"{signal_name} Width",
            0.2, 3.0, 1.0, 0.1,
            key=f"{signal_name}_width"
        )

        return amplitude, 1.0, 0.0, center, width

    else:

        frequency = st.sidebar.slider(
            f"{signal_name} Frequency",
            0.1, 3.0, 1.0, 0.1,
            key=f"{signal_name}_freq"
        )

        phase = st.sidebar.slider(
            f"{signal_name} Phase",
            0.0, 6.28, 0.0, 0.1,
            key=f"{signal_name}_phase"
        )

        return amplitude, frequency, phase, 0.0, 1.0


amp1, freq1, phase1, center1, width1 = signal_controls(
    "Signal 1", signal1_type
)

amp2, freq2, phase2, center2, width2 = signal_controls(
    "Signal 2", signal2_type
)


# --------------------------------------------------
# GENERATE SIGNALS
# --------------------------------------------------

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


# --------------------------------------------------
# CONVOLUTION
# --------------------------------------------------

conv_t, actual = calculate_convolution(signal1, signal2)

predicted = predict(model, signal1, signal2)

if len(actual) != len(predicted):
    st.error("Prediction and actual output dimensions do not match.")
    st.stop()

error = actual - predicted

mae = np.mean(np.abs(error))
mse = np.mean(error ** 2)
rmse = np.sqrt(mse)


# --------------------------------------------------
# METRICS
# --------------------------------------------------

st.markdown(
    '<div class="section-heading">📊 Model Performance</div>',
    unsafe_allow_html=True
)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">💜 Mean Absolute Error</div>
        <div class="metric-value">{mae:.5f}</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">💗 Mean Squared Error</div>
        <div class="metric-value">{mse:.5f}</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">💙 Root Mean Squared Error</div>
        <div class="metric-value">{rmse:.5f}</div>
    </div>
    """, unsafe_allow_html=True)


# --------------------------------------------------
# SIGNAL VISUALIZATION
# --------------------------------------------------

st.markdown(
    '<div class="section-heading">〰️ Input Signal Visualization</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

def make_signal_plot(x, y, title, color, fillcolor):

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode="lines",
        name=title,
        line=dict(color=color, width=3),
        fill="tozeroy",
        fillcolor=fillcolor
    ))

    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=340,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#F5F3FF"),
        margin=dict(l=15, r=15, t=50, b=15),
        xaxis_title="Time",
        yaxis_title="Amplitude",
        showlegend=False
    )

    return fig


with col1:
    st.plotly_chart(
        make_signal_plot(
            t, signal1, f"Signal 1 — {signal1_type}",
            "#C084FC", "rgba(192,132,252,0.15)"
        ),
        use_container_width=True
    )

with col2:
    st.plotly_chart(
        make_signal_plot(
            t, signal2, f"Signal 2 — {signal2_type}",
            "#38BDF8", "rgba(56,189,248,0.15)"
        ),
        use_container_width=True
    )


# --------------------------------------------------
# CONVOLUTION COMPARISON
# --------------------------------------------------

st.markdown(
    '<div class="section-heading">🧠 Mathematical Convolution vs AI Prediction</div>',
    unsafe_allow_html=True
)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=conv_t,
    y=actual,
    mode="lines",
    name="Actual Convolution",
    line=dict(color="#34D399", width=3)
))

fig.add_trace(go.Scatter(
    x=conv_t,
    y=predicted,
    mode="lines",
    name="Neural Network Prediction",
    line=dict(color="#F472B6", width=3, dash="dash")
))

fig.update_layout(
    template="plotly_dark",
    height=440,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#F5F3FF"),
    xaxis_title="Time Shift",
    yaxis_title="Convolution Value",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    margin=dict(l=15, r=15, t=50, b=15)
)

st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------
# ERROR ANALYSIS
# --------------------------------------------------

st.markdown(
    '<div class="section-heading">📉 Prediction Error Analysis</div>',
    unsafe_allow_html=True
)

error_fig = go.Figure()

error_fig.add_trace(go.Bar(
    x=conv_t,
    y=error,
    name="Prediction Error",
    marker=dict(
        color=error,
        colorscale="Plasma",
        showscale=True,
        colorbar=dict(title="Error")
    )
))

error_fig.add_hline(
    y=0,
    line_dash="dash",
    line_color="#FFFFFF",
    opacity=0.6
)

error_fig.update_layout(
    template="plotly_dark",
    height=320,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#F5F3FF"),
    xaxis_title="Time Shift",
    yaxis_title="Actual - Predicted",
    margin=dict(l=15, r=15, t=30, b=15)
)

st.plotly_chart(error_fig, use_container_width=True)


# --------------------------------------------------
# DATA TABLE
# --------------------------------------------------

st.markdown(
    '<div class="section-heading">📋 Numerical Results</div>',
    unsafe_allow_html=True
)

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
    use_container_width=True,
    height=300
)


# --------------------------------------------------
# DOWNLOAD
# --------------------------------------------------

csv = results.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇️ Download Results as CSV",
    data=csv,
    file_name="neuroconvolve_results.csv",
    mime="text/csv"
)


# --------------------------------------------------
# PROJECT INFORMATION
# --------------------------------------------------

st.markdown("---")

st.markdown("""
<div class="info-card">
    <h3>✨ About NeuroConvolve</h3>
    NeuroConvolve combines numerical integration and deep learning
    to explore mathematical convolution.

    <br><br>

    <b style="color:#C4B5FD;">Mathematical Engine</b> — Numerical convolution
    using trapezoidal integration.

    <br>

    <b style="color:#F9A8D4;">AI Engine</b> — A trained PyTorch MLP
    predicts the convolution output.

    <br>

    <b style="color:#7DD3FC;">Visualization</b> — Interactive Plotly
    charts compare the actual output with the model prediction.
</div>
""", unsafe_allow_html=True)


st.markdown("""
<div class="footer">
    Built with Python • PyTorch • NumPy • Streamlit • Plotly
    <br>
    NeuroConvolve | Mathematical Intelligence in Motion ✨
</div>
""", unsafe_allow_html=True)
