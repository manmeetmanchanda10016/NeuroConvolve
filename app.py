
import os
import tempfile

import gradio as gr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model import load_model, predict
from convolution import generate_signal, calculate_convolution


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

model = load_model("models/neuroconvolve_mlp.pth")


# ==========================================
# NEUROCONVOLVE ENGINE
# ==========================================

def run_neuroconvolve(
    signal1_type,
    signal2_type,
    amp1,
    amp2,
    freq1,
    freq2,
    phase1,
    phase2,
    center1,
    center2,
    width1,
    width2
):

    t = np.linspace(-5, 5, 64)

    # Generate first signal
    signal1 = generate_signal(
        t,
        signal1_type,
        amplitude=amp1,
        frequency=freq1,
        phase=phase1,
        center=center1,
        width=width1
    )

    # Generate second signal
    signal2 = generate_signal(
        t,
        signal2_type,
        amplitude=amp2,
        frequency=freq2,
        phase=phase2,
        center=center2,
        width=width2
    )

    # Mathematical convolution
    conv_t, actual = calculate_convolution(
        signal1,
        signal2
    )

    # Neural network prediction
    predicted = predict(
        model,
        signal1,
        signal2
    )

    # Prediction error
    error = actual - predicted

    # Performance metrics
    mae = float(np.mean(np.abs(error)))
    mse = float(np.mean(error ** 2))
    rmse = float(np.sqrt(mse))

    # ======================================
    # VISUALIZATION
    # ======================================

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(11, 12)
    )

    fig.patch.set_facecolor("#111827")

    for ax in axes:

        ax.set_facecolor("#1E293B")

        ax.tick_params(colors="white")

        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")

        ax.grid(
            alpha=0.2,
            linestyle="--"
        )

        for spine in ax.spines.values():
            spine.set_color("#475569")

    # Input signals
    axes[0].plot(
        t,
        signal1,
        color="#C084FC",
        linewidth=2.5,
        label="Signal 1"
    )

    axes[0].plot(
        t,
        signal2,
        color="#38BDF8",
        linewidth=2.5,
        label="Signal 2"
    )

    axes[0].set_title(
        "Input Signal Visualization"
    )

    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Amplitude")

    axes[0].legend(
        facecolor="#1E293B",
        labelcolor="white"
    )

    # Actual vs predicted convolution
    axes[1].plot(
        conv_t,
        actual,
        color="#34D399",
        linewidth=2.5,
        label="Actual Convolution"
    )

    axes[1].plot(
        conv_t,
        predicted,
        "--",
        color="#F472B6",
        linewidth=2.5,
        label="AI Prediction"
    )

    axes[1].set_title(
        "Mathematical Convolution vs AI Prediction"
    )

    axes[1].set_xlabel("Time Shift")
    axes[1].set_ylabel("Convolution")

    axes[1].legend(
        facecolor="#1E293B",
        labelcolor="white"
    )

    # Prediction error
    axes[2].plot(
        conv_t,
        error,
        color="#FBBF24",
        linewidth=2
    )

    axes[2].axhline(
        0,
        color="white",
        linestyle="--",
        alpha=0.5
    )

    axes[2].set_title(
        "Prediction Error Analysis"
    )

    axes[2].set_xlabel("Time Shift")
    axes[2].set_ylabel("Actual - Predicted")

    fig.tight_layout(pad=2)

    # ======================================
    # NUMERICAL RESULTS
    # ======================================

    results = pd.DataFrame({
        "Time Shift": conv_t,
        "Actual Convolution": actual,
        "AI Prediction": predicted,
        "Error": error
    }).round(5)

    # Create a downloadable CSV file
    file_descriptor, csv_path = tempfile.mkstemp(
        suffix=".csv"
    )

    os.close(file_descriptor)

    results.to_csv(
        csv_path,
        index=False
    )

    return (
        fig,
        f"{mae:.5f}",
        f"{mse:.5f}",
        f"{rmse:.5f}",
        results,
        csv_path
    )


# ==========================================
# DARK THEME CSS
# ==========================================

custom_css = """

.gradio-container {
    background:
        radial-gradient(
            circle at top left,
            #31245A,
            #111827 55%,
            #0B172A
        ) !important;

    color: #F8FAFC !important;
}

/* Main containers */

.gradio-container .block,
.gradio-container .form,
.gradio-container .panel,
.gradio-container .wrap,
.gradio-container .gr-box,
.gradio-container .gr-group {

    background-color: #20203D !important;
    border-color: #514078 !important;
}

/* Input fields */

.gradio-container input,
.gradio-container textarea,
.gradio-container select {

    background-color: #292747 !important;
    color: #F8FAFC !important;
    border-color: #514078 !important;
}

/* Dropdown menus */

.gradio-container [role="listbox"],
.gradio-container [role="option"] {

    background-color: #292747 !important;
    color: #F8FAFC !important;
}

/* Text and headings */

.gradio-container h1,
.gradio-container h2,
.gradio-container h3,
.gradio-container p,
.gradio-container label {

    color: #E9D5FF !important;
}

/* Header */

#main-title {

    text-align: center;
    padding: 30px;
    border-radius: 20px;

    background:
        linear-gradient(
            120deg,
            #31245A,
            #1E3A5F
        ) !important;

    border: 1px solid #6552A3;
    margin-bottom: 20px;
}

#main-title h1 {

    font-size: 42px;
    font-weight: 800;

    background:
        linear-gradient(
            90deg,
            #C4B5FD,
            #F9A8D4,
            #7DD3FC
        );

    background-clip: text;
    -webkit-background-clip: text;
    color: transparent !important;
}

/* Metric cards */

.metric-box {

    background:
        linear-gradient(
            145deg,
            #272448,
            #202D4C
        ) !important;

    border: 1px solid #514078 !important;
    border-radius: 15px !important;
}

/* Primary button */

button.primary {

    background:
        linear-gradient(
            90deg,
            #7C3AED,
            #2563EB
        ) !important;

    color: white !important;
    border: none !important;
    font-weight: bold !important;
}

/* Sliders */

input[type="range"] {
    accent-color: #A78BFA !important;
}

/* Hide footer */

footer {
    display: none !important;
}

"""


# ==========================================
# THEME
# ==========================================

theme = gr.themes.Base(
    primary_hue="violet",
    secondary_hue="blue",
    neutral_hue="slate"
)


# ==========================================
# GRADIO APPLICATION
# ==========================================

with gr.Blocks(
    title="NeuroConvolve | Signal Studio",
    theme=theme,
    css=custom_css
) as demo:

    # Header
    with gr.Column(elem_id="main-title"):

        gr.Markdown(
            "# 〰️ NeuroConvolve"
        )

        gr.Markdown(
            "Where Mathematical Convolution Meets Neural Networks."
            "\n\nExplore signals, calculate convolution and compare mathematical results with AI predictions."
        )

    # Signal selection
    gr.Markdown("## 🎛️ Signal Studio")

    with gr.Row():

        signal1_type = gr.Dropdown(
            ["Gaussian", "Sine", "Square", "Triangle"],
            value="Sine",
            label="🟣 First Signal"
        )

        signal2_type = gr.Dropdown(
            ["Gaussian", "Sine", "Square", "Triangle"],
            value="Sine",
            label="🔵 Second Signal"
        )

    # Signal parameters
    gr.Markdown("## ⚙️ Signal Parameters")

    with gr.Row():

        with gr.Column():

            gr.Markdown("### 🟣 Signal 1")

            amp1 = gr.Slider(
                0.1, 3,
                value=1,
                label="Amplitude"
            )

            freq1 = gr.Slider(
                0.1, 3,
                value=1,
                label="Frequency"
            )

            phase1 = gr.Slider(
                0, 6.28,
                value=0,
                label="Phase"
            )

            center1 = gr.Slider(
                -3, 3,
                value=0,
                label="Center"
            )

            width1 = gr.Slider(
                0.2, 3,
                value=1,
                label="Width"
            )

        with gr.Column():

            gr.Markdown("### 🔵 Signal 2")

            amp2 = gr.Slider(
                0.1, 3,
                value=1,
                label="Amplitude"
            )

            freq2 = gr.Slider(
                0.1, 3,
                value=1,
                label="Frequency"
            )

            phase2 = gr.Slider(
                0, 6.28,
                value=0,
                label="Phase"
            )

            center2 = gr.Slider(
                -3, 3,
                value=0,
                label="Center"
            )

            width2 = gr.Slider(
                0.2, 3,
                value=1,
                label="Width"
            )

    # Calculate button
    run_button = gr.Button(
        "🚀 Calculate Convolution",
        variant="primary",
        size="lg"
    )

    # Performance metrics
    gr.Markdown("## 📊 Model Performance")

    with gr.Row():

        mae_output = gr.Textbox(
            label="💜 Mean Absolute Error",
            interactive=False,
            elem_classes="metric-box"
        )

        mse_output = gr.Textbox(
            label="💗 Mean Squared Error",
            interactive=False,
            elem_classes="metric-box"
        )

        rmse_output = gr.Textbox(
            label="💙 Root Mean Squared Error",
            interactive=False,
            elem_classes="metric-box"
        )

    # Visualizations
    gr.Markdown("## 🧠 NeuroConvolve Visualizations")

    plot = gr.Plot(
        label="Signal and Convolution Analysis"
    )

    # Numerical results
    gr.Markdown("## 📋 Numerical Results")

    results_table = gr.Dataframe(
        interactive=False,
        label="Convolution Results"
    )

    download_file = gr.File(
        label="⬇️ Download Results as CSV"
    )

    # Connect function
    run_button.click(
        fn=run_neuroconvolve,

        inputs=[
            signal1_type,
            signal2_type,
            amp1,
            amp2,
            freq1,
            freq2,
            phase1,
            phase2,
            center1,
            center2,
            width1,
            width2
        ],

        outputs=[
            plot,
            mae_output,
            mse_output,
            rmse_output,
            results_table,
            download_file
        ]
    )

    # Footer
    gr.Markdown(
        """
        ---

        <center>

        Built with Python · PyTorch · NumPy · Pandas · Matplotlib · Gradio

        **NeuroConvolve | Mathematical Intelligence in Motion**

        </center>
        """
    )


# ==========================================
# LAUNCH
# ==========================================

if __name__ == "__main__":

    demo.launch()
