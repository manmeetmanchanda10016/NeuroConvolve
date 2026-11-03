
import os
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from convolution import generate_signal, calculate_convolution
from model import load_model, predict


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "neuroconvolve_mlp.pth"

model = load_model(str(MODEL_PATH))


def run_neuroconvolve(
    type1, type2,
    amp1, freq1, phase1, center1, width1,
    amp2, freq2, phase2, center2, width2
):
    t = np.linspace(-5, 5, 64)

    signal1 = generate_signal(
        t, type1, amp1, freq1, phase1, center1, width1
    )

    signal2 = generate_signal(
        t, type2, amp2, freq2, phase2, center2, width2
    )

    t_values, actual = calculate_convolution(signal1, signal2)

    predicted = predict(model, signal1, signal2)

    error = actual - predicted

    mae = float(np.mean(np.abs(error)))
    mse = float(np.mean(error ** 2))
    rmse = float(np.sqrt(mse))

    fig, axes = plt.subplots(3, 1, figsize=(11, 10))

    axes[0].plot(t, signal1, label=f"Signal 1: {type1}")
    axes[0].plot(t, signal2, label=f"Signal 2: {type2}")
    axes[0].set_title("Input Signals")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        t_values, actual,
        label="Actual Convolution",
        linewidth=2
    )
    axes[1].plot(
        t_values, predicted,
        label="Neural Network Prediction",
        linestyle="--",
        linewidth=2
    )
    axes[1].set_title("Actual vs Predicted Convolution")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Amplitude")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t_values, error, label="Prediction Error")
    axes[2].axhline(0, linestyle="--", linewidth=1)
    axes[2].set_title("Prediction Error")
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("Error")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    results = pd.DataFrame({
        "Time": t_values,
        "Actual Convolution": actual,
        "Predicted Convolution": predicted,
        "Error": error
    })

    fd, csv_path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)

    results.to_csv(csv_path, index=False)

    return fig, f"{mae:.6f}", f"{mse:.6f}", f"{rmse:.6f}", results, csv_path


signal_types = ["Gaussian", "Sine", "Square", "Triangle"]

with gr.Blocks(
    theme=gr.themes.Soft(),
    title="NeuroConvolve"
) as demo:

    gr.Markdown(
        """
        # 〰️ NeuroConvolve

        ### Where Mathematical Convolution Meets Neural Networks

        Generate signals, calculate mathematical convolution,
        and compare the result with an AI-powered prediction.
        """
    )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Signal 1")

            type1 = gr.Dropdown(
                signal_types, value="Sine", label="Signal Type"
            )
            amp1 = gr.Slider(0.1, 5, value=1, label="Amplitude")
            freq1 = gr.Slider(0.1, 5, value=1, label="Frequency")
            phase1 = gr.Slider(0, 6.28, value=0, label="Phase")
            center1 = gr.Slider(-5, 5, value=0, label="Center")
            width1 = gr.Slider(0.1, 5, value=1, label="Width")

        with gr.Column():
            gr.Markdown("### Signal 2")

            type2 = gr.Dropdown(
                signal_types, value="Sine", label="Signal Type"
            )
            amp2 = gr.Slider(0.1, 5, value=1, label="Amplitude")
            freq2 = gr.Slider(0.1, 5, value=1, label="Frequency")
            phase2 = gr.Slider(0, 6.28, value=0, label="Phase")
            center2 = gr.Slider(-5, 5, value=0, label="Center")
            width2 = gr.Slider(0.1, 5, value=1, label="Width")

    run_button = gr.Button(
        "Run NeuroConvolve",
        variant="primary"
    )

    gr.Markdown("## Performance Metrics")

    with gr.Row():
        mae_output = gr.Textbox(label="MAE", interactive=False)
        mse_output = gr.Textbox(label="MSE", interactive=False)
        rmse_output = gr.Textbox(label="RMSE", interactive=False)

    gr.Markdown("## Visualization")

    plot_output = gr.Plot(label="Convolution Analysis")

    gr.Markdown("## Numerical Results")

    dataframe_output = gr.Dataframe(
        label="Actual vs Predicted Values",
        interactive=False
    )

    csv_output = gr.File(label="Download Results (CSV)")

    run_button.click(
        fn=run_neuroconvolve,
        inputs=[
            type1, type2,
            amp1, freq1, phase1, center1, width1,
            amp2, freq2, phase2, center2, width2
        ],
        outputs=[
            plot_output,
            mae_output,
            mse_output,
            rmse_output,
            dataframe_output,
            csv_output
        ]
    )

    gr.Markdown(
        """
        ---
        **NeuroConvolve | Mathematical Intelligence in Motion**

        Built using Python, PyTorch, NumPy, Pandas and Gradio.
        """
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
