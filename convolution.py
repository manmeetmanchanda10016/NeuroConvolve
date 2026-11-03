
import numpy as np


def generate_signal(
    t,
    signal_type,
    amplitude=1,
    frequency=1,
    phase=0,
    center=0,
    width=1
):
    if signal_type == "Gaussian":
        return amplitude * np.exp(-((t - center) / width) ** 2)

    elif signal_type == "Sine":
        return amplitude * np.sin(
            2 * np.pi * frequency * t / 5 + phase
        )

    elif signal_type == "Square":
        return amplitude * np.where(
            np.sin(2 * np.pi * frequency * t / 5 + phase) >= 0,
            1,
            -1
        )

    elif signal_type == "Triangle":
        return amplitude * (2 / np.pi) * np.arcsin(
            np.sin(2 * np.pi * frequency * t / 5 + phase)
        )

    else:
        raise ValueError("Unsupported signal type")


def calculate_convolution(signal1, signal2):
    tau = np.linspace(-5, 5, 64)
    t_values = np.linspace(-5, 5, 64)

    convolution = []

    for shift in t_values:
        shifted_signal = np.interp(
            shift - tau,
            tau,
            signal2,
            left=0.0,
            right=0.0
        )

        result = np.trapezoid(
            signal1 * shifted_signal,
            tau
        )

        convolution.append(result)

    return t_values, np.array(convolution)
