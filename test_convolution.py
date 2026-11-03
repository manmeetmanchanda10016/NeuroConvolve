import numpy as np

from convolution import (
    generate_signal,
    calculate_convolution
)

t = np.linspace(-5, 5, 64)

signal1 = generate_signal(t, "Gaussian")
signal2 = generate_signal(t, "Sine")

shifts, result = calculate_convolution(
    signal1,
    signal2
)

print("Convolution calculated successfully!")
print("Output shape:", result.shape)
print("First five values:", result[:5])