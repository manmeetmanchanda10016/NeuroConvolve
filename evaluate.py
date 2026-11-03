
import numpy as np
from itertools import product

from model import load_model, predict
from convolution import generate_signal, calculate_convolution

signal_types = ["Gaussian", "Sine", "Square", "Triangle"]

t = np.linspace(-5, 5, 64)
model = load_model()

rng = np.random.default_rng(42)
results = []

print("\nROBUST MODEL EVALUATION — 100 SAMPLES PER COMBINATION\n")
print(f"{'Signal Pair':<30} {'MAE':>12} {'RMSE':>12}")
print("-" * 58)

for type1, type2 in product(signal_types, repeat=2):

    maes = []
    rmses = []

    for _ in range(100):

        amp1, amp2 = rng.uniform(0.5, 2, size=2)
        freq1, freq2 = rng.uniform(0.5, 2, size=2)
        phase1, phase2 = rng.uniform(0, 2 * np.pi, size=2)

        center1, center2 = rng.uniform(-2, 2, size=2)
        width1, width2 = rng.uniform(0.5, 2, size=2)

        signal1 = generate_signal(
            t, type1, amplitude=amp1, frequency=freq1,
            phase=phase1, center=center1, width=width1
        )

        signal2 = generate_signal(
            t, type2, amplitude=amp2, frequency=freq2,
            phase=phase2, center=center2, width=width2
        )

        prediction = predict(model, signal1, signal2)
        _, actual = calculate_convolution(signal1, signal2)

        maes.append(np.mean(np.abs(actual - prediction)))
        rmses.append(np.sqrt(np.mean((actual - prediction) ** 2)))

    avg_mae = np.mean(maes)
    avg_rmse = np.mean(rmses)

    results.append((type1, type2, avg_mae, avg_rmse))

    print(f"{type1 + ' + ' + type2:<30} {avg_mae:>12.4f} {avg_rmse:>12.4f}")

print("-" * 58)

print("\nOVERALL AVERAGE MAE:", round(np.mean([r[2] for r in results]), 4))
print("OVERALL AVERAGE RMSE:", round(np.mean([r[3] for r in results]), 4))

worst = max(results, key=lambda x: x[2])
best = min(results, key=lambda x: x[2])

print("\nBEST:", best)
print("WORST:", worst)
