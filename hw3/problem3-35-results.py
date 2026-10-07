# Course: MAE 435 (001) - Principles of Automatic Control
# Instructor: Prof. Atay
# Author: Jaey Kim (student)
# Date: 10/06/2026

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


def main():
    J = 600_000
    B = 20_000
    gains = [200, 400, 2000, 2000]
    time = np.linspace(0, 600, 60_001)
    colors = ["#2563eb", "#08916d", "#d66a13", "#d66a13"]

    fig, axes = plt.subplots(
        2, 2, figsize=(11.5, 7.8), sharex=True, sharey=True
    )
    print("K (N*m/rad)    Overshoot (%)    Approx. rise time, 1.8/omega_n (s)")

    for index, (K, ax, color) in enumerate(zip(gains, axes.flat, colors)):
        # Closed-loop transfer function: K / (J*s**2 + B*s + K).
        system = signal.TransferFunction([K], [J, B, K])
        t, theta = signal.step(system, T=time)

        # A unit reference step has final angle theta = 1 rad.
        overshoot = 100 * max(0.0, float(np.max(theta)) - 1.0)
        omega_n = np.sqrt(K / J)
        rise_time_approx = 1.8 / omega_n
        print(f"{K:<14} {overshoot:>10.3f} {rise_time_approx:>33.2f}")

        ax.plot(t, theta, color=color, linewidth=2)
        ax.axhline(1.0, color="#485461", linestyle="--", linewidth=0.9)
        title = f"K = {K} N·m/rad"
        if index == 3:
            title += " (repeated in prompt)"
        ax.set_title(title, fontweight="bold", fontsize=12, pad=12)
        ax.text(
            0.52, 0.93,
            f"Overshoot: {overshoot:.3f}%\nApprox. rise time: {rise_time_approx:.2f} s",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=4),
        )
        ax.set_xlim(0, 500)
        ax.set_ylim(0, 1.5)
        ax.set_xticks([0, 100, 200, 300, 400, 500])
        ax.set_yticks([0, 0.5, 1.0, 1.5])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Antenna angle (rad)")
        ax.tick_params(labelbottom=True, labelleft=True, labelsize=9)
        ax.grid(True, color="#e7eaee", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Satellite antenna — response to a 1-rad reference step",
        fontsize=17, fontweight="bold", y=0.99,
    )
    # fig.text(
    #     0.5, 0.015,
    #     "Dashed line: final angle = 1 rad.  Approximate rise time: "
    #     "t_r ≈ 1.8/ω_n, with ω_n = √(K/J).\n"
    #     "SciPy simulation of K / (600000 s² + 20000 s + K); "
    #     "the supplied prompt lists K = 2000 twice.",
    #     ha="center", fontsize=10, color="#485461",
    # )
    fig.tight_layout(rect=[0, 0.07, 1, 0.955], h_pad=2.4, w_pad=2)
    output = Path(__file__).resolve().with_name("problem3-35-results-plot.png")
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)
    print(f"\nSaved plot: {output}")


if __name__ == "__main__":
    main()
