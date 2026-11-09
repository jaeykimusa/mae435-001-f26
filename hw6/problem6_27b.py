import numpy as np
import matplotlib.pyplot as plt
import control as ct

s = ct.tf("s")
L = (s + 2) / ((s - 1) * (s**2 + 2*s + 2))  # loop gain for K = 1

# Part (b): phase margins and closed-loop pole check
for K in [0.8, 1.2, 1.5, 2.0, 2.5]:
    _, pm, _, wgc = ct.margin(K * L)
    poles = ct.poles(ct.feedback(K * L, 1))
    max_real = np.max(poles.real)

    if np.isclose(max_real, 0, atol=1e-8):
        status = "stability boundary"
    else:
        status = "stable" if max_real < 0 else "unstable"

    if np.isfinite(wgc):
        print(f"K={K:.1f}: PM={pm:.2f}°, wgc={wgc:.3f} rad/s, {status}")
    else:
        print(f"K={K:.1f}: no gain crossover, {status}")