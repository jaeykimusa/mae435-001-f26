import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

system = signal.TransferFunction([0.1], [1, 0.373879, 0.1])
w = np.logspace(-1, 1, 5000)  # 0.1 to 10 rad/s
w, H = signal.freqresp(system, w=w)

gain = np.abs(H)
peak = np.argmax(gain)
print(f"Peak gain = {gain[peak]:.4f} rad/V at {w[peak]:.4f} rad/s")

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, constrained_layout=True)
ax1.loglog(w, gain)
ax1.plot(w[peak], gain[peak], "ro", label=f"Peak at {w[peak]:.4f} rad/s")
ax1.set_ylabel("Magnitude (rad/V)")
ax1.set_title("Voltmeter frequency response")
ax1.legend()

ax2.semilogx(w, np.angle(H, deg=True))
ax2.set(xlabel="Angular frequency (rad/s)", ylabel="Phase (deg)")

for ax in (ax1, ax2):
    ax.grid(True, which="both", alpha=0.3)

plt.show()