import numpy as np
import matplotlib.pyplot as plt
import control as ct

s = ct.tf("s")
L = (s + 2) / ((s - 1) * (s**2 + 2*s + 2))  # loop gain for K = 1

# Part (a): Bode magnitude and phase for K = 1
w = np.logspace(-3, 2, 2000)
H = (1j*w + 2) / ((1j*w - 1) * ((1j*w)**2 + 2j*w + 2))

mag_db = 20 * np.log10(np.abs(H))
phase_deg = np.rad2deg(np.unwrap(np.angle(H)))
if phase_deg[0] > 0:
    phase_deg -= 360

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, constrained_layout=True)
ax1.semilogx(w, mag_db)
ax1.axhline(0, color="gray", linestyle="--")
ax1.plot(np.sqrt(2), -6.02, "ro", label="−6.02 dB at √2 rad/s")
ax1.set(ylabel="Magnitude (dB)", title="Bode plot for K = 1")
ax1.legend()

ax2.semilogx(w, phase_deg)
ax2.axhline(-180, color="gray", linestyle="--")
ax2.plot(np.sqrt(2), -180, "ro")
ax2.set(xlabel="Frequency (rad/s)", ylabel="Phase (deg)")

for ax in (ax1, ax2):
    ax.grid(True, which="both", alpha=0.25)
plt.show()