import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

system = signal.TransferFunction([0.1], [1, 0.373879, 0.1])
omega = 2.0  # input frequency, rad/s

# Figure 1: Bode plot with the response at 2 rad/s marked
w = np.logspace(-1, 1, 1000)
_, H = signal.freqresp(system, w=w)
_, H_at_omega = signal.freqresp(system, w=[omega])

gain = abs(H_at_omega[0])
phase = np.angle(H_at_omega[0], deg=True)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, constrained_layout=True)
ax1.loglog(w, np.abs(H))
ax1.set(title="Voltmeter Bode plot", ylabel="Magnitude (rad/V)")
ax2.semilogx(w, np.angle(H, deg=True))
ax2.set(xlabel="Frequency (rad/s)", ylabel="Phase (deg)")

for ax, value, label in [(ax1, gain, f"{gain:.5f} rad/V"),
                         (ax2, phase, f"{phase:.2f}°")]:
    ax.axvline(omega, color="tab:red", linestyle="--", linewidth=1)
    ax.hlines(value, w[0], omega, color="tab:red", linestyle="--", linewidth=1)
    ax.plot(omega, value, "o", color="tab:red", label=f"At 2 rad/s: {label}")
    ax.set_xlim(w[0], w[-1])
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()

# Figure 2: verify the Bode prediction with a 1-V sinusoidal input
t = np.linspace(0, 60, 12001)
v = np.sin(omega * t)
_, theta, _ = signal.lsim(system, U=v, T=t)
theta_predicted = gain * np.sin(omega * t + np.deg2rad(phase))

fig, ax = plt.subplots(constrained_layout=True)
ax.plot(t, theta, label="lsim output")
ax.plot(t, theta_predicted, "--", label="Bode prediction")
ax.set(xlim=(50, 60), xlabel="Time (s)", ylabel="Pointer angle (rad)",
       title="Steady-state verification")
ax.grid(alpha=0.25)
ax.legend()
plt.show()