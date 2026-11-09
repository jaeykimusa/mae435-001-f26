import numpy as np
import matplotlib.pyplot as plt

# D(s) = K (1+s/z)/(1+s/p), where K is the DC gain.
p = np.sqrt(3)
z = 1/p
K_design = 2/p
w = np.logspace(-2, 2, 1600)
s = 1j*w
G = -1/(1+w**2)
D = K_design*(1+s/z)/(1+s/p)
L = D*G

fig, (mag, phase) = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                                         constrained_layout=True)
for response, label in [(G, 'Plant G'), (D, 'Lead D'), (L, 'Loop DG')]:
    mag.semilogx(w, 20*np.log10(abs(response)), label=label)
phase.semilogx(w, np.full_like(w, -180))
phase.semilogx(w, np.angle(D, deg=True))
phase.semilogx(w, -180 + np.angle(D, deg=True))
for ax, level in [(mag, 0), (phase, -180)]:
    ax.axhline(level, color='gray', ls=':', lw=1)
    ax.axvline(1, color='gray', ls='--', lw=1)
mag.plot(1, 0, 'ko')
phase.plot(1, -150, 'ko')
phase.annotate('PM = 30°', (1, -150), xytext=(8, 8), textcoords='offset points')
mag.set(title='Inverted pendulum: lead compensation', ylabel='Magnitude (dB)')
mag.legend(ncol=3)
phase.set(xlabel='Frequency (rad/s)', ylabel='Phase (deg)')

for ax in (mag, phase):
    ax.grid(True, which='both', alpha=.2)
fig.savefig('pendulum_bode.png', dpi=150)
plt.show()
