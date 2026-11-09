import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

def plant(w):
    s = 1j*w
    return 5/(s*(1+s)*(1+s/5))

def lead(w):
    return (1+1j*w/1.5)/(1+1j*w/20)

def loop(w):
    return plant(w)*lead(w)

def closed(w):
    return loop(w)/(1+loop(w))

w = np.logspace(-1, 2, 1600)
db = lambda h: 20*np.log10(np.abs(h))
fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True, constrained_layout=True)
for response, label in [(plant, 'Plant G'), (lead, 'Lead Dc'), (loop, 'Loop Dc G')]:
    h = response(w)
    axes[0].semilogx(w, db(h), label=label)
    axes[1].semilogx(w, np.degrees(np.unwrap(np.angle(h))))
for response, color in [(plant, 'C0'), (loop, 'C2')]:
    wc = brentq(lambda x: abs(response(x))-1, 1, 5)
    angle = np.degrees(np.angle(response(wc)))
    axes[0].plot(wc, 0, 'o', color=color)
    axes[1].plot(wc, angle, 'o', color=color)
bw = brentq(lambda x: abs(closed(x))-1/np.sqrt(2), 1, 10)
axes[2].semilogx(w, db(closed(w)), color='C4', label='Closed-loop T')
axes[2].plot(bw, db(1/np.sqrt(2)), 'o', color='C4', label=f'Bandwidth {bw:.2f} rad/s')
for ax, level, label in zip(axes, [0, -180, db(1/np.sqrt(2))], ['Magnitude (dB)', 'Phase (deg)', 'Magnitude (dB)']):
    ax.axhline(level, color='gray', ls='--', lw=1)
    ax.set_ylabel(label)
    ax.grid(True, which='both', alpha=.22)
axes[0].legend(ncol=3)
axes[0].set_title('Open-loop Bode design')
axes[2].set_title('Closed-loop bandwidth')
axes[2].legend()
axes[2].set_xlabel('Angular frequency (rad/s)')
fig.savefig('lead_design_bode.png', dpi=150)
plt.show()
