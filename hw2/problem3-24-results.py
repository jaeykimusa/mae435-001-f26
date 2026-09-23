# Course: MAE 435 (001) - Principles of Automatic Control
# Instructor: Prof. Atay
# Author: Jaey Kim (student)
# Date: 09/21/2026


import numpy as np
import matplotlib.pyplot as plt


# simulation results settings
USE_LSIM = True # True -> lsim results / False -> RK numerical results


# given electrical component variables for the system circuit
R = 1
L = 1
C = 1


# time and input
t = np.linspace(0, 15, 1501)


# unit-step input: v1(t) = 1 V for t >= 0, given conditions
v1 = np.ones_like(t)


# ============================================================
# analytical solution:
#
# i(t) = (2/sqrt(3)) * exp(-t/2) *
#        sin(sqrt(3)*t/2)
# ============================================================
i_analytical = (
    2 / np.sqrt(3)
    * np.exp(-t / 2)
    * np.sin(np.sqrt(3) * t / 2)
)


# simulation
if USE_LSIM:

    # method 1: suggested library by prof. atay scipy.signal.lsim
    from scipy import signal

    # fransfer function:
    #
    # I(s)/V1(s) = C*s / (L*C*s^2 + R*C*s + 1)
    #
    # with R = L = C = 1:
    #
    # I(s)/V1(s) = s / (s^2 + s + 1)

    numerator = [C, 0]
    denominator = [L*C, R*C, 1]

    system = signal.TransferFunction(numerator, denominator)

    # simulate response to v1(t)
    t_sim, i_sim, _ = signal.lsim(
        system,
        U=v1,
        T=t
    )

    simulation_method = "SciPy lsim"


else:

    # method 2: rk4 numerical simulation
    i_sim = np.zeros_like(t)
    v2_sim = np.zeros_like(t)

    dt = t[1] - t[0]

    # state equations:
    #
    # di/dt  = (v1 - R*i - v2) / L
    # dv2/dt = i / C

    def derivatives(x, u):
        i, v2 = x

        di_dt = (u - R*i - v2) / L
        dv2_dt = i / C

        return np.array([di_dt, dv2_dt])

    # initial conditions:
    # i(0) = 0
    # v2(0) = 0

    x = np.array([0.0, 0.0])

    for k in range(len(t) - 1):

        u = v1[k]

        k1 = derivatives(x, u)
        k2 = derivatives(x + dt*k1/2, u)
        k3 = derivatives(x + dt*k2/2, u)
        k4 = derivatives(x + dt*k3, u)

        x = x + dt/6 * (
            k1 + 2*k2 + 2*k3 + k4
        )

        i_sim[k+1] = x[0]
        v2_sim[k+1] = x[1]

    t_sim = t

    simulation_method = "Numerical Solution via Runge-Kutta 4"


# plot
plt.figure(figsize=(8, 5))

plt.plot(
    t,
    i_analytical,
    label="Analytical Solution",
    linewidth=2
)

plt.plot(
    t_sim,
    i_sim,
    "--",
    label=simulation_method,
    linewidth=2
)

plt.xlabel("Time, t [s]")
plt.ylabel("Current, i(t) [A]")
plt.title("Current Response to Unit-Step Input")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# error validation in case
error = np.max(np.abs(i_analytical - i_sim))
print("Simulation method:", simulation_method)
print(f"Maximum absolute error: {error:.3e} A")