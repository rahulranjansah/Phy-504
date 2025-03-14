import numpy as np
import matplotlib.pyplot as plt

omega_i = 1.5 * 10 ** 16 # heartz
omega = [0.5, 1.0, 1.5]

t_0 = 5 * 10 ** (-15) # seconds
t_max = 100 * 10 ** (-15) # seconds
q_electron = - 1.6 * 10 ** (-19) # charge
m_electron = 9.1 * 10 ** (-31) # kg
gamma = 0.01 * omega_i

# E vs t graph

delta_t = 1.0e-17
num_steps = int(t_0 / delta_t) * 100

# print(num_steps)

t = np.linspace(-t_0*10, t_max, num_steps)
E_0 = [1.0, 1.0 * 10**10] # N/C

def dampend_oscillations(omega: float, E_0: float) -> float:

    """
    Simulates damped oscillations of a driven harmonic oscillator.

    Parameters:
    omega (float): Angular frequency of the oscillator.
    E_0 (float): Amplitude of the electric field.

    Returns:
    np.ndarray: Array of positions x over time.
    """
    x = np.zeros(num_steps)
    v = np.zeros(num_steps)

    x[0] = 0
    v[0] = 0

    # compute for given num steps
    for i in range(1, num_steps-1):

        # Electric field
        E = E_0 * np.exp(-(t[i] / t_0) ** 2) * np.cos(omega * omega_i * t[i])

        # ODE
        x[i+1] = (2 * q_electron *  E * delta_t ** 2) / (m_electron * (2 + (gamma * delta_t))) + \
                    ((4 * x[i]) - (2 * (x[i - 1])) + (x[i - 1] * delta_t) - (2 * x[i] * delta_t ** 2)) / \
                        (2 + gamma * delta_t)

    return x

plots = []
plt.style.use("dark_background")

# plots all the plots with nested omega vals
for E0_val in E_0:

    fig, ax = plt.subplots(figsize=(12, 8))

    for omega_val in omega:
        x = dampend_oscillations(omega_val, E0_val)

        ax.plot(t, x,
                alpha=0.5,
                label=f"$\\omega = {omega_val} \\, \\omega_i$")

    ax.set_xlabel("Time $t$ (s)")
    ax.set_ylabel("Position $x(t)$")
    ax.set_title(f"Damped-Driven Harmonic Oscillator (E0 = {E0_val} N/C)")
    ax.legend()
    ax.grid(ls="dashed", alpha=0.4)

    plots.append(fig)

for fig in plots:
    plt.show()


