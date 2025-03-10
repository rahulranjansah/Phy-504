
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Constants
a0 = 5.29e-11 # bohr radius meters
e0 = 1.602e-19  # charge in Coulombs
hbar = 1.0545718e-34  # Reduced Planck constant in J·s
E_i = 1e10  # electric field V/m

# Energy levels
def energy_vals(n):
    e_n = (-13.6 / (n ** 2)) * 1.6e-19
    return e_n

# Transition matrix element
def z_element(n, l, m, nprime, lprime, mprime):
    if m == mprime and (l == lprime + 1 or l == lprime - 1) and n != nprime:
        return a0
    else:
        return 0

# Electric field
def E_L(omega_i, t, tau):
    return E_i * np.cos(omega_i * t) * np.exp(-(t / tau) ** 2)

# ODE equation
def dcdt(t, c, states, energies, e0, hbar, omega_i, tau):
    dc = np.zeros_like(c)
    for i, (n, l, m) in enumerate(states):
        for j, (n_prime, l_prime, m_prime) in enumerate(states):
            z = z_element(n, l, m, n_prime, l_prime, m_prime)
            energy_diff = energies[n_prime] - energies[n]
            dc[i] += -1j * e0 * E_L(omega_i, t, tau) * c[j] * np.exp(-1j * energy_diff * t / hbar) * z
    return dc

# Setup
e1 = energy_vals(1)
e2 = energy_vals(2)
e3 = energy_vals(3)
omega_i = (e2 - e1) / hbar
tau = 5e-14
t_max = 10e-13
delta_t = 1.0e-14
num_steps = int(t_max / delta_t)
t = np.linspace(-tau, t_max, num_steps)

# Define states (n, l, m)
states = [(1, 0, 0), (2, 1, 0)]
energies = {1: e1, 2: e2}

# Initial conditions: C_{100}(t=-∞) = 1, all others are 0
c0 = np.zeros(len(states), dtype=complex)
c0[0] = 1.0

# Solve ODE
sol = solve_ivp(dcdt, [-tau, t_max], c0, args=(states, energies, e0, hbar, omega_i, tau),
                t_eval=t, method='RK45')

# Extract probabilities
prob_100 = np.abs(sol.y[0]) ** 2  # |C_{100}(t)|^2
prob_210 = np.abs(sol.y[1]) ** 2  # |C_{210}(t)|^2

print(prob_100)
print(prob_210)

