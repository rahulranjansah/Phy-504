import numpy as np
import matplotlib.pyplot as plt

# Constants in atomic units
hbar = 1.0  # Planck's constant
e0 = 1.0    # charge
a0 = 1.0    # Bohr radius

# Energy levels of hydrogen in atomic units using -1/2n^2
epsilon = {
    1: -0.5,
    2: -0.125,
    3: -0.0556
}

# Laser parameters tau in (fs), E_i, omega_i all relative to its atomic units
tau = 100.0
E_i = 0.05
omega_i = epsilon[2] - epsilon[1]

# Define the states and their indices
states = [(1, 0, 0), (2, 0, 0), (2, 1, 1), (2, 1, 0), (2, 1, -1), (3, 0, 0), (3, 1, 0), (3, 1, 1), (3, 1, -1), (3, 2, 0), (3, 2, 1), (3, 2, -1), (3, 2, 2), (3, 2, -2)]
# states = [(1, 0, 0), (2, 1, 0)]

state_indices = {state: i for i, state in enumerate(states)}
# print(state_indices)

# Initial conditions with probability amplitudes
c_initial = np.zeros(len(states), dtype=complex)

# part 1
# c_initial[state_indices[(1, 0, 0)]] = 1

# part 2 states
c_initial[state_indices[(1, 0, 0)]] = 1 / np.sqrt(2)
c_initial[state_indices[(2, 1, 1)]] = 1 / np.sqrt(2)


# laser electric field
def E_L(t):
    return E_i * np.cos(omega_i * t) * np.exp(-(t / tau) ** 2)

# matrix element
def z_matrix_element(n, l, m, n_prime, l_prime, m_prime):
    if m == m_prime and abs(l - l_prime) == 1 and n != n_prime:
        return a0
    else:
        return 0.0

# probability amplitudes time deriv
def dc_dt(t, c):
    dc = np.zeros_like(c, dtype=complex)
    for i, (n, l, m) in enumerate(states):
        for j, (n_prime, l_prime, m_prime) in enumerate(states):
            z_element = z_matrix_element(n, l, m, n_prime, l_prime, m_prime)
            energy_diff = epsilon[n_prime] - epsilon[n]
            phase = np.exp(-1j * energy_diff * t / hbar)
            dc[i] += -1j / hbar * e0 * E_L(t) * c[j] * z_element * phase
    return dc


# Runge-Kutta 4th order
def rk4_step(t, c, dt):
    k1 = dc_dt(t, c)
    k2 = dc_dt(t + 0.5 * dt, c + 0.5 * dt * k1)
    k3 = dc_dt(t + 0.5 * dt, c + 0.5 * dt * k2)
    k4 = dc_dt(t + dt, c + dt * k3)
    return c + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

# Time spacing in atomic units
t_span = (-10 * tau, 10 * tau)
dt = 0.5
t_eval = np.arange(t_span[0], t_span[1], dt)

# Initialize the solution array
c_solution = np.zeros((len(t_eval), len(states)), dtype=complex)
c_solution[0] = c_initial

# Solve the ODE using RK4
for i in range(1, len(t_eval)):
    c_solution[i] = rk4_step(t_eval[i - 1], c_solution[i - 1], dt)

# Extract the solution
c_solution_dict = {state: c_solution[:, i] for i, state in enumerate(states)}


# Plot the results
plt.figure(figsize=(10, 6))
for (n, l, m), c_nlm in c_solution_dict.items():
    if np.max(np.abs(c_nlm)) > 1e-3:  # Only plot states with significant population
        plt.plot(t_eval, np.abs(c_nlm) ** 2, label=f"$|c_{{{n},{l},{m}}}|^2$")

plt.xlabel("Time (atomic units)")
plt.ylabel("Population $|c_{n,l,m}|^2$")
plt.title("Time Evolution of Probability Amplitudes")
plt.legend()
plt.grid()
plt.show()
