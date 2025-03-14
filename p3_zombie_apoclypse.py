import matplotlib.pyplot as plt
import numpy as np
import math

# Define your functions F1, F2, ..., F5
def F1(f1, f2, f4):
    return -(f1 * f4 * B) + (f1 * f2 * R)

def F2(f1, f2, f4):
    return (f1 * f4 * B) - (f1 * f2 * R) - (f2 * gamma_2)

def F3(f1, f2, f3):
    return (f2 * gamma_2) - (f3 * gamma_3) - (f1 * f3 * K_I)

def F4(f1, f3, f4):
    return (-f1 * f4 * K_z) + (f3 * gamma_3)

def F5(f1, f3, f4):
    return (f1 * f4 * K_z) + (f1 * f3 * K_I)

# System of equations
def system_eq(y, t):
    f1, f2, f3, f4, f5 = y
    df1 = F1(f1, f2, f4)
    df2 = F2(f1, f2, f4)
    df3 = F3(f1, f2, f3)
    df4 = F4(f1, f3, f4)
    df5 = F5(f1, f3, f4)
    return np.array([df1, df2, df3, df4, df5])

# Runge-Kutta 4th order step
def rk4_step(y, t, dt, derivs_func):
    k1 = derivs_func(y, t)
    k2 = derivs_func(y + (dt/2) * k1, t + dt/2)
    k3 = derivs_func(y + (dt/2) * k2, t + dt/2)
    k4 = derivs_func(y + dt * k3, t + dt)
    return y + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

def euler_method(y, t, dt, derivs_func):
    return y + (derivs_func(y, t) * dt)

# Constants
gamma_2 = 1/20
gamma_3 = 1/5
B = 1/2
K_I = 1/30
K_z = 1/16
R = 1/math.exp(1)

# Time parameters
n_steps = 100
ti = 0
tf = 20 * 24
dt = (tf - ti) / n_steps
timestamps = np.linspace(ti, tf, n_steps + 1)

# Initial conditions
y = np.array([0.85, 0.1, 0.01, 0.01, 0.03])
runga_states = [y.copy()]
euler_states = [y.copy()]

# Run simulation
y_rk, y_euler = y.copy(), y.copy()
t = ti

for _ in range(n_steps):
    y_rk = rk4_step(y_rk, t, dt, system_eq)
    y_euler = euler_method(y_euler, t, dt, system_eq)
    t += dt
    runga_states.append(y_rk.copy())
    euler_states.append(y_euler.copy())

# Convert to numpy array
runga_states_array = np.array(runga_states)
euler_states_array = np.array(euler_states)

# Extract populations
f1_rk, f2_rk, f3_rk, f4_rk, f5_rk = runga_states_array.T
f1_eu, f2_eu, f3_eu, f4_eu, f5_eu = euler_states_array.T


fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].plot(timestamps, f1_rk, label="Healthy (F1)", linestyle='solid')
ax[0].plot(timestamps, f2_rk, label="Infected (F2)", linestyle='dashed')
ax[0].plot(timestamps, f3_rk, label="Removed (F3)", linestyle='dotted')
ax[0].plot(timestamps, f4_rk, label="Zombies (F4)", linestyle='dashdot')
ax[0].plot(timestamps, f5_rk, label="Dead (F5)", linestyle='solid')
ax[0].set_title("Runge-Kutta (RK4) Method")
ax[0].set_xlabel("Time (hours)")
ax[0].set_ylabel("Population Fraction")
ax[0].legend()
ax[0].grid()

# Plot Euler method
ax[1].plot(timestamps, f1_eu, label="Healthy (F1)", linestyle='solid')
ax[1].plot(timestamps, f2_eu, label="Infected (F2)", linestyle='dashed')
ax[1].plot(timestamps, f3_eu, label="Removed (F3)", linestyle='dotted')
ax[1].plot(timestamps, f4_eu, label="Zombies (F4)", linestyle='dashdot')
ax[1].plot(timestamps, f5_eu, label="Dead (F5)", linestyle='solid')
ax[1].set_title("Euler Method")
ax[1].set_xlabel("Time (hours)")
ax[1].set_ylabel("Population Fraction")
ax[1].legend()
ax[1].grid()

# Show the plots
plt.tight_layout()
plt.show()