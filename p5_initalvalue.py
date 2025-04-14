"""
Numerically solve the 1D wave equation for the laser field in project 1 (wavelength = 800 nm, 5 fs pulsewidth)
propagating in vacuum as it strikes a 10 µm dielectric where:

(i) the dielectric has a constant index of refraction of 1.5.
(ii) the dielectric is dispersive and the polarization is given by: ∂^2P(x, t) + ωi^2 P(x, t) = ϵ_0 * ω0^2 * E(x, t),

where ωi = 2.35 × 1016 rad/s and ω0 = 2.63 × 1016 rad/s.

For each case, graph many picture of the field as it propagates across your grid (in other words, graph the
entire E(x, t) at many different times), showing how it strikes the dielectric, is reflected and transmitted.
Justify your chosen boundary conditions in space!

"""

# part (i)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

# -----------------------------
# Parameters
# -----------------------------
c0 = 3e8                   # Speed of light in vacuum (m/s)
n_dielectric = 1.5         # Refractive index of dielectric
x_min, x_max = -10e-6, 20e-6  # Domain limits in meters
Nx = 1000                  # Number of spatial points
dx = (x_max - x_min) / (Nx - 1)
x = np.linspace(x_min, x_max, Nx)

# Time parameters
T = 200e-15                # Total time in seconds
dt = 0.99 * dx / c0        # Time step (CFL condition)
Nt = int(T / dt)
t = np.linspace(0, T, Nt)

# Pulse parameters
wavelength = 800e-9
omega = 2 * np.pi * c0 / wavelength
x0 = -5e-6                 # Initial pulse center
tau = 5e-15                # Pulse width

# -----------------------------
# Wave speed profile v(x), normal speed propagation
# -----------------------------
v = np.ones_like(x) * c0

# Dielectric slab region boolean mask slow speed propagation
v[(x >= 0) & (x <= 10e-6)] = c0 / n_dielectric

# -----------------------------
# Initialization
# -----------------------------
E = np.zeros((Nx, Nt))

# Initial condition: Gaussian-modulated cosine
E[:, 0] = np.exp(-((x - x0) / (c0 * tau))**2) * np.cos(omega * x / c0)

# Assuming zero initial velocity: E[:, 1] = E[:, 0]
E[:, 1] = E[:, 0].copy()

# -----------------------------
# Finite differencing loop for time = T, n is time and i is spatial
# -----------------------------
for n in range(1, Nt - 1):
    for i in range(1, Nx - 1):
        c_ratio = (v[i] * dt / dx) ** 2
        E[i, n+1] = (2 * E[i, n] - E[i, n-1] +
                     c_ratio * (E[i+1, n] - 2 * E[i, n] + E[i-1, n]))

    # Dirichlet boundaries (Boundary value conditions)
    E[0, n+1] = 0
    E[-1, n+1] = 0

# print(E)

dielectric_start = 0
dielectric_end = 10e-6

# -----------------------------
# Animation / Plotting (with dielectric box)
# -----------------------------
fig, ax = plt.subplots()
line, = ax.plot(x * 1e6, E[:, 0], lw=2)
ax.set_xlim(x_min * 1e6, x_max * 1e6)
ax.set_ylim(-1, 1)
ax.set_xlabel("x (μm)")
ax.set_ylabel("E(x, t)")
ax.set_title("Wave Propagation through Dielectric Slab")

# # Add shaded region for dielectric slab
ax.axvspan(dielectric_start * 1e6, dielectric_end * 1e6, color='gray', alpha=0.3, label='Dielectric slab')
ax.legend()

def update(frame):
    line.set_ydata(E[:, frame])
    ax.set_title(f"Wave Propagation — t = {frame * dt * 1e15:.2f} fs")
    return line,

ani = animation.FuncAnimation(fig, update, frames=range(0, Nt, 5), interval=20)
plt.tight_layout()
plt.show()

# -----------------------------
# Snapshots at Selected Times
# -----------------------------
snapshot_times_fs = [0, 50, 100, 150, 200]  # times in femtoseconds
snapshot_indices = [min(int(t_fs * 1e-15 / dt), Nt - 1) for t_fs in snapshot_times_fs]

fig, ax = plt.subplots(figsize=(10, 6))

for idx, n in enumerate(snapshot_indices):
    ax.plot(x * 1e6, E[:, n], label=f't = {snapshot_times_fs[idx]} fs')

# Highlight dielectric region
ax.axvspan(dielectric_start * 1e6, dielectric_end * 1e6, color='gray', alpha=0.3, label='Dielectric slab')

ax.set_xlim(x_min * 1e6, x_max * 1e6)
ax.set_ylim(-1, 1)
ax.set_xlabel("x (μm)")
ax.set_ylabel("E(x)")
ax.set_title("Electric Field Snapshots at Different Times")
ax.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# -----------------------------
# Plot each snapshot individually
# -----------------------------
fig, axes = plt.subplots(len(snapshot_indices), 1, figsize=(10, 2.5 * len(snapshot_indices)), sharex=True)

for i, idx in enumerate(snapshot_indices):
    ax = axes[i]
    ax.plot(x * 1e6, E[:, idx], color='blue')
    ax.axvspan(dielectric_start * 1e6, dielectric_end * 1e6, color='gray', alpha=0.3, label='Dielectric slab')
    ax.set_ylabel("E(x)")
    ax.set_title(f"t = {snapshot_times_fs[i]} fs")
    ax.set_ylim(-1, 1)
    ax.grid(True)
    if i == 0:
        ax.legend()

axes[-1].set_xlabel("x (μm)")
plt.tight_layout()
plt.show()
