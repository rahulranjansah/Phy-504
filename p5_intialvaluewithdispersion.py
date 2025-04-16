# """
# Numerically solve the 1D wave equation for the laser field in project 1 (wavelength = 800 nm, 5 fs pulsewidth)
# propagating in vacuum as it strikes a 10 µm dielectric where:

# (ii) the dielectric is dispersive and the polarization is given by: ∂^2P(x, t) + ωi^2 P(x, t) = ϵ_0 * ω0^2 * E(x, t),

# where ωi = 2.35 × 10^16 rad/s and ω0 = 2.63 × 10^16 rad/s.

# For each case, graph many picture of the field as it propagates across your grid (in other words, graph the
# entire E(x, t) at many different times), showing how it strikes the dielectric, is reflected and transmitted.
# Justify your chosen boundary conditions in space!

# """
# # part (ii)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# -----------------------------
# Constants and Parameters
# -----------------------------
c0 = 3e8  # (m/s)
epsilon_0 = 8.854187817e-12  #  permittivity (F/m)
mu_0 = 4e-7 * np.pi

# Dispersive polarization params
omega_0 = 2.63e16 # Resonant frequency
omega_i = 2.35e16  # Oscillator frequency

# Spatial domain
x_min, x_max = -10e-6, 30e-6
Nx = 1000
x = np.linspace(x_min, x_max, Nx)
dx = x[1] - x[0]

# Time domain
T = 200e-15  # Total simulation time
dt = 0.25 * dx / c0  # dt << pi * 10^-17 to deal w/ dispersion
print(dt)
Nt = int(T / dt)
t = np.linspace(0, T, Nt)

# Dielectric region
a = 10e-6
b = 20e-6
is_dielectric = (x >= a) & (x <= b)

# Pulse parameters
x0 = -2e-6
tau = 5e-15
wavelength = 800e-9
omega = 2 * np.pi * c0 / wavelength

# -----------------------------
# Field Initialization
# -----------------------------
E = np.zeros((Nx, Nt))
P = np.zeros((Nx, Nt))

# Initial electric field
E[:, 0] = np.exp(-((x - x0) / (c0 * tau))**2) * np.cos(omega * (x-x0) / c0)
E[:, 1] = np.exp(-((x - x0 - c0*dt) / (c0 * tau))**2) * np.cos(omega * (x-x0-c0*dt) / c0)

# Initial polarization (assume at rest)
P[:, 0] = 0
P[:, 1] = 0

# -----------------------------
# Time Evolution Loop
# -----------------------------
for n in range(1, Nt - 1):
    for i in range(1, Nx - 1):
        laplacian_E = (E[i+1, n] - 2 * E[i, n] + E[i-1, n]) / dx**2
        if is_dielectric[i]:
            # print(is_dielectric)
            # P[i,n] = 0
            polarization_term = omega_i**2 * P[i, n]

            # print(polarization_term)
            driving_term = omega_0**2 * E[i, n] * epsilon_0

            E[i, n+1] = 2 * E[i, n] - E[i, n-1] + c0**2 * dt**2 * (laplacian_E - (mu_0 * (driving_term - polarization_term )))

        else:
            E[i, n+1] = 2 * E[i, n] - E[i, n-1] + c0**2 * dt**2 * laplacian_E

    for i in range(1, Nx - 1):
        if is_dielectric[i]:
            P[i, n+1] = 2 * P[i, n] - P[i, n-1] + dt**2 * (epsilon_0 * omega_0**2 * E[i, n] - omega_i**2 * P[i, n])

    # Dirichlet boundary conditions (optional: test with ABCs too)
    E[0, n+1] = 0
    E[-1, n+1] = 0


# -----------------------------
# Plot Snapshots
# -----------------------------
snapshot_times_fs = [10, 30, 50, 70, 90, 120, 150]  # femtoseconds
snapshot_indices = [min(int((fs * 1e-15) / dt), Nt - 1) for fs in snapshot_times_fs]

fig, ax = plt.subplots(figsize=(10, 6))
for idx in snapshot_indices:
    ax.plot(x * 1e6, (E[:, idx]), label=f't = {t[idx]*1e15:.1f} fs')


ax.axvspan(a * 1e6, b * 1e6, color='cyan', alpha=0.1, label='Dielectric slab')
ax.set_xlabel("x (μm)")
ax.set_ylabel("E(x)")
ax.set_title("Snapshots of Electric Field Over Time")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()

# Animation
# plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(10, 4))
line, = ax.plot(x * 1e6, E[:, 0], color='#00FFFF', lw=2)

# Dielectric visualization
ax.axvspan(a * 1e6, b * 1e6, color='cyan', alpha=0.15, label='Dielectric slab')
ax.set_xlim(x_min * 1e6, x_max * 1e6)
ax.set_ylim(-1, 1)
ax.set_xlabel("x (μm)")
ax.set_ylabel("E(x, t)")
ax.set_title("Dispersive Dielectric Simulation")
time_text = ax.text(0.02, 0.9, '', transform=ax.transAxes, color='white')
ax.legend()

def update(frame):
    line.set_ydata(E[:, frame])
    time_text.set_text(f"t = {frame * dt * 1e15:.1f} fs")
    return line, time_text

ani = animation.FuncAnimation(fig, update, frames=range(0, Nt, 3), interval=20, blit=True)
plt.tight_layout()
plt.show()


# -----------------------------
# Plot each snapshot individually
# -----------------------------


dielectric_start = 10e-6
dielectric_end = 20e-6
fig, axes = plt.subplots(len(snapshot_indices), 1, figsize=(10, 2.5 * len(snapshot_indices)), sharex=True)

for i, idx in enumerate(snapshot_indices):
    ax = axes[i]
    ax.plot(x * 1e6, E[:, idx], color='blue')
    ax.axvspan(dielectric_start * 1e6, dielectric_end * 1e6, color='gray', alpha=0.3, label='Dispersive Dielectric Polarized slab')
    ax.set_ylabel("E(x)")
    ax.set_title(f"t = {snapshot_times_fs[i]} fs")
    ax.set_ylim(-1, 1)
    ax.grid(True)
    if i == 0:
        ax.legend()


axes[-1].set_xlabel("x (μm)")
plt.tight_layout()
plt.show()
