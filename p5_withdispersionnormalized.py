import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# -----------------------------
# Dimensionless Parameters
# -----------------------------
omega_0 = 1.0       # Resonant frequency (normalized)
omega_i = 0.8       # Damping oscillator frequency (normalized)
c = 1.0             # Normalized speed of light
epsilon_0 = 1.0     # Normalized permittivity

# Spatial domain (dimensionless)
x_min, x_max = -2.0, 6.0
Nx = 800
x = np.linspace(x_min, x_max, Nx)
dx = x[1] - x[0]

# Time domain (dimensionless)
T = 16.0
dt = 0.01
Nt = int(T / dt)
t = np.linspace(0, T, Nt)

# Dielectric region (dimensionless)
a, b = 0.0, 2.0
dielectric_mask = (x >= a) & (x <= b)

# Pulse parameters
x0 = -1.5
tau = 0.5
omega = omega_0  # Match driving frequency to resonance

# -----------------------------
# Field Initialization
# -----------------------------
E = np.zeros((Nx, Nt))
P = np.zeros((Nx, Nt))

# Gaussian-modulated cosine (dimensionless units)
E[:, 0] = np.exp(-((x - x0) / tau)**2) * np.cos(omega * x)
E[:, 1] = E[:, 0].copy()
P[:, 0] = 0.0
P[:, 1] = 0.0

# -----------------------------
# Finite differencing Evolution Loop
# -----------------------------
for n in range(1, Nt - 1):
    for i in range(1, Nx - 1):
        if dielectric_mask[i]:
            P[i, n+1] = (
                2 * P[i, n] - P[i, n-1] +
                dt**2 * (epsilon_0 * omega_0**2 * E[i, n] - omega_i**2 * P[i, n])
            )
        else:
            P[i, n+1] = 0.0

    for i in range(1, Nx - 1):

        laplacian_E = E[i+1, n] - 2 * E[i, n] + E[i-1, n]
        second_deriv_P = P[i, n+1] - 2 * P[i, n] + P[i, n-1]
        E[i, n+1] = (
            2 * E[i, n] - E[i, n-1] +
            (c * dt / dx)**2 * laplacian_E +
            (1 / epsilon_0) * second_deriv_P
        )

    # dirichlet boundary conditions
    E[0, n+1] = 0.0
    E[-1, n+1] = 0.0

# Prepare data for visualization
snapshot_times = [2, 4, 6, 8, 10, 15]
# snapshot_indices = [int(ts / dt) for ts in snapshot_times]
snapshot_indices = [min(int(ts / dt), Nt - 1) for ts in snapshot_times]
E_snapshots = [E[:, idx] for idx in snapshot_indices]

# -----------------------------
# Plot Snapshots
# -----------------------------
fig, ax = plt.subplots(figsize=(10, 6))

for idx, Esnap in enumerate(E_snapshots):
    ax.plot(x, Esnap, label=f't = {snapshot_times[idx]}')

ax.axvspan(a, b, color='cyan', alpha=0.15, label='Dielectric slab')
ax.set_xlabel("x (dimensionless)")
ax.set_ylabel("E(x, t)")
ax.set_title("Snapshots of Electric Field Over Time (Dimensionless)")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()


# Plot each snapshot from E_snapshots separately in its own subplot
fig, axes = plt.subplots(len(E_snapshots), 1, figsize=(10, 2.5 * len(E_snapshots)), sharex=True)

for i, Esnap in enumerate(E_snapshots):
    ax = axes[i]
    ax.plot(x, Esnap, color='blue')
    ax.axvspan(a, b, color='cyan', alpha=0.15, label='Dielectric slab')
    ax.set_ylabel("E(x)")
    ax.set_title(f"t = {snapshot_times[i]}")
    ax.grid(True)
    if i == 0:
        ax.legend()

axes[-1].set_xlabel("x (dimensionless)")
plt.tight_layout()
plt.show()


# Animate

fig, ax = plt.subplots(figsize=(10, 4))
line, = ax.plot(x, E[:, 0], color='blue', lw=2)

# Highlight the dielectric region
ax.axvspan(a, b, color='cyan', alpha=0.1, label='Dielectric slab')
ax.set_xlim(x_min, x_max)
ax.set_ylim(-0.5, 2.3)
ax.set_xlabel("x (dimensionless)")
ax.set_ylabel("E(x, t)")
ax.set_title("Electric Field Propagation (Dispersive Medium)")
time_text = ax.text(0.02, 0.9, '', transform=ax.transAxes)
ax.legend()

def update(frame):
    line.set_ydata(E[:, frame])
    time_text.set_text(f"t = {frame * dt:.2f}")
    return line, time_text

ani = animation.FuncAnimation(fig, update, frames=range(0, Nt, 3), interval=20, blit=True)
plt.tight_layout()
plt.show()
