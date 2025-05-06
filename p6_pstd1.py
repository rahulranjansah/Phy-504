import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, fftfreq
from matplotlib.animation import FuncAnimation

# -----------------------------
# PARAMETERS
# -----------------------------
c0 = 3e8
n_dielectric = 1.5
x_min, x_max = -10e-6, 20e-6
Nx = 1000
dx = (x_max - x_min) / (Nx - 1)
x = np.linspace(x_min, x_max, Nx)

T = 200e-15
dt = 0.25 * dx / c0
Nt = int(T / dt)

# Pulse parameters
wavelength = 800e-9
omega = 2 * np.pi * c0 / wavelength
x0 = -5e-6
tau = 5e-15

# -----------------------------
# SPECTRAL GRID & PROFILE
# -----------------------------
k = fftfreq(Nx, d=dx) * 2 * np.pi
k2 = -k**2

v = np.ones(Nx) * c0
slab = (x >= 0) & (x <= 10e-6)
v[slab] = c0 / n_dielectric

# -----------------------------
# PML CONDUCTIVITY PROFILE σ(x)
# -----------------------------
pml_width = int(0.1 * Nx)    # 10%
sigma_max = 1e14
sigma = np.zeros(Nx)
xl = np.linspace(1, 0, pml_width)
sigma[:pml_width] = sigma_max * xl**2
sigma[-pml_width:] = sigma_max * xl[::-1]**2

b = np.exp(-sigma * dt)
# avoid divide-by-zero: set a = dt where sigma == 0
a = np.where(sigma > 0, (1 - b) / sigma, dt)

# -----------------------------
# INITIALIZE FIELD
# -----------------------------
E = np.zeros((Nt, Nx))
E[0] = np.exp(-((x - x0)/(c0*tau))**2) * np.cos(omega*(x - x0)/c0)
E[1] = np.exp(-((x - x0 - c0*dt)/(c0*tau))**2) * np.cos(omega*(x - x0 - c0*dt)/c0)

# -----------------------------
# PSTD TIME LOOP with PML
# -----------------------------
for n in range(1, Nt-1):
    E_k = fft(E[n])
    d2E = ifft(k2 * E_k).real
    E_new = 2*E[n] - E[n-1] + (dt**2)*(v**2)*d2E
    E[n+1] = b * E_new + a * E[n]
    # clamp any NaNs/Infs
    E[n+1] = np.nan_to_num(E[n+1])

# -----------------------------
# SNAPSHOT PARAMETERS
# -----------------------------
snapshot_times_fs = [0, 10, 20, 30, 40, 50, 60, 70, 80]
snapshot_indices = [min(int(tfs*1e-15/dt), Nt-1) for tfs in snapshot_times_fs]

# -----------------------------
# ANIMATION
# -----------------------------
fig, ax = plt.subplots(figsize=(10, 4))
line, = ax.plot(x*1e6, E[0], color='#00FFFF', lw=2)
ax.set_xlim(x_min*1e6, x_max*1e6)
ax.set_ylim(-1, 1)
ax.set_xlabel("x (μm)")
ax.set_ylabel("E(x, t)")
ax.set_title("PSTD Wave with PML Absorbing Boundaries")
ax.axvspan(0, 10, color='cyan', alpha=0.3, label='Dielectric slab')
time_text = ax.text(0.02, 0.9, '', transform=ax.transAxes)

def update(frame):
    line.set_ydata(E[frame])
    time_text.set_text(f"t = {frame*dt*1e15:.1f} fs")
    return line, time_text

ani = FuncAnimation(fig, update, frames=range(0, Nt, 3), interval=30, blit=True)
ax.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# Snapshots Plot
# -----------------------------
fig, axes = plt.subplots(len(snapshot_indices), 1, figsize=(10, 2.5*len(snapshot_indices)), sharex=True)
for i, idx in enumerate(snapshot_indices):
    ax = axes[i]
    ax.plot(x*1e6, E[idx], color='blue')
    ax.axvspan(0, 10, color='gray', alpha=0.3, label='Dielectric slab')
    ax.set_ylabel("E(x)")
    ax.set_title(f"t = {snapshot_times_fs[i]} fs")
    ax.set_ylim(-1, 1)
    ax.grid(True)
    if i == 0:
        ax.legend()
axes[-1].set_xlabel("x (μm)")
plt.tight_layout()
plt.show()
