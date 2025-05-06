import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, fftfreq
from matplotlib.animation import FuncAnimation

# -----------------------------
# PARAMETERS
# -----------------------------
c0         = 3e8
epsilon_0  = 8.854187817e-12
omega_0    = 2.63e16
omega_i    = 2.35e16

# Spatial & time grid
x_min, x_max = -10e-6, 30e-6
Nx           = 1000
x            = np.linspace(x_min, x_max, Nx)
dx           = x[1] - x[0]

T   = 200e-15
dt  = 0.25 * dx / c0
Nt  = int(T / dt)

# Pulse & slab
x0      = -2e-6
tau     = 5e-15
wlen    = 800e-9
omega   = 2*np.pi*c0/wlen
a, b    = 10e-6, 20e-6
is_slab  = (x >= a) & (x <= b)

# PML
pml_width = int(0.1 * Nx)
sigma_max = 1e14
sigma     = np.zeros(Nx)
taper     = np.linspace(0, 1, pml_width)
sigma[:pml_width] = sigma_max * taper**2
sigma[-pml_width:] = sigma_max * taper[::-1]**2
b_pml = np.exp(-sigma * dt)
# avoid division by zero
a_pml = np.where(sigma>0, (1 - b_pml)/sigma, dt)

# spectral k²
k  = fftfreq(Nx, d=dx)*2*np.pi
k2 = -k**2

# -----------------------------
# INITIALIZE FIELDS
# -----------------------------
# E[t, x], P[t, x]
E = np.zeros((Nt, Nx))
P = np.zeros((Nt, Nx))

# initial Gaussian‐cosine pulse
E[0, :] = np.exp(-((x-x0)/(c0*tau))**2)*np.cos(omega*(x-x0)/c0)
E[1, :] = np.exp(-((x-x0-c0*dt)/(c0*tau))**2)*np.cos(omega*(x-x0-c0*dt)/c0)

# -----------------------------
# TIME‐STEPPING (PSTD + Lorentz + PML)
# -----------------------------
for n in range(1, Nt-1):
    # 1) Spectral second derivative of E at time n
    E_hat      = fft(E[n, :], n=Nx)
    d2E_spectral = ifft(k2 * E_hat, n=Nx).real

    # 2) Step P forward (only inside slab)
    P[n+1, :] = (
        2*P[n, :] - P[n-1, :]
        + dt**2*(epsilon_0*omega_0**2 * E[n, :] - omega_i**2 * P[n, :]) * is_slab
    )

    # 3) Compute P̈ at step n+½
    P_tt = (P[n+1, :] - 2*P[n, :] + P[n-1, :]) / dt**2

    # 4) Step E forward using PSTD formula
    E_new = (
        2*E[n, :] - E[n-1, :]
        + dt**2*(c0**2 * d2E_spectral - (1/epsilon_0) * P_tt)
    )

    # 5) Apply PML damping
    E[n+1, :] = b_pml * E_new + a_pml * E[n, :]
    E[n+1, :] = np.nan_to_num(E[n+1, :])

# -----------------------------
# SNAPSHOTS
# -----------------------------
snapshot_fs = [10, 30, 50, 70, 90, 120, 150]
snapshot_idx = [min(int(fs*1e-15/dt), Nt-1) for fs in snapshot_fs]

fig, axes = plt.subplots(len(snapshot_idx), 1, figsize=(8, 2.5*len(snapshot_idx)), sharex=True)
for ax, fs, idx in zip(axes, snapshot_fs, snapshot_idx):
    ax.plot(x*1e6, E[idx, :], 'b')
    ax.axvspan(a*1e6, b*1e6, color='gray', alpha=0.3)
    ax.set_ylabel('E(x)')
    ax.set_title(f't = {fs} fs')
    ax.set_ylim(-1, 1)
    ax.grid(True)
axes[-1].set_xlabel('x (μm)')
plt.tight_layout()
plt.show()

# -----------------------------
# ANIMATION
# -----------------------------
fig2, ax2 = plt.subplots(figsize=(8,4))
line, = ax2.plot(x*1e6, E[0, :], 'r', lw=2)
ax2.axvspan(a*1e6, b*1e6, color='gray', alpha=0.3)
ax2.set(xlim=(x_min*1e6, x_max*1e6), ylim=(-1,1),
        xlabel='x (μm)', ylabel='E(x,t)')
time_text = ax2.text(0.02, 0.9, '', transform=ax2.transAxes)

def update(frame):
    line.set_ydata(E[frame, :])
    time_text.set_text(f't = {frame*dt*1e15:.1f} fs')
    return line, time_text

anim = FuncAnimation(fig2, update, frames=range(0, Nt, 5), interval=50, blit=True)
plt.show()
