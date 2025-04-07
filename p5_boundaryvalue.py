import numpy as np
import matplotlib.pyplot as plt

# relaxation method for the problem
def relaxation_solver(V, tol=1e-4, max_iter=10000):
    V_new = V.copy()
    for _ in range(max_iter):
        V_old = V_new.copy()

        # finite differencing V_(i,j) = 1/4 (V_(i+1,j) + V_(i-1,j) + V_(i,j+1), V_(i,j-1))

        V_new[1:-1,1:-1] = (1/4) * (V_old[:-2,1:-1] + V_old[2:,1:-1] +
                                   V_old[1:-1,:-2] + V_old[1:-1,2:])

        # tolerance convergance condition max_(i,j) = | V_(i,j) - V_(i,j) | < tol
        if np.max(np.abs(V_new - V_old)) < tol:
            break

    return V_new

def plot_potential(V, title="Electric Potential"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # creating the 3D surface plot V(x,y) -- explicit and implictly we compute (∇^2)V(x,y)
    X, Y = np.meshgrid(np.arange(V.shape[0]), np.arange(V.shape[1]))


    ax.plot_surface(X, Y, V, cmap='viridis')
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Potential (V)')
    plt.tight_layout()
    plt.show()

# Grid size
N = 50
V = np.zeros((N, N))

# Case (i): Fixed boundaries matrix-style grid orientation
V[0, :] = 5       # Top (Va)
V[-1, :] = 0      # Bottom (Vb)
V[:, 0] = 3       # Left (Vc)
V[:, -1] = 1      # Right (Vd)
V_case1 = relaxation_solver(V)
plot_potential(V_case1, "Case: Fixed Boundary Conditions")

# Case (ii): "Crazy" boundaries
V2 = np.zeros((N, N))
x = np.linspace(0, 2 * np.pi, N)

V2[0, :] = 5 * np.sin(x)              # Top
V2[-1, :] = 2 * np.cos(x * 2)         # Bottom
V2[:, 0] = 10 * np.random.rand(N)     # Left
V2[:, -1] = np.linspace(0, 10, N)     # Right

V_case2 = relaxation_solver(V2)
plot_potential(V_case2, "Case: Crazy Boundary Conditions")
