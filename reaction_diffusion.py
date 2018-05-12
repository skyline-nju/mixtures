"""
A reaction - diffusion model for pattern formation

Solve a system of reacton-diffusion PDEs in two dimensions:
    u_t = sigma D1 Delta u + f(u, v)
    v_t = sigma D2 Delta v + g(u, v)
where Delta u = u_xx + u_yy denotes the Laplacian and f, g represent reaction
terms.
Consider the square domain [-1, 1] * [-1, 1] with periodic boundary condition:
u(-1, y, t) = u(1, y, t)
u(x, -1, t) = u(x, 1, t)
with correspoding conditions on v.

@file reaction_diffusion.py
@author skyline-nju
@date 2018-05-12
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse


def f(u, v):
    return alpha * u * (1 - tau1 * v**2) + v * (1 - tau2 * u)


def g(u, v):
    return beta * v + alpha * tau1 * u * v**2 + u * (gamma + tau2 * v)


def five_pt_laplacian_sparse_periodic(m, a, b):
    """Construct a sparse matrix that applies the 5-point laplacian discretization
       with periodic BCs on all sides."""
    e = np.ones(m**2)
    e2 = ([1] * (m - 1) + [0]) * m
    e3 = ([0] + [1] * (m - 1)) * m
    h = (b - a) / (m + 1)
    A = sparse.spdiags([-4 * e, e2, e3, e, e], [0, -1, 1, -m, m], m**2, m**2)
    # Top & bottom BCs:
    A_periodic = sparse.spdiags([e, e], [m - m**2, m**2 - m], m**2, m
                                ** 2).tolil()
    # Left & right BCs:
    for i in range(m):
        A_periodic[i * m, (i + 1) * m - 1] = 1.
        A_periodic[(i + 1) * m - 1, i * m] = 1.
    A = A + A_periodic
    A /= h**2
    A = A.todia()
    return A


def one_step_forward(u, v, k, A, delta, D1=0.5, D2=1.0):
    u_new = u + k * (delta * D1 * A * u + f(u, v))
    v_new = v + k * (
        delta * D2 * A * v + g(u, v))
    return u_new, v_new


def step_size(h, delta):
    return h**2 / (5. * delta)


def pattern_formation(m=10, T=1000):
    """
    Model pattern formation by solving a reaction-diffusion PDE on a periodic
    square domain with an m x m grid.
    """
    D1 = 0.5
    D2 = 1.0

    # Set up the grid
    a = -1.
    b = 1.
    h = (b - a) / m
    # Grid spacing
    x = np.linspace(a, b, m)  # Coordinates
    y = np.linspace(a, b, m)
    Y, X = np.meshgrid(y, x)

    # Initial data
    u = np.random.randn(m, m) / 2.
    v = np.random.randn(m, m) / 2.

    plt.clf()
    plt.pcolormesh(x, y, u)
    plt.colorbar()
    plt.axis('image')
    plt.draw()

    u = u.reshape(-1)
    v = v.reshape(-1)

    A = five_pt_laplacian_sparse_periodic(m, -1., 1.)

    t = 0.  # Initial time
    k = step_size(h, delta) / 2  # Time step size
    N = int(round(T / k))  # Number of steps to take

    # Now step forward in time
    next_plot = 0
    for j in range(N):
        u, v = one_step_forward(u, v, k, A, delta, D1, D2)
        t = t + k
        # Plot every t=5 units
        if t > next_plot:
            next_plot = next_plot + 5
            U = u.reshape((m, m))
            plt.clf()
            plt.pcolormesh(x, y, U)
            plt.colorbar()
            plt.axis('image')
            plt.title(r"$t=%.3f$" % t)
            plt.show()
            plt.pause(0.5)
            plt.clf()


if __name__ == "__main__":
    delta = 0.0021
    tau1 = 3.5
    tau2 = 0
    alpha = 0.899
    beta = -0.91
    gamma = -alpha
    A = five_pt_laplacian_sparse_periodic(5, -1., 1.)
    plt.ion()
    pattern_formation(m=120)
    # plt.spy(A)
    # plt.show()
    # plt.pause(2)
    # plt.close()
