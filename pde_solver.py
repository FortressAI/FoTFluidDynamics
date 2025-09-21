"""
pde_solver.py
=================

This module provides simple numerical solvers for two partial differential equations: the
heat (diffusion) equation and the Poisson equation.  These solvers are meant as
illustrative examples and are designed to demonstrate how classical, verified
numerical methods can be implemented in Python.  They avoid the unphysical
constructs of the FoT vQbit framework and instead rely on finite‑difference
schemes that are widely used in computational physics.

Functions
---------

* ``solve_heat_equation`` – solves the 2D heat equation on a square domain using an
  explicit finite‑difference time stepping scheme.
* ``solve_poisson_equation`` – solves the 2D Poisson equation on a square grid
  with Dirichlet boundary conditions using Jacobi iteration.

The solvers return the final solution array and, for the heat equation, a list
of intermediate snapshots for visualization.  Both functions accept simple
Python/numpy arrays as inputs and are intentionally kept lightweight so they
can run interactively within a Streamlit app.
"""

from __future__ import annotations

import numpy as np
from typing import Callable, List, Tuple


def solve_heat_equation(
    initial: np.ndarray,
    diffusion_coeff: float,
    dx: float,
    dt: float,
    steps: int,
    snapshot_interval: int = 10,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Solve the 2D heat equation on a square grid.

    Parameters
    ----------
    initial : np.ndarray
        A 2D array of shape (N, N) containing the initial temperature field.
    diffusion_coeff : float
        The diffusion coefficient (thermal diffusivity) ``α``.  This controls the
        rate at which heat spreads across the domain.
    dx : float
        The spatial grid spacing.  The grid is assumed to be uniformly spaced
        in both x and y directions.
    dt : float
        The time step size.  For stability of the explicit scheme, the
        Courant–Friedrichs–Lewy (CFL) condition requires ``dt < dx**2 / (4*α)``.
    steps : int
        The number of time steps to perform.
    snapshot_interval : int, optional
        The interval (in time steps) at which intermediate solutions are
        recorded.  A snapshot of the temperature field will be appended to the
        returned list every ``snapshot_interval`` steps.

    Returns
    -------
    final : np.ndarray
        The temperature field after ``steps`` time steps.
    snapshots : List[np.ndarray]
        A list of intermediate temperature fields captured at the specified
        ``snapshot_interval``.  The first snapshot corresponds to the state
        at time ``snapshot_interval * dt``.

    Notes
    -----
    The solver imposes homogeneous Dirichlet boundary conditions (temperature
    fixed at zero) by assuming the values at the edges of the array remain
    unchanged.  For other boundary conditions, consider modifying how the
    boundary values are updated each time step.
    """
    # Ensure input is a copy to avoid modifying the caller's array
    u = initial.copy().astype(float)
    snapshots: List[np.ndarray] = []

    # Precompute the coefficient used in the update
    alpha = diffusion_coeff
    coef = alpha * dt / (dx * dx)

    # Check stability condition; warn if the scheme may be unstable
    if coef >= 0.25:
        raise ValueError(
            f"Unstable time step: alpha*dt/dx^2 = {coef:.3f} >= 0.25. "
            "Reduce dt or increase dx to satisfy the CFL condition."
        )

    n, m = u.shape
    for step in range(1, steps + 1):
        # Compute new temperature field; use array slicing for efficiency
        u_new = u.copy()
        # Interior points update; skip boundaries (assumed fixed)
        u_new[1:-1, 1:-1] = (
            u[1:-1, 1:-1]
            + coef
            * (
                u[2:, 1:-1]
                + u[:-2, 1:-1]
                + u[1:-1, 2:]
                + u[1:-1, :-2]
                - 4 * u[1:-1, 1:-1]
            )
        )
        u = u_new

        # Record snapshots periodically
        if snapshot_interval > 0 and step % snapshot_interval == 0:
            snapshots.append(u.copy())

    return u, snapshots


def solve_poisson_equation(
    rhs: np.ndarray,
    dx: float,
    max_iter: int = 5000,
    tol: float = 1e-6,
) -> np.ndarray:
    """Solve the 2D Poisson equation on a square grid using Jacobi iteration.

    We seek ``φ`` satisfying ``∇²φ = rhs`` on an interior grid, with φ fixed
    to zero on the boundary.  The Jacobi method repeatedly averages neighboring
    points plus a scaled right‑hand side until convergence.

    Parameters
    ----------
    rhs : np.ndarray
        A 2D array of shape (N, N) containing the source term.  Nonzero
        values represent sources (positive) or sinks (negative) in the domain.
    dx : float
        The spatial grid spacing.
    max_iter : int, optional
        The maximum number of Jacobi iterations to perform.
    tol : float, optional
        The convergence tolerance.  Iteration stops when the maximum change
        between successive iterates is less than ``tol``.

    Returns
    -------
    phi : np.ndarray
        A 2D array approximating the solution to ``∇²φ = rhs`` with zero
        Dirichlet boundary conditions.
    """
    n, m = rhs.shape
    phi = np.zeros_like(rhs, dtype=float)
    phi_new = np.zeros_like(rhs, dtype=float)

    # Precompute constant factor
    factor = dx * dx / 4.0

    for it in range(max_iter):
        # Perform a full Jacobi sweep over interior points
        phi_new[1:-1, 1:-1] = (
            phi[2:, 1:-1]
            + phi[:-2, 1:-1]
            + phi[1:-1, 2:]
            + phi[1:-1, :-2]
            - rhs[1:-1, 1:-1] * dx * dx
        ) / 4.0

        # Compute residual (maximum absolute difference)
        diff = np.max(np.abs(phi_new - phi))
        phi, phi_new = phi_new, phi

        if diff < tol:
            break

    return phi


if __name__ == "__main__":
    # Example usage: demonstrate the solvers with simple initial data
    import matplotlib.pyplot as plt

    # Heat equation example
    N = 50
    init = np.zeros((N, N))
    # Initial hotspot in the center
    init[N // 2 - 5 : N // 2 + 5, N // 2 - 5 : N // 2 + 5] = 1.0

    final_heat, snapshots = solve_heat_equation(
        initial=init,
        diffusion_coeff=1.0,
        dx=1.0 / (N - 1),
        dt=0.1 * (1.0 / (N - 1)) ** 2,
        steps=100,
        snapshot_interval=25,
    )

    # Plot a snapshot of the temperature field
    plt.figure(figsize=(6, 5))
    plt.imshow(snapshots[-1], origin="lower", extent=(0, 1, 0, 1), cmap="hot")
    plt.title("Heat equation intermediate snapshot")
    plt.colorbar(label="Temperature")
    plt.tight_layout()
    plt.show()

    # Poisson equation example
    rhs = np.zeros((N, N))
    # Point source at the centre
    rhs[N // 2, N // 2] = 100.0
    phi = solve_poisson_equation(rhs=rhs, dx=1.0 / (N - 1))
    plt.figure(figsize=(6, 5))
    plt.imshow(phi, origin="lower", extent=(0, 1, 0, 1), cmap="viridis")
    plt.title("Poisson equation solution")
    plt.colorbar(label="Potential")
    plt.tight_layout()
    plt.show()