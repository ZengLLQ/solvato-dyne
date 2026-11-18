"""Holonomic RATTLE bond constraints and L-BFGS-B energy minimization."""

from __future__ import annotations

from typing import List, Tuple
import numpy as np
from scipy.optimize import minimize

from solvato_dyne.forcefield.bonded import minimum_image_displacement
from solvato_dyne.forcefield.engine import ForceFieldEngine
from solvato_dyne.forcefield.polarizable import relax_drude_scf
from solvato_dyne.types import ConstraintPair, MolecularSystem


def apply_rattle_positions(
    positions: np.ndarray,
    ref_positions: np.ndarray,
    inv_masses: np.ndarray,
    constraints: List[ConstraintPair],
    dt_ps: float,
    box=None,
    tol: float = 1e-10,
    max_iter: int = 120,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Enforce ||r_i - r_j|| = d0 via iterative RATTLE coordinate projection.

    Returns corrected positions, updated half/full step velocities delta_v = (r_new - r_old)/dt,
    and convergence status.
    """
    if not constraints:
        return positions, np.zeros_like(positions), True

    pos = positions.copy()
    converged = False
    for _ in range(max_iter):
        max_err = 0.0
        for c in constraints:
            rij = minimum_image_displacement(pos[c.j] - pos[c.i], box)
            r2 = float(np.dot(rij, rij))
            d2 = c.d0 * c.d0
            diff = r2 - d2
            rel_err = abs(diff) / max(d2, 1e-12)
            if rel_err > max_err:
                max_err = rel_err
            if rel_err <= tol:
                continue

            rij_ref = minimum_image_displacement(ref_positions[c.j] - ref_positions[c.i], box)
            denom = 2.0 * (inv_masses[c.i] + inv_masses[c.j]) * float(np.dot(rij, rij_ref))
            if abs(denom) < 1e-14:
                continue
            lam = diff / denom
            # Note: rij = r_j - r_i, so r_i <- r_i + lam * inv_m_i * rij_ref; r_j <- r_j - lam * inv_m_j * rij_ref
            pos[c.i] += lam * inv_masses[c.i] * rij_ref
            pos[c.j] -= lam * inv_masses[c.j] * rij_ref
        if max_err <= tol:
            converged = True
            break

    dv = (pos - positions) / max(dt_ps, 1e-12)
    return pos, dv, converged


def apply_rattle_velocities(
    positions: np.ndarray,
    velocities: np.ndarray,
    inv_masses: np.ndarray,
    constraints: List[ConstraintPair],
    box=None,
    tol: float = 1e-10,
    max_iter: int = 120,
) -> Tuple[np.ndarray, bool]:
    """Project velocities onto constraint tangent manifold: (r_j - r_i) . (v_j - v_i) = 0."""
    if not constraints:
        return velocities, True

    vel = velocities.copy()
    converged = False
    for _ in range(max_iter):
        max_err = 0.0
        for c in constraints:
            rij = minimum_image_displacement(positions[c.j] - positions[c.i], box)
            vij = vel[c.j] - vel[c.i]
            dot_rv = float(np.dot(rij, vij))
            err = abs(dot_rv)
            if err > max_err:
                max_err = err
            if err <= tol:
                continue
            r2 = float(np.dot(rij, rij))
            denom = (inv_masses[c.i] + inv_masses[c.j]) * max(r2, 1e-14)
            mu = dot_rv / denom
            vel[c.i] += mu * inv_masses[c.i] * rij
            vel[c.j] -= mu * inv_masses[c.j] * rij
        if max_err <= tol:
            converged = True
            break
    return vel, converged


def minimize_energy(
    system: MolecularSystem,
    maxiter: int = 150,
    gtol: float = 1e-3,
    relax_drude: bool = True,
) -> Tuple[MolecularSystem, float, float]:
    """Minimize potential energy using L-BFGS-B over atomic positions.

    Returns a cloned MolecularSystem at the local minimum, initial energy, and final energy.
    """
    work = system.clone()
    engine = ForceFieldEngine(work)
    eb0, _, _ = engine.evaluate(scf_drude=relax_drude)
    e_init = eb0.total_potential

    def _objective(x_flat: np.ndarray) -> Tuple[float, np.ndarray]:
        pos = x_flat.reshape(-1, 3)
        dpos = work.drude_positions
        if relax_drude and work.n_drude > 0:
            dpos, _ = relax_drude_scf(work, pos, dpos, gtol=1e-5, maxiter=40)
            work.drude_positions = dpos
        eb, f_atom, _ = engine.evaluate(positions=pos, drude_positions=dpos)
        return float(eb.total_potential), -f_atom.ravel()

    res = minimize(
        _objective,
        work.positions.ravel().copy(),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": maxiter, "gtol": gtol, "ftol": 1e-12},
    )
    work.positions = res.x.reshape(-1, 3)
    if relax_drude and work.n_drude > 0:
        dpos_opt, _ = relax_drude_scf(work, work.positions, work.drude_positions, gtol=1e-6)
        work.drude_positions = dpos_opt

    eb_fin, _, _ = engine.evaluate()
    return work, float(e_init), float(eb_fin.total_potential)
