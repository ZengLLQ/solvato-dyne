"""Eckart-projected Cartesian Hessian Normal Mode Analysis (NMA) and ANM.

Computes mass-weighted Hessian eigenvalues, projects out the 6 rigid-body
translational and rotational zero modes, evaluates vibrational wavenumbers (cm^-1),
quantum harmonic ZPVE, vibrational entropy S_vib, Helmholtz free energy F_vib,
and predicted Debye-Waller crystallographic B-factors.
"""

from __future__ import annotations

import numpy as np

from solvato_dyne.dynamics.constraints import minimize_energy
from solvato_dyne.forcefield.engine import ForceFieldEngine
from solvato_dyne.forcefield.polarizable import relax_drude_scf
from solvato_dyne.types import (
    CM_INV_TO_KCAL_MOL,
    FORCE_TO_A_PS2_PER_AMU,
    KB_KCAL_MOL_K,
    MolecularSystem,
    NMAResult,
    SPEED_OF_LIGHT_A_PS,
)


def build_eckart_basis(positions: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Construct orthonormal 3N x 6 Eckart basis spanning rigid translations and rotations."""
    n = len(masses)
    sqrt_m = np.sqrt(masses)
    com = np.sum(positions * masses[:, None], axis=0) / np.sum(masses)
    rel = positions - com

    # Align to principal axes of inertia so rotational basis vectors are orthogonal
    inertia = np.zeros((3, 3), dtype=np.float64)
    for i in range(n):
        r = rel[i]
        inertia += masses[i] * (float(np.dot(r, r)) * np.eye(3) - np.outer(r, r))
    _, eigvecs = np.linalg.eigh(inertia)

    basis = np.zeros((3 * n, 6), dtype=np.float64)
    # 3 Translational modes
    for alpha in range(3):
        vec = np.zeros((n, 3), dtype=np.float64)
        vec[:, alpha] = sqrt_m
        basis[:, alpha] = vec.ravel()

    # 3 Rotational modes about principal axes
    for alpha in range(3):
        axis = eigvecs[:, alpha]
        vec = np.zeros((n, 3), dtype=np.float64)
        for i in range(n):
            vec[i] = sqrt_m[i] * np.cross(axis, rel[i])
        basis[:, 3 + alpha] = vec.ravel()

    # Gram-Schmidt orthonormalization (handles linear molecules gracefully)
    q_cols = []
    for k in range(6):
        v = basis[:, k].copy()
        for q in q_cols:
            v -= float(np.dot(v, q)) * q
        norm = float(np.linalg.norm(v))
        if norm > 1e-10:
            q_cols.append(v / norm)
    return np.column_stack(q_cols) if q_cols else np.zeros((3 * n, 0), dtype=np.float64)


def _thermodynamic_nma_summary(
    method: str,
    raw_evals: np.ndarray,
    modes_mw: np.ndarray,
    masses: np.ndarray,
    eckart_basis: np.ndarray,
    hessian_mw: np.ndarray,
    temperature_k: float,
) -> NMAResult:
    """Project out rigid modes, convert eigenvalues to cm^-1, and compute quantum thermo."""
    n = len(masses)
    # Measure residual along rigid-body Eckart subspace
    if eckart_basis.shape[1] > 0:
        proj_rigid = eckart_basis.T @ hessian_mw @ eckart_basis
        zero_mode_residual = float(np.max(np.abs(np.linalg.eigvalsh(proj_rigid))))
    else:
        zero_mode_residual = 0.0

    # Construct orthogonal complement projector P = I - Q Q^T
    proj = np.eye(3 * n, dtype=np.float64)
    if eckart_basis.shape[1] > 0:
        proj -= eckart_basis @ eckart_basis.T
    h_vib = 0.5 * (proj @ hessian_mw @ proj + (proj @ hessian_mw @ proj).T)

    evals, evecs = np.linalg.eigh(h_vib)

    # Identify internal vibrational modes by smallest overlap with Eckart rigid subspace
    if eckart_basis.shape[1] > 0:
        rigid_overlap = np.sum((eckart_basis.T @ evecs) ** 2, axis=0)
        n_rigid = eckart_basis.shape[1]
        vib_order = np.argsort(rigid_overlap)[: 3 * n - n_rigid]
        vib_mask = np.zeros(3 * n, dtype=bool)
        vib_mask[vib_order] = True
    else:
        vib_mask = np.ones(3 * n, dtype=bool)

    vib_evals = evals[vib_mask]
    vib_evecs = evecs[:, vib_mask]

    # Sort ascending
    order = np.argsort(vib_evals)
    vib_evals = vib_evals[order]
    vib_evecs = vib_evecs[:, order]

    # Convert eigenvalue lambda [kcal/(mol*amu*A^2)] -> omega [rad/ps] -> wavenumber [cm^-1]
    omega_sq = vib_evals * FORCE_TO_A_PS2_PER_AMU
    freqs_cm = np.zeros_like(vib_evals)
    pos_mask = omega_sq > 1e-8
    neg_mask = omega_sq < -1e-8
    # cm^-1 = omega / (2*pi*c) where c in A/ps = 2.99792458e6, 1 cm = 1e8 A -> 1 A^-1 = 1e8 cm^-1
    factor_cm = 1e8 / (2.0 * np.pi * SPEED_OF_LIGHT_A_PS)
    freqs_cm[pos_mask] = np.sqrt(omega_sq[pos_mask]) * factor_cm
    freqs_cm[neg_mask] = -np.sqrt(-omega_sq[neg_mask]) * factor_cm

    # Quantum harmonic thermodynamics for real positive modes
    kbt = KB_KCAL_MOL_K * float(temperature_k)
    zpve = 0.0
    s_vib = 0.0
    f_vib = 0.0
    b_factors = np.zeros(n, dtype=np.float64)

    sqrt_m_inv = 1.0 / np.sqrt(masses)
    # Un-mass-weight eigenvectors
    modes_cart = np.zeros_like(vib_evecs)
    for k in range(vib_evecs.shape[1]):
        vec = vib_evecs[:, k].reshape(n, 3) * sqrt_m_inv[:, None]
        nrm = float(np.linalg.norm(vec))
        if nrm > 0.0:
            vec /= nrm
        modes_cart[:, k] = vec.ravel()

    for k, nu in enumerate(freqs_cm):
        if nu <= 1.0:
            continue
        hv = CM_INV_TO_KCAL_MOL * nu
        zpve += 0.5 * hv
        x = hv / kbt
        if x < 60.0:
            ex = float(np.exp(-x))
            s_vib += KB_KCAL_MOL_K * (x * ex / (1.0 - ex) - float(np.log(1.0 - ex)))
            f_vib += 0.5 * hv + kbt * float(np.log(1.0 - ex))
        else:
            f_vib += 0.5 * hv

        # Classical equipartition mean-square displacement contribution:
        # <dq_k^2> = k_B T / lambda_k (in amu*A^2)
        lam_k = float(vib_evals[k])
        if lam_k > 1e-6:
            mw_mode = vib_evecs[:, k].reshape(n, 3)
            msd_atom = (kbt / lam_k) * np.sum((mw_mode * sqrt_m_inv[:, None]) ** 2, axis=1)
            b_factors += (8.0 * (np.pi ** 2) / 3.0) * msd_atom

    _ = raw_evals, modes_mw
    return NMAResult(
        method=method,
        eigenvalues=vib_evals,
        frequencies_cm_inv=freqs_cm,
        modes_cartesian=modes_cart,
        zero_mode_residual=zero_mode_residual,
        zpve_kcal_mol=float(zpve),
        vibrational_entropy_kcal_mol_k=float(s_vib),
        helmholtz_vib_kcal_mol=float(f_vib),
        b_factors_a2=b_factors,
    )


def run_hessian_nma(
    system: MolecularSystem,
    temperature_k: float = 300.0,
    minimize_first: bool = True,
    step_a: float = 1e-4,
) -> NMAResult:
    """Perform Eckart-projected full Cartesian Hessian Normal Mode Analysis.

    Computes the 3N x 3N Hessian by central finite differences of exact analytical
    forces, mass-weights H_mw = M^(-1/2) H M^(-1/2), and extracts vibrational modes.
    """
    work = system.clone()
    if minimize_first:
        work, _, _ = minimize_energy(work, maxiter=180, gtol=1e-4, relax_drude=True)

    engine = ForceFieldEngine(work)
    n = work.n_atoms
    dim = 3 * n
    pos0 = work.positions.copy()
    dpos0 = work.drude_positions.copy() if work.n_drude > 0 else None

    hess = np.zeros((dim, dim), dtype=np.float64)
    for idx in range(dim):
        a_i = idx // 3
        c_i = idx % 3

        p_plus = pos0.copy()
        p_plus[a_i, c_i] += step_a
        dp_p = dpos0
        if work.n_drude > 0:
            dp_p, _ = relax_drude_scf(work, p_plus, dpos0, gtol=1e-6, maxiter=30)
        _, f_plus, _ = engine.evaluate(positions=p_plus, drude_positions=dp_p)

        p_minus = pos0.copy()
        p_minus[a_i, c_i] -= step_a
        dp_m = dpos0
        if work.n_drude > 0:
            dp_m, _ = relax_drude_scf(work, p_minus, dpos0, gtol=1e-6, maxiter=30)
        _, f_minus, _ = engine.evaluate(positions=p_minus, drude_positions=dp_m)

        # H[:, idx] = d^2 E / (dr dr_idx) = -(F_plus - F_minus) / (2 * step)
        hess[:, idx] = -(f_plus - f_minus).ravel() / (2.0 * step_a)

    hess = 0.5 * (hess + hess.T)
    masses = np.array([a.mass for a in work.atoms], dtype=np.float64)
    inv_sqrt_m = np.repeat(1.0 / np.sqrt(masses), 3)
    hess_mw = hess * np.outer(inv_sqrt_m, inv_sqrt_m)

    eckart = build_eckart_basis(pos0, masses)
    evals, evecs = np.linalg.eigh(hess_mw)
    return _thermodynamic_nma_summary(
        method="hessian_nma",
        raw_evals=evals,
        modes_mw=evecs,
        masses=masses,
        eckart_basis=eckart,
        hessian_mw=hess_mw,
        temperature_k=temperature_k,
    )


def run_anm_analysis(
    system: MolecularSystem,
    cutoff_a: float = 8.0,
    gamma_k: float = 10.0,
    temperature_k: float = 300.0,
) -> NMAResult:
    """Perform Anisotropic Network Model (ANM) elastic normal mode analysis."""
    n = system.n_atoms
    pos = system.positions
    masses = np.array([a.mass for a in system.atoms], dtype=np.float64)
    hess = np.zeros((3 * n, 3 * n), dtype=np.float64)

    for i in range(n):
        for j in range(i + 1, n):
            dr = pos[j] - pos[i]
            r2 = float(np.dot(dr, dr))
            r = float(np.sqrt(r2))
            if r <= cutoff_a and r > 1e-8:
                block = -gamma_k * np.outer(dr, dr) / r2
                si = slice(3 * i, 3 * i + 3)
                sj = slice(3 * j, 3 * j + 3)
                hess[si, sj] += block
                hess[sj, si] += block
                hess[si, si] -= block
                hess[sj, sj] -= block

    inv_sqrt_m = np.repeat(1.0 / np.sqrt(masses), 3)
    hess_mw = hess * np.outer(inv_sqrt_m, inv_sqrt_m)
    eckart = build_eckart_basis(pos, masses)
    evals, evecs = np.linalg.eigh(hess_mw)
    return _thermodynamic_nma_summary(
        method="anm",
        raw_evals=evals,
        modes_mw=evecs,
        masses=masses,
        eckart_basis=eckart,
        hessian_mw=hess_mw,
        temperature_k=temperature_k,
    )
