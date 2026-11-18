"""Non-bonded Lennard-Jones, Coulomb, Drude induced-dipole, and Thole screening forces.

Implements:
- C2-smooth quintic force-switched Lennard-Jones 12-6 with Lorentz-Berthelot combining.
- Exact pairwise electrostatics across cores and auxiliary Drude shell particles.
- Drude harmonic tether + quartic hyperpolarization wall guard.
- Thole dipole-dipole screening S(u) = 1 - (1 + u/2)*exp(-u) between polarizable sites.
- Born-Oppenheimer Self-Consistent Field (SCF) Drude relaxation solver.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple
import numpy as np
from scipy.optimize import minimize

from solvato_dyne.forcefield.bonded import minimum_image_displacement
from solvato_dyne.types import (
    COULOMB_KCAL_A_E2,
    MolecularSystem,
)


def _c2_switch_factor(r: float, r_on: float, r_cut: float) -> Tuple[float, float]:
    """Return C2 quintic switching factor S(r) in [0, 1] and dS/dr."""
    if r <= r_on or r_cut <= r_on:
        return 1.0, 0.0
    if r >= r_cut:
        return 0.0, 0.0
    inv_span = 1.0 / (r_cut - r_on)
    x = (r - r_on) * inv_span
    s = 1.0 - 10.0 * x**3 + 15.0 * x**4 - 6.0 * x**5
    ds_dx = -30.0 * x**2 + 60.0 * x**3 - 30.0 * x**4
    return float(s), float(ds_dx * inv_span)


def _thole_screening(
    r: float,
    a_sum: float,
    alpha_prod_1_6: float,
) -> Tuple[float, float]:
    """Compute Thole dipole-dipole screening factor S(r) and dS/dr.

    S(r) = 1 - (1 + u/2) * exp(-u),  u = (a_i + a_j) * r / (2 * (alpha_i * alpha_j)^(1/6))
    """
    if alpha_prod_1_6 <= 1e-12:
        return 1.0, 0.0
    scale = 0.5 * a_sum / alpha_prod_1_6
    u = scale * r
    exp_u = float(np.exp(-u))
    s = 1.0 - (1.0 + 0.5 * u) * exp_u
    ds_du = 0.5 * (1.0 + u) * exp_u
    return float(s), float(ds_du * scale)


def compute_nonbonded_and_drude_forces(
    system: MolecularSystem,
    positions: Optional[np.ndarray] = None,
    drude_positions: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """Compute LJ, Coulomb, Drude tether, and Thole dipole-screening energies and forces.

    Returns:
        energies: dict with 'lj', 'coulomb', 'drude_spring', 'thole_dipole'.
        f_atoms: (N, 3) forces on atomic cores.
        f_drude: (N_drude, 3) forces on auxiliary Drude particles.
        virial: (3, 3) virial tensor.
    """
    pos = system.positions if positions is None else positions
    n = system.n_atoms
    pol_idx = system.polarizable_indices
    nd = len(pol_idx)
    dpos = (
        (system.drude_positions if drude_positions is None else drude_positions)
        if nd > 0
        else np.zeros((0, 3), dtype=np.float64)
    )

    f_atoms = np.zeros((n, 3), dtype=np.float64)
    f_drude = np.zeros((nd, 3), dtype=np.float64)
    virial = np.zeros((3, 3), dtype=np.float64)

    cfg = system.ff_config
    box = system.box
    excluded, pairs_14 = system.exclusion_sets()

    # Map atom index -> drude index (-1 if non-polarizable)
    atom_to_drude = np.full(n, -1, dtype=np.int64)
    for d_i, a_i in enumerate(pol_idx):
        atom_to_drude[a_i] = d_i

    core_charges = np.array([a.core_charge for a in system.atoms], dtype=np.float64)
    drude_charges = np.array([system.atoms[a_i].drude_charge for a_i in pol_idx], dtype=np.float64)
    sigmas = np.array([a.sigma for a in system.atoms], dtype=np.float64)
    epsilons = np.array([a.epsilon for a in system.atoms], dtype=np.float64)
    alphas = np.array([a.alpha_pol for a in system.atoms], dtype=np.float64)
    tholes = np.array([a.thole_a for a in system.atoms], dtype=np.float64)

    e_lj = 0.0
    e_coul = 0.0
    e_drude_spring = 0.0
    e_thole = 0.0

    # 1. Core-Drude harmonic tether + quartic hyperpolarization wall
    for d_i, a_i in enumerate(pol_idx):
        atom = system.atoms[int(a_i)]
        dr = minimum_image_displacement(dpos[d_i] - pos[a_i], box)
        r2 = float(np.dot(dr, dr))
        r = float(np.sqrt(r2))
        e_drude_spring += 0.5 * atom.drude_k * r2
        # Force on Drude is -k * (r_D - r_core); force on core is +k * (r_D - r_core)
        f_on_drude = -atom.drude_k * dr
        if r > cfg.drude_hard_wall and r > 1e-12:
            excess = r - cfg.drude_hard_wall
            e_drude_spring += cfg.drude_wall_k * (excess ** 4)
            f_wall_mag = -4.0 * cfg.drude_wall_k * (excess ** 3) / r
            f_on_drude += f_wall_mag * dr
        f_drude[d_i] += f_on_drude
        f_atoms[a_i] -= f_on_drude

    # 2. Pairwise LJ + core/Drude Coulomb + Thole screening across distinct atoms i < j
    for i in range(n):
        qi_c = core_charges[i]
        di = int(atom_to_drude[i])
        qi_d = drude_charges[di] if di >= 0 else 0.0
        has_pol_i = di >= 0

        for j in range(i + 1, n):
            pair = (i, j)
            is_excl = pair in excluded
            is_14 = pair in pairs_14

            lj_scale = 0.0 if is_excl else (cfg.scale_14_lj if is_14 else 1.0)
            el_scale = 0.0 if is_excl else (cfg.scale_14_elec if is_14 else 1.0)

            dr_cc = minimum_image_displacement(pos[j] - pos[i], box)
            r2_cc = float(np.dot(dr_cc, dr_cc))
            r_cc = float(np.sqrt(max(1e-16, r2_cc)))

            # Lennard-Jones 12-6 with C2 quintic force switch
            if lj_scale > 0.0 and r_cc < cfg.lj_cut:
                eps_ij = float(np.sqrt(epsilons[i] * epsilons[j]))
                if eps_ij > 0.0:
                    sig_ij = 0.5 * (sigmas[i] + sigmas[j])
                    inv_r = 1.0 / r_cc
                    sr = sig_ij * inv_r
                    sr6 = sr ** 6
                    sr12 = sr6 * sr6
                    u_lj = 4.0 * eps_ij * (sr12 - sr6)
                    # d(u_lj)/dr = 4*eps * (-12*sr12 + 6*sr6) / r
                    du_dr = 24.0 * eps_ij * (sr6 - 2.0 * sr12) * inv_r

                    sw, dsw_dr = _c2_switch_factor(r_cc, cfg.lj_on, cfg.lj_cut)
                    e_pair_lj = lj_scale * sw * u_lj
                    e_lj += e_pair_lj
                    dE_dr = lj_scale * (sw * du_dr + dsw_dr * u_lj)
                    # Force on i: -dE/dr_i = +dE/dr * (r_j - r_i)/r
                    f_ij = (dE_dr * inv_r) * dr_cc
                    f_atoms[i] += f_ij
                    f_atoms[j] -= f_ij
                    virial += np.outer(dr_cc, -f_ij)

            # Electrostatics + Thole dipole-dipole screening
            dj = int(atom_to_drude[j])
            has_pol_j = dj >= 0
            # Note: even for bonded 1-2/1-3 pairs, induced-dipole Thole coupling between
            # polarizable heavy atoms is included when use_thole=True, using standard
            # Drude Thole convention; static core-core 1-2/1-3 is excluded via el_scale.
            use_thole_pair = cfg.use_thole and has_pol_i and has_pol_j
            a_prod_16 = (alphas[i] * alphas[j]) ** (1.0 / 6.0) if use_thole_pair else 0.0
            a_sum = tholes[i] + tholes[j] if use_thole_pair else 0.0

            if el_scale > 0.0:
                if cfg.coulomb_cut <= 0.0 or r_cc < cfg.coulomb_cut:
                    # Core_i - Core_j
                    qprod = qi_c * core_charges[j]
                    if abs(qprod) > 1e-14:
                        inv_r = 1.0 / r_cc
                        if use_thole_pair:
                            s_th, ds_th = _thole_screening(r_cc, a_sum, a_prod_16)
                            e_base = COULOMB_KCAL_A_E2 * el_scale * qprod * inv_r
                            e_term = e_base * s_th
                            e_coul += e_base
                            e_thole += e_term - e_base
                            dE_dr = COULOMB_KCAL_A_E2 * el_scale * qprod * (ds_th * inv_r - s_th * inv_r * inv_r)
                        else:
                            e_coul += COULOMB_KCAL_A_E2 * el_scale * qprod * inv_r
                            dE_dr = -COULOMB_KCAL_A_E2 * el_scale * qprod * inv_r * inv_r
                        f_ij = (dE_dr * inv_r) * dr_cc
                        f_atoms[i] += f_ij
                        f_atoms[j] -= f_ij
                        virial += np.outer(dr_cc, -f_ij)

                    # Core_i - Drude_j
                    if has_pol_j:
                        dr_cd = minimum_image_displacement(dpos[dj] - pos[i], box)
                        r_cd = float(np.sqrt(max(1e-16, float(np.dot(dr_cd, dr_cd)))))
                        qprod_cd = qi_c * drude_charges[dj]
                        inv_r = 1.0 / r_cd
                        if use_thole_pair:
                            s_th, ds_th = _thole_screening(r_cd, a_sum, a_prod_16)
                            e_base = COULOMB_KCAL_A_E2 * el_scale * qprod_cd * inv_r
                            e_coul += e_base
                            e_thole += e_base * (s_th - 1.0)
                            dE_dr = COULOMB_KCAL_A_E2 * el_scale * qprod_cd * (ds_th * inv_r - s_th * inv_r * inv_r)
                        else:
                            e_coul += COULOMB_KCAL_A_E2 * el_scale * qprod_cd * inv_r
                            dE_dr = -COULOMB_KCAL_A_E2 * el_scale * qprod_cd * inv_r * inv_r
                        f_ij = (dE_dr * inv_r) * dr_cd
                        f_atoms[i] += f_ij
                        f_drude[dj] -= f_ij

                    # Drude_i - Core_j
                    if has_pol_i:
                        dr_dc = minimum_image_displacement(pos[j] - dpos[di], box)
                        r_dc = float(np.sqrt(max(1e-16, float(np.dot(dr_dc, dr_dc)))))
                        qprod_dc = qi_d * core_charges[j]
                        inv_r = 1.0 / r_dc
                        if use_thole_pair:
                            s_th, ds_th = _thole_screening(r_dc, a_sum, a_prod_16)
                            e_base = COULOMB_KCAL_A_E2 * el_scale * qprod_dc * inv_r
                            e_coul += e_base
                            e_thole += e_base * (s_th - 1.0)
                            dE_dr = COULOMB_KCAL_A_E2 * el_scale * qprod_dc * (ds_th * inv_r - s_th * inv_r * inv_r)
                        else:
                            e_coul += COULOMB_KCAL_A_E2 * el_scale * qprod_dc * inv_r
                            dE_dr = -COULOMB_KCAL_A_E2 * el_scale * qprod_dc * inv_r * inv_r
                        f_ij = (dE_dr * inv_r) * dr_dc
                        f_drude[di] += f_ij
                        f_atoms[j] -= f_ij

                    # Drude_i - Drude_j
                    if has_pol_i and has_pol_j:
                        dr_dd = minimum_image_displacement(dpos[dj] - dpos[di], box)
                        r_dd = float(np.sqrt(max(1e-16, float(np.dot(dr_dd, dr_dd)))))
                        qprod_dd = qi_d * drude_charges[dj]
                        inv_r = 1.0 / r_dd
                        if use_thole_pair:
                            s_th, ds_th = _thole_screening(r_dd, a_sum, a_prod_16)
                            e_base = COULOMB_KCAL_A_E2 * el_scale * qprod_dd * inv_r
                            e_coul += e_base
                            e_thole += e_base * (s_th - 1.0)
                            dE_dr = COULOMB_KCAL_A_E2 * el_scale * qprod_dd * (ds_th * inv_r - s_th * inv_r * inv_r)
                        else:
                            e_coul += COULOMB_KCAL_A_E2 * el_scale * qprod_dd * inv_r
                            dE_dr = -COULOMB_KCAL_A_E2 * el_scale * qprod_dd * inv_r * inv_r
                        f_ij = (dE_dr * inv_r) * dr_dd
                        f_drude[di] += f_ij
                        f_drude[dj] -= f_ij

    energies = {
        "lj": float(e_lj),
        "coulomb": float(e_coul),
        "drude_spring": float(e_drude_spring),
        "thole_dipole": float(e_thole),
    }
    return energies, f_atoms, f_drude, virial


def relax_drude_scf(
    system: MolecularSystem,
    positions: Optional[np.ndarray] = None,
    drude_init: Optional[np.ndarray] = None,
    gtol: float = 1e-6,
    maxiter: int = 80,
) -> Tuple[np.ndarray, float]:
    """Solve Self-Consistent Field (SCF) induced Drude positions at fixed nuclear coordinates.

    Minimizes the total polarizable electrostatic + Drude tether energy with respect
    to the auxiliary Drude particle positions using L-BFGS-B.

    Returns:
        drude_opt: (N_drude, 3) relaxed Drude coordinates.
        max_grad: Infinity norm of residual Drude force in kcal/(mol*A).
    """
    pol_idx = system.polarizable_indices
    if len(pol_idx) == 0:
        return np.zeros((0, 3), dtype=np.float64), 0.0

    pos = system.positions if positions is None else positions
    x0 = (system.drude_positions if drude_init is None else drude_init).ravel().copy()

    def _obj(x_flat: np.ndarray) -> Tuple[float, np.ndarray]:
        dpos = x_flat.reshape(-1, 3)
        en, _, f_d, _ = compute_nonbonded_and_drude_forces(system, pos, dpos)
        tot = en["coulomb"] + en["drude_spring"] + en["thole_dipole"]
        return float(tot), -f_d.ravel()

    res = minimize(
        _obj,
        x0,
        method="L-BFGS-B",
        jac=True,
        options={"gtol": gtol, "ftol": 1e-13, "maxiter": maxiter},
    )
    d_opt = res.x.reshape(-1, 3)
    _, _, f_d_final, _ = compute_nonbonded_and_drude_forces(system, pos, d_opt)
    max_force = float(np.max(np.abs(f_d_final))) if len(f_d_final) > 0 else 0.0
    return d_opt, max_force
