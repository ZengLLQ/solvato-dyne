"""OBC-II Generalized Born (GBSA) polar solvation and smooth SASA nonpolar energy.

Implements the Onufriev-Bashford-Case (OBC-II) pairwise Hawkins-Cramer-Truhlar
descreening integral, effective Born radii, GB polarization energy, and a
C2-smooth exponential-overlap Solvent Accessible Surface Area (SASA) cavitation
model with exact analytical Cartesian gradients.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple
import numpy as np

from solvato_dyne.forcefield.bonded import minimum_image_displacement
from solvato_dyne.types import COULOMB_KCAL_A_E2, MolecularSystem


_DIELECTRIC_OFFSET = 0.09  # standard Bondi dielectric offset in Angstroms
_OBC_ALPHA = 1.0
_OBC_BETA = 0.8
_OBC_GAMMA = 4.85
_PROBE_RADIUS = 1.4


def _hct_integral_and_deriv(r: float, rho_i: float, s_j: float) -> Tuple[float, float]:
    """Compute pairwise HCT descreening integral I(r, rho_i, s_j) and dI/dr.

    Here rho_i = R_i - 0.09 A and s_j = S_j * (R_j - 0.09 A).
    Uses a smooth floor on r for numerical stability.
    """
    if r >= rho_i + s_j:
        pass  # standard outer regime works smoothly up to r -> inf
    if r + s_j <= rho_i:
        return 0.0, 0.0

    diff = abs(r - s_j)
    if rho_i >= diff:
        l_val = rho_i
        dl_dr = 0.0
    elif r > s_j:
        l_val = r - s_j
        dl_dr = 1.0
    else:
        l_val = s_j - r
        dl_dr = -1.0

    u_val = r + s_j
    du_dr = 1.0

    inv_l = 1.0 / l_val
    inv_u = 1.0 / u_val
    inv_l2 = inv_l * inv_l
    inv_u2 = inv_u * inv_u
    inv_r = 1.0 / max(r, 1e-8)
    s2 = s_j * s_j

    log_lu = float(np.log(l_val * inv_u))
    term_diff2 = inv_l2 - inv_u2

    integral = 0.5 * (
        inv_l
        - inv_u
        + 0.25 * r * (inv_u2 - inv_l2)
        + 0.5 * inv_r * log_lu
        + 0.25 * s2 * inv_r * term_diff2
    )

    # Extra term when atom i is fully engulfed inside scaled sphere j (r < s_j - rho_i)
    if r < s_j - rho_i:
        integral += 2.0 * ((1.0 / rho_i) - inv_l)

    # Exact partial derivatives w.r.t. (r, l_val, u_val)
    dI_dr_explicit = 0.5 * (
        0.25 * (inv_u2 - inv_l2)
        - 0.5 * inv_r * inv_r * log_lu
        - 0.25 * s2 * inv_r * inv_r * term_diff2
    )
    dI_dL = 0.5 * (
        -inv_l2
        + 0.5 * r * (inv_l2 * inv_l)
        + 0.5 * inv_r * inv_l
        - 0.5 * s2 * inv_r * (inv_l2 * inv_l)
    )
    if r < s_j - rho_i:
        dI_dL += 2.0 * inv_l2

    dI_dU = 0.5 * (
        inv_u2
        - 0.5 * r * (inv_u2 * inv_u)
        - 0.5 * inv_r * inv_u
        + 0.5 * s2 * inv_r * (inv_u2 * inv_u)
    )

    dI_dr = dI_dr_explicit + dI_dL * dl_dr + dI_dU * du_dr
    return float(integral), float(dI_dr)


def _c2_buried_overlap(r: float, r1: float, r2: float) -> Tuple[float, float]:
    """Smooth C2-continuous pairwise sphere-overlap area fraction and derivative."""
    r_sum = r1 + r2
    if r >= r_sum:
        return 0.0, 0.0
    r_min = abs(r1 - r2)
    if r <= r_min:
        # Fully overlapped up to smooth cap
        area = np.pi * min(r1, r2) ** 2
        return float(area), 0.0
    # Quintic smoothstep on t in [0, 1] where t = (r_sum - r) / (r_sum - r_min)
    span = r_sum - r_min
    t = (r_sum - r) / span
    s = 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5
    ds_dr = -(30.0 * t**2 - 60.0 * t**3 + 30.0 * t**4) / span
    area_max = np.pi * min(r1, r2) ** 2
    return float(area_max * s), float(area_max * ds_dr)


def compute_gbsa_forces(
    system: MolecularSystem,
    positions: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, float], np.ndarray]:
    """Compute OBC-II Generalized Born polar energy and smooth SASA nonpolar forces.

    Uses total atomic partial charges q_i (or core+Drude combined charge) so
    continuum solvation cleanly couples with bonded/non-bonded geometries.
    """
    cfg = system.ff_config
    n = system.n_atoms
    forces = np.zeros((n, 3), dtype=np.float64)
    if not cfg.use_gbsa or n == 0:
        return {"gb_polar": 0.0, "sasa_nonpolar": 0.0}, forces

    pos = system.positions if positions is None else positions
    box = system.box

    radii = np.array([max(0.5, a.gb_radius) for a in system.atoms], dtype=np.float64)
    rho = np.maximum(0.3, radii - _DIELECTRIC_OFFSET)
    scales = np.array([a.gb_scale for a in system.atoms], dtype=np.float64)
    s_rho = scales * rho
    charges = np.array([a.charge for a in system.atoms], dtype=np.float64)

    # 1. Accumulate pairwise descreening integrals psi_i = rho_i * sum_{j!=i} I_ij
    psi = np.zeros(n, dtype=np.float64)
    pair_r = np.zeros((n, n), dtype=np.float64)
    pair_dr = np.zeros((n, n, 3), dtype=np.float64)
    dI_dr_ij = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(i + 1, n):
            dr = minimum_image_displacement(pos[j] - pos[i], box)
            r = float(np.sqrt(max(1e-14, float(np.dot(dr, dr)))))
            pair_r[i, j] = pair_r[j, i] = r
            pair_dr[i, j] = dr
            pair_dr[j, i] = -dr

            I_ij, dI_ij = _hct_integral_and_deriv(r, rho[i], s_rho[j])
            I_ji, dI_ji = _hct_integral_and_deriv(r, rho[j], s_rho[i])
            psi[i] += rho[i] * I_ij
            psi[j] += rho[j] * I_ji
            dI_dr_ij[i, j] = rho[i] * dI_ij
            dI_dr_ij[j, i] = rho[j] * dI_ji

    # 2. Compute effective Born radii alpha_i and d(alpha_i)/d(psi_i)
    poly = _OBC_ALPHA * psi - _OBC_BETA * psi**2 + _OBC_GAMMA * psi**3
    tanh_poly = np.tanh(poly)
    inv_alpha = (1.0 / rho) - (1.0 / radii) * tanh_poly
    inv_alpha = np.maximum(inv_alpha, 0.02)
    alpha = 1.0 / inv_alpha

    sech2 = 1.0 - tanh_poly**2
    dpoly_dpsi = _OBC_ALPHA - 2.0 * _OBC_BETA * psi + 3.0 * _OBC_GAMMA * psi**2
    # d(1/alpha_i)/d(psi_i) = -(1/R_i) * sech^2(poly) * dpoly_dpsi
    dinv_alpha_dpsi = -(1.0 / radii) * sech2 * dpoly_dpsi
    dalpha_dpsi = -(alpha ** 2) * dinv_alpha_dpsi

    # 3. Generalized Born energy E_GB = -0.5 * tau * k_e * sum_{i,j} q_i q_j / f_GB
    tau = (1.0 / cfg.dielectric_in) - (1.0 / cfg.dielectric_out)
    pref = -0.5 * COULOMB_KCAL_A_E2 * tau

    e_gb = 0.0
    dE_dalpha = np.zeros(n, dtype=np.float64)

    # Diagonal self-terms (i == j, r_ij = 0 -> f_GB = alpha_i)
    for i in range(n):
        qi2 = charges[i] * charges[i]
        if qi2 > 0.0:
            e_gb += pref * qi2 / alpha[i]
            dE_dalpha[i] -= pref * qi2 / (alpha[i] ** 2)

    # Off-diagonal pairs (factor of 2 cancels 0.5)
    for i in range(n):
        for j in range(i + 1, n):
            qq = charges[i] * charges[j]
            if abs(qq) < 1e-15:
                continue
            r = pair_r[i, j]
            r2 = r * r
            aa = alpha[i] * alpha[j]
            exp_term = float(np.exp(-r2 / (4.0 * aa)))
            f_gb2 = r2 + aa * exp_term
            f_gb = float(np.sqrt(f_gb2))
            inv_fgb = 1.0 / f_gb
            inv_fgb3 = inv_fgb / f_gb2

            e_gb += 2.0 * pref * qq * inv_fgb

            # Direct radial derivative: df_GB^2 / dr = 2*r - 0.5*r*exp_term
            df2_dr = 2.0 * r * (1.0 - 0.25 * exp_term)
            dE_dr_direct = -pref * qq * inv_fgb3 * df2_dr
            # F_i = -dE/dr_i = +dE/dr * (r_j - r_i)/r
            f_dir = (dE_dr_direct / r) * pair_dr[i, j]
            forces[i] += f_dir
            forces[j] -= f_dir

            # Born-radius chain rule: df_GB^2 / d(aa) = exp_term * (1 + r2 / (4*aa))
            df2_daa = exp_term * (1.0 + r2 / (4.0 * aa))
            dE_daa = -pref * qq * inv_fgb3 * df2_daa
            dE_dalpha[i] += dE_daa * alpha[j]
            dE_dalpha[j] += dE_daa * alpha[i]

    # Propagate Born-radius gradients dE/d(psi_k) -> forces
    dE_dpsi = dE_dalpha * dalpha_dpsi
    for i in range(n):
        for j in range(i + 1, n):
            r = pair_r[i, j]
            if r < 1e-12:
                continue
            dE_dr_born = dE_dpsi[i] * dI_dr_ij[i, j] + dE_dpsi[j] * dI_dr_ij[j, i]
            f_born = (dE_dr_born / r) * pair_dr[i, j]
            forces[i] += f_born
            forces[j] -= f_born

    # 4. Smooth SASA nonpolar cavitation energy
    e_sasa = 0.0
    if cfg.sasa_gamma > 0.0:
        r_probe = radii + _PROBE_RADIUS
        s_max = 4.0 * np.pi * (r_probe ** 2)
        buried = np.zeros(n, dtype=np.float64)
        db_dr = np.zeros((n, n), dtype=np.float64)

        for i in range(n):
            for j in range(i + 1, n):
                ov_i, dov_i = _c2_buried_overlap(pair_r[i, j], r_probe[i], r_probe[j])
                ov_j, dov_j = _c2_buried_overlap(pair_r[i, j], r_probe[j], r_probe[i])
                buried[i] += 0.35 * ov_i
                buried[j] += 0.35 * ov_j
                db_dr[i, j] = 0.35 * dov_i
                db_dr[j, i] = 0.35 * dov_j

        sasa_i = s_max * np.exp(-buried / s_max)
        e_sasa = float(cfg.sasa_gamma * np.sum(sasa_i))

        for i in range(n):
            for j in range(i + 1, n):
                r = pair_r[i, j]
                if r < 1e-12:
                    continue
                dE_dr_sasa = cfg.sasa_gamma * (
                    -(sasa_i[i] / s_max[i]) * db_dr[i, j]
                    - (sasa_i[j] / s_max[j]) * db_dr[j, i]
                )
                f_s = (dE_dr_sasa / r) * pair_dr[i, j]
                forces[i] += f_s
                forces[j] -= f_s

    return {"gb_polar": float(e_gb), "sasa_nonpolar": float(e_sasa)}, forces
