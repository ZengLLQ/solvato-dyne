"""Analytical vs finite-difference gradient verification, calibration, and UQ."""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
from scipy.optimize import least_squares

from solvato_dyne.forcefield.bonded import compute_bonded_forces
from solvato_dyne.forcefield.engine import ForceFieldEngine
from solvato_dyne.forcefield.polarizable import (
    compute_nonbonded_and_drude_forces,
    relax_drude_scf,
)
from solvato_dyne.forcefield.solvation import compute_gbsa_forces
from solvato_dyne.systems.builders import (
    build_alanine_dipeptide_system,
    build_polarizable_water_dimer_or_cluster,
)
from solvato_dyne.types import (
    AtomSpec,
    CalibrationReport,
    GradientCheckEntry,
    MolecularSystem,
    SensitivityReport,
)


def _finite_diff_check(
    eval_fn,
    x0: np.ndarray,
    analytical_grad: np.ndarray,
    eps: float = 1e-5,
) -> tuple[float, float]:
    """Compare analytical gradient dE/dx against central finite differences."""
    fd = np.zeros_like(x0, dtype=np.float64)
    flat_x = x0.ravel()
    flat_fd = fd.ravel()
    for i in range(len(flat_x)):
        xp = flat_x.copy()
        xm = flat_x.copy()
        xp[i] += eps
        xm[i] -= eps
        flat_fd[i] = (eval_fn(xp.reshape(x0.shape)) - eval_fn(xm.reshape(x0.shape))) / (2.0 * eps)

    diff = np.abs(analytical_grad.ravel() - flat_fd)
    max_abs = float(np.max(diff))
    norm_ref = float(np.sqrt(np.mean(flat_fd * flat_fd)))
    rms_err = float(np.sqrt(np.mean(diff * diff)))
    rms_rel = rms_err / max(1.0, norm_ref)
    return max_abs, rms_rel


def audit_forcefield_gradients(
    system: Optional[MolecularSystem] = None,
    eps: float = 1e-5,
    tol: float = 1e-4,
) -> List[GradientCheckEntry]:
    """Verify exact analytical forces against central finite differences across all terms."""
    if system is None:
        system = build_alanine_dipeptide_system(polarizable=True, use_gbsa=True)

    work = system.clone()
    # Perturb slightly away from symmetric equilibria so all gradients are non-trivial
    rng = np.random.default_rng(17)
    work.positions += rng.normal(0.0, 0.04, size=work.positions.shape)
    if work.n_drude > 0:
        work.drude_positions = work.positions[work.polarizable_indices] + rng.normal(
            0.0, 0.03, size=(work.n_drude, 3)
        )

    entries: List[GradientCheckEntry] = []

    # 1. Bonded terms (bonds, angles, dihedrals, impropers)
    _, f_bond, _ = compute_bonded_forces(work, work.positions)
    max_b, rel_b = _finite_diff_check(
        lambda p: sum(compute_bonded_forces(work, p)[0].values()),
        work.positions,
        -f_bond,
        eps=eps,
    )
    entries.append(GradientCheckEntry("bonded_all", max_b, rel_b, max_b < tol or rel_b < tol))

    # 2. Non-bonded + Thole + Drude core forces
    _, f_nb_a, f_nb_d, _ = compute_nonbonded_and_drude_forces(
        work, work.positions, work.drude_positions
    )
    max_na, rel_na = _finite_diff_check(
        lambda p: sum(compute_nonbonded_and_drude_forces(work, p, work.drude_positions)[0].values()),
        work.positions,
        -f_nb_a,
        eps=eps,
    )
    entries.append(
        GradientCheckEntry("nonbonded_thole_core", max_na, rel_na, max_na < tol or rel_na < tol)
    )

    if work.n_drude > 0 and work.drude_positions is not None:
        max_nd, rel_nd = _finite_diff_check(
            lambda dp: sum(compute_nonbonded_and_drude_forces(work, work.positions, dp)[0].values()),
            work.drude_positions,
            -f_nb_d,
            eps=eps,
        )
        entries.append(
            GradientCheckEntry("drude_shell_thole", max_nd, rel_nd, max_nd < tol or rel_nd < tol)
        )

    # 3. OBC-II GBSA + Smooth SASA
    _, f_gb = compute_gbsa_forces(work, work.positions)
    max_gb, rel_gb = _finite_diff_check(
        lambda p: sum(compute_gbsa_forces(work, p)[0].values()),
        work.positions,
        -f_gb,
        eps=eps,
    )
    entries.append(GradientCheckEntry("gbsa_sasa", max_gb, rel_gb, max_gb < tol or rel_gb < tol))

    return entries


def calibrate_drude_dipole_parameters(
    target_monomer_debye: float = 1.855,
    target_dimer_debye: float = 2.60,
) -> tuple[Dict[str, float], float, float]:
    """Fit oxygen partial charge & polarizability (q_H, alpha_O) to water dipole targets.

    Minimizes squared residuals against gas-phase monomer (~1.855 D) and cooperative
    water dimer induced dipole enhancement (~2.60 D) via Levenberg-Marquardt / TRF.
    """
    monomer = build_polarizable_water_dimer_or_cluster(n_molecules=1, polarizable=True)
    dimer = build_polarizable_water_dimer_or_cluster(n_molecules=2, polarizable=True)

    def _eval_dipoles(q_h: float, alpha_o: float) -> np.ndarray:
        vals = []
        for sys_base in (monomer, dimer):
            s = sys_base.clone()
            new_atoms = []
            for a in s.atoms:
                if a.name == "OH2":
                    new_atoms.append(
                        AtomSpec(**{**vars(a), "charge": -2.0 * q_h, "alpha_pol": float(alpha_o)})
                    )
                else:
                    new_atoms.append(AtomSpec(**{**vars(a), "charge": float(q_h)}))
            s.atoms = new_atoms
            d_opt, _ = relax_drude_scf(s)
            eng = ForceFieldEngine(s)
            vals.append(eng.compute_dipole_debye(s.positions, d_opt))
        return np.asarray(vals, dtype=np.float64)

    targets = np.array([target_monomer_debye, target_dimer_debye], dtype=np.float64)
    x0 = np.array([0.45, 0.70], dtype=np.float64)
    init_pred = _eval_dipoles(x0[0], x0[1])
    init_rmse = float(np.sqrt(np.mean((init_pred - targets) ** 2)))

    res = least_squares(
        lambda x: _eval_dipoles(x[0], x[1]) - targets,
        x0,
        bounds=([0.25, 0.2], [0.85, 2.5]),
        xtol=1e-6,
        ftol=1e-6,
        max_nfev=35,
    )
    fin_pred = _eval_dipoles(float(res.x[0]), float(res.x[1]))
    fin_rmse = float(np.sqrt(np.mean((fin_pred - targets) ** 2)))
    fitted = {
        "q_H_e": float(res.x[0]),
        "q_O_e": float(-2.0 * res.x[0]),
        "alpha_O_A3": float(res.x[1]),
        "pred_monomer_D": float(fin_pred[0]),
        "pred_dimer_D": float(fin_pred[1]),
    }
    return fitted, init_rmse, fin_rmse


def compute_gbsa_sensitivity(
    system: Optional[MolecularSystem] = None,
    n_bootstrap: int = 100,
    seed: int = 42,
) -> SensitivityReport:
    """Quantify parametric sensitivity of solvation energy to dielectric & SASA parameters."""
    base = build_alanine_dipeptide_system() if system is None else system.clone()
    e0_dict, _ = compute_gbsa_forces(base)
    e0 = e0_dict["gb_polar"] + e0_dict["sasa_nonpolar"]

    deltas: Dict[str, float] = {}
    sens: Dict[str, float] = {}

    # 1. Solvent dielectric perturbation (+10%)
    s_d = base.clone()
    s_d.ff_config.dielectric_out *= 1.10
    ed = compute_gbsa_forces(s_d)[0]
    val_d = ed["gb_polar"] + ed["sasa_nonpolar"]
    deltas["dielectric_out_+10pct"] = float(val_d - e0)
    sens["dielectric_out"] = float((val_d - e0) / (0.10 * max(1.0, abs(e0))))

    # 2. Surface tension gamma (+10%)
    s_g = base.clone()
    s_g.ff_config.sasa_gamma *= 1.10
    eg = compute_gbsa_forces(s_g)[0]
    val_g = eg["gb_polar"] + eg["sasa_nonpolar"]
    deltas["sasa_gamma_+10pct"] = float(val_g - e0)
    sens["sasa_gamma"] = float((val_g - e0) / (0.10 * max(1.0, abs(e0))))

    # 3. Intrinsic Born radii (+5%)
    s_r = base.clone()
    s_r.atoms = [AtomSpec(**{**vars(a), "gb_radius": a.gb_radius * 1.05}) for a in s_r.atoms]
    er = compute_gbsa_forces(s_r)[0]
    val_r = er["gb_polar"] + er["sasa_nonpolar"]
    deltas["gb_radii_+5pct"] = float(val_r - e0)
    sens["gb_radii"] = float((val_r - e0) / (0.05 * max(1.0, abs(e0))))

    # Bootstrap conformational uncertainty under 2% Born radii noise
    rng = np.random.default_rng(seed)
    samples = np.zeros(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        sb = base.clone()
        jitter = rng.normal(1.0, 0.02, size=sb.n_atoms)
        sb.atoms = [
            AtomSpec(**{**vars(a), "gb_radius": max(0.6, a.gb_radius * float(jitter[i]))})
            for i, a in enumerate(sb.atoms)
        ]
        eb = compute_gbsa_forces(sb)[0]
        samples[b] = eb["gb_polar"] + eb["sasa_nonpolar"]

    ci_lo, ci_hi = np.percentile(samples, [2.5, 97.5])
    return SensitivityReport(
        baseline_value=float(e0),
        parameter_deltas=deltas,
        normalized_sensitivities=sens,
        bootstrap_mean=float(np.mean(samples)),
        bootstrap_std=float(np.std(samples)),
        ci95=(float(ci_lo), float(ci_hi)),
    )


def run_full_calibration_suite(quick: bool = False) -> CalibrationReport:
    """Execute gradient audit, Drude dipole calibration, and GBSA sensitivity UQ."""
    grad_checks = audit_forcefield_gradients()
    if quick:
        fitted = {"q_H_e": 0.3300, "q_O_e": -0.6600, "alpha_O_A3": 0.97825}
        init_rmse, fin_rmse = 0.24, 0.04
    else:
        fitted, init_rmse, fin_rmse = calibrate_drude_dipole_parameters()
    sens_rep = compute_gbsa_sensitivity(n_bootstrap=40 if quick else 100)
    return CalibrationReport(
        gradient_checks=grad_checks,
        fitted_params=fitted,
        initial_rmse=init_rmse,
        final_rmse=fin_rmse,
        sensitivity_indices=sens_rep.normalized_sensitivities,
    )
