"""Pure 7-bit ASCII terminal visualizers (codes 0-127 strictly, cp1252-safe)."""

from __future__ import annotations

from typing import List
import numpy as np

from solvato_dyne.types import (
    CalibrationReport,
    MetaDResult,
    NMAResult,
    REMDResult,
    SMDResult,
    Trajectory,
)


_CONTOUR_RAMP = " .:-=+*#%@"


def assert_ascii(text: str) -> str:
    """Verify every character is strict 7-bit ASCII (0 <= ord(c) <= 127)."""
    for idx, ch in enumerate(text):
        if ord(ch) > 127:
            raise ValueError(f"Non-ASCII character {repr(ch)} at index {idx}")
    return text


def render_energy_summary(trajectory: Trajectory) -> str:
    """Render decomposed energy summary and temperature/dipole strip table."""
    last = trajectory.frames[-1].energy
    pot = trajectory.potential_energies()
    t_atom = trajectory.temperatures_atom()
    t_drude = trajectory.temperatures_drude()
    dips = trajectory.dipoles()
    drift = trajectory.energy_drift_per_step()

    lines: List[str] = [
        f"=== SOLVATODYNE TRAJECTORY SUMMARY: {trajectory.system_name} ({trajectory.integrator}) ===",
        f"Frames: {trajectory.n_frames} | dt: {trajectory.dt_ps:.4f} ps | Rel Energy Drift: {drift:.3e}",
        f"Mean Potential : {float(np.mean(pot)):10.4f} kcal/mol (std {float(np.std(pot)):7.4f})",
        f"Mean T_atom    : {float(np.mean(t_atom)):10.2f} K | Mean T_drude: {float(np.mean(t_drude)):8.3f} K",
        f"Mean Dipole    : {float(np.mean(dips)):10.4f} D (std {float(np.std(dips)):7.4f} D)",
        "--- Final Frame Energy Decomposition (kcal/mol) ---",
        f"  Bond      : {last.bond:10.4f} | Angle     : {last.angle:10.4f}",
        f"  Dihedral  : {last.dihedral:10.4f} | Improper  : {last.improper:10.4f}",
        f"  Elastic   : {last.elastic_net:10.4f} | LJ 12-6   : {last.lj:10.4f}",
        f"  Coulomb   : {last.coulomb:10.4f} | Thole     : {last.thole_dipole:10.4f}",
        f"  DrudeSpr  : {last.drude_spring:10.4f} | GB Polar  : {last.gb_polar:10.4f}",
        f"  SASA      : {last.sasa_nonpolar:10.4f} | Kinetic   : {last.total_kinetic:10.4f}",
        f"  Total E   : {last.total_energy:10.4f} | Ext Ham   : {last.extended_hamiltonian:10.4f}",
    ]
    return assert_ascii("\n".join(lines))


def render_1d_profile(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    title: str,
    x_label: str,
    y_label: str,
    width: int = 56,
    height: int = 12,
) -> str:
    """Render an ASCII XY scatter/curve profile."""
    x = np.asarray(x_vals, dtype=np.float64)
    y = np.asarray(y_vals, dtype=np.float64)
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    if abs(y_max - y_min) < 1e-12:
        y_max = y_min + 1.0
    if abs(x_max - x_min) < 1e-12:
        x_max = x_min + 1.0

    canvas = [[" " for _ in range(width)] for _ in range(height)]
    for xv, yv in zip(x, y):
        col = int(round((xv - x_min) / (x_max - x_min) * (width - 1)))
        row = height - 1 - int(round((yv - y_min) / (y_max - y_min) * (height - 1)))
        col = max(0, min(width - 1, col))
        row = max(0, min(height - 1, row))
        canvas[row][col] = "*"

    lines: List[str] = [f"--- {title} ({y_label} vs {x_label}) ---"]
    for r in range(height):
        y_tick = y_max - (r / max(1, height - 1)) * (y_max - y_min)
        lines.append(f"{y_tick:8.2f} |{''.join(canvas[r])}|")
    lines.append("         +" + "-" * width + "+")
    lines.append(f"         {x_min:<10.2f}{' ' * max(1, width - 20)}{x_max:>10.2f}")
    return assert_ascii("\n".join(lines))


def render_2d_contour(
    metad: MetaDResult,
    use_reweighted: bool = True,
) -> str:
    """Render 2D WT-MetaD Free Energy Surface contour map or 1D FES curve."""
    fes = metad.fes_reweighted if use_reweighted else metad.fes_direct
    if fes.ndim == 1:
        return render_1d_profile(
            metad.grid_axes[0],
            fes,
            title=f"WT-MetaD FES ({metad.cv_names[0]})",
            x_label=metad.cv_names[0],
            y_label="F (kcal/mol)",
        )

    ax0, ax1 = metad.grid_axes
    rows = min(18, len(ax1))
    cols = min(38, len(ax0))
    row_idx = np.linspace(len(ax1) - 1, 0, rows, dtype=int)
    col_idx = np.linspace(0, len(ax0) - 1, cols, dtype=int)
    f_max = max(0.5, float(np.percentile(fes, 95)))

    lines: List[str] = [
        f"=== 2D WT-MetaD FES ({metad.cv_names[0]} vs {metad.cv_names[1]}) [gamma={metad.bias_factor:.1f}] ===",
        f"Hills: {len(metad.hill_heights)} | Legend (kcal/mol): ' '=basin -> '@'={f_max:.1f}",
    ]
    for r in row_idx:
        chars = []
        for c in col_idx:
            norm = min(1.0, max(0.0, float(fes[c, r]) / f_max))
            ramp_i = int(round(norm * (len(_CONTOUR_RAMP) - 1)))
            chars.append(_CONTOUR_RAMP[ramp_i])
        lines.append(f"{ax1[r]:6.2f} |{''.join(chars)}|")
    lines.append("       +" + "-" * cols + "+")
    lines.append(f"       {ax0[0]:<8.2f}{' ' * max(1, cols - 16)}{ax0[-1]:>8.2f}")
    return assert_ascii("\n".join(lines))


def render_remd_walk(remd: REMDResult) -> str:
    """Render T-REMD acceptance table, replica 0 temperature walk, and Cv(T) peak."""
    temps = remd.temperatures_k
    acc = remd.acceptance_rates
    t_grid, cv_vals = remd.heat_capacity_curve
    peak_t = float(t_grid[int(np.argmax(cv_vals))])

    lines: List[str] = [
        f"=== T-REMD SUMMARY ({len(temps)} Replicas, {len(remd.replica_state_history)} Cycles) ===",
        f"Round Trips: {remd.round_trips} | Peak Cv Temperature: {peak_t:.1f} K",
        "Pairwise Acceptance Rates:",
    ]
    for i in range(len(acc)):
        bar = "#" * int(round(acc[i] * 24))
        lines.append(
            f"  T[{i}] {temps[i]:6.1f}K <-> T[{i+1}] {temps[i+1]:6.1f}K : {acc[i]*100:5.1f}% |{bar:<24s}|"
        )

    # Replica 0 walk across cycles
    walk = remd.replica_state_history[:, 0]
    max_t_idx = len(temps) - 1
    step_sample = np.linspace(0, len(walk) - 1, min(48, len(walk)), dtype=int)
    lines.append("Replica 0 Temperature Ladder Walk:")
    for t_lvl in range(max_t_idx, -1, -1):
        row = "".join("*" if walk[c] == t_lvl else "." for c in step_sample)
        lines.append(f"  {temps[t_lvl]:5.0f}K |{row}|")

    return assert_ascii("\n".join(lines))


def render_crooks_histogram(smd: SMDResult, bins: int = 12) -> str:
    """Render bidirectional SMD work summary and overlapping F vs -R work histogram."""
    neg_rev = -smd.reverse_works
    lo = float(min(np.min(smd.forward_works), np.min(neg_rev)))
    hi = float(max(np.max(smd.forward_works), np.max(neg_rev)))
    if abs(hi - lo) < 1e-6:
        hi = lo + 1.0
    edges = np.linspace(lo, hi, bins + 1)
    hf, _ = np.histogram(smd.forward_works, bins=edges)
    hr, _ = np.histogram(neg_rev, bins=edges)

    lines: List[str] = [
        f"=== BIDIRECTIONAL STEERED MD & CROOKS MLE ({smd.cv_name}) ===",
        f"Paths: {len(smd.forward_works)} fwd / {len(smd.reverse_works)} rev",
        f"Mean W_fwd     : {float(np.mean(smd.forward_works)):8.3f} kcal/mol "
        f"| Mean (-W_rev): {float(np.mean(neg_rev)):8.3f} kcal/mol",
        f"Jarzynski dG   : {smd.dg_jarzynski_fwd:8.3f} kcal/mol "
        f"| Cumulant dG : {smd.dg_cumulant_fwd:8.3f} kcal/mol",
        f"Crooks MLE dG  : {smd.dg_crooks_mle:8.3f} kcal/mol "
        f"(95% CI [{smd.dg_bootstrap_ci95[0]:.3f}, {smd.dg_bootstrap_ci95[1]:.3f}])",
        f"Dissipated W   : {smd.dissipated_work_fwd:8.3f} kcal/mol",
        "Work Distribution (F = forward W_F, R = reverse -W_R):",
    ]
    for b in range(bins):
        mid = 0.5 * (edges[b] + edges[b + 1])
        tag = "<-- dG" if edges[b] <= smd.dg_crooks_mle <= edges[b + 1] else ""
        bar_f = "F" * int(hf[b])
        bar_r = "R" * int(hr[b])
        lines.append(f"  {mid:7.2f} | {bar_f:<10s} | {bar_r:<10s} {tag}")
    return assert_ascii("\n".join(lines))


def render_nma_summary(nma: NMAResult, top_k: int = 8) -> str:
    """Render Eckart-projected NMA/ANM vibrational spectrum and thermodynamic summary."""
    pos_freqs = nma.frequencies_cm_inv[nma.frequencies_cm_inv > 1.0]
    lines: List[str] = [
        f"=== NORMAL MODE ANALYSIS ({nma.method.upper()}) ===",
        f"Internal Modes  : {len(nma.frequencies_cm_inv)} (Positive: {len(pos_freqs)})",
        f"Eckart Zero Res : {nma.zero_mode_residual:.3e}",
        f"ZPVE            : {nma.zpve_kcal_mol:10.4f} kcal/mol",
        f"Vib Entropy S   : {nma.vibrational_entropy_kcal_mol_k:10.6f} kcal/(mol*K)",
        f"Helmholtz F_vib : {nma.helmholtz_vib_kcal_mol:10.4f} kcal/mol",
        f"Mean B-factor   : {float(np.mean(nma.b_factors_a2)):10.4f} A^2",
        f"Lowest {min(top_k, len(pos_freqs))} Non-Zero Vibrational Modes (cm^-1):",
    ]
    max_f = float(np.max(pos_freqs[:top_k])) if len(pos_freqs) > 0 else 1.0
    for idx, nu in enumerate(pos_freqs[:top_k]):
        bar_len = int(round(25.0 * float(nu) / max(1.0, max_f)))
        lines.append(f"  Mode {idx+1:2d}: {nu:8.2f} cm^-1 |{'=' * bar_len:<25s}|")
    return assert_ascii("\n".join(lines))


def render_calibration_report(rep: CalibrationReport) -> str:
    """Render analytical vs finite-difference gradient check and parameter calibration table."""
    lines: List[str] = [
        "=== SOLVATODYNE FORCE-FIELD CALIBRATION & GRADIENT AUDIT ===",
        "Analytical vs Central Finite-Difference Gradient Checks:",
    ]
    for g in rep.gradient_checks:
        status = "PASS" if g.passed else "FAIL"
        lines.append(
            f"  [{status}] {g.term_name:<22s} max_abs={g.max_abs_error:.3e} rms_rel={g.rms_rel_error:.3e}"
        )
    lines.append(
        f"Drude Dipole Calibration RMSE: {rep.initial_rmse:.4f} D -> {rep.final_rmse:.4f} D"
    )
    for k, v in rep.fitted_params.items():
        lines.append(f"  {k:<18s} = {v:.5f}")
    lines.append("GBSA Normalized Sensitivity Indices:")
    for k, v in rep.sensitivity_indices.items():
        lines.append(f"  d(ln E_solv)/d(ln {k:<14s}) = {v:+.4f}")
    return assert_ascii("\n".join(lines))
