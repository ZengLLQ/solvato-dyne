"""Tests for Eckart-projected Hessian NMA, ANM, Kabsch RMSD, PCA, and RDF."""

import numpy as np

from solvato_dyne.analysis.normal_modes import (
    build_eckart_basis,
    run_anm_analysis,
    run_hessian_nma,
)
from solvato_dyne.analysis.trajectory import (
    compute_dipole_susceptibility,
    compute_pca_essential_dynamics,
    compute_radial_distribution,
    compute_ramachandran_angles,
    compute_rmsd_and_rmsf,
    kabsch_align,
)
from solvato_dyne.dynamics.runner import MDSimulator
from solvato_dyne.systems.builders import (
    build_alanine_dipeptide_system,
    build_coarse_grained_elastic_protein,
    build_polarizable_water_dimer_or_cluster,
)


def test_eckart_basis_orthonormality_and_zero_mode_projection() -> None:
    cg = build_coarse_grained_elastic_protein(n_beads=8)
    masses = np.array([a.mass for a in cg.atoms])
    basis = build_eckart_basis(cg.positions, masses)
    assert basis.shape == (3 * cg.n_atoms, 6)
    assert np.allclose(basis.T @ basis, np.eye(6), atol=1e-10)

    anm = run_anm_analysis(cg, cutoff_a=9.0, temperature_k=300.0)
    assert anm.zero_mode_residual < 1e-8
    assert len(anm.frequencies_cm_inv) == 3 * cg.n_atoms - 6
    assert np.all(anm.frequencies_cm_inv > -1e-4)
    assert anm.zpve_kcal_mol > 0.0
    assert anm.vibrational_entropy_kcal_mol_k > 0.0
    assert np.all(anm.b_factors_a2 >= 0.0)


def test_full_cartesian_hessian_nma_on_minimized_ala_dipeptide() -> None:
    ala = build_alanine_dipeptide_system(polarizable=False, use_gbsa=True)
    nma = run_hessian_nma(ala, temperature_k=300.0, minimize_first=True)
    assert nma.zero_mode_residual < 1e-2
    assert len(nma.frequencies_cm_inv) == 3 * ala.n_atoms - 6
    pos_modes = nma.frequencies_cm_inv[nma.frequencies_cm_inv > 1.0]
    # Minimized structure should have overwhelmingly positive real vibrational modes
    assert len(pos_modes) >= 3 * ala.n_atoms - 8
    assert nma.zpve_kcal_mol > 1.0


def test_kabsch_so3_invariance_and_rmsf() -> None:
    ala = build_alanine_dipeptide_system(polarizable=False)
    # Apply arbitrary 3D rotation + translation to coordinates
    theta = 0.85
    rot = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta), np.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ])
    transformed = ala.positions @ rot.T + np.array([12.5, -4.2, 7.1])
    aligned, rmsd = kabsch_align(transformed, ala.positions)
    assert rmsd < 1e-10
    assert np.allclose(aligned, ala.positions, atol=1e-9)


def test_trajectory_pca_rdf_ramachandran_and_dipole_susceptibility() -> None:
    ala = build_alanine_dipeptide_system(polarizable=True, use_gbsa=True)
    sim = MDSimulator(ala, integrator="baoab", temperature_k=300.0, dt_ps=0.001, seed=8)
    traj = sim.run(n_steps=100, save_every=10)

    rmsds, rmsf = compute_rmsd_and_rmsf(traj)
    assert len(rmsds) == traj.n_frames
    assert rmsds[0] < 1e-10
    assert len(rmsf) == ala.n_atoms and np.all(rmsf >= 0.0)

    evals, ratio, proj = compute_pca_essential_dynamics(traj, n_components=3)
    assert len(evals) == 3
    assert np.all(np.diff(evals) <= 1e-10)
    assert np.isclose(np.sum(ratio) <= 1.0001, True)
    assert proj.shape == (traj.n_frames, 3)

    phis, psis = compute_ramachandran_angles(traj, (1, 3, 5, 7), (3, 5, 7, 9))
    assert np.all((phis >= -180.0) & (phis <= 180.0))
    assert np.all((psis >= -180.0) & (psis <= 180.0))

    water = build_polarizable_water_dimer_or_cluster(n_molecules=4, polarizable=True)
    w_traj = MDSimulator(water, integrator="drude", dt_ps=0.0005, seed=4).run(n_steps=60, save_every=10)
    o_idx = [a.index for a in water.atoms if a.name == "OH2"]
    r_grid, g_r = compute_radial_distribution(w_traj, o_idx, o_idx, r_max=8.0, n_bins=20)
    assert len(r_grid) == 20 and np.all(g_r >= 0.0)

    susc = compute_dipole_susceptibility(w_traj)
    assert susc["mean_dipole_debye"] >= 0.0
    assert susc["dielectric_susceptibility_proxy"] >= 0.0


def test_anm_temperature_entropy_monotonicity() -> None:
    cg = build_coarse_grained_elastic_protein(n_beads=8)
    nma_cold = run_anm_analysis(cg, temperature_k=100.0)
    nma_warm = run_anm_analysis(cg, temperature_k=400.0)
    assert nma_warm.vibrational_entropy_kcal_mol_k > nma_cold.vibrational_entropy_kcal_mol_k
    assert np.all(nma_warm.b_factors_a2 >= nma_cold.b_factors_a2)


def test_wrap_positions_to_periodic_box() -> None:
    from solvato_dyne.forcefield.bonded import wrap_positions_to_box
    from solvato_dyne.types import BoxSpec

    box = BoxSpec.orthorhombic(10.0, 12.0, 15.0)
    pts = np.array([[-2.0, 13.5, 30.2], [11.1, -0.5, 7.5]])
    wrapped = wrap_positions_to_box(pts, box)
    assert np.all((wrapped[:, 0] >= 0.0) & (wrapped[:, 0] < 10.0))
    assert np.all((wrapped[:, 1] >= 0.0) & (wrapped[:, 1] < 12.0))
    assert np.all((wrapped[:, 2] >= 0.0) & (wrapped[:, 2] < 15.0))


def test_nma_summary_and_energy_summary_ascii_render() -> None:
    from solvato_dyne.viz.ascii_viz import assert_ascii, render_energy_summary, render_nma_summary

    cg = build_coarse_grained_elastic_protein(n_beads=6)
    nma = run_anm_analysis(cg, temperature_k=300.0)
    txt_nma = render_nma_summary(nma)
    assert_ascii(txt_nma)
    assert "ZPVE" in txt_nma

    traj = MDSimulator(cg, integrator="baoab", dt_ps=0.001, seed=2).run(n_steps=20, save_every=5)
    txt_en = render_energy_summary(traj)
    assert_ascii(txt_en)
    assert "Bond" in txt_en
