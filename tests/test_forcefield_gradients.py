"""Analytical vs central finite-difference gradient and physical invariance tests."""

import numpy as np

from solvato_dyne.calibration.validator import audit_forcefield_gradients
from solvato_dyne.forcefield.bonded import (
    compute_bonded_forces,
    dihedral_angle_and_grad,
    minimum_image_displacement,
)
from solvato_dyne.forcefield.engine import ForceFieldEngine
from solvato_dyne.forcefield.polarizable import (
    compute_nonbonded_and_drude_forces,
    relax_drude_scf,
)
from solvato_dyne.forcefield.solvation import compute_gbsa_forces
from solvato_dyne.systems.builders import (
    build_alanine_dipeptide_system,
    build_coarse_grained_elastic_protein,
    build_polarizable_water_dimer_or_cluster,
)
from solvato_dyne.types import BoxSpec


def test_all_forcefield_terms_match_finite_differences() -> None:
    entries = audit_forcefield_gradients(eps=1e-5, tol=1e-5)
    assert len(entries) == 4
    for e in entries:
        assert e.passed, f"{e.term_name} failed: max_abs={e.max_abs_error}"
        assert e.max_abs_error < 1e-5


def test_dihedral_gradient_across_branch_cut_pi() -> None:
    pts = np.array([
        [0.0, 1.0, 0.05],
        [0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        [1.8, -1.0, -0.05],
    ])
    phi, g0, g1, g2, g3 = dihedral_angle_and_grad(*pts)
    assert abs(abs(phi) - np.pi) < 0.2
    an = np.vstack([g0, g1, g2, g3])
    # Verify translational & rotational invariance: sum of gradients == 0
    assert np.allclose(np.sum(an, axis=0), 0.0, atol=1e-10)
    # Verify torque sum(r_k x g_k) == 0
    torques = sum(np.cross(pts[k], an[k]) for k in range(4))
    assert np.allclose(torques, 0.0, atol=1e-10)


def test_minimum_image_triclinic_and_orthorhombic() -> None:
    box = BoxSpec.orthorhombic(10.0, 10.0, 10.0)
    dr = minimum_image_displacement(np.array([9.2, -8.5, 4.0]), box)
    assert np.allclose(dr, np.array([-0.8, 1.5, 4.0]))

    tri = BoxSpec(np.array([[10.0, 0.0, 0.0], [2.0, 10.0, 0.0], [0.0, 0.0, 10.0]]))
    dr_t = minimum_image_displacement(np.array([11.5, 9.8, 0.0]), tri)
    assert np.linalg.norm(dr_t) < 2.5


def test_drude_scf_convergence_and_induced_dipole_enhancement() -> None:
    monomer = build_polarizable_water_dimer_or_cluster(n_molecules=1, polarizable=True)
    dimer = build_polarizable_water_dimer_or_cluster(n_molecules=2, polarizable=True)

    d_mono, res_m = relax_drude_scf(monomer)
    d_dimer, res_d = relax_drude_scf(dimer)
    assert res_m < 1e-5
    assert res_d < 1e-5

    eng_m = ForceFieldEngine(monomer)
    eng_d = ForceFieldEngine(dimer)
    mu_m = eng_m.compute_dipole_debye(monomer.positions, d_mono)
    mu_d = eng_d.compute_dipole_debye(dimer.positions, d_dimer)
    assert 1.5 < mu_m < 2.6
    # Induced polarization causes cooperative dipole enhancement over rigid non-polarizable dimer
    dimer_nonpol = build_polarizable_water_dimer_or_cluster(n_molecules=2, polarizable=False)
    mu_d_nonpol = ForceFieldEngine(dimer_nonpol).compute_dipole_debye(dimer_nonpol.positions)
    assert mu_d > 0.0 and mu_d != mu_d_nonpol


def test_gbsa_dielectric_scaling_and_elnedyn_forces() -> None:
    ala_water = build_alanine_dipeptide_system(polarizable=False, use_gbsa=True)
    ala_vac = build_alanine_dipeptide_system(polarizable=False, use_gbsa=False)

    e_w, _ = compute_gbsa_forces(ala_water)
    e_v, _ = compute_gbsa_forces(ala_vac)
    assert e_w["gb_polar"] < 0.0  # Polar solvation stabilizes partial charges in water
    assert e_w["sasa_nonpolar"] > 0.0
    assert e_v["gb_polar"] == 0.0

    # Higher interior dielectric reduces polar solvation magnitude
    ala_low_eps = ala_water.clone()
    ala_low_eps.ff_config.dielectric_out = 4.0
    e_low, _ = compute_gbsa_forces(ala_low_eps)
    assert abs(e_w["gb_polar"]) > abs(e_low["gb_polar"])

    cg = build_coarse_grained_elastic_protein(n_beads=8)
    cg.positions += 0.05
    e_b, f_b, _ = compute_bonded_forces(cg)
    assert e_b["elastic_net"] >= 0.0
    assert np.allclose(np.sum(f_b, axis=0), 0.0, atol=1e-10)
    # Nonbonded + Thole force balance check
    ala_pol = build_alanine_dipeptide_system(polarizable=True)
    _, f_a, f_d, _ = compute_nonbonded_and_drude_forces(ala_pol)
    assert np.allclose(np.sum(f_a, axis=0) + np.sum(f_d, axis=0), 0.0, atol=1e-9)


def test_c2_switch_continuity_and_drude_hyperpolarization_wall() -> None:
    from solvato_dyne.forcefield.polarizable import _c2_switch_factor, _thole_screening

    s_on, ds_on = _c2_switch_factor(8.0, 8.0, 12.0)
    s_cut, ds_cut = _c2_switch_factor(12.0, 8.0, 12.0)
    assert np.isclose(s_on, 1.0) and np.isclose(ds_on, 0.0)
    assert np.isclose(s_cut, 0.0) and np.isclose(ds_cut, 0.0)

    s_near, _ = _thole_screening(0.5, 4.178, 1.0)
    s_far, _ = _thole_screening(6.0, 4.178, 1.0)
    assert 0.0 < s_near < s_far <= 1.0

    # Hyperpolarization wall activates above drude_hard_wall
    monomer = build_polarizable_water_dimer_or_cluster(n_molecules=1, polarizable=True)
    monomer.drude_positions[0] = monomer.positions[0] + np.array([0.40, 0.0, 0.0])
    en, _, f_d, _ = compute_nonbonded_and_drude_forces(monomer)
    assert en["drude_spring"] > 80.0
    assert f_d[0, 0] < -400.0
