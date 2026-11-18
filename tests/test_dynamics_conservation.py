"""Tests for symplectic NVE energy conservation, RATTLE constraints, BAOAB, and Dual-NHC Drude."""

import numpy as np

from solvato_dyne.dynamics.constraints import (
    apply_rattle_positions,
    apply_rattle_velocities,
    minimize_energy,
)
from solvato_dyne.dynamics.runner import MDSimulator, initialize_velocities
from solvato_dyne.forcefield.engine import ForceFieldEngine
from solvato_dyne.systems.builders import (
    build_alanine_dipeptide_system,
    build_polarizable_water_dimer_or_cluster,
)


def test_minimize_energy_strictly_decreases_potential() -> None:
    ala = build_alanine_dipeptide_system(polarizable=True)
    rng = np.random.default_rng(3)
    ala.positions += rng.normal(0.0, 0.08, ala.positions.shape)
    min_sys, e_init, e_final = minimize_energy(ala, maxiter=100)
    assert e_final < e_init
    assert min_sys.n_atoms == ala.n_atoms


def test_nve_symplectic_energy_conservation_and_rattle() -> None:
    water = build_polarizable_water_dimer_or_cluster(n_molecules=2, polarizable=False)
    sim = MDSimulator(water, integrator="nve", temperature_k=200.0, dt_ps=0.0005, seed=10)
    traj = sim.run(n_steps=200, save_every=10)

    e_tot = traj.total_energies()
    rel_spread = np.std(e_tot) / max(1.0, abs(np.mean(e_tot)))
    assert rel_spread < 5e-3, f"NVE energy drift too large: {rel_spread}"

    # Verify RATTLE bond lengths are preserved to < 1e-8 A across all frames
    for frame in traj.frames:
        for c in water.constraints:
            d = float(np.linalg.norm(frame.positions[c.j] - frame.positions[c.i]))
            assert abs(d - c.d0) < 1e-8


def test_rattle_velocity_orthogonality() -> None:
    water = build_polarizable_water_dimer_or_cluster(n_molecules=1, polarizable=False)
    masses = np.array([a.mass for a in water.atoms])
    inv_m = 1.0 / masses
    pos_pert = water.positions.copy()
    pos_pert[1, 0] += 0.03
    r_corr, _, conv = apply_rattle_positions(pos_pert, water.positions, inv_m, water.constraints, 0.001)
    assert conv
    vel = np.array([[0.5, -0.2, 0.1], [-0.4, 0.7, 0.3], [0.2, -0.3, -0.4]])
    v_corr, conv_v = apply_rattle_velocities(r_corr, vel, inv_m, water.constraints)
    assert conv_v
    for c in water.constraints:
        rij = r_corr[c.j] - r_corr[c.i]
        vij = v_corr[c.j] - v_corr[c.i]
        assert abs(np.dot(rij, vij)) < 1e-8


def test_baoab_langevin_thermal_stability() -> None:
    ala = build_alanine_dipeptide_system(polarizable=False, use_gbsa=True)
    sim = MDSimulator(ala, integrator="baoab", temperature_k=300.0, dt_ps=0.001, gamma_ps_inv=10.0, seed=5)
    traj = sim.run(n_steps=300, save_every=10)
    mean_t = float(np.mean(traj.temperatures_atom()[5:]))
    assert 150.0 < mean_t < 480.0
    assert np.all(np.isfinite(traj.potential_energies()))


def test_dual_thermostat_drude_temperature_separation_and_kinetic_identity() -> None:
    dimer = build_polarizable_water_dimer_or_cluster(n_molecules=2, polarizable=True)
    initialize_velocities(dimer, temperature_k=300.0, drude_temperature_k=1.0, seed=12)

    # Verify exact kinetic identity 0.5*m_c*v_c^2 + 0.5*m_D*v_D^2 == K_atom + K_drude
    eng = ForceFieldEngine(dimer)
    eb, _, _ = eng.evaluate()
    pol_idx = set(dimer.polarizable_indices.tolist())
    atom_to_d = {int(a_i): d_i for d_i, a_i in enumerate(dimer.polarizable_indices)}
    raw_k = 0.0
    from solvato_dyne.types import AMU_A2_PS2_TO_KCAL_MOL

    for idx, atom in enumerate(dimer.atoms):
        if idx in pol_idx:
            d_i = atom_to_d[idx]
            raw_k += 0.5 * AMU_A2_PS2_TO_KCAL_MOL * atom.core_mass * float(np.sum(dimer.velocities[idx] ** 2))
            raw_k += 0.5 * AMU_A2_PS2_TO_KCAL_MOL * atom.drude_mass * float(np.sum(dimer.drude_velocities[d_i] ** 2))
        else:
            raw_k += 0.5 * AMU_A2_PS2_TO_KCAL_MOL * atom.mass * float(np.sum(dimer.velocities[idx] ** 2))
    assert np.isclose(raw_k, eb.total_kinetic, atol=1e-10)

    sim = MDSimulator(dimer, integrator="drude", temperature_k=300.0, drude_temperature_k=1.0, dt_ps=0.0005, seed=12)
    traj = sim.run(n_steps=200, save_every=10)
    t_atom = float(np.mean(traj.temperatures_atom()))
    t_drude = float(np.mean(traj.temperatures_drude()))
    assert t_atom > 100.0
    # Dual thermostat keeps internal Drude oscillators cold (~1-15 K) vs ~300 K atoms
    assert t_drude < 25.0
    assert t_atom > 5.0 * max(0.5, t_drude)


def test_nve_second_order_timestep_convergence_and_restraints() -> None:
    import pytest
    from solvato_dyne.systems.builders import build_host_guest_unbinding_system

    ala = build_alanine_dipeptide_system(polarizable=False, use_gbsa=False)
    traj_coarse = MDSimulator(ala, integrator="nve", temperature_k=250.0, dt_ps=0.002, seed=9).run(
        n_steps=80, save_every=4
    )
    traj_fine = MDSimulator(ala, integrator="nve", temperature_k=250.0, dt_ps=0.0005, seed=9).run(
        n_steps=320, save_every=16
    )
    assert traj_fine.energy_drift_per_step() < traj_coarse.energy_drift_per_step()

    hg = build_host_guest_unbinding_system(polarizable=False)
    ref = hg.positions[[0, 1, 2]].copy()
    sim_rest = MDSimulator(
        hg, integrator="baoab", temperature_k=300.0, dt_ps=0.001,
        restraint_indices=np.array([0, 1, 2]), restraint_ref=ref, restraint_k=50.0, seed=4,
    )
    traj_r = sim_rest.run(n_steps=60, save_every=20)
    assert np.max(np.abs(traj_r.frames[-1].positions[[0, 1, 2]] - ref)) < 0.6

    with pytest.raises(ValueError):
        MDSimulator(ala, integrator="invalid-integrator")
