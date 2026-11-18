"""Tests for T-REMD detailed balance, feedback ladder, Jarzynski, and Crooks MLE."""

import numpy as np

from solvato_dyne.sampling.colvars import DistanceCV
from solvato_dyne.sampling.remd_and_smd import (
    SteeredMDSimulator,
    TemperatureReplicaExchange,
    build_geometric_temperature_ladder,
    crooks_mle_free_energy,
    jarzynski_free_energy,
    optimize_feedback_temperature_ladder,
)
from solvato_dyne.systems.builders import (
    build_alanine_dipeptide_system,
    build_host_guest_unbinding_system,
)
from solvato_dyne.types import KB_KCAL_MOL_K


def test_temperature_ladder_geometric_and_ktn_feedback() -> None:
    temps = build_geometric_temperature_ladder(280.0, 560.0, 5)
    assert len(temps) == 5
    assert np.all(np.diff(temps) > 0.0)
    # Geometric ratios must be constant
    ratios = temps[1:] / temps[:-1]
    assert np.allclose(ratios, ratios[0])

    # Simulated bottleneck between T[1] and T[2] pulls them closer together
    acc = np.array([0.55, 0.10, 0.55, 0.55])
    opt = optimize_feedback_temperature_ladder(temps, acc)
    assert np.isclose(opt[0], temps[0])
    assert np.isclose(opt[-1], temps[-1])
    assert np.all(np.diff(opt) > 0.0)
    assert (opt[2] - opt[1]) < (temps[2] - temps[1])


def test_remd_permutation_bijection_and_heat_capacity() -> None:
    ala = build_alanine_dipeptide_system(polarizable=False, use_gbsa=True)
    remd = TemperatureReplicaExchange(ala, t_min=280.0, t_max=480.0, n_replicas=4, seed=21)
    res = remd.run(n_cycles=12, steps_per_cycle=15)

    assert res.replica_state_history.shape == (12, 4)
    # Every cycle must be a valid bijection over replica indices {0, 1, 2, 3}
    for row in res.replica_state_history:
        assert sorted(row.tolist()) == [0, 1, 2, 3]
    assert np.all((res.acceptance_rates >= 0.0) & (res.acceptance_rates <= 1.0))
    t_grid, cv = res.heat_capacity_curve
    assert len(t_grid) == len(cv) and np.all(cv >= 0.0)


def test_crooks_mle_exact_gaussian_fluctuation_theorem() -> None:
    # By Crooks Fluctuation Theorem for Gaussian work:
    # P_F(W) = N(dG + W_diss, 2 * kT * W_diss), P_R(-W) = N(dG - W_diss, 2 * kT * W_diss)
    rng = np.random.default_rng(101)
    temp = 300.0
    kbt = KB_KCAL_MOL_K * temp
    true_dg = 6.25
    w_diss = 1.40
    sigma = np.sqrt(2.0 * kbt * w_diss)

    wf = rng.normal(true_dg + w_diss, sigma, size=400)
    # Note reverse_works is W_R (so -W_R has mean true_dg - w_diss)
    wr = -rng.normal(true_dg - w_diss, sigma, size=400)

    dg_mle, ci95 = crooks_mle_free_energy(wf, wr, temperature_k=temp, n_bootstrap=150, seed=5)
    assert abs(dg_mle - true_dg) < 0.15
    assert ci95[0] <= dg_mle <= ci95[1]

    # Second Law / Jarzynski: <W_fwd> >= dG and dG_Jarz close to true_dg
    dg_jarz, dg_cum = jarzynski_free_energy(wf, temperature_k=temp)
    assert float(np.mean(wf)) > true_dg
    assert abs(dg_jarz - true_dg) < 0.6
    assert abs(dg_cum - true_dg) < 0.4


def test_bidirectional_steered_md_host_guest_unbinding() -> None:
    complex_sys = build_host_guest_unbinding_system(polarizable=False)
    cv = DistanceCV(
        complex_sys.metadata["host_indices"],
        complex_sys.metadata["guest_indices"],
        name="pocket_ligand_A",
    )
    smd = SteeredMDSimulator(complex_sys, cv=cv, k_pull=40.0, temperature_k=300.0, seed=17)
    res = smd.run_bidirectional(n_paths=4, n_steps=80)

    assert len(res.forward_works) == 4
    assert len(res.reverse_works) == 4
    assert np.all(np.isfinite(res.forward_works))
    assert np.isfinite(res.dg_crooks_mle)
    # Unbinding a bound guest requires positive mean forward work
    assert float(np.mean(res.forward_works)) > 0.0


def test_crooks_unequal_sample_sizes_and_jarzynski_dissipation() -> None:
    rng = np.random.default_rng(33)
    temp = 300.0
    kbt = KB_KCAL_MOL_K * temp
    true_dg = -3.10
    w_diss = 0.90
    sigma = np.sqrt(2.0 * kbt * w_diss)

    wf = rng.normal(true_dg + w_diss, sigma, size=500)
    wr = -rng.normal(true_dg - w_diss, sigma, size=200)
    dg_mle, ci = crooks_mle_free_energy(wf, wr, temperature_k=temp, n_bootstrap=100, seed=9)
    assert abs(dg_mle - true_dg) < 0.2
    assert ci[0] < ci[1]

    dg_j, _ = jarzynski_free_energy(wf, temperature_k=temp)
    # Second law: dissipated work >= 0
    assert float(np.mean(wf)) - dg_j >= 0.0
