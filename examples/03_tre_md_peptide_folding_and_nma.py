"""Example 03: T-REMD thermal unfolding of alpha-helix peptide & Eckart Hessian NMA."""

from solvato_dyne.analysis.normal_modes import run_anm_analysis, run_hessian_nma
from solvato_dyne.analysis.trajectory import compute_pca_essential_dynamics
from solvato_dyne.sampling.remd_and_smd import (
    TemperatureReplicaExchange,
    optimize_feedback_temperature_ladder,
)
from solvato_dyne.systems.builders import build_alpha_helix_peptide, map_all_atom_to_cg
from solvato_dyne.viz.ascii_viz import render_nma_summary, render_remd_walk


def main() -> None:
    helix = build_alpha_helix_peptide(n_res=6, polarizable=False, use_gbsa=True)
    remd = TemperatureReplicaExchange(helix, t_min=280.0, t_max=500.0, n_replicas=4, seed=11)
    remd_res = remd.run(n_cycles=14, steps_per_cycle=18)
    print(render_remd_walk(remd_res))

    opt_ladder = optimize_feedback_temperature_ladder(remd_res.temperatures_k, remd_res.acceptance_rates)
    print(f"KTN Feedback-Optimized Ladder (K): {[round(float(t), 1) for t in opt_ladder]}")

    evals, ratio, _ = compute_pca_essential_dynamics(remd_res.trajectories[0], n_components=3)
    print(f"PCA Essential Dynamics Top-3 Variance Ratio: {[round(float(r), 3) for r in ratio]} (evals={evals.round(3).tolist()})")

    cg_helix = map_all_atom_to_cg(helix)
    nma_hess = run_hessian_nma(cg_helix, temperature_k=300.0)
    nma_anm = run_anm_analysis(helix, temperature_k=300.0)
    print(render_nma_summary(nma_hess, top_k=6))
    print(render_nma_summary(nma_anm, top_k=6))


if __name__ == "__main__":
    main()
