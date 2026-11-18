"""Example 02: Implicit-solvent OBC-II GBSA alanine dipeptide 2D WT-Metadynamics."""

from solvato_dyne.analysis.trajectory import compute_ramachandran_angles
from solvato_dyne.sampling.colvars import DihedralCV
from solvato_dyne.sampling.metadynamics import WellTemperedMetadynamics
from solvato_dyne.systems.builders import build_alanine_dipeptide_system
from solvato_dyne.viz.ascii_viz import render_2d_contour


def main() -> None:
    ala = build_alanine_dipeptide_system(polarizable=False, use_gbsa=True)
    phi_cv = DihedralCV(1, 3, 5, 7, name="phi")
    psi_cv = DihedralCV(3, 5, 7, 9, name="psi")

    metad = WellTemperedMetadynamics(
        cvs=[phi_cv, psi_cv],
        sigmas=[0.35, 0.35],
        initial_height=0.75,
        bias_factor=8.0,
        temperature_k=300.0,
        deposit_stride=15,
        grid_bins=29,
    )
    res = metad.run(ala, n_steps=240, save_every=15, seed=7)
    phis, psis = compute_ramachandran_angles(res.trajectory, (1, 3, 5, 7), (3, 5, 7, 9))
    print(f"Sampled Phi range: [{phis.min():.1f}, {phis.max():.1f}] deg")
    print(f"Sampled Psi range: [{psis.min():.1f}, {psis.max():.1f}] deg")
    print(f"Final Tiwary-Parrinello c(t): {res.c_of_t[-1]:.4f} kcal/mol")
    print(render_2d_contour(res, use_reweighted=True))


if __name__ == "__main__":
    main()
