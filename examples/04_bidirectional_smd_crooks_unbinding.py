"""Example 04: Bidirectional Steered MD host-guest unbinding and Crooks MLE free energy."""

from solvato_dyne.sampling.colvars import DistanceCV
from solvato_dyne.sampling.remd_and_smd import SteeredMDSimulator
from solvato_dyne.systems.builders import build_host_guest_unbinding_system
from solvato_dyne.viz.ascii_viz import render_crooks_histogram


def main() -> None:
    complex_sys = build_host_guest_unbinding_system(polarizable=False)
    host_idx = complex_sys.metadata["host_indices"]
    guest_idx = complex_sys.metadata["guest_indices"]

    cv = DistanceCV(host_idx, guest_idx, name="pocket_ligand_A")
    smd = SteeredMDSimulator(complex_sys, cv=cv, k_pull=45.0, temperature_k=300.0, seed=19)
    res = smd.run_bidirectional(n_paths=5, n_steps=100)
    print(render_crooks_histogram(res, bins=10))


if __name__ == "__main__":
    main()
