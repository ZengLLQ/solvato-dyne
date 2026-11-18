"""Example 01: Drude-polarizable water dimer, SCF dipole cooperativity, and dual-NHC MD."""

from solvato_dyne.calibration.validator import calibrate_drude_dipole_parameters
from solvato_dyne.dynamics.runner import MDSimulator
from solvato_dyne.forcefield.engine import ForceFieldEngine
from solvato_dyne.forcefield.polarizable import relax_drude_scf
from solvato_dyne.systems.builders import build_polarizable_water_dimer_or_cluster
from solvato_dyne.viz.ascii_viz import render_energy_summary


def main() -> None:
    monomer = build_polarizable_water_dimer_or_cluster(n_molecules=1, polarizable=True)
    dimer = build_polarizable_water_dimer_or_cluster(n_molecules=2, polarizable=True)

    d_mono, res_m = relax_drude_scf(monomer)
    d_dimer, res_d = relax_drude_scf(dimer)
    mu_mono = ForceFieldEngine(monomer).compute_dipole_debye(monomer.positions, d_mono)
    mu_dimer = ForceFieldEngine(dimer).compute_dipole_debye(dimer.positions, d_dimer)

    print(f"SCF Residual (monomer/dimer): {res_m:.2e} / {res_d:.2e} kcal/(mol*A)")
    print(f"Water Monomer Dipole: {mu_mono:.4f} D | Dimer Dipole: {mu_dimer:.4f} D")

    fitted, init_rmse, fin_rmse = calibrate_drude_dipole_parameters()
    print(f"Calibrated Drude Parameters (RMSE {init_rmse:.4f} -> {fin_rmse:.4f} D): {fitted}")

    sim = MDSimulator(dimer, integrator="drude", temperature_k=300.0, drude_temperature_k=1.0, dt_ps=0.0005)
    traj = sim.run(n_steps=160, save_every=20)
    print(render_energy_summary(traj))


if __name__ == "__main__":
    main()
