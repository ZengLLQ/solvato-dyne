"""Command-line interface for SolvatoDyne (`solvato-dyne` and `python -m solvato_dyne.cli`)."""

from __future__ import annotations

import argparse
import json
from typing import List, Optional

from solvato_dyne.analysis.normal_modes import run_anm_analysis, run_hessian_nma
from solvato_dyne.calibration.validator import run_full_calibration_suite
from solvato_dyne.dynamics.runner import MDSimulator
from solvato_dyne.sampling.colvars import DihedralCV, DistanceCV
from solvato_dyne.sampling.metadynamics import WellTemperedMetadynamics
from solvato_dyne.sampling.remd_and_smd import (
    SteeredMDSimulator,
    TemperatureReplicaExchange,
)
from solvato_dyne.systems.builders import build_system_by_name
from solvato_dyne.viz.ascii_viz import (
    render_2d_contour,
    render_calibration_report,
    render_crooks_histogram,
    render_energy_summary,
    render_nma_summary,
    render_remd_walk,
)


def _cmd_simulate(args: argparse.Namespace) -> int:
    sys_obj = build_system_by_name(args.system, polarizable=not args.non_polarizable)
    sim = MDSimulator(
        sys_obj,
        integrator=args.integrator,
        temperature_k=args.temperature,
        dt_ps=args.dt,
        seed=args.seed,
    )
    traj = sim.run(n_steps=args.steps, save_every=max(1, args.steps // 20))
    print(render_energy_summary(traj))
    if args.json_out:
        payload = {
            "system": traj.system_name,
            "integrator": traj.integrator,
            "n_frames": traj.n_frames,
            "energy_drift": traj.energy_drift_per_step(),
            "final_potential_kcal_mol": traj.frames[-1].energy.total_potential,
            "mean_temp_atom_k": float(traj.temperatures_atom().mean()),
            "mean_dipole_debye": float(traj.dipoles().mean()),
        }
        with open(args.json_out, "w", encoding="ascii") as fh:
            json.dump(payload, fh, indent=2)
    return 0


def _cmd_remd(args: argparse.Namespace) -> int:
    sys_obj = build_system_by_name(args.system, polarizable=False)
    remd = TemperatureReplicaExchange(
        sys_obj,
        t_min=args.t_min,
        t_max=args.t_max,
        n_replicas=args.replicas,
        dt_ps=args.dt,
        seed=args.seed,
    )
    res = remd.run(n_cycles=args.cycles, steps_per_cycle=args.steps_per_cycle)
    print(render_remd_walk(res))
    return 0


def _cmd_metad(args: argparse.Namespace) -> int:
    sys_obj = build_system_by_name(args.system, polarizable=False)
    if args.system == "ala-dipeptide":
        phi_cv = DihedralCV(1, 3, 5, 7, name="phi")
        psi_cv = DihedralCV(3, 5, 7, 9, name="psi")
        cvs = [phi_cv, psi_cv]
        sigmas = [0.35, 0.35]
    else:
        cvs = [DistanceCV([0], [sys_obj.n_atoms - 1], name="end_to_end_A", bounds=(2.0, 16.0))]
        sigmas = [0.45]

    metad = WellTemperedMetadynamics(
        cvs=cvs,
        sigmas=sigmas,
        initial_height=args.height,
        bias_factor=args.bias_factor,
        temperature_k=args.temperature,
        deposit_stride=args.stride,
        grid_bins=31,
    )
    res = metad.run(sys_obj, n_steps=args.steps, save_every=max(5, args.stride), seed=args.seed)
    print(render_2d_contour(res, use_reweighted=True))
    return 0


def _cmd_steer(args: argparse.Namespace) -> int:
    sys_obj = build_system_by_name(args.system, polarizable=False)
    host_idx = sys_obj.metadata.get("host_indices", [0, 1, 2])
    guest_idx = sys_obj.metadata.get("guest_indices", [sys_obj.n_atoms - 1])
    cv = DistanceCV(host_idx, guest_idx, name="pocket_ligand_A")
    smd = SteeredMDSimulator(
        sys_obj,
        cv=cv,
        k_pull=args.k_pull,
        temperature_k=args.temperature,
        seed=args.seed,
    )
    res = smd.run_bidirectional(n_paths=args.n_paths, n_steps=args.steps)
    print(render_crooks_histogram(res))
    return 0


def _cmd_modes(args: argparse.Namespace) -> int:
    sys_obj = build_system_by_name(args.system, polarizable=False)
    if args.method == "hessian":
        res = run_hessian_nma(sys_obj, temperature_k=args.temperature)
    else:
        res = run_anm_analysis(sys_obj, temperature_k=args.temperature)
    print(render_nma_summary(res))
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    rep = run_full_calibration_suite(quick=args.quick)
    print(render_calibration_report(rep))
    all_passed = all(g.passed for g in rep.gradient_checks)
    return 0 if all_passed else 1


def build_parser() -> argparse.ArgumentParser:
    """Create argparse CLI parser with 6 operational subcommands."""
    parser = argparse.ArgumentParser(
        prog="solvato-dyne",
        description="Polarizable Drude MD, GBSA implicit solvation, and enhanced sampling CLI.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_sim = sub.add_parser("simulate", help="Run NVE, BAOAB NVT, or dual-NHC Drude MD.")
    p_sim.add_argument("--system", default="water-dimer")
    p_sim.add_argument("--integrator", default="drude", choices=["nve", "baoab", "drude"])
    p_sim.add_argument("--temperature", type=float, default=300.0)
    p_sim.add_argument("--steps", type=int, default=200)
    p_sim.add_argument("--dt", type=float, default=None)
    p_sim.add_argument("--seed", type=int, default=42)
    p_sim.add_argument("--non-polarizable", action="store_true")
    p_sim.add_argument("--json-out", default=None)
    p_sim.set_defaults(func=_cmd_simulate)

    p_remd = sub.add_parser("remd", help="Run multi-replica Temperature Replica-Exchange MD.")
    p_remd.add_argument("--system", default="ala-dipeptide")
    p_remd.add_argument("--t-min", type=float, default=280.0)
    p_remd.add_argument("--t-max", type=float, default=520.0)
    p_remd.add_argument("--replicas", type=int, default=4)
    p_remd.add_argument("--cycles", type=int, default=16)
    p_remd.add_argument("--steps-per-cycle", type=int, default=20)
    p_remd.add_argument("--dt", type=float, default=0.001)
    p_remd.add_argument("--seed", type=int, default=42)
    p_remd.set_defaults(func=_cmd_remd)

    p_metad = sub.add_parser("metad", help="Run Well-Tempered Metadynamics with Tiwary-Parrinello FES.")
    p_metad.add_argument("--system", default="ala-dipeptide")
    p_metad.add_argument("--steps", type=int, default=250)
    p_metad.add_argument("--stride", type=int, default=20)
    p_metad.add_argument("--height", type=float, default=0.6)
    p_metad.add_argument("--bias-factor", type=float, default=8.0)
    p_metad.add_argument("--temperature", type=float, default=300.0)
    p_metad.add_argument("--seed", type=int, default=42)
    p_metad.set_defaults(func=_cmd_metad)

    p_steer = sub.add_parser("steer", help="Run bidirectional Steered MD and Crooks MLE dG.")
    p_steer.add_argument("--system", default="host-guest")
    p_steer.add_argument("--n-paths", type=int, default=5)
    p_steer.add_argument("--steps", type=int, default=100)
    p_steer.add_argument("--k-pull", type=float, default=40.0)
    p_steer.add_argument("--temperature", type=float, default=300.0)
    p_steer.add_argument("--seed", type=int, default=42)
    p_steer.set_defaults(func=_cmd_steer)

    p_modes = sub.add_parser("modes", help="Run Eckart-projected Hessian NMA or ANM analysis.")
    p_modes.add_argument("--system", default="ala-dipeptide")
    p_modes.add_argument("--method", default="hessian", choices=["hessian", "anm"])
    p_modes.add_argument("--temperature", type=float, default=300.0)
    p_modes.set_defaults(func=_cmd_modes)

    p_cal = sub.add_parser("calibrate", help="Run force-field gradient audit and Drude calibration.")
    p_cal.add_argument("--quick", action="store_true")
    p_cal.set_defaults(func=_cmd_calibrate)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
