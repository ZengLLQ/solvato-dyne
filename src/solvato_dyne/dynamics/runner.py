"""High-level MD simulation runner, Maxwell-Boltzmann initialization, and COM cleanup."""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np

from solvato_dyne.dynamics.constraints import (
    apply_rattle_positions,
    apply_rattle_velocities,
    minimize_energy,
)
from solvato_dyne.dynamics.integrators import (
    BAOABIntegrator,
    DrudeDualThermostatIntegrator,
    VelocityVerletIntegrator,
)
from solvato_dyne.forcefield.engine import ExternalBiasFn, ForceFieldEngine
from solvato_dyne.forcefield.polarizable import relax_drude_scf
from solvato_dyne.types import (
    AMU_A2_PS2_TO_KCAL_MOL,
    KB_KCAL_MOL_K,
    MolecularSystem,
    Trajectory,
    TrajectoryFrame,
)


def initialize_velocities(
    system: MolecularSystem,
    temperature_k: float = 300.0,
    drude_temperature_k: float = 1.0,
    remove_com_drift: bool = True,
    seed: Optional[int] = 42,
) -> None:
    """Draw Maxwell-Boltzmann atomic COM and cold-Drude relative velocities in-place."""
    rng = np.random.default_rng(seed)
    masses = np.array([a.mass for a in system.atoms], dtype=np.float64)
    sigma_v = np.sqrt(
        (KB_KCAL_MOL_K * temperature_k / AMU_A2_PS2_TO_KCAL_MOL) / masses
    )[:, None]
    v_com = sigma_v * rng.normal(size=(system.n_atoms, 3))

    if remove_com_drift and system.n_atoms > 1:
        p_tot = np.sum(masses[:, None] * v_com, axis=0)
        v_com -= p_tot / np.sum(masses)

    if system.constraints:
        v_com, _ = apply_rattle_velocities(
            system.positions, v_com, 1.0 / masses, system.constraints, box=system.box
        )

    pol_idx = system.polarizable_indices
    if len(pol_idx) > 0:
        mc = np.array([system.atoms[int(i)].core_mass for i in pol_idx], dtype=np.float64)
        md = np.array([system.atoms[int(i)].drude_mass for i in pol_idx], dtype=np.float64)
        mt = mc + md
        mu = (mc * md) / mt
        sigma_vd = np.sqrt(
            (KB_KCAL_MOL_K * drude_temperature_k / AMU_A2_PS2_TO_KCAL_MOL) / mu
        )[:, None]
        v_rel = sigma_vd * rng.normal(size=(len(pol_idx), 3))

        v_core = v_com.copy()
        v_drude = np.zeros((len(pol_idx), 3), dtype=np.float64)
        for d_i, a_i in enumerate(pol_idx):
            v_core[int(a_i)] = v_com[int(a_i)] - (md[d_i] / mt[d_i]) * v_rel[d_i]
            v_drude[d_i] = v_com[int(a_i)] + (mc[d_i] / mt[d_i]) * v_rel[d_i]
        system.velocities = v_core
        system.drude_velocities = v_drude
    else:
        system.velocities = v_com


class MDSimulator:
    """Configurable orchestrator for NVE, Langevin NVT, and Drude dual-thermostat MD."""

    def __init__(
        self,
        system: MolecularSystem,
        integrator: str = "baoab",
        temperature_k: float = 300.0,
        drude_temperature_k: float = 1.0,
        dt_ps: Optional[float] = None,
        gamma_ps_inv: float = 5.0,
        restraint_indices: Optional[np.ndarray] = None,
        restraint_ref: Optional[np.ndarray] = None,
        restraint_k: float = 0.0,
        seed: int = 42,
    ) -> None:
        self.system = system.clone()
        self.integrator_name = integrator.lower()
        self.temperature_k = float(temperature_k)
        self.drude_temperature_k = float(drude_temperature_k)
        self.seed = seed

        if dt_ps is None:
            dt_ps = 0.0005 if self.integrator_name == "drude" else 0.001
        self.dt_ps = float(dt_ps)

        self.engine = ForceFieldEngine(
            self.system,
            restraint_indices=restraint_indices,
            restraint_ref=restraint_ref,
            restraint_k=restraint_k,
        )

        if self.integrator_name == "nve":
            self.stepper = VelocityVerletIntegrator(dt_ps=self.dt_ps, use_rattle=True, scf_drude=True)
        elif self.integrator_name == "baoab":
            self.stepper = BAOABIntegrator(
                temperature_k=self.temperature_k,
                gamma_ps_inv=gamma_ps_inv,
                dt_ps=self.dt_ps,
                use_rattle=True,
                scf_drude=True,
                seed=seed,
            )
        elif self.integrator_name == "drude":
            self.stepper = DrudeDualThermostatIntegrator(
                temperature_k=self.temperature_k,
                drude_temperature_k=self.drude_temperature_k,
                dt_ps=self.dt_ps,
            )
        else:
            raise ValueError(
                f"Unknown integrator '{integrator}'. Choose 'nve', 'baoab', or 'drude'."
            )

    def run(
        self,
        n_steps: int = 200,
        save_every: int = 10,
        minimize_first: bool = True,
        init_velocities: bool = True,
        external_bias: Optional[ExternalBiasFn] = None,
        cv_evaluators: Optional[Dict[str, object]] = None,
    ) -> Trajectory:
        """Execute MD simulation and return Trajectory of sampled frames."""
        if minimize_first:
            min_sys, _, _ = minimize_energy(self.system, maxiter=80, gtol=1e-2)
            self.system.positions = min_sys.positions
            self.system.drude_positions = min_sys.drude_positions
        elif self.system.n_drude > 0:
            d_opt, _ = relax_drude_scf(self.system)
            self.system.drude_positions = d_opt

        if self.system.constraints:
            masses = np.array([a.mass for a in self.system.atoms], dtype=np.float64)
            r_corr, _, _ = apply_rattle_positions(
                self.system.positions,
                self.system.positions,
                1.0 / masses,
                self.system.constraints,
                self.dt_ps,
                box=self.system.box,
            )
            self.system.positions = r_corr
            if self.system.n_drude > 0:
                d_opt, _ = relax_drude_scf(self.system)
                self.system.drude_positions = d_opt

        if init_velocities:
            initialize_velocities(
                self.system,
                temperature_k=self.temperature_k,
                drude_temperature_k=self.drude_temperature_k,
                seed=self.seed,
            )

        frames: List[TrajectoryFrame] = []
        eb0, cached_f, _ = self.engine.evaluate(external_bias=external_bias)
        eb0.extended_hamiltonian = eb0.total_energy
        frames.append(self._make_frame(0, 0.0, eb0, cv_evaluators))

        for step_idx in range(1, n_steps + 1):
            if isinstance(self.stepper, DrudeDualThermostatIntegrator):
                eb = self.stepper.step(self.system, self.engine, external_bias=external_bias)
            else:
                eb, cached_f = self.stepper.step(
                    self.system,
                    self.engine,
                    external_bias=external_bias,
                    cached_forces=cached_f if external_bias is None else None,
                )

            if step_idx % save_every == 0 or step_idx == n_steps:
                frames.append(
                    self._make_frame(step_idx, step_idx * self.dt_ps, eb, cv_evaluators)
                )

        return Trajectory(
            system_name=self.system.name,
            integrator=self.integrator_name,
            dt_ps=self.dt_ps,
            frames=frames,
        )

    def _make_frame(
        self,
        step: int,
        time_ps: float,
        energy,
        cv_evaluators: Optional[Dict[str, object]],
    ) -> TrajectoryFrame:
        cv_vals: Dict[str, float] = {}
        if cv_evaluators:
            for name, cv in cv_evaluators.items():
                val, _ = cv.evaluate(self.system.positions)  # type: ignore[attr-defined]
                cv_vals[name] = float(val)
        return TrajectoryFrame(
            step=step,
            time_ps=float(time_ps),
            positions=self.system.positions.copy(),
            velocities=None if self.system.velocities is None else self.system.velocities.copy(),
            drude_positions=None
            if self.system.drude_positions is None
            else self.system.drude_positions.copy(),
            energy=energy,
            colvars=cv_vals,
        )
