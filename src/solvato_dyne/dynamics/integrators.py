"""Molecular dynamics integrators: NVE Velocity-Verlet, BAOAB Langevin, and Dual-NHC Drude."""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np

from solvato_dyne.dynamics.constraints import (
    apply_rattle_positions,
    apply_rattle_velocities,
)
from solvato_dyne.forcefield.engine import ExternalBiasFn, ForceFieldEngine
from solvato_dyne.forcefield.polarizable import relax_drude_scf
from solvato_dyne.types import (
    AMU_A2_PS2_TO_KCAL_MOL,
    EnergyBreakdown,
    FORCE_TO_A_PS2_PER_AMU,
    KB_KCAL_MOL_K,
    MolecularSystem,
)


class VelocityVerletIntegrator:
    """Symplectic Velocity-Verlet NVE integrator with optional RATTLE & SCF Drude."""

    def __init__(self, dt_ps: float = 0.001, use_rattle: bool = True, scf_drude: bool = True) -> None:
        if dt_ps <= 0.0:
            raise ValueError("dt_ps must be positive.")
        self.dt_ps = float(dt_ps)
        self.use_rattle = use_rattle
        self.scf_drude = scf_drude

    def step(
        self,
        system: MolecularSystem,
        engine: ForceFieldEngine,
        external_bias: Optional[ExternalBiasFn] = None,
        cached_forces: Optional[np.ndarray] = None,
    ) -> Tuple[EnergyBreakdown, np.ndarray]:
        """Advance state by one symplectic step dt_ps."""
        masses = np.array(
            [a.mass if self.scf_drude else a.core_mass for a in system.atoms],
            dtype=np.float64,
        )
        inv_masses = 1.0 / masses

        if cached_forces is None:
            _, f0, _ = engine.evaluate(
                system.positions,
                system.drude_positions,
                system.velocities,
                None if self.scf_drude else system.drude_velocities,
                external_bias=external_bias,
                scf_drude=self.scf_drude,
            )
        else:
            f0 = cached_forces

        dt = self.dt_ps
        half_dt = 0.5 * dt
        acc0 = (f0 * FORCE_TO_A_PS2_PER_AMU) * inv_masses[:, None]

        r_old = system.positions.copy()
        v_half = system.velocities + half_dt * acc0
        r_new = r_old + dt * v_half

        if self.use_rattle and system.constraints:
            r_new, dv_rattle, _ = apply_rattle_positions(
                r_new, r_old, inv_masses, system.constraints, dt, box=system.box
            )
            v_half = v_half + dv_rattle

        dpos_new = system.drude_positions
        if self.scf_drude and system.n_drude > 0:
            dpos_new, _ = relax_drude_scf(system, r_new, dpos_new)

        eb1, f1, _ = engine.evaluate(
            r_new,
            dpos_new,
            v_half,
            None,
            external_bias=external_bias,
            scf_drude=False,
        )
        acc1 = (f1 * FORCE_TO_A_PS2_PER_AMU) * inv_masses[:, None]
        v_new = v_half + half_dt * acc1

        if self.use_rattle and system.constraints:
            v_new, _ = apply_rattle_velocities(
                r_new, v_new, inv_masses, system.constraints, box=system.box
            )

        system.positions = r_new
        system.velocities = v_new
        if self.scf_drude and system.n_drude > 0:
            system.drude_positions = dpos_new
            system.drude_velocities = v_new[system.polarizable_indices].copy()

        # Recompute kinetic energy at full-step v_new
        eb_final, _, _ = engine.evaluate(
            system.positions,
            system.drude_positions,
            system.velocities,
            system.drude_velocities,
            external_bias=external_bias,
            scf_drude=False,
        )
        eb_final.extended_hamiltonian = eb_final.total_energy
        return eb_final, f1


class BAOABIntegrator:
    """BAOAB Langevin splitting integrator (Leimkuhler-Matthews) for accurate NVT sampling."""

    def __init__(
        self,
        temperature_k: float = 300.0,
        gamma_ps_inv: float = 5.0,
        dt_ps: float = 0.001,
        use_rattle: bool = True,
        scf_drude: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        if temperature_k <= 0.0 or dt_ps <= 0.0 or gamma_ps_inv < 0.0:
            raise ValueError("Invalid Langevin parameters.")
        self.temperature_k = float(temperature_k)
        self.gamma_ps_inv = float(gamma_ps_inv)
        self.dt_ps = float(dt_ps)
        self.use_rattle = use_rattle
        self.scf_drude = scf_drude
        self.rng = np.random.default_rng(seed)

    def step(
        self,
        system: MolecularSystem,
        engine: ForceFieldEngine,
        external_bias: Optional[ExternalBiasFn] = None,
        cached_forces: Optional[np.ndarray] = None,
    ) -> Tuple[EnergyBreakdown, np.ndarray]:
        """Execute B - A - O - A - B splitting step."""
        masses = np.array([a.mass for a in system.atoms], dtype=np.float64)
        inv_masses = 1.0 / masses
        dt = self.dt_ps
        half_dt = 0.5 * dt

        if cached_forces is None:
            _, f0, _ = engine.evaluate(
                system.positions,
                system.drude_positions,
                system.velocities,
                None,
                external_bias=external_bias,
                scf_drude=self.scf_drude,
            )
        else:
            f0 = cached_forces

        # B (half kick)
        v = system.velocities + half_dt * (f0 * FORCE_TO_A_PS2_PER_AMU) * inv_masses[:, None]
        # A (half drift)
        r = system.positions + half_dt * v

        # O (exact Ornstein-Uhlenbeck velocity step)
        c1 = float(np.exp(-self.gamma_ps_inv * dt))
        c2 = float(np.sqrt(max(0.0, 1.0 - c1 * c1)))
        sigma_v = np.sqrt(
            (KB_KCAL_MOL_K * self.temperature_k / AMU_A2_PS2_TO_KCAL_MOL) * inv_masses
        )[:, None]
        v = c1 * v + c2 * sigma_v * self.rng.normal(size=v.shape)

        # A (half drift)
        r_old = system.positions.copy()
        r = r + half_dt * v

        if self.use_rattle and system.constraints:
            r, dv_rattle, _ = apply_rattle_positions(
                r, r_old, inv_masses, system.constraints, dt, box=system.box
            )
            v = v + dv_rattle

        dpos_new = system.drude_positions
        if self.scf_drude and system.n_drude > 0:
            dpos_new, _ = relax_drude_scf(system, r, dpos_new)

        _, f1, _ = engine.evaluate(
            r,
            dpos_new,
            v,
            None,
            external_bias=external_bias,
            scf_drude=False,
        )
        # B (half kick)
        v = v + half_dt * (f1 * FORCE_TO_A_PS2_PER_AMU) * inv_masses[:, None]

        if self.use_rattle and system.constraints:
            v, _ = apply_rattle_velocities(r, v, inv_masses, system.constraints, box=system.box)

        system.positions = r
        system.velocities = v
        if self.scf_drude and system.n_drude > 0:
            system.drude_positions = dpos_new
            system.drude_velocities = v[system.polarizable_indices].copy()

        eb_final, _, _ = engine.evaluate(
            system.positions,
            system.drude_positions,
            system.velocities,
            system.drude_velocities,
            external_bias=external_bias,
            scf_drude=False,
        )
        eb_final.extended_hamiltonian = eb_final.total_energy
        return eb_final, f1


class DrudeDualThermostatIntegrator:
    """Extended-Lagrangian dual-thermostat Nosé-Hoover Drude integrator (Lamoureux-Roux).

    Transforms core + Drude degrees of freedom into atomic center-of-mass (COM)
    coordinates coupled to a warm thermostat at `temperature_k` (e.g. 300 K) and
    relative internal Drude displacements coupled to a cold thermostat at
    `drude_temperature_k` (e.g. 1 K), enabling stable sub-femtosecond polarizable
    dynamics without expensive per-step SCF optimization.
    """

    def __init__(
        self,
        temperature_k: float = 300.0,
        drude_temperature_k: float = 1.0,
        tau_atom_ps: float = 0.05,
        tau_drude_ps: float = 0.05,
        dt_ps: float = 0.0005,
    ) -> None:
        self.temperature_k = float(temperature_k)
        self.drude_temperature_k = float(drude_temperature_k)
        self.tau_atom_ps = float(tau_atom_ps)
        self.tau_drude_ps = float(tau_drude_ps)
        self.dt_ps = float(dt_ps)
        self.xi_atom: float = 0.0
        self.v_xi_atom: float = 0.0
        self.xi_drude: float = 0.0
        self.v_xi_drude: float = 0.0

    def step(
        self,
        system: MolecularSystem,
        engine: ForceFieldEngine,
        external_bias: Optional[ExternalBiasFn] = None,
    ) -> EnergyBreakdown:
        """Advance coupled COM + relative-Drude phase space with dual Nose-Hoover baths."""
        pol_idx = system.polarizable_indices
        nd = len(pol_idx)
        n = system.n_atoms
        dt = self.dt_ps
        half_dt = 0.5 * dt

        if nd == 0:
            # Degenerates cleanly to single Nose-Hoover NVT Velocity-Verlet
            masses = np.array([a.mass for a in system.atoms], dtype=np.float64)
            dof = max(1, 3 * n)
            target_2k = dof * KB_KCAL_MOL_K * self.temperature_k
            q_atom = max(1e-4, target_2k * self.tau_atom_ps**2)

            eb0, f0, _ = engine.evaluate(external_bias=external_bias)
            k2 = 2.0 * eb0.kinetic_atom
            self.v_xi_atom += half_dt * (k2 - target_2k) / q_atom
            self.xi_atom += half_dt * self.v_xi_atom
            scale = float(np.exp(-half_dt * self.v_xi_atom))
            v = system.velocities * scale
            v += half_dt * (f0 * FORCE_TO_A_PS2_PER_AMU) / masses[:, None]
            r = system.positions + dt * v
            _, f1, _ = engine.evaluate(positions=r, velocities=v, external_bias=external_bias)
            v += half_dt * (f1 * FORCE_TO_A_PS2_PER_AMU) / masses[:, None]
            k2_mid = float(AMU_A2_PS2_TO_KCAL_MOL * np.sum(masses[:, None] * v * v))
            self.v_xi_atom += half_dt * (k2_mid - target_2k) / q_atom
            scale2 = float(np.exp(-half_dt * self.v_xi_atom))
            v *= scale2
            self.xi_atom += half_dt * self.v_xi_atom

            system.positions = r
            system.velocities = v
            eb_fin, _, _ = engine.evaluate(external_bias=external_bias)
            eb_fin.extended_hamiltonian = (
                eb_fin.total_energy
                + 0.5 * q_atom * self.v_xi_atom**2
                + target_2k * self.xi_atom
            )
            return eb_fin

        # Build mapping between atomic and COM/relative coordinates
        atom_to_drude = {int(a_i): d_i for d_i, a_i in enumerate(pol_idx)}
        m_tot = np.array([a.mass for a in system.atoms], dtype=np.float64)
        m_core = np.array([a.core_mass for a in system.atoms], dtype=np.float64)
        m_drude = np.array([system.atoms[int(a_i)].drude_mass for a_i in pol_idx], dtype=np.float64)
        mu_drude = (m_core[pol_idx] * m_drude) / m_tot[pol_idx]

        # Transform to COM (R, V_com) and relative (d_rel, v_rel)
        r_com = system.positions.copy()
        v_com = system.velocities.copy()
        d_rel = system.drude_positions - system.positions[pol_idx]
        v_rel = system.drude_velocities - system.velocities[pol_idx]
        for a_i, d_i in atom_to_drude.items():
            mc = m_core[a_i]
            md = m_drude[d_i]
            mt = m_tot[a_i]
            r_com[a_i] = (mc * system.positions[a_i] + md * system.drude_positions[d_i]) / mt
            v_com[a_i] = (mc * system.velocities[a_i] + md * system.drude_velocities[d_i]) / mt

        dof_atom = max(1, 3 * n)
        dof_drude = max(1, 3 * nd)
        target_2k_atom = dof_atom * KB_KCAL_MOL_K * self.temperature_k
        target_2k_drude = dof_drude * KB_KCAL_MOL_K * self.drude_temperature_k
        q_atom = max(1e-3, target_2k_atom * self.tau_atom_ps**2)
        # Floor q_drude so small-cluster transient Drude coupling never overflows v_xi_drude
        q_drude = max(2e-3, target_2k_drude * self.tau_drude_ps**2)

        def _unpack(rc: np.ndarray, dr: np.ndarray, vc: np.ndarray, vr: np.ndarray):
            pos_c = rc.copy()
            vel_c = vc.copy()
            pos_d = np.zeros((nd, 3), dtype=np.float64)
            vel_d = np.zeros((nd, 3), dtype=np.float64)
            for a_i, d_i in atom_to_drude.items():
                frac_d = m_drude[d_i] / m_tot[a_i]
                frac_c = m_core[a_i] / m_tot[a_i]
                pos_c[a_i] = rc[a_i] - frac_d * dr[d_i]
                pos_d[d_i] = rc[a_i] + frac_c * dr[d_i]
                vel_c[a_i] = vc[a_i] - frac_d * vr[d_i]
                vel_d[d_i] = vc[a_i] + frac_c * vr[d_i]
            return pos_c, pos_d, vel_c, vel_d

        def _forces_com_rel(f_core: np.ndarray, f_d: np.ndarray):
            f_com = f_core.copy()
            f_rel = np.zeros((nd, 3), dtype=np.float64)
            for a_i, d_i in atom_to_drude.items():
                f_com[a_i] = f_core[a_i] + f_d[d_i]
                f_rel[d_i] = (m_core[a_i] * f_d[d_i] - m_drude[d_i] * f_core[a_i]) / m_tot[a_i]
            return f_com, f_rel

        _, f_c0, f_d0 = engine.evaluate(external_bias=external_bias)
        f_com0, f_rel0 = _forces_com_rel(f_c0, f_d0)

        # Half-step Nose-Hoover chain on COM and relative Drude velocities
        k2_a = float(AMU_A2_PS2_TO_KCAL_MOL * np.sum(m_tot[:, None] * v_com * v_com))
        k2_d = float(AMU_A2_PS2_TO_KCAL_MOL * np.sum(mu_drude[:, None] * v_rel * v_rel))

        self.v_xi_atom += half_dt * (k2_a - target_2k_atom) / q_atom
        self.v_xi_drude += half_dt * (k2_d - target_2k_drude) / q_drude
        self.v_xi_atom = float(np.clip(self.v_xi_atom, -40.0, 40.0))
        self.v_xi_drude = float(np.clip(self.v_xi_drude, -80.0, 80.0))
        self.xi_atom += half_dt * self.v_xi_atom
        self.xi_drude += half_dt * self.v_xi_drude

        v_com *= float(np.exp(-half_dt * self.v_xi_atom))
        v_rel *= float(np.exp(-half_dt * self.v_xi_drude))

        # Half-kick + full drift
        v_com += half_dt * (f_com0 * FORCE_TO_A_PS2_PER_AMU) / m_tot[:, None]
        v_rel += half_dt * (f_rel0 * FORCE_TO_A_PS2_PER_AMU) / mu_drude[:, None]
        r_com += dt * v_com
        d_rel += dt * v_rel

        pos_c1, pos_d1, vel_c1, vel_d1 = _unpack(r_com, d_rel, v_com, v_rel)
        _, f_c1, f_d1 = engine.evaluate(
            positions=pos_c1,
            drude_positions=pos_d1,
            velocities=vel_c1,
            drude_velocities=vel_d1,
            external_bias=external_bias,
        )
        f_com1, f_rel1 = _forces_com_rel(f_c1, f_d1)

        # Second half-kick
        v_com += half_dt * (f_com1 * FORCE_TO_A_PS2_PER_AMU) / m_tot[:, None]
        v_rel += half_dt * (f_rel1 * FORCE_TO_A_PS2_PER_AMU) / mu_drude[:, None]

        # Second half-step Nose-Hoover
        k2_a_new = float(AMU_A2_PS2_TO_KCAL_MOL * np.sum(m_tot[:, None] * v_com * v_com))
        k2_d_new = float(AMU_A2_PS2_TO_KCAL_MOL * np.sum(mu_drude[:, None] * v_rel * v_rel))
        self.v_xi_atom += half_dt * (k2_a_new - target_2k_atom) / q_atom
        self.v_xi_drude += half_dt * (k2_d_new - target_2k_drude) / q_drude
        self.v_xi_atom = float(np.clip(self.v_xi_atom, -40.0, 40.0))
        self.v_xi_drude = float(np.clip(self.v_xi_drude, -80.0, 80.0))
        v_com *= float(np.exp(-half_dt * self.v_xi_atom))
        v_rel *= float(np.exp(-half_dt * self.v_xi_drude))
        self.xi_atom += half_dt * self.v_xi_atom
        self.xi_drude += half_dt * self.v_xi_drude

        pos_cf, pos_df, vel_cf, vel_df = _unpack(r_com, d_rel, v_com, v_rel)
        system.positions = pos_cf
        system.drude_positions = pos_df
        system.velocities = vel_cf
        system.drude_velocities = vel_df

        eb_fin, _, _ = engine.evaluate(external_bias=external_bias)
        eb_fin.extended_hamiltonian = (
            eb_fin.total_energy
            + 0.5 * q_atom * self.v_xi_atom**2
            + target_2k_atom * self.xi_atom
            + 0.5 * q_drude * self.v_xi_drude**2
            + target_2k_drude * self.xi_drude
        )
        return eb_fin
