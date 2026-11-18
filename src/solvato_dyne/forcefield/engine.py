"""Unified force-field engine combining bonded, non-bonded, Drude/Thole, and GBSA."""

from __future__ import annotations

from typing import Callable, Optional, Tuple
import numpy as np

from solvato_dyne.forcefield.bonded import compute_bonded_forces
from solvato_dyne.forcefield.polarizable import (
    compute_nonbonded_and_drude_forces,
    relax_drude_scf,
)
from solvato_dyne.forcefield.solvation import compute_gbsa_forces
from solvato_dyne.types import (
    AMU_A2_PS2_TO_KCAL_MOL,
    DEBYE_PER_E_A,
    EnergyBreakdown,
    KB_KCAL_MOL_K,
    KCAL_MOL_A3_TO_BAR,
    MolecularSystem,
)


ExternalBiasFn = Callable[[np.ndarray], Tuple[float, np.ndarray]]


class ForceFieldEngine:
    """Evaluates total potential energy, forces, dipole moment, and kinetic observables."""

    def __init__(
        self,
        system: MolecularSystem,
        restraint_indices: Optional[np.ndarray] = None,
        restraint_ref: Optional[np.ndarray] = None,
        restraint_k: float = 0.0,
    ) -> None:
        self.system = system
        self.restraint_indices = restraint_indices
        self.restraint_ref = restraint_ref
        self.restraint_k = float(restraint_k)

    def compute_dipole_debye(
        self,
        positions: np.ndarray,
        drude_positions: Optional[np.ndarray] = None,
    ) -> float:
        """Compute total molecular dipole magnitude in Debye around system center of geometry."""
        cog = np.mean(positions, axis=0)
        mu = np.zeros(3, dtype=np.float64)
        for a in self.system.atoms:
            mu += a.core_charge * (positions[a.index] - cog)
        pol_idx = self.system.polarizable_indices
        if len(pol_idx) > 0 and drude_positions is not None:
            for d_i, a_i in enumerate(pol_idx):
                mu += self.system.atoms[int(a_i)].drude_charge * (drude_positions[d_i] - cog)
        return float(np.linalg.norm(mu) * DEBYE_PER_E_A)

    def evaluate(
        self,
        positions: Optional[np.ndarray] = None,
        drude_positions: Optional[np.ndarray] = None,
        velocities: Optional[np.ndarray] = None,
        drude_velocities: Optional[np.ndarray] = None,
        external_bias: Optional[ExternalBiasFn] = None,
        scf_drude: bool = False,
    ) -> Tuple[EnergyBreakdown, np.ndarray, np.ndarray]:
        """Compute complete energy decomposition and forces on cores and Drude shells."""
        pos = self.system.positions if positions is None else positions
        dpos = self.system.drude_positions if drude_positions is None else drude_positions

        if scf_drude and self.system.n_drude > 0:
            dpos, _ = relax_drude_scf(self.system, pos, dpos)

        e_bond_dict, f_bond, vir_bond = compute_bonded_forces(self.system, pos)
        e_nb_dict, f_nb, f_drude, vir_nb = compute_nonbonded_and_drude_forces(
            self.system, pos, dpos
        )
        e_gb_dict, f_gb = compute_gbsa_forces(self.system, pos)

        f_total = f_bond + f_nb + f_gb
        e_restraint = 0.0
        if (
            self.restraint_k > 0.0
            and self.restraint_indices is not None
            and self.restraint_ref is not None
        ):
            idx = self.restraint_indices
            disp = pos[idx] - self.restraint_ref
            e_restraint = float(0.5 * self.restraint_k * np.sum(disp * disp))
            f_total[idx] -= self.restraint_k * disp

        e_bias = 0.0
        if external_bias is not None:
            e_b, f_b = external_bias(pos)
            e_bias = float(e_b)
            f_total += f_b

        # Kinetic energies and instantaneous temperatures
        vel = self.system.velocities if velocities is None else velocities
        dvel = self.system.drude_velocities if drude_velocities is None else drude_velocities
        masses = np.array([a.core_mass for a in self.system.atoms], dtype=np.float64)
        pol_idx = self.system.polarizable_indices

        k_atom = 0.0
        t_atom = 0.0
        k_drude = 0.0
        t_drude = 0.0

        if vel is not None and len(vel) == self.system.n_atoms:
            if len(pol_idx) > 0 and dvel is not None and len(dvel) == len(pol_idx):
                # Exact canonical decomposition:
                # 0.5*m_c*v_c^2 + 0.5*m_D*v_D^2 == 0.5*M*V_COM^2 + 0.5*mu_D*v_rel^2
                atom_to_drude = {int(a_i): d_i for d_i, a_i in enumerate(pol_idx)}
                for idx_a, atom in enumerate(self.system.atoms):
                    if idx_a in atom_to_drude:
                        d_i = atom_to_drude[idx_a]
                        mc = atom.core_mass
                        md = atom.drude_mass
                        mtot = mc + md
                        mu = (mc * md) / mtot
                        v_com = (mc * vel[idx_a] + md * dvel[d_i]) / mtot
                        v_rel = dvel[d_i] - vel[idx_a]
                        k_atom += 0.5 * AMU_A2_PS2_TO_KCAL_MOL * mtot * float(np.dot(v_com, v_com))
                        k_drude += 0.5 * AMU_A2_PS2_TO_KCAL_MOL * mu * float(np.dot(v_rel, v_rel))
                    else:
                        k_atom += 0.5 * AMU_A2_PS2_TO_KCAL_MOL * atom.mass * float(np.dot(vel[idx_a], vel[idx_a]))
                dof_d = 3 * len(pol_idx)
                t_drude = float(2.0 * k_drude / (dof_d * KB_KCAL_MOL_K))
            else:
                v_sq = np.sum(vel * vel, axis=1)
                k_atom = float(0.5 * AMU_A2_PS2_TO_KCAL_MOL * np.sum(masses * v_sq))

            dof_atom = max(1, 3 * self.system.n_atoms - len(self.system.constraints))
            t_atom = float(2.0 * k_atom / (dof_atom * KB_KCAL_MOL_K))

        pressure_bar = 0.0
        if self.system.box is not None and self.system.box.periodic:
            vol = self.system.box.volume
            vir_tot = vir_bond + vir_nb
            pressure_bar = float(
                (2.0 * k_atom - float(np.trace(vir_tot))) / (3.0 * vol) * KCAL_MOL_A3_TO_BAR
            )

        eb = EnergyBreakdown(
            bond=e_bond_dict["bond"],
            angle=e_bond_dict["angle"],
            dihedral=e_bond_dict["dihedral"],
            improper=e_bond_dict["improper"],
            elastic_net=e_bond_dict["elastic_net"],
            lj=e_nb_dict["lj"],
            coulomb=e_nb_dict["coulomb"],
            drude_spring=e_nb_dict["drude_spring"],
            thole_dipole=e_nb_dict["thole_dipole"],
            gb_polar=e_gb_dict["gb_polar"],
            sasa_nonpolar=e_gb_dict["sasa_nonpolar"],
            restraint=e_restraint,
            bias=e_bias,
            kinetic_atom=k_atom,
            kinetic_drude=k_drude,
            temperature_atom_k=t_atom,
            temperature_drude_k=t_drude,
            dipole_debye=self.compute_dipole_debye(pos, dpos),
            pressure_bar=pressure_bar,
        )
        return eb, f_total, f_drude
