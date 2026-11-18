"""Core data types, physical constants, and immutable state containers for SolvatoDyne.

All quantities follow consistent computational biophysics units:
- Length: Angstroms (A)
- Time: picoseconds (ps)
- Mass: atomic mass units (amu, g/mol)
- Energy: kcal/mol
- Charge: elementary charge (e)
- Force: kcal/(mol*A)
- Temperature: Kelvin (K)
- Pressure: bar
- Polarizability: cubic Angstroms (A^3)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


# CODATA physical constants in MD units (A, ps, amu, kcal/mol, e, K)
KB_KCAL_MOL_K: float = 0.00198720425864083
COULOMB_KCAL_A_E2: float = 332.063713299
AMU_A2_PS2_TO_KCAL_MOL: float = 10.0 / 4184.0  # exact 0.002390057361376673
FORCE_TO_A_PS2_PER_AMU: float = 418.4          # 1.0 / AMU_A2_PS2_TO_KCAL_MOL
DEBYE_PER_E_A: float = 4.803204712570263
KCAL_MOL_A3_TO_BAR: float = 69476.95457
HBAR_KCAL_MOL_PS: float = 0.015186787          # hbar * N_A in kcal/mol * ps
SPEED_OF_LIGHT_A_PS: float = 2.99792458e6      # A / ps
CM_INV_TO_KCAL_MOL: float = 0.002859144        # hc * N_A in kcal/mol per cm^-1


@dataclass(frozen=True)
class BoxSpec:
    """Periodic simulation box specification supporting orthorhombic or triclinic cells.

    Attributes:
        vectors: 3x3 matrix whose rows are lattice vectors (a, b, c) in Angstroms.
        periodic: Whether periodic boundary conditions are active.
    """

    vectors: np.ndarray
    periodic: bool = True

    def __post_init__(self) -> None:
        arr = np.asarray(self.vectors, dtype=np.float64)
        if arr.shape != (3, 3):
            raise ValueError(f"Box vectors must have shape (3, 3), got {arr.shape}")
        object.__setattr__(self, "vectors", arr.copy())

    @classmethod
    def orthorhombic(cls, lx: float, ly: float, lz: float, periodic: bool = True) -> "BoxSpec":
        """Construct an axis-aligned orthorhombic periodic box."""
        if lx <= 0 or ly <= 0 or lz <= 0:
            raise ValueError("Orthogonal box dimensions must be strictly positive.")
        return cls(vectors=np.diag([float(lx), float(ly), float(lz)]), periodic=periodic)

    @property
    def volume(self) -> float:
        """Cell volume in A^3."""
        return float(abs(np.linalg.det(self.vectors)))

    @property
    def is_orthorhombic(self) -> bool:
        """Return True if off-diagonal lattice components are numerically zero."""
        off_diag = self.vectors - np.diag(np.diag(self.vectors))
        return bool(np.allclose(off_diag, 0.0, atol=1e-10))


@dataclass(frozen=True)
class AtomSpec:
    """Single physical atom or coarse-grained bead specification.

    Attributes:
        index: Zero-based atom index.
        name: Atom or bead name (e.g. 'OH2', 'CA', 'BB').
        residue_name: Three- or four-letter residue identifier.
        residue_id: One-based residue sequence number.
        element: Chemical element symbol or 'CG' for coarse-grained beads.
        mass: Atomic/total core+shell mass in amu (for Drude pairs, core holds total - drude_mass).
        charge: Core or static partial charge in e.
        sigma: Lennard-Jones sigma parameter in A.
        epsilon: Lennard-Jones well depth in kcal/mol.
        alpha_pol: Isotropic atomic dipole polarizability in A^3 (0.0 if non-polarizable).
        thole_a: Thole dipole-dipole screening factor (dimensionless, typically 1.3 - 2.6).
        drude_k: Harmonic core-Drude spring constant in kcal/(mol*A^2).
        drude_mass: Auxiliary Drude oscillator mass in amu (typically 0.4 amu).
        gb_radius: Intrinsic Born radius in A for OBC Generalized Born solvation.
        gb_scale: OBC descreening scale factor S_j.
        sasa_sigma: Atomic solvation tension parameter in kcal/(mol*A^2) for SASA.
    """

    index: int
    name: str
    residue_name: str
    residue_id: int
    element: str
    mass: float
    charge: float
    sigma: float = 3.2
    epsilon: float = 0.1
    alpha_pol: float = 0.0
    thole_a: float = 2.089
    drude_k: float = 1000.0
    drude_mass: float = 0.4
    gb_radius: float = 1.5
    gb_scale: float = 0.8
    sasa_sigma: float = 0.0054

    @property
    def is_polarizable(self) -> bool:
        """Return True if atom carries an inducible Drude shell."""
        return self.alpha_pol > 1e-9

    @property
    def drude_charge(self) -> float:
        """Charge q_D of the auxiliary Drude particle in elementary charges (negative)."""
        if not self.is_polarizable:
            return 0.0
        return -float(np.sqrt(self.drude_k * self.alpha_pol / COULOMB_KCAL_A_E2))

    @property
    def core_charge(self) -> float:
        """Effective core charge q_core = q_total - q_D preserving total net charge."""
        return self.charge - self.drude_charge

    @property
    def core_mass(self) -> float:
        """Remaining core mass m_core = m_total - m_D when Drude is active."""
        if not self.is_polarizable:
            return self.mass
        if self.mass <= self.drude_mass:
            raise ValueError(
                f"Atom {self.index} ({self.name}) mass {self.mass} <= drude_mass {self.drude_mass}"
            )
        return self.mass - self.drude_mass


@dataclass(frozen=True)
class BondTerm:
    """Harmonic covalent or coarse-grained bond V(r) = 0.5 * k * (r - r0)^2."""

    i: int
    j: int
    k: float
    r0: float
    is_elastic: bool = False


@dataclass(frozen=True)
class AngleTerm:
    """Harmonic valence angle V(theta) = 0.5 * k * (theta - theta0)^2."""

    i: int
    j: int
    k_idx: int
    k: float
    theta0: float  # radians


@dataclass(frozen=True)
class DihedralTerm:
    """Proper periodic torsion V(phi) = k * (1 + cos(periodicity * phi - phase))."""

    i: int
    j: int
    k_idx: int
    l_idx: int
    k: float
    periodicity: int
    phase: float  # radians


@dataclass(frozen=True)
class ImproperTerm:
    """Harmonic improper torsion V(omega) = 0.5 * k * (wrap(omega - omega0))^2."""

    i: int
    j: int
    k_idx: int
    l_idx: int
    k: float
    omega0: float  # radians


@dataclass(frozen=True)
class ConstraintPair:
    """Holonomic bond-length constraint ||r_i - r_j|| = d0 for RATTLE."""

    i: int
    j: int
    d0: float


@dataclass
class ForceFieldConfig:
    """Global force-field interaction settings.

    Attributes:
        lj_on: Switching start distance in A for C2 force-switched LJ.
        lj_cut: Cutoff distance in A for Lennard-Jones interactions.
        coulomb_cut: Cutoff distance in A for direct Coulomb interactions (0 = all pairs).
        scale_14_lj: Lennard-Jones scaling factor for 1-4 bonded pairs.
        scale_14_elec: Electrostatic scaling factor for 1-4 bonded pairs.
        use_gbsa: Enable OBC-II Generalized Born + smooth SASA implicit solvent.
        dielectric_in: Solute interior dielectric constant.
        dielectric_out: Continuum solvent dielectric constant.
        sasa_gamma: Nonpolar surface tension coefficient in kcal/(mol*A^2).
        drude_hard_wall: Maximum core-Drude separation (A) before quartic wall engages.
        drude_wall_k: Quartic hyperpolarization wall stiffness in kcal/(mol*A^4).
        use_thole: Enable Thole dipole-dipole screening between polarizable pairs.
    """

    lj_on: float = 8.0
    lj_cut: float = 12.0
    coulomb_cut: float = 0.0
    scale_14_lj: float = 0.5
    scale_14_elec: float = 0.8333333333333334
    use_gbsa: bool = True
    dielectric_in: float = 1.0
    dielectric_out: float = 78.5
    sasa_gamma: float = 0.0054
    drude_hard_wall: float = 0.25
    drude_wall_k: float = 20000.0
    use_thole: bool = True


@dataclass
class MolecularSystem:
    """Complete mutable topology and phase-space state for an atomistic or CG system."""

    name: str
    atoms: List[AtomSpec]
    positions: np.ndarray
    velocities: Optional[np.ndarray] = None
    drude_positions: Optional[np.ndarray] = None
    drude_velocities: Optional[np.ndarray] = None
    bonds: List[BondTerm] = field(default_factory=list)
    angles: List[AngleTerm] = field(default_factory=list)
    dihedrals: List[DihedralTerm] = field(default_factory=list)
    impropers: List[ImproperTerm] = field(default_factory=list)
    constraints: List[ConstraintPair] = field(default_factory=list)
    box: Optional[BoxSpec] = None
    ff_config: ForceFieldConfig = field(default_factory=ForceFieldConfig)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.atoms)
        pos = np.asarray(self.positions, dtype=np.float64)
        if pos.shape != (n, 3):
            raise ValueError(f"positions must have shape ({n}, 3), got {pos.shape}")
        self.positions = pos.copy()

        if self.velocities is None:
            self.velocities = np.zeros((n, 3), dtype=np.float64)
        else:
            vel = np.asarray(self.velocities, dtype=np.float64)
            if vel.shape != (n, 3):
                raise ValueError(f"velocities must have shape ({n}, 3), got {vel.shape}")
            self.velocities = vel.copy()

        pol_idx = self.polarizable_indices
        nd = len(pol_idx)
        if self.drude_positions is None:
            self.drude_positions = self.positions[pol_idx].copy() if nd > 0 else np.zeros((0, 3), dtype=np.float64)
        else:
            dp = np.asarray(self.drude_positions, dtype=np.float64)
            if dp.shape != (nd, 3):
                raise ValueError(f"drude_positions must have shape ({nd}, 3), got {dp.shape}")
            self.drude_positions = dp.copy()

        if self.drude_velocities is None:
            self.drude_velocities = np.zeros((nd, 3), dtype=np.float64)
        else:
            dv = np.asarray(self.drude_velocities, dtype=np.float64)
            if dv.shape != (nd, 3):
                raise ValueError(f"drude_velocities must have shape ({nd}, 3), got {dv.shape}")
            self.drude_velocities = dv.copy()

    @property
    def n_atoms(self) -> int:
        return len(self.atoms)

    @property
    def polarizable_indices(self) -> np.ndarray:
        """Integer array of atom indices carrying inducible Drude oscillators."""
        return np.array([a.index for a in self.atoms if a.is_polarizable], dtype=np.int64)

    @property
    def n_drude(self) -> int:
        return int(len(self.polarizable_indices))

    @property
    def total_charge(self) -> float:
        return float(sum(a.charge for a in self.atoms))

    def clone(self) -> "MolecularSystem":
        """Return an independent deep copy of coordinates, velocities, and topology references."""
        return MolecularSystem(
            name=self.name,
            atoms=list(self.atoms),
            positions=self.positions.copy(),
            velocities=None if self.velocities is None else self.velocities.copy(),
            drude_positions=None if self.drude_positions is None else self.drude_positions.copy(),
            drude_velocities=None if self.drude_velocities is None else self.drude_velocities.copy(),
            bonds=list(self.bonds),
            angles=list(self.angles),
            dihedrals=list(self.dihedrals),
            impropers=list(self.impropers),
            constraints=list(self.constraints),
            box=self.box,
            ff_config=ForceFieldConfig(**vars(self.ff_config)),
            metadata=dict(self.metadata),
        )

    def exclusion_sets(self) -> Tuple[set[Tuple[int, int]], set[Tuple[int, int]]]:
        """Compute 1-2/1-3 excluded atom pairs and 1-4 scaled atom pairs from bond graph."""
        n = self.n_atoms
        adj: List[set[int]] = [set() for _ in range(n)]
        for b in self.bonds:
            if not b.is_elastic:
                adj[b.i].add(b.j)
                adj[b.j].add(b.i)

        excluded: set[Tuple[int, int]] = set()
        pairs_14: set[Tuple[int, int]] = set()

        for i in range(n):
            n1 = adj[i]
            n2 = set()
            for j in n1:
                pair = (min(i, j), max(i, j))
                excluded.add(pair)
                n2 |= adj[j]
            n2.discard(i)
            n2 -= n1
            for k in n2:
                excluded.add((min(i, k), max(i, k)))
            n3 = set()
            for k in n2:
                n3 |= adj[k]
            n3.discard(i)
            n3 -= n1
            n3 -= n2
            for l_idx in n3:
                pair = (min(i, l_idx), max(i, l_idx))
                if pair not in excluded:
                    pairs_14.add(pair)

        return excluded, pairs_14


@dataclass
class EnergyBreakdown:
    """Decomposed potential, kinetic, and thermodynamic observables at a single snapshot."""

    bond: float = 0.0
    angle: float = 0.0
    dihedral: float = 0.0
    improper: float = 0.0
    elastic_net: float = 0.0
    lj: float = 0.0
    coulomb: float = 0.0
    drude_spring: float = 0.0
    thole_dipole: float = 0.0
    gb_polar: float = 0.0
    sasa_nonpolar: float = 0.0
    restraint: float = 0.0
    bias: float = 0.0
    kinetic_atom: float = 0.0
    kinetic_drude: float = 0.0
    temperature_atom_k: float = 0.0
    temperature_drude_k: float = 0.0
    dipole_debye: float = 0.0
    pressure_bar: float = 0.0
    extended_hamiltonian: float = 0.0

    @property
    def total_potential(self) -> float:
        return (
            self.bond
            + self.angle
            + self.dihedral
            + self.improper
            + self.elastic_net
            + self.lj
            + self.coulomb
            + self.drude_spring
            + self.thole_dipole
            + self.gb_polar
            + self.sasa_nonpolar
            + self.restraint
            + self.bias
        )

    @property
    def total_kinetic(self) -> float:
        return self.kinetic_atom + self.kinetic_drude

    @property
    def total_energy(self) -> float:
        return self.total_potential + self.total_kinetic


@dataclass
class TrajectoryFrame:
    """Single trajectory frame storing coordinates, energy breakdown, and optional CVs."""

    step: int
    time_ps: float
    positions: np.ndarray
    velocities: Optional[np.ndarray]
    drude_positions: Optional[np.ndarray]
    energy: EnergyBreakdown
    colvars: Dict[str, float] = field(default_factory=dict)


@dataclass
class Trajectory:
    """Sequence of simulation frames with summary conservation metrics."""

    system_name: str
    integrator: str
    dt_ps: float
    frames: List[TrajectoryFrame]

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    def positions_array(self) -> np.ndarray:
        """Return (T, N, 3) array of atomic positions."""
        return np.stack([f.positions for f in self.frames], axis=0)

    def potential_energies(self) -> np.ndarray:
        return np.array([f.energy.total_potential for f in self.frames], dtype=np.float64)

    def total_energies(self) -> np.ndarray:
        return np.array([f.energy.total_energy for f in self.frames], dtype=np.float64)

    def temperatures_atom(self) -> np.ndarray:
        return np.array([f.energy.temperature_atom_k for f in self.frames], dtype=np.float64)

    def temperatures_drude(self) -> np.ndarray:
        return np.array([f.energy.temperature_drude_k for f in self.frames], dtype=np.float64)

    def dipoles(self) -> np.ndarray:
        return np.array([f.energy.dipole_debye for f in self.frames], dtype=np.float64)

    def energy_drift_per_step(self) -> float:
        """Slope of total/extended energy per frame normalized by initial |E| or N."""
        if len(self.frames) < 2:
            return 0.0
        vals = np.array(
            [
                f.energy.extended_hamiltonian if f.energy.extended_hamiltonian != 0.0 else f.energy.total_energy
                for f in self.frames
            ],
            dtype=np.float64,
        )
        return float(np.std(vals) / max(1.0, abs(float(np.mean(vals)))))


@dataclass
class REMDResult:
    """Summary of a Temperature Replica-Exchange Molecular Dynamics run."""

    temperatures_k: np.ndarray
    acceptance_rates: np.ndarray
    replica_state_history: np.ndarray  # shape (n_cycles, n_replicas)
    potential_histories: np.ndarray    # shape (n_cycles, n_replicas)
    round_trips: int
    heat_capacity_curve: Tuple[np.ndarray, np.ndarray]  # (T_grid, Cv_kcal_mol_K)
    trajectories: List[Trajectory]


@dataclass
class MetaDResult:
    """Summary of a Well-Tempered Metadynamics run with Tiwary-Parrinello reweighting."""

    cv_names: List[str]
    cv_history: np.ndarray
    hill_centers: np.ndarray
    hill_heights: np.ndarray
    hill_sigmas: np.ndarray
    bias_factor: float
    grid_axes: List[np.ndarray]
    fes_direct: np.ndarray
    fes_reweighted: np.ndarray
    c_of_t: np.ndarray
    trajectory: Trajectory


@dataclass
class SMDResult:
    """Bidirectional Steered MD work distributions and free-energy estimates."""

    forward_works: np.ndarray
    reverse_works: np.ndarray
    dg_jarzynski_fwd: float
    dg_cumulant_fwd: float
    dg_crooks_mle: float
    dg_bootstrap_ci95: Tuple[float, float]
    dissipated_work_fwd: float
    cv_name: str
    forward_paths: List[Tuple[np.ndarray, np.ndarray]]
    reverse_paths: List[Tuple[np.ndarray, np.ndarray]]


@dataclass
class NMAResult:
    """Eckart-projected Hessian or ANM Normal Mode Analysis thermodynamic output."""

    method: str
    eigenvalues: np.ndarray
    frequencies_cm_inv: np.ndarray
    modes_cartesian: np.ndarray
    zero_mode_residual: float
    zpve_kcal_mol: float
    vibrational_entropy_kcal_mol_k: float
    helmholtz_vib_kcal_mol: float
    b_factors_a2: np.ndarray


@dataclass
class GradientCheckEntry:
    """Single force-term analytical vs central finite-difference audit result."""

    term_name: str
    max_abs_error: float
    rms_rel_error: float
    passed: bool


@dataclass
class CalibrationReport:
    """Results from analytical gradient audit and force-field parameter calibration."""

    gradient_checks: List[GradientCheckEntry]
    fitted_params: Dict[str, float]
    initial_rmse: float
    final_rmse: float
    sensitivity_indices: Dict[str, float]


@dataclass
class SensitivityReport:
    """Parametric sensitivity and uncertainty quantification summary."""

    baseline_value: float
    parameter_deltas: Dict[str, float]
    normalized_sensitivities: Dict[str, float]
    bootstrap_mean: float
    bootstrap_std: float
    ci95: Tuple[float, float]
