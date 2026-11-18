"""Self-contained molecular topology builders, CG Martini mapping, and PDB I/O.

Constructs reproducible atomistic, Drude-polarizable, implicit-solvent, and
coarse-grained elastic-network systems with physically realistic force-field
parameters and zero external downloads.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np

from solvato_dyne.types import (
    AngleTerm,
    AtomSpec,
    BondTerm,
    BoxSpec,
    ConstraintPair,
    DihedralTerm,
    ForceFieldConfig,
    ImproperTerm,
    MolecularSystem,
)


_ELEMENT_MASSES: Dict[str, float] = {
    "H": 1.008,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "S": 32.06,
    "P": 30.974,
    "CG": 72.0,
}

_ELEMENT_GB_RADII: Dict[str, float] = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.50,
    "S": 1.80,
    "P": 1.85,
    "CG": 2.35,
}

_ELEMENT_GB_SCALE: Dict[str, float] = {
    "H": 0.85,
    "C": 0.72,
    "N": 0.79,
    "O": 0.85,
    "S": 0.96,
    "P": 0.86,
    "CG": 0.80,
}

_ELEMENT_LJ: Dict[str, Tuple[float, float]] = {
    "H": (2.35, 0.0157),
    "C": (3.40, 0.1094),
    "N": (3.25, 0.1700),
    "O": (3.06, 0.2100),
    "S": (3.56, 0.2500),
    "P": (3.74, 0.2000),
    "CG": (4.70, 0.8500),
}


def _make_atom(
    idx: int,
    name: str,
    resname: str,
    resid: int,
    element: str,
    charge: float,
    alpha_pol: float = 0.0,
    thole_a: float = 2.089,
    drude_k: float = 1000.0,
    sigma: Optional[float] = None,
    epsilon: Optional[float] = None,
) -> AtomSpec:
    el = element.upper()
    lj_sig, lj_eps = _ELEMENT_LJ.get(el, (3.3, 0.1))
    return AtomSpec(
        index=idx,
        name=name,
        residue_name=resname,
        residue_id=resid,
        element=el,
        mass=_ELEMENT_MASSES.get(el, 12.011),
        charge=charge,
        sigma=lj_sig if sigma is None else sigma,
        epsilon=lj_eps if epsilon is None else epsilon,
        alpha_pol=alpha_pol,
        thole_a=thole_a,
        drude_k=drude_k,
        drude_mass=0.4,
        gb_radius=_ELEMENT_GB_RADII.get(el, 1.6),
        gb_scale=_ELEMENT_GB_SCALE.get(el, 0.8),
    )


def build_polarizable_water_dimer_or_cluster(
    n_molecules: int = 2,
    box_size: Optional[float] = None,
    polarizable: bool = True,
    seed: int = 42,
) -> MolecularSystem:
    """Build an SWM4-inspired Drude-polarizable water dimer or multi-water cluster.

    Each water molecule consists of O (sigma=3.184 A, epsilon=0.211 kcal/mol,
    alpha_O=0.978 A^3 when polarizable) and two hydrogens H1, H2 at OH=0.9572 A,
    HOH=104.52 deg, preserving exact net charge neutrality (q_O + 2*q_H = 0).

    Args:
        n_molecules: Number of water molecules (>= 1).
        box_size: Optional cubic periodic box edge in A (None for gas-phase cluster).
        polarizable: Whether oxygen atoms carry inducible Drude oscillators.
        seed: Random seed for multi-molecule placement.

    Returns:
        Configured MolecularSystem ready for Drude SCF or dual-thermostat MD.
    """
    if n_molecules < 1:
        raise ValueError("n_molecules must be >= 1.")

    rng = np.random.default_rng(seed)
    atoms: List[AtomSpec] = []
    coords: List[np.ndarray] = []
    bonds: List[BondTerm] = []
    angles: List[AngleTerm] = []
    constraints: List[ConstraintPair] = []

    r_oh = 0.9572
    half_angle = np.deg2rad(104.52 / 2.0)
    # Calibrated 3-site charge so OH geometry reproduces gas-phase water dipole ~1.855 D
    q_h = 0.3300 if polarizable else 0.4170
    q_o = -2.0 * q_h
    alpha_o = 0.97825 if polarizable else 0.0

    # Place molecules along hydrogen-bonded / lattice grid spacing ~2.85 A
    n_side = max(1, int(np.ceil(n_molecules ** (1.0 / 3.0))))
    spacing = 2.90

    for m in range(n_molecules):
        if m == 0:
            center = np.array([0.0, 0.0, 0.0])
            rot = np.eye(3)
        elif m == 1 and n_molecules == 2:
            # Canonical water dimer geometry: acceptor along +x at O..O ~ 2.91 A
            center = np.array([2.91, 0.0, 0.0])
            theta = np.deg2rad(55.0)
            rot = np.array([
                [np.cos(theta), 0.0, -np.sin(theta)],
                [0.0, 1.0, 0.0],
                [np.sin(theta), 0.0, np.cos(theta)],
            ])
        else:
            ix = m % n_side
            iy = (m // n_side) % n_side
            iz = m // (n_side * n_side)
            center = np.array([ix * spacing, iy * spacing, iz * spacing]) + rng.uniform(-0.15, 0.15, 3)
            u, _, vt = np.linalg.svd(rng.normal(size=(3, 3)))
            rot = u @ vt
            if np.linalg.det(rot) < 0:
                rot[:, 0] *= -1.0

        o_local = np.array([0.0, 0.0, 0.0])
        h1_local = np.array([r_oh * np.cos(half_angle), r_oh * np.sin(half_angle), 0.0])
        h2_local = np.array([r_oh * np.cos(half_angle), -r_oh * np.sin(half_angle), 0.0])

        idx_o = len(atoms)
        atoms.append(
            _make_atom(
                idx_o, "OH2", "WAT", m + 1, "O",
                charge=q_o, alpha_pol=alpha_o, thole_a=2.60, drude_k=1000.0,
                sigma=3.184, epsilon=0.2108,
            )
        )
        atoms.append(
            _make_atom(
                idx_o + 1, "H1", "WAT", m + 1, "H",
                charge=q_h, alpha_pol=0.0, sigma=0.40, epsilon=0.001,
            )
        )
        atoms.append(
            _make_atom(
                idx_o + 2, "H2", "WAT", m + 1, "H",
                charge=q_h, alpha_pol=0.0, sigma=0.40, epsilon=0.001,
            )
        )

        coords.append(center + rot @ o_local)
        coords.append(center + rot @ h1_local)
        coords.append(center + rot @ h2_local)

        bonds.append(BondTerm(idx_o, idx_o + 1, k=450.0, r0=r_oh))
        bonds.append(BondTerm(idx_o, idx_o + 2, k=450.0, r0=r_oh))
        angles.append(AngleTerm(idx_o + 1, idx_o, idx_o + 2, k=55.0, theta0=np.deg2rad(104.52)))
        constraints.append(ConstraintPair(idx_o, idx_o + 1, r_oh))
        constraints.append(ConstraintPair(idx_o, idx_o + 2, r_oh))

    pos = np.vstack(coords)
    box = BoxSpec.orthorhombic(box_size, box_size, box_size) if box_size is not None else None
    ff_cfg = ForceFieldConfig(
        lj_on=7.0,
        lj_cut=11.0,
        use_gbsa=False,
        use_thole=True,
        drude_hard_wall=0.25,
    )
    return MolecularSystem(
        name=f"polarizable_water_{n_molecules}",
        atoms=atoms,
        positions=pos,
        bonds=bonds,
        angles=angles,
        constraints=constraints,
        box=box,
        ff_config=ff_cfg,
        metadata={"n_waters": n_molecules, "polarizable": polarizable},
    )


def build_alanine_dipeptide_system(
    polarizable: bool = True,
    use_gbsa: bool = True,
) -> MolecularSystem:
    """Build a 12-heavy-atom capped Ace-Ala-Nme alanine dipeptide conformational benchmark.

    Includes calibrated backbone (phi, psi) dihedral potentials producing realistic
    C7eq / alpha_R / C7ax basins separated by ~2-4 kcal/mol torsional barriers, making
    it ideal for Well-Tempered Metadynamics and T-REMD verification.

    Atom layout (12 atoms):
      0: CH3 (ACE), 1: C (ACE), 2: O (ACE), 3: N (ALA), 4: H (ALA),
      5: CA (ALA), 6: CB (ALA), 7: C (ALA), 8: O (ALA),
      9: N (NME), 10: H (NME), 11: CH3 (NME)
    Backbone phi = (1, 3, 5, 7), psi = (3, 5, 7, 9).
    """
    specs = [
        ("CH3", "ACE", 1, "C", -0.08, 0.0),
        ("C", "ACE", 1, "C", 0.51, 0.60 if polarizable else 0.0),
        ("O", "ACE", 1, "O", -0.51, 0.85 if polarizable else 0.0),
        ("N", "ALA", 2, "N", -0.47, 0.55 if polarizable else 0.0),
        ("HN", "ALA", 2, "H", 0.31, 0.0),
        ("CA", "ALA", 2, "C", 0.07, 0.0),
        ("CB", "ALA", 2, "C", 0.09, 0.0),
        ("C", "ALA", 2, "C", 0.51, 0.60 if polarizable else 0.0),
        ("O", "ALA", 2, "O", -0.51, 0.85 if polarizable else 0.0),
        ("N", "NME", 3, "N", -0.47, 0.55 if polarizable else 0.0),
        ("HN", "NME", 3, "H", 0.31, 0.0),
        ("CH3", "NME", 3, "C", 0.25, 0.0),
    ]
    # Enforce exact neutrality
    raw_q = sum(s[4] for s in specs)
    atoms = []
    for idx, (nm, res, rid, el, q, alpha) in enumerate(specs):
        adj_q = q - raw_q if idx == 0 else q
        atoms.append(_make_atom(idx, nm, res, rid, el, adj_q, alpha_pol=alpha))

    # Extended near-C7eq starting geometry (Angstroms)
    positions = np.array([
        [-3.45, 0.80, 0.10],   # 0 CH3(ACE)
        [-2.05, 0.30, 0.00],   # 1 C(ACE)
        [-1.70, -0.88, -0.05],  # 2 O(ACE)
        [-1.12, 1.28, 0.02],   # 3 N(ALA)
        [-1.45, 2.23, 0.08],   # 4 HN(ALA)
        [0.30, 1.02, -0.05],   # 5 CA(ALA)
        [0.90, 1.50, 1.26],    # 6 CB(ALA)
        [1.04, 1.62, -1.25],   # 7 C(ALA)
        [0.52, 2.55, -1.88],   # 8 O(ALA)
        [2.25, 1.10, -1.55],   # 9 N(NME)
        [2.60, 0.35, -0.97],   # 10 HN(NME)
        [3.05, 1.60, -2.65],   # 11 CH3(NME)
    ], dtype=np.float64)

    bond_defs = [
        (0, 1, 317.0, 1.50),
        (1, 2, 570.0, 1.23),
        (1, 3, 490.0, 1.33),
        (3, 4, 434.0, 1.01),
        (3, 5, 337.0, 1.45),
        (5, 6, 310.0, 1.52),
        (5, 7, 317.0, 1.52),
        (7, 8, 570.0, 1.23),
        (7, 9, 490.0, 1.33),
        (9, 10, 434.0, 1.01),
        (9, 11, 337.0, 1.45),
    ]
    bonds = [BondTerm(i, j, k, r0) for i, j, k, r0 in bond_defs]

    angle_defs = [
        (0, 1, 2, 80.0, 120.5),
        (0, 1, 3, 70.0, 116.5),
        (2, 1, 3, 80.0, 123.0),
        (1, 3, 4, 50.0, 119.5),
        (1, 3, 5, 50.0, 121.0),
        (4, 3, 5, 45.0, 118.5),
        (3, 5, 6, 65.0, 110.5),
        (3, 5, 7, 65.0, 111.0),
        (6, 5, 7, 65.0, 110.5),
        (5, 7, 8, 80.0, 120.5),
        (5, 7, 9, 70.0, 116.5),
        (8, 7, 9, 80.0, 123.0),
        (7, 9, 10, 50.0, 119.5),
        (7, 9, 11, 50.0, 121.0),
        (10, 9, 11, 45.0, 118.5),
    ]
    angles = [AngleTerm(i, j, k_idx, k, np.deg2rad(deg)) for i, j, k_idx, k, deg in angle_defs]

    # Peptide bond planarity (omega ~ 180 deg) + dual-basin phi/psi landscape
    dihedrals = [
        # Omega1 (0, 1, 3, 5) trans preference
        DihedralTerm(0, 1, 3, 5, k=7.5, periodicity=2, phase=np.pi),
        # Omega2 (5, 7, 9, 11) trans preference
        DihedralTerm(5, 7, 9, 11, k=7.5, periodicity=2, phase=np.pi),
        # Phi (1, 3, 5, 7) double-well torsional profile
        DihedralTerm(1, 3, 5, 7, k=1.35, periodicity=2, phase=0.0),
        DihedralTerm(1, 3, 5, 7, k=0.75, periodicity=1, phase=np.deg2rad(60.0)),
        # Psi (3, 5, 7, 9) double-well torsional profile
        DihedralTerm(3, 5, 7, 9, k=1.40, periodicity=2, phase=np.deg2rad(15.0)),
        DihedralTerm(3, 5, 7, 9, k=0.65, periodicity=1, phase=np.deg2rad(-40.0)),
    ]

    impropers = [
        ImproperTerm(0, 2, 1, 3, k=40.0, omega0=0.0),
        ImproperTerm(5, 8, 7, 9, k=40.0, omega0=0.0),
    ]

    constraints = [
        ConstraintPair(3, 4, 1.01),
        ConstraintPair(9, 10, 1.01),
    ]

    ff_cfg = ForceFieldConfig(
        lj_on=8.0,
        lj_cut=14.0,
        use_gbsa=use_gbsa,
        use_thole=True,
    )
    return MolecularSystem(
        name="alanine_dipeptide",
        atoms=atoms,
        positions=positions,
        bonds=bonds,
        angles=angles,
        dihedrals=dihedrals,
        impropers=impropers,
        constraints=constraints,
        ff_config=ff_cfg,
        metadata={"phi_indices": (1, 3, 5, 7), "psi_indices": (3, 5, 7, 9)},
    )


def build_alpha_helix_peptide(
    n_res: int = 8,
    polarizable: bool = False,
    use_gbsa: bool = True,
) -> MolecularSystem:
    """Build a helical polyalanine backbone peptide (N, CA, C, O per residue).

    Ideal alpha-helix geometry (3.6 residues/turn, pitch 5.4 A, radius 2.3 A)
    with calibrated i -> i+4 hydrogen-bond attraction via partial charges and
    backbone dihedral potentials. Ideal for T-REMD melting and Hessian NMA.
    """
    if n_res < 3:
        raise ValueError("n_res must be >= 3.")

    atoms: List[AtomSpec] = []
    coords: List[np.ndarray] = []
    bonds: List[BondTerm] = []
    angles: List[AngleTerm] = []
    dihedrals: List[DihedralTerm] = []
    impropers: List[ImproperTerm] = []

    turn_angle = np.deg2rad(100.0)
    rise = 1.50
    radius = 2.30

    for r in range(n_res):
        theta = r * turn_angle
        z_base = r * rise

        # N, CA, C, O positions around helical cylinder
        p_n = np.array([radius * np.cos(theta - 0.32), radius * np.sin(theta - 0.32), z_base - 0.48])
        p_ca = np.array([radius * np.cos(theta), radius * np.sin(theta), z_base])
        p_c = np.array([radius * np.cos(theta + 0.30), radius * np.sin(theta + 0.30), z_base + 0.50])
        p_o = np.array([(radius + 0.85) * np.cos(theta + 0.32), (radius + 0.85) * np.sin(theta + 0.32), z_base + 1.05])

        base_idx = len(atoms)
        atoms.append(_make_atom(base_idx, "N", "ALA", r + 1, "N", -0.35, alpha_pol=0.55 if polarizable else 0.0))
        atoms.append(_make_atom(base_idx + 1, "CA", "ALA", r + 1, "C", 0.15, alpha_pol=0.0))
        atoms.append(_make_atom(base_idx + 2, "C", "ALA", r + 1, "C", 0.55, alpha_pol=0.60 if polarizable else 0.0))
        atoms.append(_make_atom(base_idx + 3, "O", "ALA", r + 1, "O", -0.35, alpha_pol=0.80 if polarizable else 0.0))
        coords.extend([p_n, p_ca, p_c, p_o])

        # Intra-residue covalent bonds using exact initial distances as equilibrium r0
        d_nca = float(np.linalg.norm(p_ca - p_n))
        d_cac = float(np.linalg.norm(p_c - p_ca))
        d_co = float(np.linalg.norm(p_o - p_c))
        bonds.append(BondTerm(base_idx, base_idx + 1, k=320.0, r0=d_nca))
        bonds.append(BondTerm(base_idx + 1, base_idx + 2, k=320.0, r0=d_cac))
        bonds.append(BondTerm(base_idx + 2, base_idx + 3, k=550.0, r0=d_co))

        # Valence angles anchored to helical geometry
        def _ang(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
            v1 = a - b
            v2 = c - b
            cos_t = np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -0.999999, 0.999999)
            return float(np.arccos(cos_t))

        angles.append(AngleTerm(base_idx, base_idx + 1, base_idx + 2, k=65.0, theta0=_ang(p_n, p_ca, p_c)))
        angles.append(AngleTerm(base_idx + 1, base_idx + 2, base_idx + 3, k=75.0, theta0=_ang(p_ca, p_c, p_o)))

        if r > 0:
            prev_c = base_idx - 2
            prev_o = base_idx - 1
            prev_ca = base_idx - 3
            d_cn = float(np.linalg.norm(coords[base_idx] - coords[prev_c]))
            bonds.append(BondTerm(prev_c, base_idx, k=480.0, r0=d_cn))
            angles.append(
                AngleTerm(
                    prev_ca, prev_c, base_idx, k=70.0,
                    theta0=_ang(coords[prev_ca], coords[prev_c], coords[base_idx]),
                )
            )
            angles.append(
                AngleTerm(
                    prev_o, prev_c, base_idx, k=80.0,
                    theta0=_ang(coords[prev_o], coords[prev_c], coords[base_idx]),
                )
            )
            angles.append(
                AngleTerm(
                    prev_c, base_idx, base_idx + 1, k=60.0,
                    theta0=_ang(coords[prev_c], coords[base_idx], coords[base_idx + 1]),
                )
            )
            # Backbone phi/psi terms
            dihedrals.append(
                DihedralTerm(prev_c, base_idx, base_idx + 1, base_idx + 2, k=1.2, periodicity=1, phase=np.pi)
            )
            dihedrals.append(
                DihedralTerm(prev_ca, prev_c, base_idx, base_idx + 1, k=6.0, periodicity=2, phase=np.pi)
            )
            impropers.append(ImproperTerm(prev_ca, prev_o, prev_c, base_idx, k=35.0, omega0=0.0))

        # Native i -> i+4 stabilizing elastic hydrogen-bond proxy for folding transition
        if r >= 4:
            donor_n = base_idx
            acc_o = (r - 4) * 4 + 3
            d_no = float(np.linalg.norm(coords[donor_n] - coords[acc_o]))
            bonds.append(BondTerm(acc_o, donor_n, k=2.5, r0=d_no, is_elastic=True))

    pos = np.vstack(coords)
    ff_cfg = ForceFieldConfig(use_gbsa=use_gbsa, use_thole=True, lj_on=8.0, lj_cut=14.0)
    return MolecularSystem(
        name=f"alpha_helix_{n_res}mer",
        atoms=atoms,
        positions=pos,
        bonds=bonds,
        angles=angles,
        dihedrals=dihedrals,
        impropers=impropers,
        ff_config=ff_cfg,
        metadata={"n_residues": n_res, "helix_ca_indices": [4 * i + 1 for i in range(n_res)]},
    )


def build_host_guest_unbinding_system(
    polarizable: bool = False,
) -> MolecularSystem:
    """Build a 3D receptor pocket + guest ligand complex for bidirectional SMD unbinding.

    Receptor consists of an 8-atom ring/cavity anchored with an elastic scaffold
    (indices 0..7) and a bound 3-atom polarizable guest ligand (indices 8..10)
    sitting in an energetic well of ~5-9 kcal/mol. Pulling along the pocket-ligand
    COM separation cleanly probes forward unbinding and reverse rebinding work.
    """
    atoms: List[AtomSpec] = []
    coords: List[np.ndarray] = []
    bonds: List[BondTerm] = []
    angles: List[AngleTerm] = []

    # 8-atom host pocket ring + base forming a deep binding funnel along +z
    for k in range(6):
        ang = k * np.pi / 3.0
        coords.append(np.array([3.35 * np.cos(ang), 3.35 * np.sin(ang), 0.5]))
        atoms.append(_make_atom(k, f"HR{k+1}", "HST", 1, "C", charge=-0.10 if k % 2 == 0 else 0.10, epsilon=0.22))
    coords.append(np.array([0.0, 0.0, -2.55]))
    atoms.append(
        _make_atom(
            6, "HB1", "HST", 1, "O", charge=-0.42,
            alpha_pol=0.75 if polarizable else 0.0, epsilon=0.28,
        )
    )
    coords.append(np.array([1.25, 0.0, -2.10]))
    atoms.append(_make_atom(7, "HB2", "HST", 1, "N", charge=0.42, alpha_pol=0.50 if polarizable else 0.0, epsilon=0.24))

    # Elastic scaffold within host so pocket maintains integrity during pulling
    for i in range(8):
        for j in range(i + 1, 8):
            d0 = float(np.linalg.norm(coords[i] - coords[j]))
            bonds.append(BondTerm(i, j, k=35.0, r0=d0, is_elastic=True))

    # 3-atom guest ligand initially bound inside cavity (z ~ 0.1 to 1.6)
    g_coords = [
        np.array([0.0, 0.0, 0.20]),
        np.array([0.0, 0.0, 1.65]),
        np.array([1.20, 0.0, 2.40]),
    ]
    g_specs = [
        ("G1", "GST", 2, "N", 0.38, 0.50 if polarizable else 0.0),
        ("G2", "GST", 2, "C", -0.18, 0.0),
        ("G3", "GST", 2, "O", -0.20, 0.70 if polarizable else 0.0),
    ]
    for offset, (nm, res, rid, el, q, alpha) in enumerate(g_specs):
        atoms.append(_make_atom(8 + offset, nm, res, rid, el, charge=q, alpha_pol=alpha, epsilon=0.24))
        coords.append(g_coords[offset])

    bonds.append(BondTerm(8, 9, k=320.0, r0=float(np.linalg.norm(g_coords[1] - g_coords[0]))))
    bonds.append(BondTerm(9, 10, k=320.0, r0=float(np.linalg.norm(g_coords[2] - g_coords[1]))))
    v1 = g_coords[0] - g_coords[1]
    v2 = g_coords[2] - g_coords[1]
    ang0 = float(np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0)))
    angles.append(AngleTerm(8, 9, 10, k=65.0, theta0=ang0))

    ff_cfg = ForceFieldConfig(use_gbsa=False, use_thole=True, lj_on=9.0, lj_cut=14.0)
    return MolecularSystem(
        name="host_guest_complex",
        atoms=atoms,
        positions=np.vstack(coords),
        bonds=bonds,
        angles=angles,
        ff_config=ff_cfg,
        metadata={"host_indices": list(range(8)), "guest_indices": [8, 9, 10]},
    )


def map_all_atom_to_cg(
    system: MolecularSystem,
    cutoff_a: float = 8.5,
    k_elastic: float = 12.0,
) -> MolecularSystem:
    """Map an all-atom protein/peptide system to 1-bead-per-residue Martini/ElNeDyn CG model.

    Computes mass-weighted residue centers of mass and connects non-sequential
    beads within `cutoff_a` by harmonic ElNeDyn elastic network springs.
    """
    res_groups: Dict[int, List[AtomSpec]] = {}
    for atom in system.atoms:
        res_groups.setdefault(atom.residue_id, []).append(atom)

    sorted_rids = sorted(res_groups.keys())
    cg_atoms: List[AtomSpec] = []
    cg_coords: List[np.ndarray] = []

    for idx, rid in enumerate(sorted_rids):
        members = res_groups[rid]
        masses = np.array([m.mass for m in members], dtype=np.float64)
        pts = np.array([system.positions[m.index] for m in members], dtype=np.float64)
        com = np.sum(pts * masses[:, None], axis=0) / np.sum(masses)
        tot_mass = float(np.sum(masses))
        tot_charge = float(sum(m.charge for m in members))
        cg_atoms.append(
            AtomSpec(
                index=idx,
                name="BB",
                residue_name=members[0].residue_name,
                residue_id=rid,
                element="CG",
                mass=tot_mass,
                charge=tot_charge,
                sigma=4.7,
                epsilon=0.85,
                gb_radius=2.35,
                gb_scale=0.80,
            )
        )
        cg_coords.append(com)

    pos = np.vstack(cg_coords)
    bonds: List[BondTerm] = []
    n_cg = len(cg_atoms)
    for i in range(n_cg):
        for j in range(i + 1, n_cg):
            dist = float(np.linalg.norm(pos[i] - pos[j]))
            if j == i + 1:
                bonds.append(BondTerm(i, j, k=150.0, r0=dist, is_elastic=False))
            elif dist <= cutoff_a:
                bonds.append(BondTerm(i, j, k=k_elastic, r0=dist, is_elastic=True))

    ff_cfg = ForceFieldConfig(use_gbsa=False, lj_on=9.0, lj_cut=12.0, use_thole=False)
    return MolecularSystem(
        name=f"{system.name}_cg",
        atoms=cg_atoms,
        positions=pos,
        bonds=bonds,
        ff_config=ff_cfg,
        metadata={"coarse_grained": True, "cutoff_a": cutoff_a},
    )


def build_coarse_grained_elastic_protein(
    n_beads: int = 12,
    cutoff_a: float = 8.5,
) -> MolecularSystem:
    """Construct a coarse-grained beta-barrel / helical bundle elastic network model."""
    helix = build_alpha_helix_peptide(n_res=n_beads, polarizable=False, use_gbsa=False)
    return map_all_atom_to_cg(helix, cutoff_a=cutoff_a, k_elastic=15.0)


def parse_pdb_block(pdb_text: str, polarizable_heavy: bool = False) -> MolecularSystem:
    """Parse an ATOM/HETATM PDB text block into a MolecularSystem with distance-based bonds."""
    atoms: List[AtomSpec] = []
    coords: List[np.ndarray] = []

    for line in pdb_text.splitlines():
        if not (line.startswith("ATOM  ") or line.startswith("HETATM")):
            continue
        atom_name = line[12:16].strip()
        res_name = line[17:20].strip() or "MOL"
        try:
            res_id = int(line[22:26].strip() or "1")
        except ValueError:
            res_id = 1
        x = float(line[30:38].strip())
        y = float(line[38:46].strip())
        z = float(line[46:54].strip())
        elem = line[76:78].strip() if len(line) >= 78 else ""
        if not elem:
            elem = "".join(ch for ch in atom_name if ch.isalpha())[:1].upper() or "C"

        alpha = 0.70 if (polarizable_heavy and elem in {"N", "O", "S"}) else 0.0
        atoms.append(_make_atom(len(atoms), atom_name, res_name, res_id, elem, charge=0.0, alpha_pol=alpha))
        coords.append(np.array([x, y, z], dtype=np.float64))

    if not atoms:
        raise ValueError("No ATOM/HETATM records found in PDB input.")

    pos = np.vstack(coords)
    bonds: List[BondTerm] = []
    n = len(atoms)
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(pos[i] - pos[j]))
            is_h = atoms[i].element == "H" or atoms[j].element == "H"
            thresh = 1.25 if is_h else 1.85
            if 0.4 < d <= thresh:
                bonds.append(BondTerm(i, j, k=320.0, r0=d))

    return MolecularSystem(
        name="pdb_import",
        atoms=atoms,
        positions=pos,
        bonds=bonds,
        ff_config=ForceFieldConfig(use_gbsa=True, use_thole=True),
    )


def export_pdb_block(system: MolecularSystem, positions: Optional[np.ndarray] = None) -> str:
    """Serialize atomic coordinates to standard fixed-width PDB format (7-bit ASCII)."""
    pos = system.positions if positions is None else positions
    lines: List[str] = [f"REMARK   SOLVATODYNE {system.name}"]
    for i, atom in enumerate(system.atoms):
        x, y, z = pos[i]
        lines.append(
            f"ATOM  {i + 1:5d} {atom.name:<4s} {atom.residue_name:>3s} A{atom.residue_id:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {atom.element:>2s}"
        )
    lines.append("END")
    return "\n".join(lines)


def build_system_by_name(name: str, polarizable: bool = True) -> MolecularSystem:
    """Factory resolving standard CLI/benchmark system identifiers."""
    key = name.strip().lower()
    if key in {"water-dimer", "water_dimer", "dimer"}:
        return build_polarizable_water_dimer_or_cluster(n_molecules=2, polarizable=polarizable)
    if key in {"water-cluster", "water_cluster", "cluster"}:
        return build_polarizable_water_dimer_or_cluster(n_molecules=6, polarizable=polarizable)
    if key in {"ala-dipeptide", "ala_dipeptide", "alanine-dipeptide"}:
        return build_alanine_dipeptide_system(polarizable=polarizable, use_gbsa=True)
    if key in {"helix8", "alpha-helix", "helix"}:
        return build_alpha_helix_peptide(n_res=8, polarizable=polarizable, use_gbsa=True)
    if key in {"host-guest", "host_guest", "complex"}:
        return build_host_guest_unbinding_system(polarizable=polarizable)
    if key in {"cg-protein", "cg_protein", "elnedyn"}:
        return build_coarse_grained_elastic_protein(n_beads=10)
    raise ValueError(
        f"Unknown system '{name}'. Choose from: water-dimer, water-cluster, "
        "ala-dipeptide, helix8, host-guest, cg-protein."
    )
