"""Bonded force-field terms (bonds, angles, dihedrals, impropers, ElNeDyn) and PBC.

Every potential implements exact analytical Cartesian gradients and virial
contributions free of colinear/planar trigonometric singularities.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple
import numpy as np

from solvato_dyne.types import BoxSpec, MolecularSystem


def minimum_image_displacement(
    dr: np.ndarray,
    box: Optional[BoxSpec],
) -> np.ndarray:
    """Apply minimum-image convention to displacement vector(s) dr = r_j - r_i.

    Supports both orthorhombic and general triclinic 3x3 lattice matrices.
    """
    if box is None or not box.periodic:
        return dr
    if box.is_orthorhombic:
        lengths = np.diag(box.vectors)
        return dr - lengths * np.round(dr / lengths)
    inv_box = np.linalg.inv(box.vectors)
    frac = dr @ inv_box
    frac -= np.round(frac)
    return frac @ box.vectors


def wrap_positions_to_box(positions: np.ndarray, box: Optional[BoxSpec]) -> np.ndarray:
    """Wrap Cartesian coordinates into primary periodic unit cell [0, L)."""
    if box is None or not box.periodic:
        return positions
    if box.is_orthorhombic:
        lengths = np.diag(box.vectors)
        return np.mod(positions, lengths)
    inv_box = np.linalg.inv(box.vectors)
    frac = np.mod(positions @ inv_box, 1.0)
    return frac @ box.vectors


def _wrap_angle(delta: float) -> float:
    """Wrap angle difference into (-pi, pi]."""
    return float((delta + np.pi) % (2.0 * np.pi) - np.pi)


def dihedral_angle_and_grad(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    box: Optional[BoxSpec] = None,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute proper dihedral angle phi in [-pi, pi] and exact gradients d(phi)/dp_k.

    Uses the standard Blondel-Karplus / Bekker vector formulation:
      b1 = p1 - p0, b2 = p2 - p1, b3 = p3 - p2
      m = b1 x b2, n = b2 x b3
    which avoids 1/sin(phi) singularities at phi = 0 or pi.
    """
    b1 = minimum_image_displacement(p1 - p0, box)
    b2 = minimum_image_displacement(p2 - p1, box)
    b3 = minimum_image_displacement(p3 - p2, box)

    b2_norm = float(np.linalg.norm(b2))
    if b2_norm < 1e-12:
        zero = np.zeros(3, dtype=np.float64)
        return 0.0, zero, zero, zero, zero

    b2_hat = b2 / b2_norm
    m = np.cross(b1, b2)
    n = np.cross(b2, b3)
    m_sq = float(np.dot(m, m))
    n_sq = float(np.dot(n, n))

    if m_sq < 1e-16 or n_sq < 1e-16:
        zero = np.zeros(3, dtype=np.float64)
        return 0.0, zero, zero, zero, zero

    # IUPAC/Bekker convention: cos(phi) = (m . n) / (|m||n|), sin(phi) = (b2_hat . (m x n)) / (|m||n|)
    x = float(np.dot(m, n))
    y = float(np.dot(b2_hat, np.cross(m, n)))
    phi = float(np.arctan2(y, x))

    # Exact Bekker gradients of phi w.r.t. (p0, p1, p2, p3)
    dphi_dp0 = -(b2_norm / m_sq) * m
    dphi_dp3 = (b2_norm / n_sq) * n

    c12 = float(np.dot(b1, b2)) / (b2_norm * b2_norm)
    c32 = float(np.dot(b3, b2)) / (b2_norm * b2_norm)

    dphi_dp1 = -dphi_dp0 - c12 * dphi_dp0 + c32 * dphi_dp3
    dphi_dp2 = -dphi_dp3 + c12 * dphi_dp0 - c32 * dphi_dp3

    return phi, dphi_dp0, dphi_dp1, dphi_dp2, dphi_dp3


def compute_bonded_forces(
    system: MolecularSystem,
    positions: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Evaluate all bonded potentials, Cartesian forces, and virial tensor.

    Returns:
        energies: Dictionary with keys 'bond', 'angle', 'dihedral', 'improper', 'elastic_net'.
        forces: (N, 3) array of atomic forces in kcal/(mol*A).
        virial: (3, 3) virial contribution tensor sum(r_ij outer F_ij).
    """
    pos = system.positions if positions is None else positions
    n = system.n_atoms
    forces = np.zeros((n, 3), dtype=np.float64)
    virial = np.zeros((3, 3), dtype=np.float64)
    box = system.box

    e_bond = 0.0
    e_elastic = 0.0
    e_angle = 0.0
    e_dihedral = 0.0
    e_improper = 0.0

    # 1. Harmonic covalent and ElNeDyn elastic bonds
    for b in system.bonds:
        dr = minimum_image_displacement(pos[b.j] - pos[b.i], box)
        r = float(np.linalg.norm(dr))
        if r < 1e-12:
            continue
        diff = r - b.r0
        e_term = 0.5 * b.k * diff * diff
        if b.is_elastic:
            e_elastic += e_term
        else:
            e_bond += e_term
        # F_i = -dE/dr_i = k * (r - r0) * (r_j - r_i) / r
        f_mag = b.k * diff / r
        f_vec = f_mag * dr
        forces[b.i] += f_vec
        forces[b.j] -= f_vec
        virial += np.outer(dr, -f_vec)

    # 2. Harmonic valence angles
    for a in system.angles:
        r_ij = minimum_image_displacement(pos[a.i] - pos[a.j], box)
        r_kj = minimum_image_displacement(pos[a.k_idx] - pos[a.j], box)
        d_ij = float(np.linalg.norm(r_ij))
        d_kj = float(np.linalg.norm(r_kj))
        if d_ij < 1e-12 or d_kj < 1e-12:
            continue
        u_ij = r_ij / d_ij
        u_kj = r_kj / d_kj
        cos_theta = float(np.clip(np.dot(u_ij, u_kj), -0.999999999, 0.999999999))
        theta = float(np.arccos(cos_theta))
        sin_theta = float(np.sqrt(max(1e-16, 1.0 - cos_theta * cos_theta)))

        dtheta = theta - a.theta0
        e_angle += 0.5 * a.k * dtheta * dtheta

        # d(theta)/d(cos_theta) = -1 / sin(theta)
        pref = -a.k * dtheta / sin_theta
        dcos_di = (u_kj - cos_theta * u_ij) / d_ij
        dcos_dk = (u_ij - cos_theta * u_kj) / d_kj

        f_i = -pref * dcos_di
        f_k = -pref * dcos_dk
        f_j = -(f_i + f_k)

        forces[a.i] += f_i
        forces[a.k_idx] += f_k
        forces[a.j] += f_j
        virial += np.outer(r_ij, f_i) + np.outer(r_kj, f_k)

    # 3. Proper periodic dihedrals
    for d in system.dihedrals:
        phi, g0, g1, g2, g3 = dihedral_angle_and_grad(
            pos[d.i], pos[d.j], pos[d.k_idx], pos[d.l_idx], box
        )
        arg = d.periodicity * phi - d.phase
        e_dihedral += d.k * (1.0 + np.cos(arg))
        dE_dphi = -d.k * d.periodicity * np.sin(arg)
        forces[d.i] -= dE_dphi * g0
        forces[d.j] -= dE_dphi * g1
        forces[d.k_idx] -= dE_dphi * g2
        forces[d.l_idx] -= dE_dphi * g3

    # 4. Harmonic improper dihedrals
    for imp in system.impropers:
        omega, g0, g1, g2, g3 = dihedral_angle_and_grad(
            pos[imp.i], pos[imp.j], pos[imp.k_idx], pos[imp.l_idx], box
        )
        domega = _wrap_angle(omega - imp.omega0)
        e_improper += 0.5 * imp.k * domega * domega
        dE_domega = imp.k * domega
        forces[imp.i] -= dE_domega * g0
        forces[imp.j] -= dE_domega * g1
        forces[imp.k_idx] -= dE_domega * g2
        forces[imp.l_idx] -= dE_domega * g3

    energies = {
        "bond": float(e_bond),
        "elastic_net": float(e_elastic),
        "angle": float(e_angle),
        "dihedral": float(e_dihedral),
        "improper": float(e_improper),
    }
    return energies, forces, virial
