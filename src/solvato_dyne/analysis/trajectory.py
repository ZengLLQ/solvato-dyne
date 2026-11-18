"""Trajectory geometric, conformational, radial-distribution, and PCA diagnostics."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple
import numpy as np

from solvato_dyne.forcefield.bonded import dihedral_angle_and_grad
from solvato_dyne.types import DEBYE_PER_E_A, KB_KCAL_MOL_K, Trajectory


def kabsch_align(
    coords: np.ndarray,
    reference: np.ndarray,
    indices: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, float]:
    """Optimally superimpose `coords` onto `reference` via Kabsch SVD and return RMSD."""
    idx = np.arange(coords.shape[0]) if indices is None else np.asarray(indices, dtype=np.int64)
    p = coords[idx]
    q = reference[idx]
    p_com = np.mean(p, axis=0)
    q_com = np.mean(q, axis=0)

    p_c = p - p_com
    q_c = q - q_com
    cov = p_c.T @ q_c
    u, _, vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    s = np.diag([1.0, 1.0, d if d != 0.0 else 1.0])
    rot = vt.T @ s @ u.T

    aligned_all = (coords - p_com) @ rot.T + q_com
    diff = aligned_all[idx] - q
    rmsd = float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))
    return aligned_all, rmsd


def compute_rmsd_and_rmsf(
    trajectory: Trajectory,
    reference: Optional[np.ndarray] = None,
    indices: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-frame Kabsch-aligned RMSD and per-atom RMSF across trajectory."""
    pos_arr = trajectory.positions_array()  # (T, N, 3)
    ref = pos_arr[0] if reference is None else reference
    t_len, n_atoms, _ = pos_arr.shape

    rmsds = np.zeros(t_len, dtype=np.float64)
    aligned = np.zeros_like(pos_arr)
    for t in range(t_len):
        aligned[t], rmsds[t] = kabsch_align(pos_arr[t], ref, indices=indices)

    mean_pos = np.mean(aligned, axis=0)
    disp = aligned - mean_pos[None, :, :]
    rmsf = np.sqrt(np.mean(np.sum(disp * disp, axis=2), axis=0))
    return rmsds, rmsf


def compute_radial_distribution(
    trajectory: Trajectory,
    group_a: Sequence[int],
    group_b: Sequence[int],
    r_max: float = 10.0,
    n_bins: int = 40,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute normalized pairwise radial distribution function g(r) over trajectory."""
    pos_arr = trajectory.positions_array()
    idx_a = np.asarray(group_a, dtype=np.int64)
    idx_b = np.asarray(group_b, dtype=np.int64)

    edges = np.linspace(0.0, float(r_max), int(n_bins) + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    hist = np.zeros(n_bins, dtype=np.float64)

    pair_count = 0
    for frame_pos in pos_arr:
        dists = []
        for i in idx_a:
            for j in idx_b:
                if i == j:
                    continue
                dists.append(float(np.linalg.norm(frame_pos[j] - frame_pos[i])))
        if dists:
            h, _ = np.histogram(dists, bins=edges)
            hist += h
            pair_count += len(dists)

    if pair_count == 0:
        return centers, hist

    shell_vol = (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    ref_vol = (4.0 / 3.0) * np.pi * (r_max ** 3)
    density = pair_count / ref_vol
    g_r = hist / (density * shell_vol)
    return centers, g_r


def compute_pca_essential_dynamics(
    trajectory: Trajectory,
    n_components: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform Cartesian coordinate PCA on Kabsch-aligned trajectory.

    Returns:
        eigenvalues: Descending covariance eigenvalues (A^2).
        explained_ratio: Fraction of total positional variance per mode.
        projections: (T, n_components) trajectory projections onto principal modes.
    """
    pos_arr = trajectory.positions_array()
    t_len, n_atoms, _ = pos_arr.shape
    ref = pos_arr[0]
    aligned = np.array([kabsch_align(pos_arr[t], ref)[0] for t in range(t_len)])
    flat = aligned.reshape(t_len, 3 * n_atoms)
    mean_vec = np.mean(flat, axis=0)
    centered = flat - mean_vec[None, :]

    cov = (centered.T @ centered) / max(1, t_len - 1)
    evals, evecs = np.linalg.eigh(cov)
    idx = np.argsort(evals)[::-1]
    evals = np.maximum(0.0, evals[idx])
    evecs = evecs[:, idx]

    tot = float(np.sum(evals))
    ratio = evals / tot if tot > 1e-14 else np.zeros_like(evals)
    k = min(n_components, len(evals))
    projections = centered @ evecs[:, :k]
    return evals[:k], ratio[:k], projections


def compute_ramachandran_angles(
    trajectory: Trajectory,
    phi_indices: Tuple[int, int, int, int],
    psi_indices: Tuple[int, int, int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract (phi, psi) backbone dihedral time series in degrees."""
    pos_arr = trajectory.positions_array()
    phis = np.zeros(len(pos_arr), dtype=np.float64)
    psis = np.zeros(len(pos_arr), dtype=np.float64)
    for t, pos in enumerate(pos_arr):
        p_val, *_ = dihedral_angle_and_grad(
            pos[phi_indices[0]], pos[phi_indices[1]], pos[phi_indices[2]], pos[phi_indices[3]]
        )
        s_val, *_ = dihedral_angle_and_grad(
            pos[psi_indices[0]], pos[psi_indices[1]], pos[psi_indices[2]], pos[psi_indices[3]]
        )
        phis[t] = np.rad2deg(p_val)
        psis[t] = np.rad2deg(s_val)
    return phis, psis


def compute_dipole_susceptibility(
    trajectory: Trajectory,
    volume_a3: float = 100.0,
    temperature_k: float = 300.0,
) -> Dict[str, float]:
    """Compute dipole moment statistics and Kirkwood-Frohlich dielectric susceptibility."""
    dipoles = trajectory.dipoles()
    mean_d = float(np.mean(dipoles))
    var_d2 = float(np.var(dipoles))
    # Convert Debye^2 -> e^2 * A^2 for dimensionless susceptibility proxy
    var_ea2 = var_d2 / (DEBYE_PER_E_A ** 2)
    chi_e = (4.0 * np.pi / 3.0) * (332.0637 * var_ea2) / (volume_a3 * KB_KCAL_MOL_K * temperature_k)
    return {
        "mean_dipole_debye": mean_d,
        "std_dipole_debye": float(np.std(dipoles)),
        "dielectric_susceptibility_proxy": float(chi_e),
    }
