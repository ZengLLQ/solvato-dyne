"""Differentiable Collective Variables (CVs) with exact Cartesian gradients.

Every CV provides:
- `evaluate(positions) -> (value, grad)` where `grad` has shape `(N, 3)`.
- `is_periodic` and `period` metadata for proper Metadynamics wrapping.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Sequence, Tuple
import numpy as np

from solvato_dyne.forcefield.bonded import dihedral_angle_and_grad


class CollectiveVariable(ABC):
    """Abstract base class for differentiable structural collective variables."""

    name: str
    is_periodic: bool = False
    period: float = 0.0
    bounds: Tuple[float, float] = (0.0, 10.0)

    @abstractmethod
    def evaluate(self, positions: np.ndarray) -> Tuple[float, np.ndarray]:
        """Return scalar CV value and Cartesian gradient ds/dr with shape (N, 3)."""

    def wrap_diff(self, delta: float) -> float:
        """Compute minimal signed difference accounting for CV periodicity."""
        if not self.is_periodic or self.period <= 0.0:
            return float(delta)
        half = 0.5 * self.period
        return float((delta + half) % self.period - half)


class DistanceCV(CollectiveVariable):
    """Euclidean distance between geometric centers of two atom index groups."""

    def __init__(
        self,
        group_a: Sequence[int],
        group_b: Sequence[int],
        name: str = "distance",
        bounds: Tuple[float, float] = (0.5, 25.0),
    ) -> None:
        self.group_a = np.asarray(group_a, dtype=np.int64)
        self.group_b = np.asarray(group_b, dtype=np.int64)
        if len(self.group_a) == 0 or len(self.group_b) == 0:
            raise ValueError("DistanceCV atom groups must be non-empty.")
        self.name = name
        self.is_periodic = False
        self.bounds = bounds

    def evaluate(self, positions: np.ndarray) -> Tuple[float, np.ndarray]:
        com_a = np.mean(positions[self.group_a], axis=0)
        com_b = np.mean(positions[self.group_b], axis=0)
        diff = com_b - com_a
        dist = float(np.sqrt(max(1e-16, float(np.dot(diff, diff)))))
        grad = np.zeros_like(positions, dtype=np.float64)
        u = diff / dist
        grad[self.group_a] -= u / len(self.group_a)
        grad[self.group_b] += u / len(self.group_b)
        return dist, grad


class RadiusOfGyrationCV(CollectiveVariable):
    """Radius of gyration Rg over a subset of atoms."""

    def __init__(
        self,
        indices: Sequence[int],
        masses: Optional[Sequence[float]] = None,
        name: str = "rg",
        bounds: Tuple[float, float] = (1.0, 20.0),
    ) -> None:
        self.indices = np.asarray(indices, dtype=np.int64)
        if len(self.indices) < 2:
            raise ValueError("RadiusOfGyrationCV requires at least 2 atoms.")
        if masses is None:
            self.weights = np.ones(len(self.indices), dtype=np.float64) / len(self.indices)
        else:
            m = np.asarray(masses, dtype=np.float64)
            self.weights = m / np.sum(m)
        self.name = name
        self.is_periodic = False
        self.bounds = bounds

    def evaluate(self, positions: np.ndarray) -> Tuple[float, np.ndarray]:
        pts = positions[self.indices]
        com = np.sum(pts * self.weights[:, None], axis=0)
        rel = pts - com
        rg2 = float(np.sum(self.weights[:, None] * (rel * rel)))
        rg = float(np.sqrt(max(1e-16, rg2)))
        grad = np.zeros_like(positions, dtype=np.float64)
        # d(Rg)/dr_i = w_i * (r_i - r_com) / Rg (sum w_i (r_i - r_com) = 0 cancels COM term)
        grad[self.indices] = (self.weights[:, None] / rg) * rel
        return rg, grad


class CoordinationNumberCV(CollectiveVariable):
    """Smooth rational switching coordination number between two atom groups.

    C = sum_{i in A, j in B} (1 - (r_ij / r0)^n) / (1 - (r_ij / r0)^m)
    """

    def __init__(
        self,
        group_a: Sequence[int],
        group_b: Sequence[int],
        r0: float = 3.5,
        n_exp: int = 6,
        m_exp: int = 12,
        name: str = "coord",
        bounds: Tuple[float, float] = (0.0, 20.0),
    ) -> None:
        self.group_a = np.asarray(group_a, dtype=np.int64)
        self.group_b = np.asarray(group_b, dtype=np.int64)
        self.r0 = float(r0)
        self.n_exp = int(n_exp)
        self.m_exp = int(m_exp)
        self.name = name
        self.is_periodic = False
        self.bounds = bounds

    def evaluate(self, positions: np.ndarray) -> Tuple[float, np.ndarray]:
        coord = 0.0
        grad = np.zeros_like(positions, dtype=np.float64)
        n_e = self.n_exp
        m_e = self.m_exp
        inv_r0 = 1.0 / self.r0

        for i in self.group_a:
            for j in self.group_b:
                if i == j:
                    continue
                dr = positions[j] - positions[i]
                r = float(np.sqrt(max(1e-16, float(np.dot(dr, dr)))))
                x = r * inv_r0
                if abs(x - 1.0) < 1e-6:
                    # Smooth removable limit (n/m) and derivative at x=1
                    s_val = float(n_e) / float(m_e)
                    ds_dx = 0.5 * float(n_e) * (float(n_e - m_e)) / float(m_e)
                else:
                    xn = x ** n_e
                    xm = x ** m_e
                    num = 1.0 - xn
                    den = 1.0 - xm
                    s_val = num / den
                    ds_dx = (
                        -n_e * (x ** (n_e - 1)) * den + num * m_e * (x ** (m_e - 1))
                    ) / (den * den)

                coord += s_val
                ds_dr = ds_dx * inv_r0
                g_vec = (ds_dr / r) * dr
                grad[i] -= g_vec
                grad[j] += g_vec

        return float(coord), grad


class DihedralCV(CollectiveVariable):
    """Periodic torsion angle CV in [-pi, pi] radians."""

    def __init__(
        self,
        i: int,
        j: int,
        k_idx: int,
        l_idx: int,
        name: str = "dihedral",
    ) -> None:
        self.i = int(i)
        self.j = int(j)
        self.k_idx = int(k_idx)
        self.l_idx = int(l_idx)
        self.name = name
        self.is_periodic = True
        self.period = 2.0 * np.pi
        self.bounds = (-np.pi, np.pi)

    def evaluate(self, positions: np.ndarray) -> Tuple[float, np.ndarray]:
        phi, g0, g1, g2, g3 = dihedral_angle_and_grad(
            positions[self.i],
            positions[self.j],
            positions[self.k_idx],
            positions[self.l_idx],
        )
        grad = np.zeros_like(positions, dtype=np.float64)
        grad[self.i] = g0
        grad[self.j] = g1
        grad[self.k_idx] = g2
        grad[self.l_idx] = g3
        return float(phi), grad


class RMSDCV(CollectiveVariable):
    """Differentiable optimal-superposition Kabsch RMSD CV against a reference structure."""

    def __init__(
        self,
        indices: Sequence[int],
        reference: np.ndarray,
        name: str = "rmsd",
        bounds: Tuple[float, float] = (0.0, 10.0),
    ) -> None:
        self.indices = np.asarray(indices, dtype=np.int64)
        ref = np.asarray(reference, dtype=np.float64)
        if ref.shape != (len(self.indices), 3):
            raise ValueError("Reference coordinates must match len(indices) x 3.")
        self.ref_centered = ref - np.mean(ref, axis=0)
        self.name = name
        self.is_periodic = False
        self.bounds = bounds

    def evaluate(self, positions: np.ndarray) -> Tuple[float, np.ndarray]:
        pts = positions[self.indices]
        m = len(self.indices)
        x_c = pts - np.mean(pts, axis=0)
        y_c = self.ref_centered

        # Kabsch optimal rotation R minimizing || x_c R^T - y_c ||_F^2
        cov = x_c.T @ y_c
        u, _, vt = np.linalg.svd(cov)
        d = np.sign(np.linalg.det(vt.T @ u.T))
        s_mat = np.diag([1.0, 1.0, d if d != 0.0 else 1.0])
        rot = vt.T @ s_mat @ u.T  # maps x_c -> y_c frame: x_aligned = x_c @ rot.T

        x_aligned = x_c @ rot.T
        diff = x_aligned - y_c
        msd = float(np.sum(diff * diff) / m)
        rmsd = float(np.sqrt(max(1e-16, msd)))

        # By envelope theorem, d(RMSD)/d(x_c) removes rotation dependence:
        # grad_i = (x_aligned_i - y_c_i) @ rot / (m * rmsd)
        g_local = (diff @ rot) / (m * rmsd)
        # Centering correction already vanishes since sum(diff) = 0, but keep explicit:
        g_local -= np.mean(g_local, axis=0)

        grad = np.zeros_like(positions, dtype=np.float64)
        grad[self.indices] = g_local
        return rmsd, grad
