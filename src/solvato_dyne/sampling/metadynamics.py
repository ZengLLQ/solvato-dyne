"""Well-Tempered Metadynamics (WT-MetaD) with Tiwary-Parrinello c(t) reweighting."""

from __future__ import annotations

from typing import List, Sequence, Tuple
import numpy as np

from solvato_dyne.dynamics.runner import MDSimulator
from solvato_dyne.sampling.colvars import CollectiveVariable
from solvato_dyne.types import KB_KCAL_MOL_K, MetaDResult, MolecularSystem


class WellTemperedMetadynamics:
    """1D or 2D Well-Tempered Metadynamics driver with exact analytical bias forces.

    Implements Barducci-Bussi-Parrinello height scaling:
        w(t) = w0 * exp(-V(s(t), t) / (k_B * DeltaT))
    and Tiwary-Parrinello time-dependent offset c(t) for unbiased Boltzmann reweighting.
    """

    def __init__(
        self,
        cvs: Sequence[CollectiveVariable],
        sigmas: Sequence[float],
        initial_height: float = 0.6,
        bias_factor: float = 8.0,
        temperature_k: float = 300.0,
        deposit_stride: int = 20,
        grid_bins: int = 41,
    ) -> None:
        if not (1 <= len(cvs) <= 2):
            raise ValueError("WellTemperedMetadynamics supports 1D or 2D collective variables.")
        if len(sigmas) != len(cvs):
            raise ValueError("sigmas must match number of CVs.")
        if bias_factor <= 1.0:
            raise ValueError("bias_factor gamma must be > 1.0.")

        self.cvs = list(cvs)
        self.sigmas = np.asarray(sigmas, dtype=np.float64)
        self.initial_height = float(initial_height)
        self.bias_factor = float(bias_factor)
        self.temperature_k = float(temperature_k)
        self.deposit_stride = int(deposit_stride)
        self.grid_bins = int(grid_bins)

        self.kbt = KB_KCAL_MOL_K * self.temperature_k
        self.delta_t = self.temperature_k * (self.bias_factor - 1.0)
        self.kb_delta_t = KB_KCAL_MOL_K * self.delta_t

        self.centers: List[np.ndarray] = []
        self.heights: List[float] = []

    def bias_and_grad(self, s_vec: np.ndarray) -> Tuple[float, np.ndarray]:
        """Evaluate accumulated Gaussian bias V(s) and gradient dV/ds."""
        dim = len(self.cvs)
        if not self.centers:
            return 0.0, np.zeros(dim, dtype=np.float64)

        centers_arr = np.asarray(self.centers, dtype=np.float64)  # (K, D)
        heights_arr = np.asarray(self.heights, dtype=np.float64)  # (K,)

        diffs = np.zeros_like(centers_arr)
        for d, cv in enumerate(self.cvs):
            raw = s_vec[d] - centers_arr[:, d]
            if cv.is_periodic and cv.period > 0.0:
                half = 0.5 * cv.period
                raw = (raw + half) % cv.period - half
            diffs[:, d] = raw

        scaled = diffs / self.sigmas[None, :]
        exp_terms = np.exp(-0.5 * np.sum(scaled * scaled, axis=1))
        weighted = heights_arr * exp_terms
        v_tot = float(np.sum(weighted))
        dv_ds = -np.sum(weighted[:, None] * (diffs / (self.sigmas[None, :] ** 2)), axis=0)
        return v_tot, dv_ds

    def _compute_grid_bias(self, grid_axes: List[np.ndarray]) -> np.ndarray:
        """Evaluate bias potential V(s) on the 1D or 2D reference grid."""
        if len(grid_axes) == 1:
            ax = grid_axes[0]
            v_grid = np.zeros_like(ax, dtype=np.float64)
            for idx, x in enumerate(ax):
                v_grid[idx], _ = self.bias_and_grad(np.array([x], dtype=np.float64))
            return v_grid

        ax0, ax1 = grid_axes
        v_grid = np.zeros((len(ax0), len(ax1)), dtype=np.float64)
        for i, x0 in enumerate(ax0):
            for j, x1 in enumerate(ax1):
                v_grid[i, j], _ = self.bias_and_grad(np.array([x0, x1], dtype=np.float64))
        return v_grid

    def _compute_c_of_t(self, grid_axes: List[np.ndarray]) -> float:
        """Compute Tiwary-Parrinello instantaneous reweighting offset c(t).

        c(t) = k_B T * ln( sum_grid exp(beta * gamma * V(s) / (gamma - 1)) /
                           sum_grid exp(beta * V(s) / (gamma - 1)) )
        """
        v_grid = self._compute_grid_bias(grid_axes)
        beta = 1.0 / self.kbt
        gam_ratio = self.bias_factor / (self.bias_factor - 1.0)
        inv_gam_m1 = 1.0 / (self.bias_factor - 1.0)

        a = beta * gam_ratio * v_grid
        b = beta * inv_gam_m1 * v_grid
        # Log-sum-exp stabilization
        m_a = float(np.max(a))
        m_b = float(np.max(b))
        lse_a = m_a + float(np.log(np.sum(np.exp(a - m_a))))
        lse_b = m_b + float(np.log(np.sum(np.exp(b - m_b))))
        return float(self.kbt * (lse_a - lse_b))

    def run(
        self,
        system: MolecularSystem,
        n_steps: int = 400,
        integrator: str = "baoab",
        dt_ps: float = 0.001,
        save_every: int = 10,
        seed: int = 42,
    ) -> MetaDResult:
        """Run Well-Tempered Metadynamics and return direct & reweighted FES."""
        sim = MDSimulator(
            system,
            integrator=integrator,
            temperature_k=self.temperature_k,
            dt_ps=dt_ps,
            seed=seed,
        )
        from solvato_dyne.dynamics.constraints import minimize_energy
        from solvato_dyne.dynamics.runner import initialize_velocities

        min_sys, _, _ = minimize_energy(sim.system, maxiter=60, gtol=1e-2)
        sim.system.positions = min_sys.positions
        sim.system.drude_positions = min_sys.drude_positions
        initialize_velocities(sim.system, temperature_k=self.temperature_k, seed=seed)

        self.centers = []
        self.heights = []

        grid_axes = [
            np.linspace(cv.bounds[0], cv.bounds[1], self.grid_bins, dtype=np.float64)
            for cv in self.cvs
        ]

        cv_history: List[np.ndarray] = []
        c_t_history: List[float] = []
        v_sample_history: List[float] = []
        frames = []

        def _external_bias(pos: np.ndarray) -> Tuple[float, np.ndarray]:
            s_vals = []
            s_grads = []
            for cv in self.cvs:
                val, grad = cv.evaluate(pos)
                s_vals.append(val)
                s_grads.append(grad)
            s_vec = np.asarray(s_vals, dtype=np.float64)
            v_tot, dv_ds = self.bias_and_grad(s_vec)
            f_bias = np.zeros_like(pos, dtype=np.float64)
            for d, grad in enumerate(s_grads):
                f_bias -= dv_ds[d] * grad
            return v_tot, f_bias

        for step_idx in range(1, n_steps + 1):
            if hasattr(sim.stepper, "step"):
                res = sim.stepper.step(sim.system, sim.engine, external_bias=_external_bias)
                eb = res[0] if isinstance(res, tuple) else res
            else:
                raise RuntimeError("Invalid integrator stepper.")

            s_now = np.array([cv.evaluate(sim.system.positions)[0] for cv in self.cvs], dtype=np.float64)
            v_now, _ = self.bias_and_grad(s_now)

            if step_idx % self.deposit_stride == 0:
                h_wt = self.initial_height * float(np.exp(-v_now / self.kb_delta_t))
                self.centers.append(s_now.copy())
                self.heights.append(h_wt)

            if step_idx % save_every == 0 or step_idx == n_steps:
                c_val = self._compute_c_of_t(grid_axes) if self.centers else 0.0
                cv_history.append(s_now)
                v_sample_history.append(v_now)
                c_t_history.append(c_val)
                frames.append(
                    sim._make_frame(
                        step_idx,
                        step_idx * dt_ps,
                        eb,
                        {cv.name: cv for cv in self.cvs},
                    )
                )

        # Direct WT-MetaD FES: F(s) = -(gamma / (gamma - 1)) * V(s), shifted to min = 0
        v_final = self._compute_grid_bias(grid_axes)
        fes_direct = -(self.bias_factor / (self.bias_factor - 1.0)) * v_final
        fes_direct -= np.min(fes_direct)

        # Tiwary-Parrinello Boltzmann reweighted histogram FES on grid
        cv_hist_arr = np.asarray(cv_history, dtype=np.float64)
        v_arr = np.asarray(v_sample_history, dtype=np.float64)
        c_arr = np.asarray(c_t_history, dtype=np.float64)
        log_weights = (v_arr - c_arr) / self.kbt
        log_weights -= np.max(log_weights)
        weights = np.exp(log_weights)

        fes_reweighted = self._reweight_fes_kde(cv_hist_arr, weights, grid_axes)

        from solvato_dyne.types import Trajectory

        traj = Trajectory(
            system_name=system.name,
            integrator=f"wt_metad_{integrator}",
            dt_ps=dt_ps,
            frames=frames,
        )
        return MetaDResult(
            cv_names=[cv.name for cv in self.cvs],
            cv_history=cv_hist_arr,
            hill_centers=np.asarray(self.centers, dtype=np.float64)
            if self.centers
            else np.zeros((0, len(self.cvs))),
            hill_heights=np.asarray(self.heights, dtype=np.float64),
            hill_sigmas=self.sigmas.copy(),
            bias_factor=self.bias_factor,
            grid_axes=grid_axes,
            fes_direct=fes_direct,
            fes_reweighted=fes_reweighted,
            c_of_t=c_arr,
            trajectory=traj,
        )

    def _reweight_fes_kde(
        self,
        samples: np.ndarray,
        weights: np.ndarray,
        grid_axes: List[np.ndarray],
    ) -> np.ndarray:
        """Compute smooth weighted probability density -> -k_B T ln P(s) on grid."""
        bw = self.sigmas * 1.1
        if len(grid_axes) == 1:
            ax = grid_axes[0]
            prob = np.zeros_like(ax, dtype=np.float64)
            for idx, x in enumerate(ax):
                diff = np.array([self.cvs[0].wrap_diff(x - s[0]) for s in samples])
                kern = np.exp(-0.5 * (diff / bw[0]) ** 2)
                prob[idx] = float(np.sum(weights * kern))
            prob = np.maximum(prob, 1e-12 * float(np.max(prob) + 1e-12))
            fes = -self.kbt * np.log(prob)
            return fes - np.min(fes)

        ax0, ax1 = grid_axes
        prob = np.zeros((len(ax0), len(ax1)), dtype=np.float64)
        for i, x0 in enumerate(ax0):
            d0 = np.array([self.cvs[0].wrap_diff(x0 - s[0]) for s in samples])
            k0 = np.exp(-0.5 * (d0 / bw[0]) ** 2)
            for j, x1 in enumerate(ax1):
                d1 = np.array([self.cvs[1].wrap_diff(x1 - s[1]) for s in samples])
                k1 = np.exp(-0.5 * (d1 / bw[1]) ** 2)
                prob[i, j] = float(np.sum(weights * k0 * k1))
        prob = np.maximum(prob, 1e-12 * float(np.max(prob) + 1e-12))
        fes = -self.kbt * np.log(prob)
        return fes - np.min(fes)
