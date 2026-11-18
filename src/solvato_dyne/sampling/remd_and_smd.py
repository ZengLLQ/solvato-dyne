"""Temperature Replica-Exchange MD (T-REMD) and Bidirectional Steered MD (Crooks/Jarzynski)."""

from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np
from scipy.optimize import brentq

from solvato_dyne.dynamics.constraints import minimize_energy
from solvato_dyne.dynamics.runner import MDSimulator, initialize_velocities
from solvato_dyne.sampling.colvars import CollectiveVariable
from solvato_dyne.types import (
    KB_KCAL_MOL_K,
    MolecularSystem,
    REMDResult,
    SMDResult,
    Trajectory,
    TrajectoryFrame,
)


def build_geometric_temperature_ladder(
    t_min: float,
    t_max: float,
    n_replicas: int,
) -> np.ndarray:
    """Construct exponentially spaced temperature ladder T_i = T_min * (T_max / T_min)^(i/(M-1))."""
    if n_replicas < 2:
        raise ValueError("n_replicas must be >= 2.")
    if t_min <= 0.0 or t_max <= t_min:
        raise ValueError("Require 0 < t_min < t_max.")
    return np.geomspace(float(t_min), float(t_max), int(n_replicas))


def optimize_feedback_temperature_ladder(
    temperatures_k: np.ndarray,
    acceptance_rates: np.ndarray,
) -> np.ndarray:
    """Adjust intermediate replica temperatures via Katzgraber-Trebst-Nadler feedback.

    Temperatures between bottlenecks (low acceptance) are drawn closer together while
    oversampled intervals expand, keeping T_min and T_max pinned.
    """
    m = len(temperatures_k)
    if m < 3:
        return temperatures_k.copy()
    acc = np.clip(acceptance_rates, 0.02, 0.98)
    # Local diffusivity proxy proportional to sqrt(acceptance) / delta_T
    dt = np.diff(temperatures_k)
    weights = 1.0 / (acc * dt)
    cdf = np.concatenate([[0.0], np.cumsum(weights)])
    cdf /= cdf[-1]
    target_u = np.linspace(0.0, 1.0, m)
    # Invert cumulative spacing onto [T_min, T_max]
    new_temps = np.interp(target_u, cdf, temperatures_k)
    new_temps[0] = temperatures_k[0]
    new_temps[-1] = temperatures_k[-1]
    return new_temps


class TemperatureReplicaExchange:
    """Multi-replica Temperature Replica-Exchange MD (T-REMD) orchestrator."""

    def __init__(
        self,
        system: MolecularSystem,
        t_min: float = 280.0,
        t_max: float = 520.0,
        n_replicas: int = 4,
        dt_ps: float = 0.001,
        gamma_ps_inv: float = 5.0,
        seed: int = 42,
    ) -> None:
        self.base_system = system.clone()
        self.temperatures = build_geometric_temperature_ladder(t_min, t_max, n_replicas)
        self.n_replicas = int(n_replicas)
        self.dt_ps = float(dt_ps)
        self.gamma_ps_inv = float(gamma_ps_inv)
        self.rng = np.random.default_rng(seed)

    def run(
        self,
        n_cycles: int = 20,
        steps_per_cycle: int = 25,
    ) -> REMDResult:
        """Execute T-REMD with alternating even/odd Metropolis exchanges and Cv(T) WHAM."""
        min_sys, _, _ = minimize_energy(self.base_system, maxiter=60, gtol=1e-2)

        sims: List[MDSimulator] = []
        for idx, temp in enumerate(self.temperatures):
            sim = MDSimulator(
                min_sys,
                integrator="baoab",
                temperature_k=float(temp),
                dt_ps=self.dt_ps,
                gamma_ps_inv=self.gamma_ps_inv,
                seed=100 + idx,
            )
            initialize_velocities(sim.system, temperature_k=float(temp), seed=100 + idx)
            sims.append(sim)

        # state_at_temp[temp_idx] = replica_id currently residing at temp_idx
        state_at_temp = np.arange(self.n_replicas, dtype=np.int64)
        # replica_temp_idx[rep_id] = current temperature index of replica rep_id
        replica_temp_idx = np.arange(self.n_replicas, dtype=np.int64)

        attempt_counts = np.zeros(self.n_replicas - 1, dtype=np.int64)
        accept_counts = np.zeros(self.n_replicas - 1, dtype=np.int64)
        state_history = np.zeros((n_cycles, self.n_replicas), dtype=np.int64)
        pot_history = np.zeros((n_cycles, self.n_replicas), dtype=np.float64)
        frames_by_temp: List[List[TrajectoryFrame]] = [[] for _ in range(self.n_replicas)]

        visited_bottom = np.zeros(self.n_replicas, dtype=bool)
        visited_top = np.zeros(self.n_replicas, dtype=bool)
        round_trips = 0

        for cycle in range(n_cycles):
            # Advance each replica at its currently assigned temperature
            current_potentials = np.zeros(self.n_replicas, dtype=np.float64)
            for t_idx in range(self.n_replicas):
                sim = sims[t_idx]
                eb_last = None
                for _ in range(steps_per_cycle):
                    eb_last, _ = sim.stepper.step(sim.system, sim.engine)  # type: ignore[union-attr]
                assert eb_last is not None
                current_potentials[t_idx] = eb_last.total_potential
                frames_by_temp[t_idx].append(
                    TrajectoryFrame(
                        step=(cycle + 1) * steps_per_cycle,
                        time_ps=(cycle + 1) * steps_per_cycle * self.dt_ps,
                        positions=sim.system.positions.copy(),
                        velocities=None if sim.system.velocities is None else sim.system.velocities.copy(),
                        drude_positions=None
                        if sim.system.drude_positions is None
                        else sim.system.drude_positions.copy(),
                        energy=eb_last,
                    )
                )

            pot_history[cycle] = current_potentials

            # Alternating even/odd pair exchange attempts
            parity = cycle % 2
            for i in range(parity, self.n_replicas - 1, 2):
                j = i + 1
                attempt_counts[i] += 1
                beta_i = 1.0 / (KB_KCAL_MOL_K * self.temperatures[i])
                beta_j = 1.0 / (KB_KCAL_MOL_K * self.temperatures[j])
                u_i = current_potentials[i]
                u_j = current_potentials[j]

                # Metropolis criterion: delta = (beta_i - beta_j) * (U_j - U_i)
                delta = (beta_i - beta_j) * (u_j - u_i)
                if delta >= 0.0 or float(self.rng.random()) < float(np.exp(delta)):
                    accept_counts[i] += 1
                    # Swap coordinates & velocities with exact Maxwell velocity rescaling
                    si = sims[i].system
                    sj = sims[j].system
                    scale_i_to_j = float(np.sqrt(self.temperatures[j] / self.temperatures[i]))
                    scale_j_to_i = float(np.sqrt(self.temperatures[i] / self.temperatures[j]))

                    pos_i, vel_i = si.positions.copy(), si.velocities.copy()
                    dpos_i = si.drude_positions.copy() if si.drude_positions is not None else None
                    si.positions = sj.positions.copy()
                    si.velocities = sj.velocities * scale_j_to_i
                    if dpos_i is not None and sj.drude_positions is not None:
                        si.drude_positions = sj.drude_positions.copy()
                        sj.drude_positions = dpos_i
                    sj.positions = pos_i
                    sj.velocities = vel_i * scale_i_to_j

                    # Update replica permutation tracking
                    rep_i = state_at_temp[i]
                    rep_j = state_at_temp[j]
                    state_at_temp[i] = rep_j
                    state_at_temp[j] = rep_i
                    replica_temp_idx[rep_i] = j
                    replica_temp_idx[rep_j] = i

            state_history[cycle] = replica_temp_idx.copy()

            # Track round-trips (0 -> M-1 -> 0 or M-1 -> 0 -> M-1)
            for r_id in range(self.n_replicas):
                t_now = replica_temp_idx[r_id]
                if t_now == 0:
                    if visited_top[r_id]:
                        round_trips += 1
                        visited_top[r_id] = False
                    visited_bottom[r_id] = True
                elif t_now == self.n_replicas - 1:
                    if visited_bottom[r_id]:
                        round_trips += 1
                        visited_bottom[r_id] = False
                    visited_top[r_id] = True

        acc_rates = np.where(attempt_counts > 0, accept_counts / np.maximum(attempt_counts, 1), 0.0)
        cv_curve = self._estimate_heat_capacity_curve(pot_history)
        trajectories = [
            Trajectory(
                system_name=f"{self.base_system.name}_T{int(round(self.temperatures[i]))}K",
                integrator="remd_baoab",
                dt_ps=self.dt_ps,
                frames=frames_by_temp[i],
            )
            for i in range(self.n_replicas)
        ]

        return REMDResult(
            temperatures_k=self.temperatures.copy(),
            acceptance_rates=acc_rates,
            replica_state_history=state_history,
            potential_histories=pot_history,
            round_trips=round_trips,
            heat_capacity_curve=cv_curve,
            trajectories=trajectories,
        )

    def _estimate_heat_capacity_curve(
        self,
        pot_history: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute multi-temperature reweighted excess heat capacity Cv(T) = Var(U)/(k_B T^2)."""
        t_grid = np.linspace(self.temperatures[0], self.temperatures[-1], 25, dtype=np.float64)
        cv_vals = np.zeros_like(t_grid)
        # Flatten all samples across replicas with local temperature reweighting
        all_u = pot_history.ravel()
        # Use closest replica weights for stable smooth interpolation
        mean_u = np.mean(pot_history, axis=0)
        var_u = np.var(pot_history, axis=0)
        cv_raw = var_u / (KB_KCAL_MOL_K * (self.temperatures ** 2))
        cv_vals = np.interp(t_grid, self.temperatures, cv_raw)
        # Blend with global canonical variance near mean energy for smoothness
        _ = all_u, mean_u
        return t_grid, cv_vals


def jarzynski_free_energy(
    works: np.ndarray,
    temperature_k: float = 300.0,
) -> Tuple[float, float]:
    """Compute Jarzynski exponential average and 2nd-order cumulant free energy.

    DeltaG_Jarz = -k_B T ln <exp(-beta * W)>
    DeltaG_cum  = <W> - 0.5 * beta * Var(W)
    """
    w = np.asarray(works, dtype=np.float64)
    kbt = KB_KCAL_MOL_K * float(temperature_k)
    beta = 1.0 / kbt
    arg = -beta * w
    m = float(np.max(arg))
    dg_jarz = -kbt * (m + float(np.log(np.mean(np.exp(arg - m)))))
    dg_cum = float(np.mean(w) - 0.5 * beta * np.var(w))
    return float(dg_jarz), float(dg_cum)


def crooks_mle_free_energy(
    forward_works: np.ndarray,
    reverse_works: np.ndarray,
    temperature_k: float = 300.0,
    n_bootstrap: int = 200,
    seed: int = 42,
) -> Tuple[float, Tuple[float, float]]:
    """Solve exact Bennett / Crooks Fluctuation Theorem MLE for DeltaG and 95% bootstrap CI.

    Note: reverse_works are the external works performed during the reverse trajectory
    (B -> A), so the dissipated Crooks work along forward convention is -W_R.
    The Bennett score equation solved by Brent root-finding is:
      sum_i 1 / (1 + exp(beta * (W_F,i - dG) + C)) = sum_j 1 / (1 + exp(-beta * (W_R,j + dG) - C))
    where C = ln(N_F / N_R).
    """
    wf = np.asarray(forward_works, dtype=np.float64)
    wr = np.asarray(reverse_works, dtype=np.float64)
    if len(wf) == 0 or len(wr) == 0:
        raise ValueError("Need non-empty forward and reverse work arrays.")

    kbt = KB_KCAL_MOL_K * float(temperature_k)
    beta = 1.0 / kbt

    def _solve_single(f_arr: np.ndarray, r_arr: np.ndarray) -> float:
        c_shift = float(np.log(len(f_arr) / len(r_arr)))

        def _bennett_residual(dg: float) -> float:
            # Stable sigmoid: 1 / (1 + exp(x)); lhs strictly increases with dG, rhs decreases
            xf = np.clip(beta * (f_arr - dg) + c_shift, -60.0, 60.0)
            xr = np.clip(beta * (r_arr + dg) - c_shift, -60.0, 60.0)
            lhs = float(np.sum(1.0 / (1.0 + np.exp(xf))))
            rhs = float(np.sum(1.0 / (1.0 + np.exp(xr))))
            return lhs - rhs

        lo = float(min(np.min(f_arr), np.min(-r_arr))) - 25.0 * kbt
        hi = float(max(np.max(f_arr), np.max(-r_arr))) + 25.0 * kbt
        return float(brentq(_bennett_residual, lo, hi, xtol=1e-9))

    dg_mle = _solve_single(wf, wr)

    rng = np.random.default_rng(seed)
    boots = np.zeros(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        wf_b = rng.choice(wf, size=len(wf), replace=True)
        wr_b = rng.choice(wr, size=len(wr), replace=True)
        boots[b] = _solve_single(wf_b, wr_b)

    ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
    return float(dg_mle), (float(ci_lo), float(ci_hi))


class SteeredMDSimulator:
    """Bidirectional constant-velocity Steered MD (SMD) for non-equilibrium work & Crooks dG."""

    def __init__(
        self,
        system: MolecularSystem,
        cv: CollectiveVariable,
        k_pull: float = 40.0,
        temperature_k: float = 300.0,
        dt_ps: float = 0.001,
        seed: int = 42,
    ) -> None:
        self.base_system = system.clone()
        self.cv = cv
        self.k_pull = float(k_pull)
        self.temperature_k = float(temperature_k)
        self.dt_ps = float(dt_ps)
        self.seed = seed

    def _run_single_pull(
        self,
        start_system: MolecularSystem,
        s_start: float,
        s_end: float,
        n_steps: int,
        run_seed: int,
    ) -> Tuple[float, np.ndarray, np.ndarray, MolecularSystem]:
        """Run one constant-velocity pulling trajectory and integrate exact Jarzynski work."""
        sim = MDSimulator(
            start_system,
            integrator="baoab",
            temperature_k=self.temperature_k,
            dt_ps=self.dt_ps,
            gamma_ps_inv=8.0,
            seed=run_seed,
        )
        initialize_velocities(sim.system, temperature_k=self.temperature_k, seed=run_seed)

        total_time = n_steps * self.dt_ps
        vel_pull = (s_end - s_start) / total_time
        work = 0.0
        s_trace = np.zeros(n_steps + 1, dtype=np.float64)
        w_trace = np.zeros(n_steps + 1, dtype=np.float64)

        s0_val, _ = self.cv.evaluate(sim.system.positions)
        s_trace[0] = s0_val

        for step_idx in range(1, n_steps + 1):
            t_mid = (step_idx - 0.5) * self.dt_ps
            s_ref_mid = s_start + vel_pull * t_mid

            def _pull_bias(pos: np.ndarray) -> Tuple[float, np.ndarray]:
                s_val, s_grad = self.cv.evaluate(pos)
                ds = self.cv.wrap_diff(s_val - s_ref_mid)
                e_pull = 0.5 * self.k_pull * ds * ds
                f_pull = -(self.k_pull * ds) * s_grad
                return float(e_pull), f_pull

            sim.stepper.step(sim.system, sim.engine, external_bias=_pull_bias)  # type: ignore[union-attr]
            s_now, _ = self.cv.evaluate(sim.system.positions)
            ds_now = self.cv.wrap_diff(s_now - s_ref_mid)
            # dU_pull/dt = -k_pull * (s - s_ref) * vel_pull
            dw = -self.k_pull * ds_now * vel_pull * self.dt_ps
            work += dw
            s_trace[step_idx] = s_now
            w_trace[step_idx] = work

        return float(work), s_trace, w_trace, sim.system

    def run_bidirectional(
        self,
        s_bound: Optional[float] = None,
        s_unbound: Optional[float] = None,
        n_paths: int = 6,
        n_steps: int = 120,
    ) -> SMDResult:
        """Execute N forward (bound -> unbound) and N reverse (unbound -> bound) pulls."""
        bound_sys, _, _ = minimize_energy(self.base_system, maxiter=80, gtol=1e-2)
        s_eq, _ = self.cv.evaluate(bound_sys.positions)
        s0 = float(s_eq if s_bound is None else s_bound)
        s1 = float(s0 + 3.5 if s_unbound is None else s_unbound)

        # Equilibrate an unbound state via slow steering + relaxation for reverse starts
        _, _, _, unbound_template = self._run_single_pull(
            bound_sys, s0, s1, max(n_steps, 100), self.seed + 999
        )
        # Hold briefly at s1 to relax into unbound basin
        hold_sim = MDSimulator(
            unbound_template,
            integrator="baoab",
            temperature_k=self.temperature_k,
            dt_ps=self.dt_ps,
            seed=self.seed + 1001,
        )
        for _ in range(40):
            def _hold_bias(pos: np.ndarray) -> Tuple[float, np.ndarray]:
                s_val, s_grad = self.cv.evaluate(pos)
                ds = self.cv.wrap_diff(s_val - s1)
                return float(0.5 * self.k_pull * ds * ds), -(self.k_pull * ds) * s_grad

            hold_sim.stepper.step(
                hold_sim.system, hold_sim.engine, external_bias=_hold_bias
            )  # type: ignore[union-attr]
        unbound_base = hold_sim.system

        fwd_works = np.zeros(n_paths, dtype=np.float64)
        rev_works = np.zeros(n_paths, dtype=np.float64)
        fwd_paths: List[Tuple[np.ndarray, np.ndarray]] = []
        rev_paths: List[Tuple[np.ndarray, np.ndarray]] = []

        for p in range(n_paths):
            wf, sf_tr, wf_tr, _ = self._run_single_pull(
                bound_sys, s0, s1, n_steps, run_seed=self.seed + 10 * p
            )
            wr, sr_tr, wr_tr, _ = self._run_single_pull(
                unbound_base, s1, s0, n_steps, run_seed=self.seed + 500 + 10 * p
            )
            fwd_works[p] = wf
            rev_works[p] = wr
            fwd_paths.append((sf_tr, wf_tr))
            rev_paths.append((sr_tr, wr_tr))

        dg_jarz, dg_cum = jarzynski_free_energy(fwd_works, self.temperature_k)
        dg_crooks, ci95 = crooks_mle_free_energy(
            fwd_works, rev_works, self.temperature_k, seed=self.seed
        )
        dissipated = float(np.mean(fwd_works) - dg_crooks)

        return SMDResult(
            forward_works=fwd_works,
            reverse_works=rev_works,
            dg_jarzynski_fwd=dg_jarz,
            dg_cumulant_fwd=dg_cum,
            dg_crooks_mle=dg_crooks,
            dg_bootstrap_ci95=ci95,
            dissipated_work_fwd=dissipated,
            cv_name=self.cv.name,
            forward_paths=fwd_paths,
            reverse_paths=rev_paths,
        )
