"""Tests for Collective Variable gradients and Well-Tempered Metadynamics."""

import numpy as np

from solvato_dyne.sampling.colvars import (
    CoordinationNumberCV,
    DihedralCV,
    DistanceCV,
    RMSDCV,
    RadiusOfGyrationCV,
)
from solvato_dyne.sampling.metadynamics import WellTemperedMetadynamics
from solvato_dyne.systems.builders import build_alanine_dipeptide_system


def _check_cv_gradient(cv, positions: np.ndarray, eps: float = 1e-5) -> None:
    _, grad = cv.evaluate(positions)
    fd = np.zeros_like(positions)
    for i in range(positions.shape[0]):
        for j in range(3):
            pp = positions.copy()
            pm = positions.copy()
            pp[i, j] += eps
            pm[i, j] -= eps
            vp, _ = cv.evaluate(pp)
            vm, _ = cv.evaluate(pm)
            fd[i, j] = cv.wrap_diff(vp - vm) / (2.0 * eps)
    assert np.allclose(grad, fd, atol=1e-5), f"{cv.name} max err={np.max(np.abs(grad - fd))}"


def test_all_collective_variable_analytical_gradients() -> None:
    ala = build_alanine_dipeptide_system(polarizable=False)
    rng = np.random.default_rng(77)
    pos = ala.positions + rng.normal(0.0, 0.12, ala.positions.shape)
    ref = ala.positions[:6].copy()

    cvs = [
        DistanceCV([0, 1], [9, 11], name="dist"),
        RadiusOfGyrationCV([0, 1, 3, 5, 7, 9, 11], name="rg"),
        CoordinationNumberCV([1, 2], [9, 10, 11], r0=3.5, name="coord"),
        DihedralCV(1, 3, 5, 7, name="phi"),
        RMSDCV([0, 1, 2, 3, 4, 5], ref, name="rmsd"),
    ]
    for cv in cvs:
        _check_cv_gradient(cv, pos)


def test_well_tempered_metadynamics_1d_and_2d_reweighting() -> None:
    ala = build_alanine_dipeptide_system(polarizable=False, use_gbsa=True)
    phi_cv = DihedralCV(1, 3, 5, 7, name="phi")
    psi_cv = DihedralCV(3, 5, 7, 9, name="psi")

    # 1D WT-MetaD
    metad_1d = WellTemperedMetadynamics(
        cvs=[phi_cv],
        sigmas=[0.35],
        initial_height=0.6,
        bias_factor=6.0,
        deposit_stride=10,
        grid_bins=25,
    )
    res_1d = metad_1d.run(ala, n_steps=160, save_every=10, seed=42)
    assert len(res_1d.hill_heights) == 16
    # Well-tempered hill heights decay monotonically or fall below initial_height once hills overlap
    assert np.min(res_1d.hill_heights) <= 0.6
    assert np.isclose(np.min(res_1d.fes_direct), 0.0)
    assert np.isclose(np.min(res_1d.fes_reweighted), 0.0)
    assert np.all(np.isfinite(res_1d.c_of_t))

    # 2D WT-MetaD
    metad_2d = WellTemperedMetadynamics(
        cvs=[phi_cv, psi_cv],
        sigmas=[0.4, 0.4],
        initial_height=0.7,
        bias_factor=8.0,
        deposit_stride=15,
        grid_bins=21,
    )
    res_2d = metad_2d.run(ala, n_steps=180, save_every=15, seed=9)
    assert res_2d.fes_direct.shape == (21, 21)
    assert res_2d.fes_reweighted.shape == (21, 21)
    assert np.max(res_2d.fes_direct) > 0.0


def test_metadynamics_bias_gradient_vs_finite_difference_and_wrap() -> None:
    phi_cv = DihedralCV(1, 3, 5, 7, name="phi")
    psi_cv = DihedralCV(3, 5, 7, 9, name="psi")
    assert np.isclose(phi_cv.wrap_diff(2.0 * np.pi - 0.1), -0.1)

    metad = WellTemperedMetadynamics(cvs=[phi_cv, psi_cv], sigmas=[0.4, 0.5], bias_factor=6.0)
    metad.centers = [np.array([0.2, -0.5]), np.array([-2.9, 2.8])]
    metad.heights = [0.5, 0.35]

    s = np.array([3.0, -3.05])
    _, grad = metad.bias_and_grad(s)
    eps = 1e-6
    fd = np.zeros(2)
    for d in range(2):
        sp = s.copy()
        sm = s.copy()
        sp[d] += eps
        sm[d] -= eps
        fd[d] = (metad.bias_and_grad(sp)[0] - metad.bias_and_grad(sm)[0]) / (2.0 * eps)
    assert np.allclose(grad, fd, atol=1e-6)
