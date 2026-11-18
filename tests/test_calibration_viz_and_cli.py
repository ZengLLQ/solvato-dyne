"""Tests for calibration suite, sensitivity UQ, strict 7-bit ASCII viz, and all 6 CLI commands."""

import json
from pathlib import Path
import runpy

import pytest

from solvato_dyne.calibration.validator import (
    compute_gbsa_sensitivity,
    run_full_calibration_suite,
)
from solvato_dyne.cli import main as cli_main
from solvato_dyne.viz.ascii_viz import (
    assert_ascii,
    render_1d_profile,
)


def test_full_calibration_and_gbsa_sensitivity() -> None:
    rep = run_full_calibration_suite(quick=False)
    assert all(g.passed for g in rep.gradient_checks)
    assert rep.final_rmse <= rep.initial_rmse
    assert "alpha_O_A3" in rep.fitted_params

    sens = compute_gbsa_sensitivity(n_bootstrap=40, seed=7)
    assert sens.ci95[0] <= sens.bootstrap_mean <= sens.ci95[1]
    assert "gb_radii" in sens.normalized_sensitivities


def test_cli_all_six_subcommands_strict_ascii(tmp_path: Path, capsys) -> None:
    json_path = tmp_path / "sim_out.json"
    invocations = [
        ["simulate", "--system", "water-dimer", "--integrator", "drude", "--steps", "80", "--json-out", str(json_path)],
        ["remd", "--system", "ala-dipeptide", "--replicas", "3", "--cycles", "8", "--steps-per-cycle", "10"],
        ["metad", "--system", "ala-dipeptide", "--steps", "100", "--stride", "10"],
        ["steer", "--system", "host-guest", "--n-paths", "3", "--steps", "60"],
        ["modes", "--system", "helix8", "--method", "anm"],
        ["calibrate", "--quick"],
    ]

    for argv in invocations:
        rc = cli_main(argv)
        assert rc == 0, f"CLI failed for {argv}"
        captured = capsys.readouterr()
        assert len(captured.out.strip()) > 0
        # Enforce strict 7-bit ASCII for Windows cp1252 / CI safety
        assert_ascii(captured.out)

    assert json_path.exists()
    data = json.loads(json_path.read_text(encoding="ascii"))
    assert data["integrator"] == "drude"
    assert data["n_frames"] > 1


def test_examples_run_cleanly_as_scripts(capsys) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    example_files = sorted((repo_root / "examples").glob("*.py"))
    assert len(example_files) == 4
    for script in example_files:
        runpy.run_path(str(script), run_name="__main__")
        captured = capsys.readouterr()
        assert len(captured.out.strip()) > 0
        assert_ascii(captured.out)


def test_ascii_viz_guard_and_1d_profile() -> None:
    import numpy as np

    assert assert_ascii("hello-world_123") == "hello-world_123"
    with pytest.raises(ValueError):
        assert_ascii("bad-\u2014-dash")

    x = np.linspace(-3.0, 3.0, 20)
    y = x**2
    plot = render_1d_profile(x, y, "Parabola", "x", "x^2")
    assert_ascii(plot)
    assert "Parabola" in plot
