import json
import math
import os
import subprocess
import sys
from pathlib import Path


SUITE_DIR = Path(__file__).resolve().parents[1]
PIPELINE_SCRIPT = SUITE_DIR.parent / "pipelines" / "hardcoded_adapt_pipeline.py"
ARTIFACT_DIR = Path(os.environ.get("ADAPT_HH_ARTIFACT_DIR", SUITE_DIR / "artifacts"))
TOL = 1e-4

DEFAULT_ARGS = {
    "t": 0.2,
    "u": 0.2,
    "omega0": 0.2,
    "g_ep": 0.0,
    "n_ph_max": 1,
    "boson_encoding": "binary",
    "boundary": "open",
    "ordering": "blocked",
    "adapt_pool": "hva",
    "adapt_max_depth": 120,
    "adapt_maxiter": 1200,
    "adapt_eps_grad": 1e-12,
    "adapt_eps_energy": 1e-10,
    "initial_state_source": "hf",
}


def _default_seed() -> int:
    return int(os.environ.get("ADAPT_HH_TEST_SEED", "7"))


def _run_hh_adapt_case(L: int) -> dict:
    seed = _default_seed()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    output_json = ARTIFACT_DIR / f"hh_adapt_vqe_L{L}_seed{seed}.json"

    command = [
        sys.executable,
        str(PIPELINE_SCRIPT),
        "--L",
        str(L),
        "--problem",
        "hh",
        "--t",
        str(DEFAULT_ARGS["t"]),
        "--u",
        str(DEFAULT_ARGS["u"]),
        "--omega0",
        str(DEFAULT_ARGS["omega0"]),
        "--g-ep",
        str(DEFAULT_ARGS["g_ep"]),
        "--n-ph-max",
        str(DEFAULT_ARGS["n_ph_max"]),
        "--boson-encoding",
        DEFAULT_ARGS["boson_encoding"],
        "--boundary",
        DEFAULT_ARGS["boundary"],
        "--ordering",
        DEFAULT_ARGS["ordering"],
        "--adapt-pool",
        DEFAULT_ARGS["adapt_pool"],
        "--adapt-max-depth",
        str(DEFAULT_ARGS["adapt_max_depth"]),
        "--adapt-maxiter",
        str(DEFAULT_ARGS["adapt_maxiter"]),
        "--adapt-eps-grad",
        str(DEFAULT_ARGS["adapt_eps_grad"]),
        "--adapt-eps-energy",
        str(DEFAULT_ARGS["adapt_eps_energy"]),
        "--adapt-no-repeats",
        "--adapt-no-finite-angle-fallback",
        "--adapt-seed",
        str(seed),
        "--initial-state-source",
        DEFAULT_ARGS["initial_state_source"],
        "--skip-pdf",
        "--output-json",
        str(output_json),
        "--dv",
        "0.0",
    ]

    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0

    with output_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    adapt_vqe = payload.get("adapt_vqe", {})
    assert adapt_vqe.get("success") is True

    abs_delta_e = adapt_vqe.get("abs_delta_e")
    assert isinstance(abs_delta_e, (int, float)) and math.isfinite(abs_delta_e)
    assert abs(abs_delta_e) < TOL

    energy = adapt_vqe.get("energy")
    exact_energy = adapt_vqe.get("exact_gs_energy")
    assert isinstance(energy, (int, float)) and math.isfinite(energy)
    assert isinstance(exact_energy, (int, float)) and math.isfinite(exact_energy)

    return payload


def test_l2_hh_adapt_vqe_ground_state_within_1e_minus_4() -> None:
    payload = _run_hh_adapt_case(2)
    assert payload["settings"]["L"] == 2


def test_l3_hh_adapt_vqe_ground_state_within_1e_minus_4() -> None:
    payload = _run_hh_adapt_case(3)
    assert payload["settings"]["L"] == 3
