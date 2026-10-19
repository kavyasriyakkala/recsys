"""Smoke test: full pipeline on 50 users must complete and produce valid JSON."""
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest


REQUIRED_KEYS = {
    "meta", "data", "baseline_val", "baseline_test",
    "retriever_val", "retriever_test", "dcn_val", "dcn_test",
    "history_study", "serving",
}


@pytest.mark.slow
def test_full_pipeline_smoke():
    result = subprocess.run(
        [sys.executable, "scripts/run_pipeline.py", "--smoke"],
        capture_output=True, text=True, timeout=300,
    )
    print(result.stdout[-3000:])
    if result.returncode != 0:
        print("STDERR:", result.stderr[-2000:])
    assert result.returncode == 0, f"Pipeline failed:\n{result.stderr[-1000:]}"

    out_path = Path("reports/smoke_results.json")
    assert out_path.exists(), "reports/smoke_results.json was not created"

    with open(out_path) as f:
        data = json.load(f)

    missing = REQUIRED_KEYS - set(data.keys())
    assert not missing, f"Missing keys in smoke_results.json: {missing}"

    # Sanity checks on values
    assert data["meta"]["smoke"] is True
    assert data["data"]["n_train"] > 0
    assert data["data"]["n_val"] > 0
    assert data["data"]["n_test"] > 0

    # At least some baseline metrics exist
    assert "recall@10" in data["baseline_test"]
    assert "recall@100" in data["retriever_test"]
    assert "recall@10" in data["dcn_test"]

    # History study has all strategies
    for strategy in ["last_10", "last_50", "all", "attention"]:
        assert strategy in data["history_study"], f"Missing strategy: {strategy}"

    # Serving benchmark
    assert "p50_ms" in data["serving"]
    assert "p95_ms" in data["serving"]
