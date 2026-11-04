"""Tests for the synthetic and rotated-MNIST reprogramming benchmarks."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mnist import evaluate_from_cached_labels, make_heads, make_model
from synthetic import predictors_head_arm, predictors_input_arm, run_part_a, run_part_b


def test_synthetic_estimation_symmetry() -> None:
    for r in (0, 2, 4, 6, 8):
        assert np.array_equal(predictors_input_arm(r), predictors_head_arm(r))
    res = run_part_a(repetitions=10)
    for r in (0, 2, 4, 6, 8):
        for n in (8, 64, 512):
            assert np.array_equal(res["excess"][("input", r, n)], res["excess"][("head", r, n)])


def test_synthetic_approximation_split() -> None:
    res = run_part_b(repetitions=5)
    for m in (0, 2, 8):
        assert np.allclose(res["plugin_h"][("info_loss", m, 8192)], 0.0)
        assert np.allclose(res["plugin_i"][("head_restr", m, 8192)], 0.0)


def test_mnist_cached_reproduction() -> None:
    cache_dir = ROOT / "checkpoints" / "hard_labels"
    mlp = evaluate_from_cached_labels("mlp", cache_dir)
    lenet = evaluate_from_cached_labels("lenet", cache_dir)
    assert mlp["growth_verified"] == 960
    assert lenet["growth_verified"] == 960
    assert round(mlp["contrasts"]["Delta_in"][0], 4) == 0.3881
    assert round(mlp["contrasts"]["Delta_out"][0], 4) == 0.3514
    assert round(lenet["contrasts"]["Delta_in"][0], 4) == 0.3480
    assert round(lenet["contrasts"]["Delta_out"][0], 4) == 0.3293


def test_model_and_head_shapes() -> None:
    _, h_thr = make_heads("H_thr")
    _, h_full = make_heads("H_full")
    assert h_thr.shape == (20, 10)
    assert h_full.shape == (1024, 10)
    assert make_model("mlp") is not None
    assert make_model("lenet") is not None
