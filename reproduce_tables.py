#!/usr/bin/env python3
"""Recompute and display all numbers in Paper Table 1 and Appendix K from exact ERM."""

from __future__ import annotations

from pathlib import Path

from mnist import evaluate_from_cached_labels
from synthetic import BOOTSTRAP_DRAWS, PART_A_N, PART_A_R, PART_B_M, SEED_ROOT, bootstrap_ci, run_part_a, run_part_b, summarize_and_save
import numpy as np


def main() -> None:
    root = Path(__file__).resolve().parent
    out_dir = root / "results"
    cache_dir = root / "checkpoints" / "hard_labels"

    summarize_and_save(out_dir, repetitions=200)
    ss = np.random.SeedSequence(entropy=[SEED_ROOT], spawn_key=(3, 0, 1))
    boot_idx = np.random.default_rng(ss).integers(0, 200, size=(BOOTSTRAP_DRAWS, 200))
    part_a = run_part_a(200)
    part_b = run_part_b(200)

    print("Table 1(a): Synthetic Estimation Error E_T^est (Input == Head, 200 paired runs)")
    header_a = f"{'Sample size':<12}" + "".join(f"{'r=' + str(r):>10}" for r in PART_A_R)
    print(header_a)
    print("-" * len(header_a))
    for n in (8, 32, 64, 512):
        row_str = f"n_T = {n:<6}"
        for r in PART_A_R:
            val = part_a["excess"][("input", r, n)].mean()
            row_str += f"{val:>10.4f}"
        print(row_str)
    max_diff = max(
        float(np.max(np.abs(part_a["excess"][("input", r, n)] - part_a["excess"][("head", r, n)])))
        for r in PART_A_R for n in PART_A_N
    )
    print(f"Max paired |Input - Head| difference across all 35 cells: {max_diff:.6f}\n")

    print("Table 1(b): Synthetic Approximation Split at Matched R* (n_T = 8192, 200 runs)")
    print(f"{'Mechanism (m)':<20}{'R*':>8}{'I':>8}{'H':>8}{'I_hat':>10}{'H_hat':>10}")
    print("-" * 64)
    for m in (2, 8):
        oracle = 0.1 + 0.05 * m
        for label, arm, ex_i, ex_h in (
            (f"Merge labels ({m})", "info_loss", 0.05 * m, 0.0),
            (f"Restrict head ({m})", "head_restr", 0.0, 0.05 * m),
        ):
            ih = part_b["plugin_i"][(arm, m, 8192)].mean()
            hh = part_b["plugin_h"][(arm, m, 8192)].mean()
            print(f"{label:<20}{oracle:>8.4f}{ex_i:>8.4f}{ex_h:>8.4f}{ih:>10.4f}{hh:>10.4f}")
    print()

    mlp = evaluate_from_cached_labels("mlp", cache_dir)
    lenet = evaluate_from_cached_labels("lenet", cache_dir)
    mlp_map = {(r["wrapper_class"], r["head_class"], r["n_T"]): r["mean_risk"] for r in mlp["summary"]}
    lenet_map = {(r["wrapper_class"], r["head_class"], r["n_T"]): r["mean_risk"] for r in lenet["summary"]}

    print("Table 1(c): Pretrained Neural Sources on 90-deg Rotated MNIST Parity (Held-Out 0-1 Risk)")
    print(f"{'Wrapper (G, H)':<18}{'MLP 50':>9}{'200':>8}{'800':>8}{'3200':>8} | {'LeNet 50':>9}{'200':>8}{'800':>8}{'3200':>8}")
    print("-" * 87)
    for g, h in (("G_id", "H_thr"), ("G_id", "H_full"), ("G_rot", "H_thr"), ("G_rot", "H_full")):
        m_vals = "".join(f"{mlp_map[(g, h, n)]:>8.4f}" for n in (50, 200, 800, 3200))
        l_vals = "".join(f"{lenet_map[(g, h, n)]:>8.4f}" for n in (50, 200, 800, 3200))
        print(f"({g}, {h}){'':<4} {m_vals} |  {l_vals}")
    un_m = mlp["contrasts"]["untrained_risk"][0]
    un_l = lenet["contrasts"]["untrained_risk"][0]
    print(f"{'Untrained (n=3200)':<18}{'':>25}{un_m:>8.4f} | {'':>25}{un_l:>8.4f}")
    print(
        f"Contrasts at n_T=3200: "
        f"MLP Delta_in={mlp['contrasts']['Delta_in'][0]:.4f}, Delta_out={mlp['contrasts']['Delta_out'][0]:.4f}; "
        f"LeNet Delta_in={lenet['contrasts']['Delta_in'][0]:.4f}, Delta_out={lenet['contrasts']['Delta_out'][0]:.4f}"
    )


if __name__ == "__main__":
    main()
