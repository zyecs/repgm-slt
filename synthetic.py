#!/usr/bin/env python3
"""Synthetic exact-ERM benchmark for hard-label model reprogramming (Section 7, Study 1).

Part A tests whether placing 2^r degrees of freedom in the input transformation g
or in the output head h yields identical finite-sample estimation error (Theorem 1).
Part B separates source-label information loss I from output-head restriction H
at matched in-class optimal risk R* = 0.1 + 0.05m (Proposition 4).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

SEED_ROOT = 20260725
BOOTSTRAP_DRAWS = 10_000
PART_A_R = (0, 2, 4, 6, 8)
PART_A_N = (8, 16, 32, 64, 128, 256, 512)
PART_B_M = (0, 2, 4, 6, 8)
PART_B_N = (32, 128, 512, 2048, 8192)

PART_IDS = {"A": 1, "B": 2}
ROLE_IDS = {
    "A_train": 1, "A_test": 2, "A_tie": 3,
    "B_train": 1, "B_test": 2, "B_diag": 3, "B_tie": 4,
}


def rng_for(part: str, repetition: int, role: str) -> np.random.Generator:
    ss = np.random.SeedSequence(
        entropy=[SEED_ROOT],
        spawn_key=(PART_IDS[part], int(repetition), ROLE_IDS[role]),
    )
    return np.random.default_rng(ss)


def sample_noisy(rng: np.random.Generator, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Uniform over 16 basis vectors; clean label (x & 1) flipped with probability 0.1."""
    x = rng.integers(0, 16, size=size, dtype=np.int16)
    flip = rng.random(size) < 0.1
    y = (x & 1) ^ flip.astype(np.int16)
    return x, y


def exact_risk_num(pred: np.ndarray) -> int:
    """Exact population risk numerator over 160 (Bayes risk = 16/160 = 0.1)."""
    bayes = np.arange(16, dtype=np.int16) & 1
    return 16 + 8 * int(np.count_nonzero(pred != bayes))


def predictors_input_arm(r: int) -> np.ndarray:
    k = np.arange(16, dtype=np.int16)
    out = []
    for mask in range(1 << r):
        bit = np.array([(mask >> j) & 1 for j in range(8)], dtype=np.int16)
        bit[r:] = 0
        out.append((k & 1) ^ bit[k // 2])
    return np.stack(out)


def predictors_head_arm(r: int) -> np.ndarray:
    base = np.arange(16, dtype=np.int16) & 1
    out = []
    for mask in range(1 << r):
        p = base.copy()
        for j in range(r):
            if (mask >> j) & 1:
                p[2 * j : 2 * j + 2] ^= 1
        out.append(p)
    return np.stack(out)


def run_part_a(repetitions: int = 200) -> dict[tuple[str, int, int], dict[str, np.ndarray]]:
    """Run Part A across all (r, n_T) cells for input-side and head-side parameterizations."""
    excess = {(arm, r, n): np.zeros(repetitions) for arm in ("input", "head") for r in PART_A_R for n in PART_A_N}
    q_obs = {(arm, r, n): np.zeros(repetitions) for arm in ("input", "head") for r in PART_A_R for n in PART_A_N}
    classes_cache = {r: {"input": predictors_input_arm(r), "head": predictors_head_arm(r)} for r in PART_A_R}

    for rep in range(repetitions):
        train_x, train_y = sample_noisy(rng_for("A", rep, "A_train"), max(PART_A_N))
        priorities = rng_for("A", rep, "A_tie").bit_generator.random_raw(256)
        for r in PART_A_R:
            for n in PART_A_N:
                x, y = train_x[:n], train_y[:n]
                q = len(set((x[x // 2 < r] // 2).tolist()))
                for arm, preds in classes_cache[r].items():
                    losses = np.count_nonzero(preds[:, x] != y[None, :], axis=1)
                    minimizers = np.flatnonzero(losses == int(losses.min()))
                    sel = min((int(m) for m in minimizers), key=lambda m: (int(priorities[m]), m))
                    excess[(arm, r, n)][rep] = (exact_risk_num(preds[sel]) - 16) / 160.0
                    q_obs[(arm, r, n)][rep] = float(q)
    return {"excess": excess, "q_obs": q_obs}


def _counts(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    c = np.zeros((16, 2), dtype=np.int64)
    np.add.at(c, (x, y), 1)
    return c


def _choose_label(c_row: np.ndarray, source_idx: int, priorities: np.ndarray) -> int:
    errs = (int(c_row[1]), int(c_row[0]))
    best = min(errs)
    return min((p for p in (0, 1) if errs[p] == best), key=lambda p: (int(priorities[source_idx, p]), p))


def run_part_b(repetitions: int = 200) -> dict[tuple[str, int, int], dict[str, np.ndarray]]:
    """Run Part B across all (m, n_T) cells for source-merging and head-restricted mechanisms."""
    pop_risk = {(arm, m, n): np.zeros(repetitions) for arm in ("info_loss", "head_restr") for m in PART_B_M for n in PART_B_N}
    res_excess = {(arm, m, n): np.zeros(repetitions) for arm in ("info_loss", "head_restr") for m in PART_B_M for n in PART_B_N}
    plugin_i = {(arm, m, n): np.zeros(repetitions) for arm in ("info_loss", "head_restr") for m in PART_B_M for n in PART_B_N}
    plugin_h = {(arm, m, n): np.zeros(repetitions) for arm in ("info_loss", "head_restr") for m in PART_B_M for n in PART_B_N}

    for rep in range(repetitions):
        train_x, train_y = sample_noisy(rng_for("B", rep, "B_train"), max(PART_B_N))
        diag_x, diag_y = sample_noisy(rng_for("B", rep, "B_diag"), 20_000)
        priorities = rng_for("B", rep, "B_tie").bit_generator.random_raw((16, 2))
        cd = _counts(diag_x, diag_y)
        e_id = int(np.minimum(cd[:, 0], cd[:, 1]).sum())

        for m in PART_B_M:
            oracle = 0.1 + 0.05 * m
            e_trans_bi = sum(int(min(cd[2 * j] + cd[2 * j + 1])) for j in range(m)) + int(
                np.minimum(cd[2 * m :, 0], cd[2 * m :, 1]).sum()
            )
            fixed_bh = {2 * j for j in range(m)}
            e_actual_bh = int(sum(cd[k, 0] if k in fixed_bh else min(cd[k]) for k in range(16)))

            for n in PART_B_N:
                ct = _counts(train_x[:n], train_y[:n])
                pred_bi = np.zeros(16, dtype=np.int16)
                for j in range(m):
                    pred_bi[2 * j : 2 * j + 2] = _choose_label(ct[2 * j] + ct[2 * j + 1], 2 * j, priorities)
                for k in range(2 * m, 16):
                    pred_bi[k] = _choose_label(ct[k], k, priorities)

                pred_bh = np.zeros(16, dtype=np.int16)
                for k in range(16):
                    pred_bh[k] = 1 if k in fixed_bh else _choose_label(ct[k], k, priorities)

                for arm, pred, i_cnt, h_cnt in (
                    ("info_loss", pred_bi, e_trans_bi - e_id, 0),
                    ("head_restr", pred_bh, 0, e_actual_bh - e_id),
                ):
                    r_val = exact_risk_num(pred) / 160.0
                    pop_risk[(arm, m, n)][rep] = r_val
                    res_excess[(arm, m, n)][rep] = r_val - oracle
                    plugin_i[(arm, m, n)][rep] = i_cnt / 20_000.0
                    plugin_h[(arm, m, n)][rep] = h_cnt / 20_000.0

    return {
        "pop_risk": pop_risk,
        "res_excess": res_excess,
        "plugin_i": plugin_i,
        "plugin_h": plugin_h,
    }


def bootstrap_ci(values: np.ndarray, boot_idx: np.ndarray) -> tuple[float, float, float]:
    means = values[boot_idx].mean(axis=1)
    return (
        float(values.mean()),
        float(np.quantile(means, 0.025, method="linear")),
        float(np.quantile(means, 0.975, method="linear")),
    )


def summarize_and_save(out_dir: Path, repetitions: int = 200) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ss = np.random.SeedSequence(entropy=[SEED_ROOT], spawn_key=(3, 0, 1))
    boot_idx = np.random.default_rng(ss).integers(0, repetitions, size=(BOOTSTRAP_DRAWS, repetitions))

    part_a = run_part_a(repetitions)
    rows_a = []
    for arm in ("input", "head"):
        for r in PART_A_R:
            for n in PART_A_N:
                mean, lo, hi = bootstrap_ci(part_a["excess"][(arm, r, n)], boot_idx)
                rows_a.append({
                    "parameterization": arm, "r": r, "n": n, "repetitions": repetitions,
                    "mean_exact_estimation_excess": f"{mean:.10f}",
                    "ci025": f"{lo:.10f}", "ci975": f"{hi:.10f}",
                    "mean_exact_population_risk": f"{0.1 + mean:.10f}",
                    "mean_q_observed": f"{part_a['q_obs'][(arm, r, n)].mean():.10f}",
                })
    _write_csv(out_dir / "synthetic_part_a.csv", list(rows_a[0].keys()), rows_a)

    rows_pair = []
    for r in PART_A_R:
        for n in PART_A_N:
            diff = part_a["excess"][("input", r, n)] - part_a["excess"][("head", r, n)]
            mean, lo, hi = bootstrap_ci(diff, boot_idx)
            rows_pair.append({
                "r": r, "n": n, "repetitions": repetitions,
                "mean_input_minus_head_excess": f"{mean:.10f}",
                "ci025": f"{lo:.10f}", "ci975": f"{hi:.10f}",
                "mean_abs_diff": f"{np.abs(diff).mean():.10f}",
            })
    _write_csv(out_dir / "synthetic_part_a_paired.csv", list(rows_pair[0].keys()), rows_pair)

    part_b = run_part_b(repetitions)
    rows_b = []
    for arm in ("head_restr", "info_loss"):
        for m in PART_B_M:
            oracle = 0.1 + 0.05 * m
            for n in PART_B_N:
                mean_ex, lo, hi = bootstrap_ci(part_b["res_excess"][(arm, m, n)], boot_idx)
                rows_b.append({
                    "mechanism": arm, "m": m, "n": n, "repetitions": repetitions,
                    "class_oracle_risk": f"{oracle:.10f}",
                    "mean_exact_population_risk": f"{part_b['pop_risk'][(arm, m, n)].mean():.10f}",
                    "mean_residual_estimation_excess": f"{mean_ex:.10f}",
                    "ci025": f"{lo:.10f}", "ci975": f"{hi:.10f}",
                    "exact_I": f"{(0.05 * m if arm == 'info_loss' else 0.0):.10f}",
                    "exact_H": f"{(0.05 * m if arm == 'head_restr' else 0.0):.10f}",
                    "mean_plugin_I": f"{part_b['plugin_i'][(arm, m, n)].mean():.10f}",
                    "mean_plugin_H": f"{part_b['plugin_h'][(arm, m, n)].mean():.10f}",
                })
    _write_csv(out_dir / "synthetic_part_b.csv", list(rows_b[0].keys()), rows_b)


def _write_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fields))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Study 1 synthetic exact-ERM experiments.")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--repetitions", type=int, default=200)
    args = parser.parse_args()
    summarize_and_save(args.out_dir, args.repetitions)
    print(f"Saved synthetic benchmark results ({args.repetitions} repetitions) to {args.out_dir}")


if __name__ == "__main__":
    main()
