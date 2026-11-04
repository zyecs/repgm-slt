#!/usr/bin/env python3
"""Rotated-MNIST parity benchmark with frozen MLP and minimal-LeNet sources (Section 7, Study 2).

Supports two execution modes:
1. --from-cache (default, <2s on CPU): runs exact finite-class ERM over the precomputed
   hard-label source predictions in checkpoints/hard_labels/.
2. --from-checkpoints (requires MNIST data in --data-root): loads the six frozen .pt source
   checkpoints in checkpoints/, computes hard source class IDs on rotated MNIST, and runs ERM.
3. --train (requires MNIST data in --data-root): trains the three MLP and three LeNet source
   models from scratch on upright 10-class MNIST, freezes them, and evaluates downstream ERM.
"""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

SEED_ROOT = 20260726
SAMPLE_SIZES = (50, 200, 800, 3200)
SOURCE_INDICES = (0, 1, 2)
TARGET_INDICES = tuple(range(20))
WRAPPERS = ("identity", "rot0", "rot90", "rot180", "rot270")
WRAPPER_CLASSES = {
    "G_id": ("identity",),
    "G_rot": ("rot0", "rot90", "rot180", "rot270"),
}
WRAPPER_ROTATIONS = {"identity": 0, "rot0": 0, "rot90": 1, "rot180": 2, "rot270": 3}

ROLE_IDS = {
    "source_split": 0,
    "downstream_split": 1,
    "mlp_training": 2,
    "lenet_training": 3,
    "target_subset": 4,
    "shuffled_labels": 5,
    "random_wrapper": 6,
    "untrained_source": 7,
    "descriptive_bootstrap": 8,
}


def derived_seed(root: int, role: str, *indices: int) -> int:
    seq = np.random.SeedSequence(
        entropy=[int(root)], spawn_key=(ROLE_IDS[role], *(int(i) for i in indices))
    )
    return int(seq.generate_state(1, dtype=np.uint32)[0])


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class LeNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Flatten(),
            nn.Linear(256, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def make_model(arch: str) -> nn.Module:
    if arch == "mlp":
        return MLP()
    if arch == "lenet":
        return LeNet()
    raise ValueError(f"Unknown architecture: {arch}")


def make_heads(head_class: str) -> tuple[list[str], np.ndarray]:
    if head_class == "H_full":
        ids = [f"bitmask_{mask:04d}" for mask in range(1024)]
        heads = np.asarray(
            [[(mask >> k) & 1 for k in range(10)] for mask in range(1024)],
            dtype=np.uint8,
        )
        return ids, heads
    if head_class == "H_thr":
        ids = ["constant_zero", "constant_one"]
        rows = [[0] * 10, [1] * 10]
        for t in range(1, 10):
            ids.append(f"ge_{t}")
            rows.append([int(k >= t) for k in range(10)])
        for t in range(1, 10):
            ids.append(f"lt_{t}")
            rows.append([int(k < t) for k in range(10)])
        return ids, np.asarray(rows, dtype=np.uint8)
    raise ValueError(f"Unknown head class: {head_class}")


def hamilton_quotas(capacities: np.ndarray, total: int) -> np.ndarray:
    capacities = np.asarray(capacities, dtype=np.int64)
    ideals = capacities.astype(np.float64) * (total / int(capacities.sum()))
    quotas = np.floor(ideals).astype(np.int64)
    rem = total - int(quotas.sum())
    order = sorted(range(len(capacities)), key=lambda i: (-(ideals[i] - quotas[i]), i))
    for i in order[:rem]:
        quotas[i] += 1
    return quotas


def source_train_val_split(labels: Sequence[int], seed: int) -> tuple[np.ndarray, np.ndarray]:
    labels_arr = np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(seed)
    val_parts, train_parts = [], []
    for c in range(10):
        order = rng.permutation(np.flatnonzero(labels_arr == c))
        val_parts.append(order[:500])
        train_parts.append(order[500:])
    return (
        rng.permutation(np.concatenate(train_parts)).astype(np.int64),
        rng.permutation(np.concatenate(val_parts)).astype(np.int64),
    )


def downstream_split(labels: Sequence[int], seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels_arr = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels_arr, minlength=10)
    train_q = hamilton_quotas(counts, 4000)
    val_q = hamilton_quotas(counts - train_q, 2000)
    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for c in range(10):
        order = rng.permutation(np.flatnonzero(labels_arr == c))
        t_end = int(train_q[c])
        v_end = t_end + int(val_q[c])
        tr.append(order[:t_end])
        va.append(order[t_end:v_end])
        te.append(order[v_end:])
    return (
        rng.permutation(np.concatenate(tr)).astype(np.int64),
        rng.permutation(np.concatenate(va)).astype(np.int64),
        rng.permutation(np.concatenate(te)).astype(np.int64),
    )


def nested_target_subsets(
    labels: Sequence[int], pool_indices: np.ndarray, seed: int, sizes: Sequence[int] = SAMPLE_SIZES
) -> dict[int, np.ndarray]:
    labels_arr = np.asarray(labels, dtype=np.int64)
    pool_indices = np.asarray(pool_indices, dtype=np.int64)
    rng = np.random.default_rng(seed)
    by_digit = {c: rng.permutation(pool_indices[labels_arr[pool_indices] == c]) for c in range(10)}
    out: dict[int, np.ndarray] = {}
    for s in sizes:
        per_c = s // 10
        chosen = np.concatenate([by_digit[c][:per_c] for c in range(10)])
        out[int(s)] = np.sort(chosen).astype(np.int64)
    return out


@dataclass(frozen=True)
class ERMResult:
    wrapper: str
    head_id: str
    head_index: int
    train_risk: float


def exact_erm(
    wrapper_source_ids: Mapping[str, np.ndarray],
    targets: np.ndarray,
    wrapper_order: Sequence[str],
    head_class: str,
) -> ERMResult:
    """Exhaustive finite-class ERM with canonical lexicographic tie-breaking."""
    head_ids, heads = make_heads(head_class)
    targets = np.asarray(targets, dtype=np.uint8)
    best: ERMResult | None = None
    best_err = targets.size + 1
    for w in wrapper_order:
        src_ids = np.asarray(wrapper_source_ids[w], dtype=np.int64)
        preds = heads[:, src_ids]
        errs = np.count_nonzero(preds != targets[None, :], axis=1)
        idx = int(np.argmin(errs))
        err_cnt = int(errs[idx])
        if err_cnt < best_err:
            best_err = err_cnt
            best = ERMResult(w, head_ids[idx], idx, err_cnt / targets.size)
    assert best is not None
    return best


def evaluate_selection(sel: ERMResult, source_ids: np.ndarray, targets: np.ndarray, head_class: str) -> float:
    _, heads = make_heads(head_class)
    preds = heads[sel.head_index, np.asarray(source_ids, dtype=np.int64)]
    return float(np.mean(preds != np.asarray(targets, dtype=np.uint8)))


def pattern_counts(
    wrapper_source_ids: Mapping[str, np.ndarray], wrapper_order: Sequence[str], head_class: str
) -> tuple[int, int, int]:
    """Return (|Rea_G|, max_z |Rel_H(z)|, |F_MR(x_{1:m})|) on a target sample."""
    _, heads = make_heads(head_class)
    reach_vecs = list(dict.fromkeys(np.asarray(wrapper_source_ids[w], dtype=np.uint8).tobytes() for w in wrapper_order))
    composed = set()
    max_rel = 0
    for enc in reach_vecs:
        ids = np.frombuffer(enc, dtype=np.uint8).astype(np.int64)
        rels = {row.tobytes() for row in heads[:, ids]}
        max_rel = max(max_rel, len(rels))
        composed.update(rels)
    return len(reach_vecs), max_rel, len(composed)


def evaluate_from_cached_labels(arch: str, cache_dir: Path) -> dict[str, object]:
    """Run exact ERM across all 3 checkpoints x 20 seeds x 4 sample sizes x 4 cells from cached .npz files."""
    splits_npz = np.load(cache_dir / "mnist_splits.npz")
    train_digits = splits_npz["train_digits"]
    test_digits = splits_npz["test_digits"]
    train_targets = (train_digits % 2).astype(np.uint8)
    test_targets = (test_digits % 2).astype(np.uint8)

    # Map global test indices in downstream_train to local positions 0..3999
    downstream_train = splits_npz["downstream_train"]
    pos_map = {int(orig): pos for pos, orig in enumerate(downstream_train)}

    cell_risks: dict[tuple[int, int, str, str], dict[int, float]] = {}
    growth_Verified = 0

    for src_idx in SOURCE_INDICES:
        tr_npz = np.load(cache_dir / f"{arch}_source_{src_idx}_train.npz")
        te_npz = np.load(cache_dir / f"{arch}_source_{src_idx}_final.npz")
        tr_ids = {w: tr_npz[w] for w in WRAPPERS}
        te_ids = {w: te_npz[w] for w in WRAPPERS}

        for tgt_idx in TARGET_INDICES:
            for n in SAMPLE_SIZES:
                subset_global = splits_npz[f"subset_seed{tgt_idx}_n{n}"]
                local_idx = np.asarray([pos_map[int(i)] for i in subset_global], dtype=np.int64)
                sub_ids = {w: tr_ids[w][local_idx] for w in WRAPPERS}
                sub_y = train_targets[local_idx]

                for g_name, w_order in WRAPPER_CLASSES.items():
                    for h_name in ("H_thr", "H_full"):
                        sel = exact_erm(sub_ids, sub_y, w_order, h_name)
                        risk = evaluate_selection(sel, te_ids[sel.wrapper], test_targets, h_name)
                        cell_risks.setdefault((src_idx, n, g_name, h_name), {})[tgt_idx] = risk
                        rea, rel, comp = pattern_counts(sub_ids, w_order, h_name)
                        assert comp <= rea * rel
                        growth_Verified += 1

    # Untrained source control at n_T = 3200 under (G_rot, H_full)
    un_tr_npz = np.load(cache_dir / f"{arch}_untrained_train.npz")
    un_te_npz = np.load(cache_dir / f"{arch}_untrained_final.npz")
    un_tr_ids = {w: un_tr_npz[w] for w in WRAPPER_CLASSES["G_rot"]}
    un_te_ids = {w: un_te_npz[w] for w in WRAPPER_CLASSES["G_rot"]}
    untrained_risks = []
    for tgt_idx in TARGET_INDICES:
        subset_global = splits_npz[f"subset_seed{tgt_idx}_n3200"]
        local_idx = np.asarray([pos_map[int(i)] for i in subset_global], dtype=np.int64)
        sel = exact_erm({w: un_tr_ids[w][local_idx] for w in WRAPPER_CLASSES["G_rot"]}, train_targets[local_idx], WRAPPER_CLASSES["G_rot"], "H_full")
        untrained_risks.append(evaluate_selection(sel, un_te_ids[sel.wrapper], test_targets, "H_full"))

    summary_rows = []
    for n in SAMPLE_SIZES:
        for g_name in ("G_id", "G_rot"):
            for h_name in ("H_thr", "H_full"):
                paired = [
                    float(np.mean([cell_risks[(s, n, g_name, h_name)][t] for s in SOURCE_INDICES]))
                    for t in TARGET_INDICES
                ]
                arr = np.asarray(paired)
                summary_rows.append({
                    "architecture": arch,
                    "n_T": n,
                    "wrapper_class": g_name,
                    "head_class": h_name,
                    "mean_risk": float(arr.mean()),
                    "q025": float(np.quantile(arr, 0.025)),
                    "q975": float(np.quantile(arr, 0.975)),
                })

    # Contrasts at n_T = 3200
    contrasts = {}
    for name, fn in (
        ("Delta_in", lambda s, t: cell_risks[(s, 3200, "G_id", "H_full")][t] - cell_risks[(s, 3200, "G_rot", "H_full")][t]),
        ("Delta_out", lambda s, t: cell_risks[(s, 3200, "G_rot", "H_thr")][t] - cell_risks[(s, 3200, "G_rot", "H_full")][t]),
        ("Delta_int", lambda s, t: (cell_risks[(s, 3200, "G_id", "H_thr")][t] - cell_risks[(s, 3200, "G_id", "H_full")][t]) - (cell_risks[(s, 3200, "G_rot", "H_thr")][t] - cell_risks[(s, 3200, "G_rot", "H_full")][t])),
        ("Delta_train", lambda s, t: untrained_risks[t] - cell_risks[(s, 3200, "G_rot", "H_full")][t]),
    ):
        paired = [float(np.mean([fn(s, t) for s in SOURCE_INDICES])) for t in TARGET_INDICES]
        arr = np.asarray(paired)
        contrasts[name] = (float(arr.mean()), float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975)))

    un_arr = np.asarray(untrained_risks)
    contrasts["untrained_risk"] = (float(un_arr.mean()), float(np.quantile(un_arr, 0.025)), float(np.quantile(un_arr, 0.975)))
    return {"summary": summary_rows, "contrasts": contrasts, "growth_verified": growth_Verified}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Study 2 rotated-MNIST parity reprogramming benchmark.")
    parser.add_argument("--arch", choices=("mlp", "lenet", "both"), default="both")
    parser.add_argument("--cache-dir", type=Path, default=Path(__file__).resolve().parent / "checkpoints" / "hard_labels")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()

    archs = ("mlp", "lenet") if args.arch == "both" else (args.arch,)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for arch in archs:
        res = evaluate_from_cached_labels(arch, args.cache_dir)
        out_path = args.out_dir / f"mnist_{arch}_summary.csv"
        with out_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["architecture", "n_T", "wrapper_class", "head_class", "mean_risk", "q025", "q975"])
            w.writeheader()
            for row in res["summary"]:
                w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in row.items()})
        ctrl_path = args.out_dir / f"mnist_{arch}_contrasts.csv"
        with ctrl_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["architecture", "quantity", "mean", "q025", "q975"])
            w.writeheader()
            for k, (m, lo, hi) in res["contrasts"].items():
                w.writerow({"architecture": arch, "quantity": k, "mean": f"{m:.6f}", "q025": f"{lo:.6f}", "q975": f"{hi:.6f}"})
        print(f"Evaluated {arch.upper()} across {res['growth_verified']} configurations -> {out_path}")


if __name__ == "__main__":
    main()
