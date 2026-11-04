# On the Statistical Learnability of Model Reprogramming: A Reachability and Relabeling Perspective

Code and precomputed artifacts for the NeurIPS 2026 paper:

**On the Statistical Learnability of Model Reprogramming: A Reachability and Relabeling Perspective**  
Zesheng Ye, Pin-Yu Chen, and Feng Liu  
*Advances in Neural Information Processing Systems (NeurIPS)*, 2026.

## Contents

This repository reproduces the two controlled exact-ERM studies in Section 7 and Appendix K of the paper:

1. **Synthetic exact-ERM benchmark (`synthetic.py`)**: Evaluates finite-sample estimation error across input-side reachability and output-side relabeling parameterizations of size $2^r$ (Theorem 1, Table 1a, Table 4), and separates source-label information loss $I$ from output-head restriction $H$ at matched in-class optimal risk $R_{\mathrm{T}}^{\mathcal{F}_{\mathrm{MR}}, \ast} = 0.1 + 0.05m$ (Proposition 4, Table 1b, Table 5).
2. **Rotated-MNIST parity benchmark (`mnist.py`)**: Evaluates $2 \times 2$ wrapper-head combinations $(\mathcal{G}_{\mathrm{id}}, \mathcal{G}_{\mathrm{rot}}) \times (\mathcal{H}_{\mathrm{thr}}, \mathcal{H}_{\mathrm{full}})$ on unseen $90^\circ$-rotated MNIST digits using frozen upright-trained MLP and minimal-LeNet classifiers that expose only predicted digit IDs in $\{0, \dots, 9\}$ (Table 1c, Table 6).

## Setup

```bash
pip install -r requirements.txt
```

## Reproduce Paper Tables

To run exact finite-class ERM on both benchmarks and print Table 1(a), Table 1(b), and Table 1(c) (takes about 3 seconds on CPU):

```bash
python reproduce_tables.py
```

This script also writes the full summary tables (including $95\%$ intervals across all sample sizes $n_{\mathrm{T}}$) to `results/`:

- `results/synthetic_part_a.csv` and `results/synthetic_part_a_paired.csv` (Table 1a and Appendix Table 4)
- `results/synthetic_part_b.csv` (Table 1b and Appendix Table 5)
- `results/mnist_mlp_summary.csv` and `results/mnist_mlp_contrasts.csv` (Table 1c and Appendix Table 6, MLP)
- `results/mnist_lenet_summary.csv` and `results/mnist_lenet_contrasts.csv` (Table 1c and Appendix Table 6, LeNet)

You can also run each benchmark or the test suite individually:

```bash
python synthetic.py
python mnist.py --arch both
pytest -q
```

## Directory Layout

```text
.
├── reproduce_tables.py       # Recomputes and prints Paper Table 1(a)-(c) and saves CSVs in results/
├── synthetic.py              # Study 1: synthetic exact-ERM estimation and approximation experiments
├── mnist.py                  # Study 2: rotated-MNIST parity evaluation with frozen MLP and LeNet sources
├── checkpoints/
│   ├── mlp_source_{0,1,2}.pt     # Frozen upright-MNIST MLP source checkpoints
│   ├── lenet_source_{0,1,2}.pt   # Frozen upright-MNIST minimal-LeNet source checkpoints
│   └── hard_labels/              # Cached hard-label predictions (.npz) and stratified split indices
├── results/                  # Generated summary CSVs matching Table 1 and Appendix Tables 4-6
└── tests/
    └── test_reproduce.py     # Protocol and numerical reproduction tests
```

## Citation

```bibtex
@inproceedings{ye2026understanding,
  title     = {On the Statistical Learnability of Model Reprogramming: A Reachability and Relabeling Perspective},
  author    = {Ye, Zesheng and Chen, Pin-Yu and Liu, Feng},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## License

Code in this repository is released under the [MIT License](LICENSE). The MNIST dataset used in Study 2 is governed by the Creative Commons Attribution-ShareAlike 3.0 (CC BY-SA 3.0) license.
