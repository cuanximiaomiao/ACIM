"""Synthetic benchmarks S1-S4 for causal discovery and interaction modeling experiments.

Four scenarios covering the linear/nonlinear, Gaussian/non-Gaussian and
continuous/mixed-variable combinations:

    Scenario                  d     n     Edges (mean +/- SD)   Noise                       Nonlinearity     Variables
    S1 linear Gaussian        20    500   58.4 +/- 4.6          N(0, 1)                     none             continuous
    S2 linear non-Gaussian    20    500   58.4 +/- 4.6          Exp(1) - 1                  none             continuous
    S3 nonlinear              20    500   58.4 +/- 4.6          N(0, 0.3)                   tanh             continuous
    S4 mixed high-dimensional 50    1000  238.9 +/- 8.6         cont. N(0, 0.3); bin. Bern. tanh + sigmoid   25 cont. + 25 bin.

The edge counts above are the empirical means over the graphs produced by this
generator; the expected counts are C(d, 2) * p = 57 for d = 20, p = 0.3 and 245 for
d = 50, p = 0.2.

Construction. The skeleton is an Erdos-Renyi random graph: each of the d(d-1)/2
candidate directed edges is present independently with probability p, and the node
labels are then permuted. Writing pa(j) for the parent set of X_j and W_ij for the
entries of the weighted adjacency matrix, drawn from [0.5, 1] or [-1, -0.5] with
equal probability:

    S1, S2:  X_j = sum_{i in pa(j)} W_ij X_i + eps_j
    S3:      X_j = tanh( sum_{i in pa(j)} W_ij X_i ) + eps_j
    S4:      X_j = tanh( sum_{i in pa(j)} W_ij X_i ) + eps_j          (continuous half)
             X_j ~ Bernoulli( sigmoid( sum_{i in pa(j)} W_ij X_i ) ) (binary half)

with eps_j ~ N(0, 1) in S1, eps_j ~ Exp(1) - 1 in S2, and eps_j ~ N(0, 0.3) in S3 and
S4. Variables without parents take the sum over parents to be 0.

Each scenario is drawn with its own seed. The order in which the random number
generator is consumed is fixed, so a given seed reproduces the same graph, the same
edge weights and the same observations.

Usage
-----
    python synthetic_data.py --out synthetic --seeds 10

writes data and ground-truth graphs for every scenario and seed, and prints the edge
counts so they can be checked against the table above.
"""
import argparse
import json
import os

import numpy as np
from scipy.special import expit as sigmoid

# Candidate edge weights: drawn with equal probability from these two ranges.
POSITIVE_RANGE = (0.5, 1.0)
NEGATIVE_RANGE = (-1.0, -0.5)

SCENARIOS = ("s1", "s2", "s3", "s4")


def generate_dag(d, edge_prob, seed):
    """Erdos-Renyi skeleton, upper-triangular, then a random permutation of the nodes.

    Args:
        d: number of variables.
        edge_prob: inclusion probability of each candidate directed edge.
        seed: seed for the graph structure.

    Returns:
        B: (d, d) binary adjacency matrix, B[i, j] = 1 iff i -> j.
    """
    rng = np.random.RandomState(seed)

    upper = (rng.rand(d, d) < edge_prob).astype(float)
    upper = np.triu(upper, k=1)  # strictly upper triangular, hence acyclic

    perm = rng.permutation(d)
    return upper[perm][:, perm]


def generate_weights(B, seed):
    """Draw an edge weight for every edge of B from POSITIVE_RANGE or NEGATIVE_RANGE."""
    rng = np.random.RandomState(seed)
    d = B.shape[0]
    W = np.zeros((d, d))

    for i in range(d):
        for j in range(d):
            if B[i, j] == 1:
                if rng.rand() > 0.5:
                    W[i, j] = rng.uniform(*POSITIVE_RANGE)
                else:
                    W[i, j] = rng.uniform(*NEGATIVE_RANGE)
    return W


def topological_order(B):
    """Kahn's algorithm; falls back to ordering by in-degree if a cycle is present."""
    d = B.shape[0]
    in_degree = (B != 0).sum(axis=0).astype(int)
    order = []
    queue = [i for i in range(d) if in_degree[i] == 0]

    while queue:
        v = queue.pop(0)
        order.append(v)
        for u in range(d):
            if B[v, u] != 0:
                in_degree[u] -= 1
                if in_degree[u] == 0:
                    queue.append(u)

    if len(order) < d:
        remaining = sorted(set(range(d)) - set(order), key=lambda x: in_degree[x])
        order.extend(remaining)
    return order


def _parent_term(X, W, B, j, n):
    """sum_{i in pa(j)} W_ij X_i, or zeros when j has no parents."""
    parents = np.where(B[:, j] == 1)[0]
    if len(parents) == 0:
        return np.zeros(n)
    return X[:, parents] @ W[parents, j]


def generate_scenario1(d=20, n=500, edge_prob=0.3, seed=42):
    """S1: linear, Gaussian noise.  X_j = sum_i W_ij X_i + N(0, 1)."""
    rng = np.random.RandomState(seed)
    B = generate_dag(d, edge_prob, seed)
    W = generate_weights(B, seed)

    X = np.zeros((n, d))
    for j in topological_order(B):
        X[:, j] = _parent_term(X, W, B, j, n) + rng.randn(n)

    meta = {"scenario": "s1_linear_gaussian", "d": d, "n": n,
            "edge_prob": edge_prob, "noise": "N(0,1)", "nonlinearity": "none",
            "variable_types": "continuous"}
    return X, B, meta


def generate_scenario2(d=20, n=500, edge_prob=0.3, seed=42):
    """S2: linear, non-Gaussian noise.  X_j = sum_i W_ij X_i + (Exp(1) - 1)."""
    rng = np.random.RandomState(seed)
    B = generate_dag(d, edge_prob, seed)
    W = generate_weights(B, seed)

    X = np.zeros((n, d))
    for j in topological_order(B):
        noise = rng.exponential(scale=1.0, size=n) - 1.0
        X[:, j] = _parent_term(X, W, B, j, n) + noise

    meta = {"scenario": "s2_linear_nongaussian", "d": d, "n": n,
            "edge_prob": edge_prob, "noise": "Exp(1)-1", "nonlinearity": "none",
            "variable_types": "continuous"}
    return X, B, meta


def generate_scenario3(d=20, n=500, edge_prob=0.3, seed=42):
    """S3: nonlinear, Gaussian noise.  X_j = tanh(sum_i W_ij X_i) + N(0, 0.3)."""
    rng = np.random.RandomState(seed)
    B = generate_dag(d, edge_prob, seed)
    W = generate_weights(B, seed)

    X = np.zeros((n, d))
    for j in topological_order(B):
        X[:, j] = np.tanh(_parent_term(X, W, B, j, n)) + rng.randn(n) * 0.3

    meta = {"scenario": "s3_nonlinear", "d": d, "n": n, "edge_prob": edge_prob,
            "noise": "N(0,0.3)", "nonlinearity": "tanh",
            "variable_types": "continuous"}
    return X, B, meta


def generate_scenario4(d=50, n=1000, edge_prob=0.2, seed=42):
    """S4: mixed variables, nonlinear, high-dimensional.

    Half the variables are continuous, X_j = tanh(sum_i W_ij X_i) + N(0, 0.3);
    the other half are binary, X_j ~ Bernoulli(sigmoid(sum_i W_ij X_i)).
    """
    rng = np.random.RandomState(seed)
    B = generate_dag(d, edge_prob, seed)
    W = generate_weights(B, seed)

    n_binary = d // 2
    binary_vars = rng.choice(d, n_binary, replace=False)
    binary_mask = np.zeros(d, dtype=bool)
    binary_mask[binary_vars] = True

    X = np.zeros((n, d))
    for j in topological_order(B):
        z = _parent_term(X, W, B, j, n)
        if binary_mask[j]:
            X[:, j] = (rng.rand(n) < sigmoid(z)).astype(float)
        else:
            X[:, j] = np.tanh(z) + rng.randn(n) * 0.3

    meta = {"scenario": "s4_mixed_highdim", "d": d, "n": n, "edge_prob": edge_prob,
            "noise": "continuous N(0,0.3); binary Bernoulli(sigmoid(.))",
            "nonlinearity": "tanh + sigmoid", "variable_types": "mixed",
            "n_binary": int(n_binary), "n_continuous": int(d - n_binary)}
    return X, B, meta


GENERATORS = {
    "s1": generate_scenario1,
    "s2": generate_scenario2,
    "s3": generate_scenario3,
    "s4": generate_scenario4,
}

# Default dimensions per scenario, as used for the reported benchmarks.
DEFAULTS = {
    "s1": dict(d=20, n=500, edge_prob=0.3),
    "s2": dict(d=20, n=500, edge_prob=0.3),
    "s3": dict(d=20, n=500, edge_prob=0.3),
    "s4": dict(d=50, n=1000, edge_prob=0.2),
}


def generate(scenario, seed=42, **overrides):
    """Generate one scenario. Returns (X, B, meta)."""
    key = scenario.lower()
    if key not in GENERATORS:
        raise ValueError(f"unknown scenario '{scenario}'; expected one of {SCENARIOS}")
    kwargs = dict(DEFAULTS[key])
    kwargs.update(overrides)
    return GENERATORS[key](seed=seed, **kwargs)


def edge_statistics(n_seeds=10, seed0=42):
    """Mean +/- SD of the number of edges, per scenario, over n_seeds graphs."""
    stats = {}
    for key in SCENARIOS:
        counts = [int(generate(key, seed=seed0 + i)[1].sum()) for i in range(n_seeds)]
        stats[key] = (float(np.mean(counts)), float(np.std(counts)))
    return stats


def main():
    p = argparse.ArgumentParser(description="Generate the S1-S4 synthetic benchmarks")
    p.add_argument("--out", default="synthetic", help="output directory")
    p.add_argument("--seeds", type=int, default=10, help="number of seeds per scenario")
    p.add_argument("--seed0", type=int, default=42, help="first seed")
    args = p.parse_args()

    print(f"=== synthetic benchmarks | {args.seeds} seeds ===")
    print(f"{'scenario':<8}{'d':>4}{'n':>6}   {'edges (mean +/- SD)':<22}")

    for key in SCENARIOS:
        counts = []
        for i in range(args.seeds):
            seed = args.seed0 + i
            X, B, meta = generate(key, seed=seed)
            counts.append(int(B.sum()))

            out_dir = os.path.join(args.out, key, f"seed{seed}")
            os.makedirs(out_dir, exist_ok=True)
            np.save(os.path.join(out_dir, "data.npy"), X)
            np.save(os.path.join(out_dir, "B_true.npy"), B)
            with open(os.path.join(out_dir, "meta.json"), "w") as f:
                json.dump({**meta, "seed": seed, "n_edges": int(B.sum())}, f, indent=2)

        print(f"{key.upper():<8}{meta['d']:>4}{meta['n']:>6}   "
              f"{np.mean(counts):>6.1f} +/- {np.std(counts):<6.1f}")

    print(f"\nwritten under {args.out}/<scenario>/seed<k>/ "
          f"(data.npy, B_true.npy, meta.json)")


if __name__ == "__main__":
    main()
