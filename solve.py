#!/usr/bin/env python3
"""
I Dropped a Neural Net — Solver
================================
Recovers the correct ordering of 97 shuffled linear layers from a
48-block ResNet, given only the layer weights and 10,000 data points.

Pipeline:
  1. Classify pieces by shape into inp (96×48), out (48×96), last (1×48)
  2. Pair inp/out via diagonal dominance ratio + Hungarian algorithm
  3. Seed block ordering by ||W_out||_F
  4. Hill-climb (bubble sort) to minimize MSE

Usage:
  python solve.py

Requires: torch, pandas
"""

import sys, os, time, csv
sys.stdout.reconfigure(line_buffering=True)

import torch
import torch.nn as nn


PIECES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "historical_data_and_pieces", "pieces")
DATA_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "historical_data_and_pieces", "historical_data.csv")


# ── Architecture ─────────────────────────────────────────────────────────

class Block(nn.Module):
    """Residual block: x + W_out(ReLU(W_in @ x + b_in)) + b_out"""
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, in_dim)

    def forward(self, x):
        return x + self.out(torch.relu(self.inp(x)))


class LastLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.layer = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.layer(x)


# ── Hungarian algorithm (pure Python) ────────────────────────────────────

def hungarian(cost):
    """Solve the assignment problem: minimise total cost. Returns row→col mapping."""
    n = len(cost)
    INF = float('inf')
    u, v = [0.0] * (n + 1), [0.0] * (n + 1)
    p, way = [0] * (n + 1), [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    result = [0] * n
    for j in range(1, n + 1):
        result[p[j] - 1] = j - 1
    return result


# ── Utilities ────────────────────────────────────────────────────────────

def build_model(block_list, pieces, last_idx):
    """Assemble a full model from a list of (inp_idx, out_idx) pairs."""
    modules = []
    for inp_idx, out_idx in block_list:
        blk = Block(48, 96)
        blk.inp.weight = nn.Parameter(pieces[inp_idx]['weight'])
        blk.inp.bias   = nn.Parameter(pieces[inp_idx]['bias'])
        blk.out.weight = nn.Parameter(pieces[out_idx]['weight'])
        blk.out.bias   = nn.Parameter(pieces[out_idx]['bias'])
        modules.append(blk)
    ll = LastLayer(48, 1)
    ll.layer.weight = nn.Parameter(pieces[last_idx]['weight'])
    ll.layer.bias   = nn.Parameter(pieces[last_idx]['bias'])
    modules.append(ll)
    return nn.Sequential(*modules)


def eval_mse(block_list, pieces, last_idx, X, y):
    model = build_model(block_list, pieces, last_idx)
    model.eval()
    with torch.no_grad():
        return ((model(X).squeeze() - y) ** 2).mean().item()


# ── Step 1: Pair via diagonal dominance + Hungarian ──────────────────────

def pair_blocks(pieces, inp_pieces, out_pieces):
    """
    For each candidate (inp_i, out_j) pair, compute the diagonal dominance
    ratio of M = W_out @ W_in (Equation 1 in the paper):

        d(i, j) = |tr(W_out^(j) W_in^(i))| / ||W_out^(j) W_in^(i)||_F

    Well-trained ResNet blocks exhibit near-identity Jacobians (dynamic
    isometry), causing M for correctly paired layers to have a dominant
    negative diagonal. This ratio is maximised for true pairs, so we
    solve a maximum-weight bipartite matching via the Hungarian algorithm.
    """
    n = len(inp_pieces)
    ratio_matrix = [[0.0] * n for _ in range(n)]

    for i in range(n):
        W_in = pieces[inp_pieces[i]]['weight']
        for j in range(n):
            W_out = pieces[out_pieces[j]]['weight']
            M = W_out @ W_in
            ratio_matrix[i][j] = abs(M.trace().item()) / (M.norm().item() + 1e-10)

    mx = max(ratio_matrix[i][j] for i in range(n) for j in range(n)) + 1.0
    cost = [[mx - ratio_matrix[i][j] for j in range(n)] for i in range(n)]
    asgn = hungarian(cost)

    paired = [(inp_pieces[i], out_pieces[asgn[i]]) for i in range(n)]
    matched_ratios = [ratio_matrix[i][asgn[i]] for i in range(n)]
    return paired, matched_ratios


# ── Step 2: Seed ordering by ||W_out||_F ─────────────────────────────────

def seed_order(paired, pieces):
    """
    Sort blocks by the Frobenius norm of their output weight. In ResNets,
    earlier blocks tend to make smaller perturbations, providing a rough
    depth proxy.
    """
    decorated = []
    for idx, (inp_idx, out_idx) in enumerate(paired):
        norm = pieces[out_idx]['weight'].norm().item()
        decorated.append((norm, idx))
    decorated.sort()
    return [paired[i] for _, i in decorated]


# ── Step 3: Hill-climb via bubble sort ───────────────────────────────────

def hillclimb(order, pieces, last_idx, X_sub, y_sub, max_rounds=20):
    """
    Iteratively swap adjacent blocks if the swap decreases MSE.
    Then try wider gap swaps (2–5) for any remaining misplacements.
    """
    current = list(order)
    cur_mse = eval_mse(current, pieces, last_idx, X_sub, y_sub)
    total_swaps = 0

    for rnd in range(1, max_rounds + 1):
        rnd_swaps = 0
        for i in range(len(current) - 1):
            trial = list(current)
            trial[i], trial[i + 1] = trial[i + 1], trial[i]
            m = eval_mse(trial, pieces, last_idx, X_sub, y_sub)
            if m < cur_mse - 1e-10:
                current = trial; cur_mse = m; rnd_swaps += 1
        for i in range(len(current) - 2, -1, -1):
            trial = list(current)
            trial[i], trial[i + 1] = trial[i + 1], trial[i]
            m = eval_mse(trial, pieces, last_idx, X_sub, y_sub)
            if m < cur_mse - 1e-10:
                current = trial; cur_mse = m; rnd_swaps += 1

        total_swaps += rnd_swaps
        print(f"    Round {rnd:2d}: {rnd_swaps:3d} swaps, MSE = {cur_mse:.10f}")
        if rnd_swaps == 0:
            print(f"    Converged.")
            break

    for gap in range(2, 6):
        for i in range(len(current) - gap):
            trial = list(current)
            trial[i], trial[i + gap] = trial[i + gap], trial[i]
            m = eval_mse(trial, pieces, last_idx, X_sub, y_sub)
            if m < cur_mse - 1e-10:
                current = trial; cur_mse = m; total_swaps += 1

    return current, cur_mse, total_swaps


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  I DROPPED A NEURAL NET — SOLVER")
    print("=" * 70)

    # Load data
    rows_X, rows_pred = [], []
    with open(DATA_CSV) as f:
        for row in csv.DictReader(f):
            rows_X.append([float(row[f"measurement_{i}"]) for i in range(48)])
            rows_pred.append(float(row["pred"]))
    X = torch.tensor(rows_X, dtype=torch.float32)
    y_pred = torch.tensor(rows_pred, dtype=torch.float32)
    print(f"\nData: {X.shape[0]} samples x {X.shape[1]} features")

    # Load pieces
    pieces = {}
    for i in range(97):
        pieces[i] = torch.load(os.path.join(PIECES_DIR, f"piece_{i}.pth"),
                                map_location="cpu", weights_only=True)

    inp_pieces = sorted(i for i in pieces if pieces[i]['weight'].shape == torch.Size([96, 48]))
    out_pieces = sorted(i for i in pieces if pieces[i]['weight'].shape == torch.Size([48, 96]))
    last_piece = next(i for i in pieces if pieces[i]['weight'].shape == torch.Size([1, 48]))

    print(f"Pieces: {len(inp_pieces)} inp (96x48), "
          f"{len(out_pieces)} out (48x96), 1 last (1x48)")

    # ── Step 1 ──
    print(f"\n{'─' * 70}")
    print("Step 1: Pair layers via diagonal dominance + Hungarian algorithm")
    print(f"{'─' * 70}")
    t0 = time.time()
    paired, ratios = pair_blocks(pieces, inp_pieces, out_pieces)
    t1 = time.time()
    print(f"  Time: {t1 - t0:.2f}s")
    print(f"  Matched ratios: min={min(ratios):.3f}, max={max(ratios):.3f}, "
          f"mean={sum(ratios)/len(ratios):.3f}")

    # ── Step 2 ──
    print(f"\n{'─' * 70}")
    print("Step 2: Seed initial order by ||W_out||_F")
    print(f"{'─' * 70}")
    ordered = seed_order(paired, pieces)
    X_sub, y_sub = X[:1000], y_pred[:1000]
    seed_mse = eval_mse(ordered, pieces, last_piece, X_sub, y_sub)
    print(f"  Seed MSE (N=1000): {seed_mse:.6f}")

    # ── Step 3 ──
    print(f"\n{'─' * 70}")
    print("Step 3: Hill-climb (bubble sort + gap swaps)")
    print(f"{'─' * 70}")
    t2 = time.time()
    ordered, final_sub_mse, total_swaps = hillclimb(
        ordered, pieces, last_piece, X_sub, y_sub)
    t3 = time.time()
    print(f"  Total swaps: {total_swaps}, time: {t3 - t2:.1f}s")

    # ── Final ──
    final_mse = eval_mse(ordered, pieces, last_piece, X, y_pred)
    print(f"\n{'=' * 70}")
    print("  RESULT")
    print(f"{'=' * 70}")
    print(f"  Total time:  {t3 - t0:.1f}s")
    print(f"  Final MSE:   {final_mse:.12f}")

    flat = []
    for inp_idx, out_idx in ordered:
        flat.append(inp_idx)
        flat.append(out_idx)
    flat.append(last_piece)
    print(f"  Permutation: [{', '.join(str(x) for x in flat)}]")

    if final_mse < 1e-10:
        print(f"\n  PERFECT RECONSTRUCTION (MSE = 0)")
    else:
        print(f"\n  MSE is non-zero — reconstruction may be incomplete.")

    return flat, final_mse


if __name__ == "__main__":
    main()
