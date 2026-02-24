# I Dropped a Neural Net

A solver for the Jane Street puzzle: given 97 shuffled linear layers from a 48-block ResNet and 10,000 data points, recover the exact original ordering.

The combined search space is $(48!)^2 \approx 10^{122}$ — more than the atoms in the observable universe. Our solver reconstructs the network in under 30 seconds with **MSE = 0**.

## The Problem

A 48-block Residual Network was trained on a financial dataset. Each block consists of two linear layers (input projection 48→96 and output projection 96→48) connected by ReLU, with a residual connection. There is also a final linear layer (48→1) that produces the prediction.

All 97 layers were extracted and shuffled. The task: put them back in order.

## The Solution

### Step 1 — Pairing (weight-only, <1s)

Dynamic isometry in well-trained ResNets causes the product $W_{out} W_{in}$ for correctly paired layers to exhibit a dominant negative diagonal. We compute the **diagonal dominance ratio** (Eq. 1 in the paper):

$$d(i, j) = \frac{|\text{tr}(W_{out}^{(j)} W_{in}^{(i)})|}{\|W_{out}^{(j)} W_{in}^{(i)}\|_F}$$

for all 48×48 candidate pairings, then solve a maximum-weight bipartite matching via the **Hungarian algorithm**. This yields all 48 correct pairs.

### Step 2 — Seed ordering (weight-only, instant)

Sort blocks by $\|W_{out}\|_F$ (Frobenius norm of the output weight). Earlier blocks in a ResNet tend to make smaller perturbations, providing a rough depth proxy.

### Step 3 — Hill-climb (~20s)

Starting from the seed ordering, iteratively swap adjacent blocks if the swap decreases the MSE on a 1,000-sample subset. Then try wider gap swaps (gaps 2–5) for any remaining misplacements.

## Architecture

```
Input (48) → [Block_1 → Block_2 → ... → Block_48] → LastLayer → Output (1)

Block_k:  x → x + W_out(ReLU(W_in @ x + b_in)) + b_out
          W_in: (96, 48)   W_out: (48, 96)
```

## Usage

```bash
pip install torch pandas
python solve.py
```

### Expected output

```
Step 1: Pair layers via diagonal dominance + Hungarian algorithm
  Time: 0.05s
  Matched ratios: min=1.764, max=3.232, mean=2.785

Step 2: Seed initial order by ||W_out||_F
  Seed MSE (N=1000): 0.075716

Step 3: Hill-climb (bubble sort + gap swaps)
    Round  1:  52 swaps, MSE = 0.0021263792
    Round  2:  15 swaps, MSE = 0.0003789639
    Round  3:   5 swaps, MSE = 0.0000000000
    Converged.

  RESULT
  Total time:  10.4s
  Final MSE:   0.000000000000
  Permutation: [43, 34, 65, 22, 69, 89, 28, 12, 27, 76, 81, 8, 5, 21,
                62, 79, 64, 70, 94, 96, 4, 17, 48, 9, 23, 46, 14, 33,
                95, 26, 50, 66, 1, 40, 15, 67, 41, 92, 16, 83, 77, 32,
                10, 20, 3, 53, 45, 19, 87, 71, 88, 54, 39, 38, 18, 25,
                56, 30, 91, 29, 44, 82, 35, 24, 61, 80, 86, 57, 31, 36,
                13, 7, 59, 52, 68, 47, 84, 63, 74, 90, 0, 75, 73, 11,
                37, 6, 58, 78, 42, 55, 49, 72, 2, 51, 60, 93, 85]

  PERFECT RECONSTRUCTION (MSE = 0)
```

## Repository Structure

```
├── solve.py                              # The solver
├── README.md
├── requirements.txt
└── historical_data_and_pieces/
    ├── historical_data.csv               # 10,000 samples × 48 features + targets
    └── pieces/
        ├── piece_0.pth ... piece_96.pth  # The 97 shuffled layers
```

## Paper

Hyunwoo Park. *I Dropped a Neural Net.* 2026. [[arXiv:2602.19845]](https://arxiv.org/abs/2602.19845)

## License

MIT
