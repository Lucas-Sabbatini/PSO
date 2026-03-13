# HCEPSO Implementation Plan
## Two-Dimensional Bin-Packing Problem with Conflicts and Load Balancing

**Reference:** Sun, Li, Wang, Xie (2025). *Computers & Industrial Engineering*, 200, 110851.

---

## 1. Problem Overview

The 2D-BPPCL (2D Bin-Packing Problem with Conflicts and Load Balancing) aims to pack `n` rectangular items into the minimum number of bins `P`, subject to:

- Each item assigned to exactly one bin
- No geometric overlap between items inside a bin
- Items fit within bin dimensions `W × H`
- Total item weight per bin ≤ `Q`
- Center-of-mass displacement ≤ `∆` (load balancing)
- Conflicting item pairs `g_{ij} = 1` cannot share a bin

---

## 2. Project Structure

```
hcepso/
│
├── main.py                  # Entry point: run experiments, collect results
├── config.py                # All hyperparameters and experiment settings
│
├── instance/
│   ├── item.py              # Item dataclass (id, w, h, weight, rotation flag)
│   ├── instance.py          # Instance dataclass (items, bin dims, conflict graph)
│   └── loader.py            # Parse Bengtsson dataset + synthetic dimension generation
│
├── pso/
│   ├── particle.py          # Particle: position X, velocity V, pBest
│   ├── swarm.py             # Swarm: population, gBest, PSO update equations
│   ├── velocity.py          # Velocity update: ω decay, c1·r1·(pBest-X) + c2·r2·(gBest-X)
│   └── encoding.py          # Encoding/decoding: integer array → permutation → packing order
│
├── operators/
│   ├── crossover.py         # Elitist crossover (gBest × random particle, 2-point)
│   ├── mutation.py          # Mutation type 1 (swap genes) and type 2 (flip sign)
│   └── chaos.py             # Chaotic local search via Logistic Map (μ=4)
│
├── packing/
│   ├── heuristic.py         # 2D packing heuristic (Bottom-Left or Guillotine cut)
│   └── fitness.py           # Fitness function: f = Σ(W·H - ΣE_ik) + α·PC_k + λ·w_k + δ·b_k
│
└── results/
    ├── reporter.py          # Collect, average, print results per β and n
    └── logger.py            # CSV / log file output
```

---

## 3. Module Descriptions

### 3.1 `config.py`
Centralized configuration. No magic numbers elsewhere.

```python
# PSO hyperparameters (Table 2, Sun et al. 2025)
POPULATION     = 10
MAX_ITER       = 200        # k_max — only stopping criterion
C1             = 2.0        # cognitive coefficient
C2             = 2.0        # social coefficient
OMEGA_INI      = 1.2        # initial inertia weight
OMEGA_FIN      = 0.8        # final inertia weight

# Fitness penalty weights
ALPHA          = 1000       # conflict penalty
LAMBDA         = 100        # load imbalance penalty
DELTA          = 100        # center-of-mass displacement penalty

# Chaotic search
MU             = 4.0        # Logistic Map parameter

# Bin dimensions (to be set per dataset)
BIN_W          = 100
BIN_H          = 100
BIN_Q          = ...        # from dataset C field
BIN_DELTA_MAX  = ...        # max barycenter displacement

# Experiment settings
N_RUNS         = 10
BETA_VALUES    = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
```

---

### 3.2 `instance/item.py`

```python
@dataclass
class Item:
    id: int
    width: float          # geometric width (synthetic if from Bengtsson)
    height: float         # geometric height (synthetic if from Bengtsson)
    weight: float         # from dataset field w_i
    can_rotate: bool      # flag: item can be placed rotated 90°
```

---

### 3.3 `instance/loader.py`
Reads the Bengtsson `.txt` format and augments with synthetic geometry.

**Dataset format (from `0_format_description.txt`):**
```
N   C
i   w   a1  a2  ...  ak
```

**Synthetic geometry generation:**  
Since the Bengtsson dataset provides only `(weight, conflicts)`, dimensions `(w_i, h_i)` must be generated. Two strategies:

- **Uniform:** `w_i, h_i ~ Uniform(10, 50)` — independent of weight
- **Correlated:** `w_i * h_i ∝ weight_i` — heavier items occupy more area

Use a fixed random seed per instance for reproducibility.

```python
def load_instance(filepath: str, seed: int = 42) -> Instance:
    # 1. Parse N, C, weights, conflict list
    # 2. Generate (width, height) synthetically
    # 3. Build conflict adjacency matrix g[i][j]
    # 4. Return Instance object
```

---

### 3.4 `pso/encoding.py`

The particle position is an integer array `X` of size `n`:

```
X = [2, 1, 3, -4, 7, 9, 5, 6, -8, 10]
```

- **Value at position `p`:** item ID to be placed at step `p`
- **Negative value:** item is rotated 90° before placement
- **Decoding:** iterate through `X`, place each item using the 2D packing heuristic

```python
def decode(X: np.ndarray, items: list[Item]) -> list[Bin]:
    """Convert particle position to bin assignment via sequential placement."""

def encode_random(n: int) -> np.ndarray:
    """Initialize a random valid particle: permutation of ±[1..n]."""
```

> **Implementation note:** The article uses integer encoding and references Liu et al. (2008) for the PSO adaptation to permutation space. The velocity equation is applied as continuous arithmetic; the resulting real-valued array is then converted to integer permutation by ranking. Negative values below a threshold indicate rotation.

---

### 3.5 `pso/velocity.py`

Velocity update with linearly decaying inertia:

```
V^{k+1} = ω^k · V^k + c1·r1·(pBest - X^k) + c2·r2·(gBest - X^k)
X^{k+1} = X^k + V^{k+1}
```

Inertia decay:
```
ω^k = ω_ini - (ω_ini - ω_fin) / k_max · k
```

```python
def update_velocity(V, X, pBest, gBest, omega, c1, c2) -> np.ndarray:
    r1, r2 = np.random.rand(len(X)), np.random.rand(len(X))
    return omega * V + c1 * r1 * (pBest - X) + c2 * r2 * (gBest - X)

def update_position(X, V) -> np.ndarray:
    return X + V

def decay_omega(k, k_max, omega_ini, omega_fin) -> float:
    return omega_ini - (omega_ini - omega_fin) / k_max * k
```

---

### 3.6 `operators/crossover.py`

**Elitist 2-point crossover:**

1. Parent 1 = `pBest` from the random particle selected
2. Parent 2 = random particle from swarm
3. Select two random cut points `c1 < c2`
4. Offspring = `pBest[c1:c2]` + remaining genes of Parent 2 (in order, removing duplicates)
5. Accept offspring if `f(offspring) < f(parent2)`

```python
def elitist_crossover(pBest: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    """Returns offspring. Handles sign (rotation) preservation."""
```

---

### 3.7 `operators/mutation.py`

Two mutation types applied with some probability:

- **Type 1 — Swap:** select positions `i ≠ j`, swap `X[i] ↔ X[j]` (reorders packing sequence)
- **Type 2 — Flip sign:** select position `i`, set `X[i] = -X[i]` (toggles 90° rotation)

```python
def mutation_swap(X: np.ndarray) -> np.ndarray:
    """Swap two random genes."""

def mutation_flip_sign(X: np.ndarray) -> np.ndarray:
    """Invert sign of one random gene."""
```

---

### 3.8 `operators/chaos.py`

Chaotic local search using the Logistic Map to escape local optima:

```
X_{l+1} = μ · X_l · (1 - X_l),   μ = 4
```

Applied in a neighborhood `[X* - R, X* + R]` around each particle's current best.

```python
def logistic_map(x: float, mu: float = 4.0) -> float:
    return mu * x * (1 - x)

def chaotic_search(X: np.ndarray, fitness_fn, n_steps: int = 10) -> np.ndarray:
    """Generate chaotic sequence around X; return best found solution."""
```

---

### 3.9 `packing/heuristic.py`

Converts the decoded permutation into actual bin assignments with 2D coordinates.

**Recommended approach: Bottom-Left Fill (BLF)**

1. For each item in permutation order:
   - Try to place it in current bin at the lowest-leftmost feasible position
   - Check: no overlap, fits within `W × H`, no conflict with existing items in bin
   - If cannot fit: open new bin
2. Return list of bins with item placements and `(x, y)` coordinates

```python
def pack_items(permutation: list[Item], bin_W, bin_H, conflicts) -> list[Bin]:
    """Bottom-Left Fill heuristic for 2D bin packing with conflicts."""
```

---

### 3.10 `packing/fitness.py`

```
f = Σ_k [ W·H - Σ_i E_{ik} + α·PC_k + λ·w_k + δ·b_k ]
```

Where:
- `W·H - Σ E_{ik}` = unused area in bin `k`
- `PC_k` = number of conflict pairs in bin `k` (should be 0 if feasible)
- `w_k` = weight constraint violation in bin `k`
- `b_k` = center-of-mass displacement violation in bin `k`

```python
def compute_fitness(bins: list[Bin], W, H, Q, delta_max,
                    alpha, lambda_, delta) -> float:
```

---

### 3.11 `pso/swarm.py`

Main HCEPSO loop:

```python
def run(instance: Instance, config: Config) -> Solution:
    # 1. Initialize M particles (encode_random)
    # 2. Evaluate fitness for all particles
    # 3. Set pBest[i] = X[i], gBest = argmin fitness
    # 4. For k in range(k_max):               ← ONLY stopping criterion
    #     a. Update ω (decay)
    #     b. For each particle i:
    #         - update_velocity(V[i], X[i], pBest[i], gBest)
    #         - update_position(X[i], V[i])
    #         - elitist_crossover(gBest, X[i])  → maybe replace X[i]
    #         - mutation_swap or mutation_flip_sign
    #         - chaotic_search(X[i])
    #         - evaluate fitness(X[i])
    #         - update pBest[i] if improved
    #     c. Update gBest
    # 5. Return gBest as final solution
```

---

## 4. Dataset Adaptation Pipeline

```
Bengtsson .txt
      │
      ▼
loader.py  ──────────────────────────────────────────────┐
  - parse N, C, w_i, conflicts                           │
  - generate (width_i, height_i) synthetically           │
  - seed-controlled for reproducibility                   │
      │                                                   │
      ▼                                                   ▼
Instance object                               250 / 500 item instances
  (n=120, β ∈ {0.1,...,0.9})                (generated with same distributions)
```

**Synthetic dimension generation strategy:**
```python
rng = np.random.default_rng(seed)
widths  = rng.integers(10, 51, size=N)   # Uniform[10, 50]
heights = rng.integers(10, 51, size=N)   # Uniform[10, 50]
# Item weight already provided by dataset
```

---

## 5. Experiment Setup

| Parameter | Value |
|-----------|-------|
| Dataset 1 | Bengtsson (1982), n=120 |
| Dataset 2 | Synthetic, n=250 |
| Dataset 3 | Synthetic, n=500 |
| Conflict degrees β | 0.1, 0.2, ..., 0.9 |
| Runs per instance | 10 |
| Max iterations | 200 |
| Population size | 10 |

**Metrics to collect (matching Table 3–5 of the paper):**
- Best fitness found per run
- Average fitness over 10 runs
- Standard deviation over 10 runs
- Number of bins used (feasible solution)

---

## 6. Known Open Questions / Limitations in the Paper

1. **PSO on permutation space:** the paper applies the continuous velocity equation to an integer encoding. The exact discretization method is not explicitly described. Recommended approach: treat `X` as real-valued internally, derive permutation by argsort ranking, use sign for rotation.

2. **Single stopping criterion:** only `k_max = 200` is used. No early stopping on stagnation. This is computationally inefficient for easy instances; consider adding an optional stagnation check.

3. **Packing heuristic not specified:** the article does not state which 2D placement heuristic is used after decoding the permutation. Bottom-Left Fill is the standard choice for this class of problems.

4. **Rotation policy:** it is unclear whether all items or only a subset are rotation-eligible. Treat all as rotation-eligible unless otherwise specified.

---

## 7. Suggested Implementation Order

1. `config.py` — define all constants
2. `instance/item.py` + `instance/instance.py` — data structures
3. `instance/loader.py` — parse Bengtsson + synthetic geometry
4. `packing/heuristic.py` — BLF packing (testable independently)
5. `packing/fitness.py` — evaluate a solution
6. `pso/encoding.py` — encode/decode particles
7. `pso/velocity.py` — PSO update equations
8. `operators/crossover.py` + `operators/mutation.py`
9. `operators/chaos.py` — chaotic local search
10. `pso/swarm.py` — full HCEPSO loop
11. `results/reporter.py` + `main.py` — experiments and output
