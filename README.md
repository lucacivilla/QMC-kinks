# QMC with Kinks — Asian Option Pricing via Monte Carlo & Quasi-Monte Carlo

**Authors:** Burzio, Civilla, Vinciguerra

A Python implementation of Monte Carlo (MC) and Quasi-Monte Carlo (QMC) methods for pricing **arithmetic** and **digital Asian options**, with a focus on variance reduction in the presence of discontinuities ("kinks") in the payoff function.

---

## Overview

Asian options have payoffs that depend on the arithmetic average of an asset's price over time, making them path-dependent and high-dimensional. This project investigates how different variance reduction strategies improve the convergence rate of Monte Carlo estimators, particularly for out-of-the-money (OTM) options where the payoff surface has a kink or a sharp discontinuity.

Two option types are studied:

| Option | Payoff |
|--------|--------|
| **Arithmetic Asian** | `e^{-rT} * max(S_mean - K, 0)` |
| **Digital Asian** | `e^{-rT} * 1[S_mean > K]` |

---

## Methods Compared

### Base estimators
- **Crude MC** — standard Monte Carlo with i.i.d. normal samples
- **Plain RQMC** — Randomized QMC using scrambled Sobol sequences

### Variance reduction techniques
- **ODIS** (Optimal Drift Importance Sampling) — shifts the sampling distribution toward the exercise region by solving an optimization problem for the optimal drift vector μ*
- **Pre-Integration (z1)** — analytically integrates out the first coordinate, smoothing the discontinuity
- **Pre-Integration (AS)** — same as above but along the **active subspace** direction, the leading eigenvector of the gradient covariance matrix `C = E[∇Ψ ∇Ψᵀ]`
- Combined methods: **RQMC + ODIS**, **Pre-Int (AS) + ODIS**

For OTM options (K=120), ODIS is applied on top of RQMC and pre-integration to further reduce variance.

---

## Model Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `S0` | 100 | Initial stock price |
| `T` | 1.0 | Time to maturity (years) |
| `r` | 0.1 | Risk-free interest rate |
| `sigma` | 0.1 | Volatility |
| `K` | 100 / 120 | Strike price (ATM / OTM) |

### Simulation settings

| Parameter | Value |
|-----------|-------|
| Dimensions `d` | 32, 64, 128, 256, 512 |
| Sample sizes `N` | 2⁷ to 2¹³ (128 – 8192) |
| Repetitions | 50 (for stable RMSE estimation) |

---

## Path Generation

Asset paths are simulated using Geometric Brownian Motion over `d` time steps. The covariance matrix Σ ∈ ℝ^{d×d} with entries `Σ_{ij} = min(t_i, t_j)` is factored via **Cholesky decomposition** (Σ = AAᵀ), and paths are generated as:

```
W = A z,    z ~ N(0, I)
S_{t_i} = S0 * exp((r - σ²/2) t_i + σ W_{t_i})
```

---

## Repository Structure

```
QMC-kinks/
├── arithmetic_asian_option_final.py   # Pricing for arithmetic Asian options
├── digital_asian_option_final.py      # Pricing for digital Asian options
├── plots_arithmetic_asian_final/      # Output plots & CSVs (arithmetic)
│   ├── d_32/
│   ├── d_64/
│   ├── d_128/
│   ├── d_256/
│   └── d_512/
├── plots_digital_asian_final/         # Output plots & CSVs (digital)
│   ├── d_32/
│   ├── d_64/
│   ├── d_128/
│   ├── d_256/
│   └── d_512/
└── Burzio_Civilla_Vinciguerra_Report.pdf
```

Each dimension subfolder contains:
- **Convergence plots** — log-log RMSE vs N, with fitted convergence rates
- **Efficiency plots** — RMSE vs average computation time
- **CSV results** — full numerical results per method
- **Required N** — minimum sample size to reach 1% relative error (OTM case)

---

## Running the Simulations

### Prerequisites

```bash
pip install numpy scipy pandas matplotlib tqdm
```

### Arithmetic Asian option

```bash
python arithmetic_asian_option_final.py
```

### Digital Asian option

```bash
python digital_asian_option_final.py
```

Each script loops over all dimensions, runs experiments for K=100 and K=120, saves results to CSV, and generates plots. It also finds the minimum sample size N required to achieve a 1% relative error for the OTM case using a sequential stopping criterion.

---

## Output

For each dimension `d` and strike `K`, the scripts produce:

- **`*_K100_d{d}_Analysis.png`** — convergence + efficiency plots (ATM)
- **`*_K120_d{d}_Comprehensive.png`** — all methods compared (OTM)
- **`*_K120_d{d}_Variance.png`** — pairwise comparison showing the effect of ODIS
- **`*_results_d{d}.csv`** — raw RMSE and timing data
- **`required_N_*_d{d}.txt`** — required N for 1% relative error

---

## Key Algorithmic Components

### Active Subspace
The dominant direction of variation in the payoff is identified by estimating the gradient covariance matrix via finite differences on a pilot sample of M=128 quasi-random points. The leading eigenvector serves as the integration direction for pre-integration.

### Householder Rotation
The active subspace direction is aligned with the first coordinate axis via a numerically stable Householder reflection, enabling efficient decomposition into active (1D) and perpendicular ((d-1)D) subspaces.

### ODIS (Optimal Drift Importance Sampling)
- **Arithmetic option**: minimizes `J(z) = ½‖z‖² - log Ψ(z)` via L-BFGS-B with multiple restarts
- **Digital option**: finds the point on the exercise boundary `{z : S_mean(z) = K}` closest to the origin via SLSQP

### Pre-Integration
For each sample in the (d-1)-dimensional perpendicular subspace, the critical threshold `v*` is found via Brent's root-finding method, and the 1D integral over the active direction is computed in closed form:
- **Arithmetic**: Gaussian integral giving E[(S_mean - K)⁺ | z_perp]
- **Digital**: tail probability Φ(-v*)

### Adaptive N Finder
A sequential stopping rule monitors the relative error of a running mean estimator and terminates when the 95% confidence interval half-width falls below 1% of the estimated price.

---

## Reference

See `Burzio_Civilla_Vinciguerra_Report.pdf` for the full mathematical derivations, convergence analysis, and numerical results.
