import os
import time
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import qmc
from scipy.optimize import minimize, brentq
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from tqdm import tqdm

np.random.seed(42)

"""
This module implements Monte Carlo and Quasi-Monte Carlo simulations for pricing arithmetic Asian options.

Arithmetic Asian options have payoffs based on the arithmetic average of the underlying asset prices over time.

The code uses various variance reduction techniques including:
- Quasi-Monte Carlo (RQMC) with Sobol sequences
- Importance sampling (ODIS - Optimal Drift for Importance Sampling)
- Pre-integration estimators using closed-form solutions
- Active subspace methods for dimension reduction

The simulations are performed for different dimensions (number of time steps) and strike prices,
with convergence analysis and efficiency comparisons.

"""

# ==========================================
# Setup and Configuration
# ==========================================

# Model Parameters
S0 = 100  # Initial stock price
T = 1.0   # Time to maturity (in years)
r = 0.1   # Risk-free interest rate
sigma = 0.1  # Volatility of the stock

# Dimensions to test (number of time steps in the Asian option)
DIMENSIONS = [32, 64, 128, 256, 512]

# Simulation Parameters
POWERS = np.arange(7, 14)  # Powers from 7 to 13, so sample sizes 2^7 to 2^13
SAMPLE_SIZES = 2**POWERS  # Actual sample sizes: 128, 256, ..., 8192
N_REPEATS = 50  # Number of repetitions for each simulation to compute stable RMSE

# ==========================================
# Adaptive N Finder
# ==========================================

def find_N_specific_arithmetic(model, d, tol=0.01):
    """
    find_n_adaptive(model, ..., tol): Implements the stopping criterion derived in Eq. (19). Rather than
    fixing N a priori, this routine increases the sample size in batches, monitoring the standard error of
    the estimator. It terminates the simulation only when the relative RMSE falls below the specified
    tolerance (e.g., ε = 10^{-2}), ensuring the reliability of the OTM price estimates.
    
    Parameters:
    - model: An instance of AsianOptionCholesky representing the option model.
    - d: Dimension (number of time steps).
    - tol: Tolerance for relative error (default 0.01, i.e., 1%).
    
    Returns:
    - N: The required sample size.
    - mu: The estimated mean payoff.
    """
    z_score = 1.96  # 95% Confidence interval z-score
    
    N = 128  # Starting sample size
    # Generate initial batch of standard normal random variables
    z_init = np.random.standard_normal((N, model.d))
    Z_values = model.payoff(z_init)  # Compute payoffs for initial batch
    
    # Initial statistics: mean and variance
    mu = np.mean(Z_values)
    var = np.var(Z_values, ddof=1)  # Sample variance with ddof=1
    
    print(f"\n--- Specific Recursive Search (Arithmetic, d={d}, Start N={N}) ---")
    print(f"{'N':<10} | {'Mean':<10} | {'Var':<10} | {'RelErr':<10}")
    print("-" * 55)
    
    while True:
        # Generate one new standard normal sample
        z_new = np.random.standard_normal((1, model.d))
        Z_next = model.payoff(z_new)[0]  # Payoff for the new sample
        
        # Save previous mean for variance update
        mu_old = mu
        
        # Recursive update of mean and variance
        mu = (N / (N + 1)) * mu + Z_next / (N + 1)
        var = ((N - 1) / N) * var + (Z_next - mu_old)**2 / (N + 1)
        
        N += 1  # Increment sample size
        
        # Check for convergence
        sigma_est = np.sqrt(max(var, 0.0))  # Estimated standard deviation
        
        if abs(mu) > 1e-12 and sigma_est > 0:
            half_width = z_score * sigma_est / np.sqrt(N)  # Half-width of confidence interval
            rel_err = half_width / abs(mu)  # Relative error
            
            if rel_err <= tol:
                print("-" * 55)
                print(f"Converged at N = {N} for d={d}")
                print(f"Final Mean: {mu:.6f}")
                print(f"Final Var:  {var:.6f}")
                return N, mu
            
            # Print progress every 5000 samples
            if N % 5000 == 0:
                print(f"{N:<10} | {mu:<10.4f} | {var:<10.4f} | {rel_err:<10.2%}")

# ==========================================
# Model: Asian Option with Cholesky Path
# ==========================================

class AsianOptionCholesky:
    """
    Represents an arithmetic Asian option using Cholesky decomposition for path simulation.
    
    This class models the underlying asset paths using correlated Brownian motions,
    discretized into 'd' time steps. The payoff is based on the arithmetic average
    of the stock prices at these time steps.
    
    Attributes:
    - S0: Initial stock price
    - K: Strike price
    - T: Time to maturity
    - r: Risk-free rate
    - sigma: Volatility
    - d: Number of time steps
    - dt: Time step size
    - A: Cholesky matrix for correlation
    - drift_term: Precomputed drift term for each time step
    """
    def __init__(self, S0=100, K=100, T=1.0, r=0.1, sigma=0.1, d=32):
        """
        create_model(S0, K, T, r, σ, m): This initialization routine constructs the discrete time grid
        t_i = i*T/m where i = 1, ..., m and the associated covariance matrix Σ ∈ R^{m×m} with entries
        Σ_{ij} = min(t_i, t_j). Crucially, it performs the Cholesky decomposition Σ = A A^T (via numpy.linalg.cholesky).
        The resulting lower-triangular matrix A is stored to facilitate the linear mapping W = A z required for
        path generation, as described in Section 2.1.
        
        Parameters:
        - S0: Initial stock price
        - K: Strike price
        - T: Time to maturity
        - r: Risk-free interest rate
        - sigma: Volatility
        - d: Number of time steps (m in the explanation)
        """
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.d = d
        self.dt = T / d  # Time step size
        
        # Covariance Matrix construction: Σ_{ij} = min(t_i, t_j)
        times = np.linspace(self.dt, T, d)
        C = np.minimum(times[:, None], times[None, :])
        
        # Cholesky Decomposition: A A^T = Σ
        self.A = np.linalg.cholesky(C)
        
        # Precompute the drift term for each time step: (r - 0.5 σ^2) * t_i
        self.drift_term = (self.r - 0.5 * self.sigma**2) * times

    def payoff(self, z):
        """
        payoff_arithmetic(z, model): Evaluates the discounted payoffs e^{-r T} Ψ_i(z).
        Implements Eq. (4), returning e^{-r T} (S_mean - K)+.
        
        Parameters:
        - z: Array of standard normal random variables, shape (N, d)
        
        Returns:
        - Discounted payoffs, shape (N,)
        """
        if z.ndim == 1: z = z.reshape(1, -1)  # Ensure 2D
        # Correlated Brownian increments: W = A z
        B = z @ self.A.T
        # Stock prices: S_{t_i} = S0 * exp(drift + σ * W_{t_i})
        S = self.S0 * np.exp(self.drift_term + self.sigma * B)
        # Arithmetic average: (1/m) Σ S_{t_i}
        S_mean = np.mean(S, axis=1)
        # Payoff: max(S_mean - K, 0), discounted
        return np.exp(-self.r * self.T) * np.maximum(S_mean - self.K, 0)

# ==========================================
# Directions & Shifts
# ==========================================

def get_gradient(f, z, epsilon=1e-5):
    d = len(z)
    grad = np.zeros(d)
    base = f(z)
    for i in range(d):
        z_p = z.copy()
        z_p[i] += epsilon
        grad[i] = (f(z_p) - base) / epsilon
    return grad

def get_active_subspace(model, M=128):
    """
    get_active_subspace(model, payoff_fn, M): Approximates the solution to the maximization problem (14).
    It estimates the gradient covariance matrix C = E[∇Ψ ∇Ψ^T] using a pilot sample of size M = 128.
    Gradients are computed via finite differences (get_gradient). The function returns the leading eigenvector
    u1 of C, identifying the direction of the kink or jump.
    
    Parameters:
    - model: Option model instance
    - M: Number of pilot samples (default 128)
    
    Returns:
    - u_as: Dominant direction vector (normalized)
    """
    sampler = qmc.Sobol(d=model.d, scramble=True)  # Scrambled Sobol sequence
    z_pilot = stats.norm.ppf(sampler.random(M))  # Transform to normal
    
    grads = []
    for i in range(M):
        g = get_gradient(model.payoff, z_pilot[i])  # Finite difference gradient
        grads.append(g)
    grads = np.array(grads)
    
    # Covariance matrix of gradients: C ≈ (1/M) ∇Ψ^T ∇Ψ
    C_hat = (grads.T @ grads) / M
    evals, evecs = np.linalg.eigh(C_hat)  # Eigen decomposition
    u_as = evecs[:, -1]  # Leading eigenvector (largest eigenvalue)
    if np.sum(u_as) < 0: u_as = -u_as  # Ensure consistent orientation
    return u_as

def get_z1_direction(d):
    """Direction for z1: [1, 0, ... 0]"""
    u = np.zeros(d)
    u[0] = 1.0
    return u

def householder_matrix(u):
    """
    householder_matrix(u): Constructs the orthogonal matrix Q required to align the active direction u1 with
    the first canonical coordinate axis. It implements the Householder reflection Q = I - 2vv^T / ||v||^2
    with v = u - ||u|| e1, ensuring the numerically stable rotation of the integration domain.
    
    Parameters:
    - u: Input vector to rotate
    
    Returns:
    - Q: Orthogonal matrix
    """
    d = len(u)
    e1 = np.zeros(d); e1[0] = 1.0  # First standard basis vector
    sign = np.sign(u[0]) if u[0] != 0 else 1
    v = u + sign * np.linalg.norm(u) * e1  # Householder vector
    v = v / np.linalg.norm(v)  # Normalize
    H = np.eye(d) - 2 * np.outer(v, v)  # Householder reflection
    return H * (-sign)  # Adjust sign

def get_odis_shift(model):
    """
    get_odis_shift_arithmetic(model): Solves the unconstrained optimization problem (23) for the call option.
    It minimizes the objective function J(z) = (1/2) ||z||^2 - ln(Ψ(z)) using the L-BFGS-B algorithm.
    To ensure global optimality for the non-convex objective, the optimizer is initialized from multiple
    starting points (including 0 and scaled vectors 1).
    
    Parameters:
    - model: Option model
    
    Returns:
    - μ*: Optimal shift vector
    """
    def obj(z):
        p = model.payoff(z)[0]
        if p <= 1e-12: return 1e6  # Penalty for zero payoff
        return 0.5 * np.sum(z**2) - np.log(p)  # Objective function J(z)
    
    # Multi-start optimization to avoid local minima
    best_res = None
    best_val = np.inf
    starts = [np.zeros(model.d), np.ones(model.d)*0.5, np.ones(model.d)*1.5]  # Starting points
    
    for x0 in starts:
        res = minimize(obj, x0, method='L-BFGS-B')  # Local optimization
        if res.fun < best_val:
            best_val = res.fun
            best_res = res
    return best_res.x  # Optimal μ

# ==========================================
# Estimators
# ==========================================

def standard_estimator(model, N, method='MC', mu_shift=None):
    """
    standard_estimator(model, N, payoff_fn, method, μ): A unified driver for the estimators ˆV_{CMC} and ˆV_{QMC}.
    • Sequence Generation: If method='RQMC', it utilizes a scrambled Sobol sequence generator (scipy.stats.qmc.Sobol)
      to produce points U^{(i)} ∈ [0,1]^m, which are mapped to R^m via the inverse normal CDF Φ^{-1}.
      To preserve numerical stability, inputs to Φ^{-1} are clipped to [10^{-10}, 1 - 10^{-10}].
    • Change of Measure: To support ODIS (Section 2.7), the function accepts an optimal drift vector μ.
      It shifts the samples Z' = Z + μ (simulating from q(z; μ)) and computes the Radon-Nikodym derivative
      w(z; μ) = exp(-μ^T Z' + (1/2)||μ||^2).
    • Estimation: Returns the sample mean of the product Ψ(Z') w(Z'; μ), providing an unbiased estimate of V.
    
    Parameters:
    - model: Option model
    - N: Number of samples
    - method: 'MC' for Monte Carlo, 'RQMC' for Quasi-Monte Carlo
    - mu_shift: Shift vector μ for importance sampling (optional)
    
    Returns:
    - Estimated price (mean of weighted payoffs)
    """
    if method == 'MC':
        z = np.random.standard_normal((N, model.d))  # Standard normal samples
    elif method == 'RQMC':
        sampler = qmc.Sobol(d=model.d, scramble=True)  # Scrambled Sobol
        u = sampler.random(N)  # Uniform [0,1]^m
        # Clip to avoid Φ^{-1} issues at boundaries
        u = np.clip(u, 1e-10, 1 - 1e-10)
        z = stats.norm.ppf(u)  # Inverse CDF to normal
    
    weights = np.ones(N)  # Default weights
    if mu_shift is not None:
        # Importance Sampling: shift samples and adjust weights
        X = z + mu_shift  # Z' = Z + μ
        dot = np.sum(X * mu_shift, axis=1)  # μ^T Z'
        mu_sq = 0.5 * np.sum(mu_shift**2)  # (1/2) ||μ||^2
        weights = np.exp(-dot + mu_sq)  # w(Z'; μ)
        payoffs = model.payoff(X)  # Ψ(Z')
    else:
        payoffs = model.payoff(z)  # Ψ(Z)
       
    return np.mean(payoffs * weights)  # Sample mean of Ψ w

def pre_int_estimator(model, N, u, Q, mu_perp=None, method='RQMC'):
    """
    pre_int_estimator(model, N, u, Q, μ_perp, method): Implements the pre-integrated estimator ˆV_{PI} for arithmetic Asian options (Section 2.8).
    • Pre-Integration: Smooths the payoff by analytically integrating over the last time step, reducing variance for large m.
    • Closed-Form Smoothing: For arithmetic average A, the conditional expectation E[max(A - K, 0) | Z_perp] is computed
      using the closed-form expression (Eq. (25)), where A is approximated as a log-normal random variable.
    • Active Subspace: Uses the active subspace direction u and rotation matrix Q to reduce dimensionality.
      Samples are generated in the perpendicular subspace Z_perp ∈ R^{m-1}.
    • Change of Measure: Applies importance sampling in the perpendicular subspace if μ_perp is provided.
    • Estimation: For each sample, solves for v* such that the expected payoff equals K, then computes the integral
      using Φ functions, providing a smoothed estimate of the option price.
    
    Parameters:
    - model: Option model
    - N: Number of samples
    - u: Active subspace direction vector
    - Q: Rotation matrix from Householder transformation
    - mu_perp: Shift vector in perpendicular subspace for importance sampling (optional)
    - method: 'MC' or 'RQMC'
    
    Returns:
    - Estimated price (mean of smoothed payoffs)
    """
    d_perp = model.d - 1
    if method == 'MC':
        z_perp = np.random.standard_normal((N, d_perp))
    else:
        sampler = qmc.Sobol(d=d_perp, scramble=True)
        z_perp = stats.norm.ppf(sampler.random(N))
        
    weights = np.ones(N)
    if mu_perp is not None:
        X_perp = z_perp + mu_perp
        dot = np.sum(X_perp * mu_perp, axis=1)
        mu_sq = 0.5 * np.sum(mu_perp**2)
        weights = np.exp(-dot + mu_sq)
        z_perp = X_perp # Use shifted samples

    # Geometric Constants
    U_perp = Q[:, 1:]
    Au = model.A @ u 
    AU_perp = model.A @ U_perp
    beta = model.sigma * Au
    const_part = np.log(model.S0) + model.drift_term
    
    estimates = np.zeros(N)
    exponent_perps = model.sigma * (z_perp @ AU_perp.T)
    
    # We search for root in [-30, 30] to cover the relevant probability mass
    for i in range(N):
        alpha = np.exp(const_part + exponent_perps[i])
        def g(v): return np.mean(alpha * np.exp(beta * v)) - model.K
        
        # Root Finding
        try: v_star = brentq(g, -30, 30)
        except ValueError: v_star = -30 if g(0) > 0 else 30
            
        # Closed-Form Integration
        if v_star < 25:
            d1 = beta - v_star
            term1 = np.mean(alpha * np.exp(0.5 * beta**2) * stats.norm.cdf(d1))
            term2 = model.K * stats.norm.cdf(-v_star)
            val = np.exp(-model.r * model.T) * (term1 - term2)
        else:
            val = 0.0
        estimates[i] = val * weights[i]
        
    return np.mean(estimates)

# ==========================================
# Simulation Logic
# ==========================================

def run_experiment(K_target, d_val):
    """
    run_experiment(K_target, d_val): Runs the full simulation experiment for a given strike K and dimension d.
    • Model Setup: Creates an AsianOptionCholesky model with the specified parameters.
    • Directions: Computes the z1 direction (first coordinate) and active subspace direction u_as, along with their
      rotation matrices Q_z1 and Q_as.
    • ODIS Shift: For K=120 (out-of-the-money), computes the optimal shift μ_opt using get_odis_shift and projects
      it to the perpendicular subspace for active subspace methods.
    • Ground Truth: Generates a high-accuracy reference value using pre_int_estimator with 2^17 samples.
    • Methods: Defines a dictionary of estimators to test, varying by K (K=100: basic methods; K=120: includes ODIS).
    • Simulation Loop: For each sample size N and method, repeats N_REPEATS times to compute RMSE and average time.
    • Output: Returns a DataFrame with results for plotting and analysis.
    
    Parameters:
    - K_target: Strike price (100 or 120)
    - d_val: Dimension (number of time steps)
    
    Returns:
    - DataFrame with columns: K, N, Method, RMSE, Time, d
    """
    print(f"\n--- Running Experiment for K = {K_target}, d = {d_val} ---")
    model = AsianOptionCholesky(S0=S0, K=K_target, T=T, r=r, sigma=sigma, d=d_val)
    
    # Directions
    u_z1 = get_z1_direction(d_val)
    Q_z1 = np.eye(d_val)
    u_as = get_active_subspace(model)
    Q_as = householder_matrix(u_as)
    
    # ODIS Shift (Only needed for K=120)
    mu_opt = None
    mu_perp_as = None
    if K_target == 120:
        mu_opt = get_odis_shift(model)
        mu_local = Q_as.T @ mu_opt 
        mu_perp_as = mu_local[1:] 

    # Ground Truth Generation
    if K_target == 120:
        true_val = pre_int_estimator(model, 2**17, u_as, Q_as, mu_perp_as, 'RQMC')
    else:
        true_val = pre_int_estimator(model, 2**17, u_as, Q_as, None, 'RQMC')
    print(f"  -> Truth: {true_val:.6f}")

    methods = {}
    if K_target == 100:
        methods['Crude MC'] = lambda n: standard_estimator(model, n, 'MC')
        methods['Plain RQMC'] = lambda n: standard_estimator(model, n, 'RQMC')
        methods['Pre-Int (z1)'] = lambda n: pre_int_estimator(model, n, u_z1, Q_z1, None, 'RQMC')
        methods['Pre-Int (AS)'] = lambda n: pre_int_estimator(model, n, u_as, Q_as, None, 'RQMC')
        
    elif K_target == 120:
        methods['Crude MC'] = lambda n: standard_estimator(model, n, 'MC')
        methods['MC + ODIS'] = lambda n: standard_estimator(model, n, 'MC', mu_opt)
        methods['Plain RQMC'] = lambda n: standard_estimator(model, n, 'RQMC')
        methods['RQMC + ODIS'] = lambda n: standard_estimator(model, n, 'RQMC', mu_opt)
        methods['Pre-Int (AS)'] = lambda n: pre_int_estimator(model, n, u_as, Q_as, None, 'RQMC')
        methods['Pre-Int (AS) + ODIS'] = lambda n: pre_int_estimator(model, n, u_as, Q_as, mu_perp_as, 'RQMC')

    results = []
    total_ops = len(SAMPLE_SIZES) * len(methods) * N_REPEATS
    
    with tqdm(total=total_ops, desc=f"Simulating d={d_val}") as pbar:
        for N in SAMPLE_SIZES:
            for name, func in methods.items():
                errs = []
                times = []
                for _ in range(N_REPEATS):
                    t0 = time.time()
                    est = func(N)
                    t1 = time.time()
                    errs.append(est)
                    times.append(t1 - t0)
                    pbar.update(1)
                
                rmse = np.sqrt(np.mean((np.array(errs) - true_val)**2))
                avg_time = np.mean(times)
                results.append({'K': K_target, 'N': N, 'Method': name, 'RMSE': rmse, 'Time': avg_time, 'd': d_val})
                
    return pd.DataFrame(results)

# ==========================================
# Plotting Functions
# ==========================================

def get_convergence_rate(N, RMSE):
    # Fit log(RMSE) = a + b * log(N)
    # Slope b is the convergence rate
    slope, intercept = np.polyfit(np.log(N), np.log(RMSE), 1)
    return slope

def plot_k100(df, d_val, save_dir):
    """
    plot_k100(df, d_val, save_dir): Generates plots for K=100 experiments.
    • Convergence Plot (Left): Log-log plot of RMSE vs N for each method, with fitted convergence rates.
      Includes reference lines for O(N^{-0.5}) (MC) and O(N^{-1}) (QMC).
    • Efficiency Plot (Right): Log-log plot of RMSE vs computation time to assess cost-effectiveness.
    • Output: Saves the figure as PNG in the specified directory.
    
    Parameters:
    - df: DataFrame with simulation results
    - d_val: Dimension
    - save_dir: Directory to save the plot
    """
    df = df[(df['K'] == 100) & (df['d'] == d_val)]
    if df.empty: return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Convergence Plot
    ax = axes[0]
    methods = ['Crude MC', 'Plain RQMC', 'Pre-Int (z1)', 'Pre-Int (AS)']
    markers = ['o', 's', 'x', '^']
    
    for m, mark in zip(methods, markers):
        sub = df[df['Method'] == m]
        if sub.empty: continue
        
        # Calculate Slope
        slope = get_convergence_rate(sub['N'], sub['RMSE'])
        label_str = f"{m} (Rate $\\approx N^{{{slope:.2f}}}$)"
        
        ax.loglog(sub['N'], sub['RMSE'], marker=mark, linestyle='-', label=label_str, base=2)

    # Reference Lines
    Ns = df['N'].unique()
    ref_mc = Ns**(-0.5) * (df[df['Method']=='Crude MC']['RMSE'].iloc[0] * Ns[0]**0.5)
    ax.loglog(Ns, ref_mc, 'k--', alpha=0.3, label='$O(N^{-0.5})$', base=2)
    
    ref_qmc = Ns**(-1.0) * (df[df['Method']=='Plain RQMC']['RMSE'].iloc[0] * Ns[0]**1.0)
    ax.loglog(Ns, ref_qmc, 'k:', alpha=0.3, label='$O(N^{-1.0})$', base=2)

    ax.set_title(f'K=100, d={d_val}: Convergence Analysis', fontsize=14)
    ax.set_xlabel('Sample Size $N$ (log2 scale)', fontsize=12)
    ax.set_ylabel('RMSE', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())

    # 2. Computational Cost Plot
    ax = axes[1]
    for m, mark in zip(methods, markers):
        sub = df[df['Method'] == m]
        if sub.empty: continue
        # Total time for N samples vs RMSE
        ax.loglog(sub['Time'], sub['RMSE'], marker=mark, linestyle='-', label=m)

    ax.set_title(f'K=100, d={d_val}: Efficiency (Error vs Time)', fontsize=14)
    ax.set_xlabel('Avg Computation Time (s)', fontsize=12)
    ax.set_ylabel('RMSE', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", ls="-", alpha=0.2)

    plt.tight_layout()
    fname = os.path.join(save_dir, f'Arithmetic_Asian_K100_d{d_val}_Analysis.png')
    plt.savefig(fname, dpi=300)
    print(f"Saved {fname}")
    plt.close()

def plot_k120(df, d_val, save_dir):
    """
    plot_k120(df, d_val, save_dir): Generates plots for K=120 experiments (out-of-the-money case).
    • Plot A: Comprehensive comparison of all methods with convergence rates.
    • Plot B: Focus on variance reduction by comparing base methods vs methods with ODIS.
      Pairs: Crude MC vs MC+ODIS, Plain RQMC vs RQMC+ODIS, Pre-Int (AS) vs Pre-Int (AS)+ODIS.
    • Output: Saves two PNG figures in the specified directory.
    
    Parameters:
    - df: DataFrame with simulation results
    - d_val: Dimension
    - save_dir: Directory to save the plots
    """
    df = df[(df['K'] == 120) & (df['d'] == d_val)]
    if df.empty: return

    # Plot A: Comprehensive Comparison
    plt.figure(figsize=(10, 7))
    methods = df['Method'].unique()
    
    for m in methods:
        sub = df[df['Method'] == m]
        slope = get_convergence_rate(sub['N'], sub['RMSE'])
        label_str = f"{m} ($N^{{{slope:.2f}}}$)"
        plt.loglog(sub['N'], sub['RMSE'], marker='o', label=label_str, base=2)

    plt.title(f'K=120, d={d_val}: Comprehensive Comparison', fontsize=14)
    plt.xlabel('Sample Size $N$ (log2)', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    plt.grid(True, which="both", alpha=0.2)
    plt.legend()
    fname_A = os.path.join(save_dir, f'Arithmetic_Asian_K120_d{d_val}_Comprehensive.png')
    plt.savefig(fname_A, dpi=300)
    print(f"Saved {fname_A}")
    plt.close()

    # Plot B: Variance Reduction Focus
    plt.figure(figsize=(10, 7))
    
    pairs = [
        ('Crude MC', 'MC + ODIS', 'red'),
        ('Plain RQMC', 'RQMC + ODIS', 'blue'),
        ('Pre-Int (AS)', 'Pre-Int (AS) + ODIS', 'green')
    ]
    
    for base, odis, color in pairs:
        sub_b = df[df['Method'] == base]
        if not sub_b.empty:
            plt.loglog(sub_b['N'], sub_b['RMSE'], color=color, linestyle='--', marker='o', label=base, base=2, alpha=0.5)
        
        sub_o = df[df['Method'] == odis]
        if not sub_o.empty:
            slope = get_convergence_rate(sub_o['N'], sub_o['RMSE'])
            label_str = f"{odis} ($N^{{{slope:.2f}}}$)"
            plt.loglog(sub_o['N'], sub_o['RMSE'], color=color, linestyle='-', marker='D', label=label_str, base=2)

    plt.title(f'K=120, d={d_val}: Impact of Variance Reduction', fontsize=14)
    plt.xlabel('Sample Size $N$ (log2)', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    plt.grid(True, which="both", alpha=0.2)
    plt.legend()
    fname_B = os.path.join(save_dir, f'Arithmetic_Asian_K120_d{d_val}_Variance.png')
    plt.savefig(fname_B, dpi=300)
    print(f"Saved {fname_B}")
    plt.close()

# ==========================================
# Main Execution
# ==========================================

if __name__ == "__main__":
    """
    Main execution block: Runs the full analysis for arithmetic Asian options.
    • Directory Setup: Creates 'plots_arithmetic_asian' and subdirectories for each dimension.
    • Experiment Loop: For each dimension d in DIMENSIONS, runs experiments for K=100 and K=120,
      saves results to CSV, generates plots, and computes required N for K=120 with tolerance 0.01.
    • Output: Saves plots, CSVs, and required N files; prints summary of required N across dimensions.
    """
    
    if not os.path.exists('plots_arithmetic_asian'):
        os.makedirs('plots_arithmetic_asian')
        
    required_n_results = []
    
    # Loop over the dimensions
    for d_val in DIMENSIONS:
        print("\n" + "#"*60)
        print(f"PROCESSING DIMENSION: {d_val}")
        print("#"*60)
        
        # Create directory for this dimension
        curr_dir = os.path.join('plots_arithmetic_asian', f'd_{d_val}')
        if not os.path.exists(curr_dir):
            os.makedirs(curr_dir)
            
        # Run Experiments
        df100 = run_experiment(100, d_val)
        df120 = run_experiment(120, d_val)
        
        full_df = pd.concat([df100, df120])
        csv_name = os.path.join(curr_dir, f'arithmetic_asian_results_d{d_val}.csv')
        full_df.to_csv(csv_name, index=False)
        print(f"\nResults for d={d_val} saved to {csv_name}.")
        
        # Generate Plots
        plot_k100(full_df, d_val, curr_dir)
        plot_k120(full_df, d_val, curr_dir)

        # Find N for K=120
        print(f"\nFind N (Arithmetic) for d={d_val}")
        model_test = AsianOptionCholesky(S0=S0, K=120, T=T, r=r, sigma=sigma, d=d_val)
        
        final_N, final_price = find_N_specific_arithmetic(model_test, d=d_val, tol=0.01)
        required_n_results.append({'d': d_val, 'Required_N': final_N, 'Estimated_Price': final_price})
        
        with open(os.path.join(curr_dir, f'required_N_arithmetic_d{d_val}.txt'), 'w') as f:
            f.write(str(final_N))
    
    print("\nArithmetic Option Analysis Complete..")