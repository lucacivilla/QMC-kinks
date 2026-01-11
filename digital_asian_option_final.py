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
This module implements Monte Carlo and Quasi-Monte Carlo simulations for pricing digital Asian options.

Digital Asian options have binary payoffs based on whether the arithmetic average of the underlying asset prices
exceeds a strike price. The code uses various variance reduction techniques including Quasi-Monte Carlo (RQMC),
importance sampling (ODIS), pre-integration with closed-form smoothing, and active subspaces.

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

def find_N_specific_digital(model, d, tol=0.01):
    """
    Implements the stopping criterion derived in Eq. (19). Rather than fixing N a priori, this routine
    increases the sample size in batches, monitoring the standard error of the estimator. It terminates
    the simulation only when the relative RMSE falls below the specified tolerance (e.g., ε = 10^{-2}),
    ensuring the reliability of the OTM price estimates.
    
    Parameters:
    - model: An instance of DigitalAsianOption representing the option model.
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
    Z_values = model.payoff_digital(z_init)  # Compute payoffs for initial batch
    
    # Initial statistics: mean and variance
    mu = np.mean(Z_values)
    var = np.var(Z_values, ddof=1)  # Sample variance with ddof=1
    
    print(f"\n--- Specific Recursive Search (Digital, d={d}, Start N={N}) ---")
    print(f"{'N':<10} | {'Mean':<10} | {'Var':<10} | {'RelErr':<10}")
    print("-" * 55)
    
    while True:
        # Generate one new standard normal sample
        z_new = np.random.standard_normal((1, model.d))
        Z_next = model.payoff_digital(z_new)[0]  # Payoff for the new sample
        
        mu_old = mu  # Save previous mean
        
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
# Model: Digital Asian Option
# ==========================================

class DigitalAsianOption:
    """
    Represents a digital Asian option using Cholesky decomposition for path generation.
    
    This class models the underlying asset paths using correlated Brownian motions,
    discretized into 'd' time steps. The payoff is binary based on the arithmetic average
    of the stock prices exceeding the strike.
    
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
        
        # 1. Covariance Matrix construction: Σ_{ij} = min(t_i, t_j)
        times = np.linspace(self.dt, T, d)
        C = np.minimum(times[:, None], times[None, :])
        
        # Cholesky Decomposition: A A^T = Σ
        self.A = np.linalg.cholesky(C)
        
        # Precompute drift for simulation: (r - 0.5 σ^2) * t_i
        self.drift_term = (self.r - 0.5 * self.sigma**2) * times

    def get_S_mean(self, z):
        """
        get_s_mean(z, model): Computes the realization of the random variable (1/m) Σ_{i=1}^m S_{t_i}.
        Given a standard normal vector z ∈ R^m, it first recovers the Brownian motion path W = A z.
        It then applies the geometric Brownian motion mapping S_{t_i} = S0 exp((r − σ^2/2) t_i + σ W_{t_i})
        in a vectorized manner. This function serves as the numerical evaluation of the underlying asset process.
        
        Parameters:
        - z: Array of standard normal random variables, shape (N, d)
        
        Returns:
        - Arithmetic mean of stock prices, shape (N,)
        """
        if z.ndim == 1: z = z.reshape(1, -1)  # Ensure 2D
        # Correlated Brownian increments: W = A z
        B = z @ self.A.T
        # Stock prices: S_{t_i} = S0 * exp(drift + σ * W_{t_i})
        S = self.S0 * np.exp(self.drift_term + self.sigma * B)
        # Arithmetic mean: (1/m) Σ S_{t_i}
        return np.mean(S, axis=1)

    def payoff_digital(self, z):
        """
        payoff_digital(z, model): Evaluates the discounted payoffs e^{-r T} Ψ_i(z).
        Implements Eq. (5), returning e^{-r T} if the arithmetic mean exceeds K, and 0 otherwise.
        
        Parameters:
        - z: Array of standard normal random variables, shape (N, d)
        
        Returns:
        - Discounted binary payoffs, shape (N,)
        """
        S_mean = self.get_S_mean(z)  # Compute arithmetic mean
        p = np.where(S_mean > self.K, 1.0, 0.0)  # Binary payoff
        return np.exp(-self.r * self.T) * p  # Discounted
   
    def payoff_arithmetic(self, z):
        """
        payoff_arithmetic(z, model): Evaluates the discounted payoffs e^{-r T} Ψ_i(z) for the arithmetic option.
        Returns e^{-r T} max(S_mean - K, 0). Used as a proxy for gradient estimation in active subspaces.
        
        Parameters:
        - z: Array of standard normal random variables, shape (N, d)
        
        Returns:
        - Discounted arithmetic payoffs, shape (N,)
        """
        S_mean = self.get_S_mean(z)  # Compute arithmetic mean
        return np.exp(-self.r * self.T) * np.maximum(S_mean - self.K, 0)  # Discounted payoff

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
    # Use Arithmetic payoff for gradient estimation (as proxy for digital)
    for i in range(M):
        g = get_gradient(model.payoff_arithmetic, z_pilot[i])  # Finite difference gradient
        grads.append(g)
    grads = np.array(grads)
    
    # Covariance matrix of gradients: C ≈ (1/M) ∇Ψ^T ∇Ψ
    C_hat = (grads.T @ grads) / M
    evals, evecs = np.linalg.eigh(C_hat)  # Eigen decomposition
    u_as = evecs[:, -1]  # Leading eigenvector (largest eigenvalue)
    
    # Standardize sign: ensure mean moves up with positive u
    test_z = u_as * 0.1
    if model.get_S_mean(test_z)[0] < model.get_S_mean(-test_z)[0]:
        u_as = -u_as
    return u_as

def get_z1_direction(d):
    """Direction for z1: [1, 0, ... 0]"""
    u = np.zeros(d)
    u[0] = 1.0
    return u

def householder_matrix(u):
    """Orthogonal matrix Q where first column is u"""
    d = len(u)
    e1 = np.zeros(d); e1[0] = 1.0
    sign = np.sign(u[0]) if u[0] != 0 else 1
    v = u + sign * np.linalg.norm(u) * e1
    v = v / np.linalg.norm(v)
    H = np.eye(d) - 2 * np.outer(v, v)
    return H * (-sign)

def get_odis_shift(model):
    """
    get_odis_shift_digital(model): Determines the optimal drift μ* for the binary option.
    Instead of the generic likelihood maximization, it solves a constrained geometric problem:
    it minimizes the distance ||z||^2 / 2 subject to Ψ(z) > 0. This identifies the point on the limit
    surface φ(z) = 0 with the highest probability density, solved via the SLSQP algorithm.
    
    Parameters:
    - model: Option model
    
    Returns:
    - Optimal shift vector μ*
    """
    def constr(z):
        # Constraint: S_mean(z) - K = 0 (on the exercise boundary)
        return model.get_S_mean(z)[0] - model.K
   
    def obj(z):
        # Objective: minimize ||z||^2 / 2
        return 0.5 * np.sum(z**2)

    x0 = np.ones(model.d) * 0.1  # Initial guess
    cons = ({'type': 'eq', 'fun': constr})  # Equality constraint
    res = minimize(obj, x0, method='SLSQP', constraints=cons, tol=1e-4)  # Constrained optimization
    
    if not res.success:
        return np.zeros(model.d)  # Fallback to zero shift
       
    return res.x  # Optimal μ

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
        payoffs = model.payoff_digital(X)  # Ψ(Z')
    else:
        payoffs = model.payoff_digital(z)  # Ψ(Z)
       
    return np.mean(payoffs * weights)  # Sample mean of Ψ w

def pre_int_estimator_closed_form(model, N, u, Q, mu_perp=None, method='RQMC'):
    """
    vector_pre_int_digital(...): These functions implement the smoothed estimators by evaluating the inner
    integral p(x_{-1}) defined in Eq. (6).
    • Root Finding: For every sample z_{-1} in the subspace R^{m-1}, the routine defines the implicit function
      g(v) = Mean(S(v u1 + Q_{-1} z_{-1})) - K. It employs Brent's method to find the critical value v* = ψ(z_{-1})
      where the payoff activates.
    • Analytic Smoothing: For the Digital option, the discontinuous indicator is replaced by the smooth tail
      probability Φ(-v*). For the Arithmetic option, the function computes the conditional expectation
      E[(S_mean - K)+ | z_{-1}] analytically using Gaussian integrals, removing the derivative discontinuity.
    
    Parameters:
    - model: Option model
    - N: Number of samples in perpendicular subspace
    - u: Direction vector (e.g., active subspace or z1)
    - Q: Orthogonal matrix from Householder
    - mu_perp: Shift in perpendicular direction (optional)
    - method: 'MC' or 'RQMC'
    
    Returns:
    - Estimated price using pre-integration
    """
    d_perp = model.d - 1  # Dimension of perpendicular subspace R^{m-1}
    if method == 'MC':
        z_perp = np.random.standard_normal((N, d_perp))  # Samples in R^{m-1}
    else:
        sampler = qmc.Sobol(d=d_perp, scramble=True)
        z_perp = stats.norm.ppf(sampler.random(N))  # Quasi-random
    
    weights = np.ones(N)  # IS weights
    if mu_perp is not None:
        # Apply IS in perpendicular direction
        X_perp = z_perp + mu_perp
        dot = np.sum(X_perp * mu_perp, axis=1)
        mu_sq = 0.5 * np.sum(mu_perp**2)
        weights = np.exp(-dot + mu_sq)
        z_perp = X_perp  # Use shifted samples

    # Precompute matrices
    U_perp = Q[:, 1:]  # Perpendicular directions
    Au = model.A @ u  # A u
    AU_perp_z_perp = (model.A @ U_perp) @ z_perp.T  # A U_{-1} z_{-1}
    
    beta = model.sigma * Au  # Coefficient for exponential
    const_log_S = np.log(model.S0) + model.drift_term  # Log S0 + drift terms
    
    estimates = np.zeros(N)
    
    # Search range for v*
    BOUND = 30.0
    
    for i in range(N):
        # Contribution from perpendicular part
        perp_contribution = model.sigma * AU_perp_z_perp[:, i]
        alpha = np.exp(const_log_S + perp_contribution)  # Alpha coefficients
        
        def g(v):
            # g(v) = E[S_mean | z_{-1}] - K, where S_mean depends on v
            return np.mean(alpha * np.exp(beta * v)) - model.K
        
        # Find v* such that g(v*) = 0
        try:
            if g(-BOUND) * g(BOUND) < 0:
                v_star = brentq(g, -BOUND, BOUND)  # Root in [-BOUND, BOUND]
            else:
                v_star = -BOUND if g(0) > 0 else BOUND  # Boundary case
        except ValueError:
             v_star = -BOUND if g(0) > 0 else BOUND
        
        # Closed-form smoothing: replace indicator with Φ(-v*)
        val = np.exp(-model.r * model.T) * stats.norm.cdf(-v_star)
        estimates[i] = val * weights[i]  # Weighted estimate
        
    return np.mean(estimates)  # Average over samples

# ==========================================
# Simulation Logic
# ==========================================

def run_experiment(K_target, d_val):
    """
    Runs simulations for a given strike price K_target and dimension d_val.
    
    Computes RMSE for various estimators and returns results as a DataFrame.
    """
    print(f"\n--- Running Experiment for K = {K_target}, d = {d_val} (Digital) ---")
    model = DigitalAsianOption(S0=S0, K=K_target, T=T, r=r, sigma=sigma, d=d_val)  # Initialize model
    
    # Compute directions for pre-integration
    u_z1 = get_z1_direction(d_val)  # z1 direction
    Q_z1 = np.eye(d_val)  # Identity for z1
    u_as = get_active_subspace(model)  # Active subspace direction
    Q_as = householder_matrix(u_as)  # Rotation matrix
    
    # Compute ODIS shift only for OTM case (K=120)
    mu_opt = None
    mu_perp_as = None
    if K_target == 120:
        print("  -> Computing ODIS shift...")
        mu_opt = get_odis_shift(model)  # Optimal shift
        mu_local = Q_as.T @ mu_opt  # Transform to local coords
        mu_perp_as = mu_local[1:]  # Perpendicular component

    # Compute ground truth using high-accuracy pre-integration
    print("  -> Computing Ground Truth...")
    if K_target == 120:
        true_val = pre_int_estimator_closed_form(model, 2**17, u_as, Q_as, mu_perp_as, 'RQMC')
    else:
        true_val = pre_int_estimator_closed_form(model, 2**17, u_as, Q_as, None, 'RQMC')
    print(f"  -> Truth: {true_val:.6f}")

    # Define methods based on strike
    methods = {}
    if K_target == 100:  # ATM
        methods['Crude MC'] = lambda n: standard_estimator(model, n, 'MC')
        methods['Plain RQMC'] = lambda n: standard_estimator(model, n, 'RQMC')
        methods['Pre-Int (z1)'] = lambda n: pre_int_estimator_closed_form(model, n, u_z1, Q_z1, None, 'RQMC')
        methods['Pre-Int (AS)'] = lambda n: pre_int_estimator_closed_form(model, n, u_as, Q_as, None, 'RQMC')
       
    elif K_target == 120:  # OTM
        methods['Crude MC'] = lambda n: standard_estimator(model, n, 'MC')
        methods['MC + ODIS'] = lambda n: standard_estimator(model, n, 'MC', mu_opt)
        methods['Plain RQMC'] = lambda n: standard_estimator(model, n, 'RQMC')
        methods['RQMC + ODIS'] = lambda n: standard_estimator(model, n, 'RQMC', mu_opt)
        methods['Pre-Int (AS)'] = lambda n: pre_int_estimator_closed_form(model, n, u_as, Q_as, None, 'RQMC')
        methods['Pre-Int (AS) + ODIS'] = lambda n: pre_int_estimator_closed_form(model, n, u_as, Q_as, mu_perp_as, 'RQMC')

    results = []
    total_ops = len(SAMPLE_SIZES) * len(methods) * N_REPEATS  # Progress tracking
    
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
    slope, intercept = np.polyfit(np.log(N), np.log(RMSE), 1)
    return slope

def plot_k100(df, d_val, save_dir):
    """
    K=100 Plot: Convergence and efficiency analysis for digital option.
    Left: Convergence (Log2 N vs RMSE)
    Right: Cost (Time vs RMSE)
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
       
        slope = get_convergence_rate(sub['N'], sub['RMSE'])
        label_str = f"{m} (Rate $\\approx N^{{{slope:.2f}}}$)"
        ax.loglog(sub['N'], sub['RMSE'], marker=mark, linestyle='-', label=label_str, base=2)

    Ns = df['N'].unique()
    # Reference Lines
    if not df[df['Method']=='Crude MC'].empty:
        ref_mc = Ns**(-0.5) * (df[df['Method']=='Crude MC']['RMSE'].iloc[0] * Ns[0]**0.5)
        ax.loglog(Ns, ref_mc, 'k--', alpha=0.3, label='$O(N^{-0.5})$', base=2)
   
    if not df[df['Method']=='Plain RQMC'].empty:
        ref_qmc = Ns**(-1.0) * (df[df['Method']=='Plain RQMC']['RMSE'].iloc[0] * Ns[0]**1.0)
        ax.loglog(Ns, ref_qmc, 'k:', alpha=0.3, label='$O(N^{-1.0})$', base=2)

    ax.set_title(f'K=100 (Digital No-Quad, d={d_val}): Convergence Analysis', fontsize=14)
    ax.set_xlabel('Sample Size $N$ (log2 scale)', fontsize=12)
    ax.set_ylabel('RMSE', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())

    # Computational Cost Plot
    ax = axes[1]
    for m, mark in zip(methods, markers):
        sub = df[df['Method'] == m]
        if sub.empty: continue
        # Total time for N samples vs RMSE
        ax.loglog(sub['Time'], sub['RMSE'], marker=mark, linestyle='-', label=m)

    ax.set_title(f'K=100 (Digital No-Quad, d={d_val}): Efficiency', fontsize=14)
    ax.set_xlabel('Avg Computation Time (s)', fontsize=12)
    ax.set_ylabel('RMSE', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", ls="-", alpha=0.2)

    plt.tight_layout()
    fname = os.path.join(save_dir, f'Digital_NoQuad_K100_d{d_val}_Analysis.png')
    plt.savefig(fname, dpi=300)
    print(f"Saved {fname}")
    plt.close()

def plot_k120(df, d_val, save_dir):
    """
    K=120 Plots: Comprehensive comparison and variance reduction focus for digital option.
    Plot A: Comprehensive (All methods).
    Plot B: Variance Reduction Focus (Method vs Method+ODIS).
    """
    df = df[(df['K'] == 120) & (df['d'] == d_val)]
    if df.empty: return

    # Plot A: Comprehensive Comparison
    plt.figure(figsize=(10, 7))
    methods = df['Method'].unique()
   
    for m in methods:
        sub = df[df['Method'] == m]
        if sub.empty: continue
        slope = get_convergence_rate(sub['N'], sub['RMSE'])
        label_str = f"{m} ($N^{{{slope:.2f}}}$)"
        plt.loglog(sub['N'], sub['RMSE'], marker='o', label=label_str, base=2)

    plt.title(f'K=120 (Digital No-Quad, d={d_val}): Comprehensive Comparison', fontsize=14)
    plt.xlabel('Sample Size $N$ (log2)', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    plt.grid(True, which="both", alpha=0.2)
    plt.legend()
    fname_A = os.path.join(save_dir, f'Digital_NoQuad_K120_d{d_val}_Comprehensive.png')
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

    plt.title(f'K=120 (Digital No-Quad, d={d_val}): Variance Reduction Impact', fontsize=14)
    plt.xlabel('Sample Size $N$ (log2)', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    plt.grid(True, which="both", alpha=0.2)
    plt.legend()
    fname_B = os.path.join(save_dir, f'Digital_NoQuad_K120_d{d_val}_Variance.png')
    plt.savefig(fname_B, dpi=300)
    print(f"Saved {fname_B}")
    plt.close()

# ==========================================
# Main Execution
# ==========================================

if __name__ == "__main__":
    # Main execution for digital Asian option analysis
    if not os.path.exists('plots_digital_asian_2'):
        os.makedirs('plots_digital_asian_2')

    required_n_results = []
   
    # Loop over the dimensions
    for d_val in DIMENSIONS:
        print("\n" + "#"*60)
        print(f"PROCESSING DIMENSION: {d_val}")
        print("#"*60)
       
        # Create directory for this dimension
        curr_dir = os.path.join('plots_digital_asian_2', f'd_{d_val}')
        if not os.path.exists(curr_dir):
            os.makedirs(curr_dir)
           
        # Run Experiments
        df100 = run_experiment(100, d_val)
        df120 = run_experiment(120, d_val)
       
        full_df = pd.concat([df100, df120])
        csv_name = os.path.join(curr_dir, f'digital_option_results_2_d{d_val}.csv')
        full_df.to_csv(csv_name, index=False)
        print(f"\nResults for d={d_val} saved to {csv_name}.")
       
        # Generate Plots
        plot_k100(full_df, d_val, curr_dir)
        plot_k120(full_df, d_val, curr_dir)

        # Find N for K=120
        print(f"\nFind N (Digital) for d={d_val}")
        model_test = DigitalAsianOption(S0=S0, K=120, T=T, r=r, sigma=sigma, d=d_val)
       
        final_N, final_price = find_N_specific_digital(model_test, d=d_val, tol=0.01)
        required_n_results.append({'d': d_val, 'Required_N': final_N, 'Estimated_Price': final_price})
       
        with open(os.path.join(curr_dir, f'required_N_digital_d{d_val}.txt'), 'w') as f:
            f.write(str(final_N))
   
    print("\nDigital Option Analysis Complete.")