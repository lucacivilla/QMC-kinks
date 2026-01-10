import os
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import qmc
from scipy.optimize import minimize, brentq
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(42)


# ==========================================
# Setup and Configuration
# ==========================================

if not os.path.exists('plots_project_solutions'):
    os.makedirs('plots_project_solutions')

# Global parameters
S0 = 100   # Initial asset price
K = 100   # Strike price (Placeholder, changes per question)
T = 1.0   # Time to maturity
r = 0.1   # Risk-free interest rate
sigma = 0.1   # Volatility
M_VALUES = [32, 64, 128]   # Discretization parameters

# Numbers of samples
POWERS = np.arange(7, 14)
SAMPLE_SIZES = 2**POWERS
# Repeats for RMSE estimation
N_REPEATS = 10  


# ==========================================
# Model Definition
# ==========================================

def create_model(S0, K, T, r, sigma, m):
    """
    Initializes the financial model parameters and precomputes the
    Cholesky decomposition required for generating Brownian motion paths.
    """
    dt = T / m   # Time step size
    times = np.linspace(dt, T, m)   # Time grid
    
    # Covariance Matrix for Brownian motion
    C = np.minimum(times[:, None], times[None, :])
    
    # Cholesky decomposition (C = A @ A.T)
    A = np.linalg.cholesky(C)
    
    # Precompute the deterministic drift term
    drift_term = (r - 0.5 * sigma**2) * times
    
    # Return model as a dictionary
    return {
        'S0': S0, 'K': K, 'T': T, 'r': r, 'sigma': sigma, 'm': m,
        'dt': dt, 'A': A, 'drift_term': drift_term
    }


# ==========================================
# Path generation and payoffs
# ==========================================

def get_s_mean(z, model):
    """Generates the arithmetic mean of the asset price path S(t)."""
    # Ensure z is 2D
    if z.ndim == 1: z = z.reshape(1, -1)
    
    # Get model parameters
    A = model['A']
    S0 = model['S0']
    drift = model['drift_term']
    sigma = model['sigma']
    
    # Generate Brownian Motion path: B = A * z
    B = z @ A.T
    
    # Calculate Asset Price S(t)
    S = S0 * np.exp(drift + sigma * B)
    
    # Return the mean across time steps
    return np.mean(S, axis=1)


def payoff_digital(z, model):
    """Binary Digital Option payoff: 1 if Mean(S) > K, else 0."""
    S_mean = get_s_mean(z, model)
    # Return discounted payoff
    # Psi = exp(-rT) * 1_{Mean(S) > K}
    return np.exp(-model['r'] * model['T']) * np.where(S_mean > model['K'], 1.0, 0.0)


def payoff_arithmetic(z, model):
    """Arithmetic Call Option payoff: max(Mean(S) - K, 0)."""
    S_mean = get_s_mean(z, model)
    # Return discounted payoff
    # Psi = exp(-rT) * max(Mean(S) - K, 0)
    return np.exp(-model['r'] * model['T']) * np.maximum(S_mean - model['K'], 0)


# ==========================================
# Mathematical Utilities
# ==========================================

def householder_matrix(u):
    """Constructs an orthogonal Householder matrix H."""
    dim = len(u)
    # Unit vector e1
    e1 = np.zeros(dim)
    e1[0] = 1.0
    
    # Determine sign to avoid cancellation
    sign = np.sign(u[0]) if u[0] != 0 else 1
    # Construct vector v
    v = u + sign * np.linalg.norm(u) * e1
    v = v / np.linalg.norm(v)
    # H = I - 2 * v * v.T
    H = np.eye(dim) - 2 * np.outer(v, v)

    # H reflects e1 to -sign*u, adjust sign to get +u
    return H * (-sign)


def get_gradient(f, z, model, epsilon=1e-5):
    """Approximates the gradient of function f at point z."""
    dim = len(z)
    grad = np.zeros(dim)   # initialize gradient vector
    
    # Extract scalar value using .item()
    base = f(z, model).item()   
    
    # Finite difference approximation
    for i in range(dim):
        z_p = z.copy()
        z_p[i] += epsilon
        
        # Extract scalar value for the perturbed point
        val_p = f(z_p, model).item()
        
        grad[i] = (val_p - base) / epsilon
        
    return grad


def get_convergence_rate(N, RMSE):
    """Calculates the empirical convergence rate."""

    # Convert to numpy arrays
    N_arr = np.array(N)
    RMSE_arr = np.array(RMSE)
    
    # Create a mask to filter out NaNs, Infs, and Zeros (log(0) is -inf)
    mask = np.isfinite(RMSE_arr) & (RMSE_arr > 0)
    
    # Check if we have enough valid points left
    if np.sum(mask) < 2: 
        return 0.0
    
    # Fit line only on valid data
    slope, intercept = np.polyfit(np.log(N_arr[mask]), np.log(RMSE_arr[mask]), 1)

    return slope


# ==========================================
# Active Subspace & Optimal Drift Importance Sampling
# ==========================================

def get_active_subspace(model, payoff_fn, M=128):
    """Identifies the Active Subspace (dominant gradient direction)."""
    m = model['m']
    # Generate pilot samples
    sampler = qmc.Sobol(d=m, scramble=True)
    z_pilot = stats.norm.ppf(sampler.random(M))
    
    # Compute gradients at pilot samples
    grads = []
    for i in range(M):
        grads.append(get_gradient(payoff_fn, z_pilot[i], model))
    grads = np.array(grads)
    
    # C = E[grad(psi) * grad(psi)^T]
    # We compute an approximation, C_hat
    C_hat = (grads.T @ grads) / M
    # Eigen-decomposition
    evals, evecs = np.linalg.eigh(C_hat)
    # Active subspace direction is the eigenvector of largest eigenvalue
    u_as = evecs[:, -1]
    
    # Normalize sign
    test_z = u_as * 0.1
    if payoff_fn(test_z, model)[0] < payoff_fn(-test_z, model)[0]:
        u_as = -u_as
        
    return u_as


def get_odis_shift_digital(model):
    """ODIS for Digital Options: Boundary surface minimization."""
    m = model['m']
    # Constraint: E[Psi] = target (K)
    def constr(z): return get_s_mean(z, model)[0] - model['K']
    # Objective: Minimize 0.5 * ||z||^2
    def obj(z): return 0.5 * np.sum(z**2)

    # Initial guess
    x0 = np.ones(m) * 0.1
    # Define constraints
    cons = ({'type': 'eq', 'fun': constr})
    # Run optimization
    res = minimize(obj, x0, method='SLSQP', constraints=cons, tol=1e-4)

    return res.x if res.success else np.zeros(m)


def get_odis_shift_arithmetic(model):
    """ODIS for Arithmetic Options: Second moment minimization."""
    m = model['m']
    # Objective: Minimize 0.5 * ||z||^2 - log(E[Psi])
    def obj(z):
        p = payoff_arithmetic(z, model)[0]
        if p <= 1e-12: return 1e6
        return 0.5 * np.sum(z**2) - np.log(p)
    
    # Multiple random starts to avoid local minima
    best_val = np.inf
    best_res = None
    starts = [np.zeros(m), np.ones(m)*0.5, np.ones(m)*1.5]
    
    # Optimize from multiple starting points
    for x0 in starts:
        res = minimize(obj, x0, method='L-BFGS-B')
        if res.fun < best_val:
            best_val = res.fun
            best_res = res
            
    return best_res.x if best_res is not None else np.zeros(m)


# ==========================================
# Estimators (Standard & Pre-Integration)
# ==========================================

def standard_estimator(model, N, payoff_fn, method='MC', mu_shift=None):
    """Standard MC or RQMC estimator with optional Importance Sampling."""
    m = model['m']
    
    # Generate samples
    if method == 'MC':
        z = np.random.standard_normal((N, m))
    elif method == 'RQMC':
        sampler = qmc.Sobol(d=m, scramble=True)
        # Generate Uniform samples
        u_rnd = sampler.random(N)
        # Clip to avoid 0 or 1
        u_rnd = np.clip(u_rnd, 1e-10, 1.0 - 1e-10)
        
        z = stats.norm.ppf(u_rnd)
    
    # Importance sampling weights (initially 1)
    weights = np.ones(N)
    
    # Apply shift if provided
    if mu_shift is not None:
        # shift samples
        X = z + mu_shift
        # compute weights
        dot = np.sum(X * mu_shift, axis=1)
        mu_sq = 0.5 * np.sum(mu_shift**2)
        weights = np.exp(-dot + mu_sq)
        sim_z = X
    else:
        sim_z = z

    # Compute payoffs
    payoffs = payoff_fn(sim_z, model)

    return np.mean(payoffs * weights)


def vector_pre_int_digital(model, N, u, Q, mu_perp, method):
    """Pre-integration estimator for Digital Option."""
    m = model['m']
    d_perp = m - 1

    # Generate perpendicular samples
    if method == 'MC':
        z_perp = np.random.standard_normal((N, d_perp))
    else:
        sampler = qmc.Sobol(d=d_perp, scramble=True)
        # Generate Uniform samples
        u_rnd = sampler.random(N)
        u_rnd = np.clip(u_rnd, 1e-10, 1.0 - 1e-10)
        z_perp = stats.norm.ppf(u_rnd)
    
    # Importance sampling weights (set to 1)
    weights = np.ones(N)
    
    # Apply shift if provided
    if mu_perp is not None:
        X_perp = z_perp + mu_perp
        dot = np.sum(X_perp * mu_perp, axis=1)
        mu_sq = 0.5 * np.sum(mu_perp**2)
        weights = np.exp(-dot + mu_sq)
        z_perp = X_perp 

    # Precompute constants
    U_perp = Q[:, 1:]
    Au = model['A'] @ u 
    AU_perp_z_perp = (model['A'] @ U_perp) @ z_perp.T
    beta = model['sigma'] * Au
    const_log_S = np.log(model['S0']) + model['drift_term']
    estimates = np.zeros(N)

    # Root finding bounds
    BOUND = 30.0
    
    # Loop over samples to compute estimates
    for i in range(N):
        perp_contribution = model['sigma'] * AU_perp_z_perp[:, i]
        alpha = np.exp(const_log_S + perp_contribution)
        
        # Define g(v) for root finding
        def g(v): return np.mean(alpha * np.exp(beta * v)) - model['K']
        
        # Find v_star such that g(v_star) = 0
        try:
            # Check if root exists in interval
            if g(-BOUND)*g(BOUND) < 0:
                v_star = brentq(g, -BOUND, BOUND)
            else:
                v_star = -BOUND if g(0) > 0 else BOUND
        except: 
            v_star = -BOUND if g(0) > 0 else BOUND
        
        # Compute estimate for this sample
        val = np.exp(-model['r'] * model['T']) * stats.norm.cdf(-v_star)
        estimates[i] = val * weights[i]
        
    return estimates


def vector_pre_int_arithmetic(model, N, u, Q, mu_perp, method):
    """Pre-integration estimator for Arithmetic Option."""
    m = model['m']
    d_perp = m - 1
    
    # Generate perpendicular samples
    if method == 'MC':
        z_perp = np.random.standard_normal((N, d_perp))
    else:
        sampler = qmc.Sobol(d=d_perp, scramble=True)
        # Generate Uniform samples
        u_rnd = sampler.random(N)
        u_rnd = np.clip(u_rnd, 1e-10, 1.0 - 1e-10)
        z_perp = stats.norm.ppf(u_rnd)
    
    # Importance sampling weights (initially 1)
    weights = np.ones(N)

    # Apply shift if provided
    if mu_perp is not None:
        X_perp = z_perp + mu_perp
        dot = np.sum(X_perp * mu_perp, axis=1)
        mu_sq = 0.5 * np.sum(mu_perp**2)
        weights = np.exp(-dot + mu_sq)
        z_perp = X_perp

    # Precompute constants
    U_perp = Q[:, 1:]
    Au = model['A'] @ u 
    AU_perp = model['A'] @ U_perp
    beta = model['sigma'] * Au
    const_part = np.log(model['S0']) + model['drift_term']
    estimates = np.zeros(N)
    exponent_perps = model['sigma'] * (z_perp @ AU_perp.T)
    
    # Loop over samples
    for i in range(N):
        alpha = np.exp(const_part + exponent_perps[i])
        def g(v): return np.mean(alpha * np.exp(beta * v)) - model['K']
        
        # Find v_star such that g(v_star) = 0
        try: v_star = brentq(g, -30, 30)
        except: v_star = -30 if g(0) > 0 else 30
        
        # Analytic integral for the first dimension
        if v_star < 25:
            d1 = beta - v_star
            term1 = np.mean(alpha * np.exp(0.5 * beta**2) * stats.norm.cdf(d1))
            term2 = model['K'] * stats.norm.cdf(-v_star)
            val = np.exp(-model['r'] * model['T']) * (term1 - term2)
        else:
            val = 0.0
        estimates[i] = val * weights[i]
        
    return estimates

# ==========================================
# Adaptive sample size search
# ==========================================

def find_n_adaptive(model, payoff_type, estimator_type='standard', 
                    mu_shift=None, u_dir=None, Q_mat=None, mu_perp=None, 
                    tol=0.01, label="Generic"):
    """
    Determines N required for relative RMSE < 0.01 (Mean Squared Sense).
    Criteria: (StdErr / Mean) < tol
    """
    BATCH_SIZE = 128
    history_means = []
    
    print(f"    Searching N for {label}...")
    
    while True:
        # Generate batch (no pre-integration)
        if estimator_type == 'standard':
            if 'RQMC' in label:
                sampler = qmc.Sobol(d=model['m'], scramble=True)
                # --- FIX: Clip values to avoid 0.0 or 1.0 (which cause inf in ppf) ---
                u_rnd = sampler.random(BATCH_SIZE)
                u_rnd = np.clip(u_rnd, 1e-10, 1.0 - 1e-10)
                z_batch = stats.norm.ppf(u_rnd)
            else:
                z_batch = np.random.standard_normal((BATCH_SIZE, model['m']))
            
            # Importance sampling weights (initially 1)
            weights = np.ones(BATCH_SIZE)

            # Apply shift if provided
            if mu_shift is not None:
                X = z_batch + mu_shift
                dot = np.sum(X * mu_shift, axis=1)
                mu_sq = 0.5 * np.sum(mu_shift**2)
                weights = np.exp(-dot + mu_sq)
                sim_z = X
            else:
                sim_z = z_batch
            
            # Compute payoffs
            if payoff_type == 'digital':
                vals = payoff_digital(sim_z, model) * weights
            else:
                vals = payoff_arithmetic(sim_z, model) * weights
        
        # With pre-integration
        elif estimator_type == 'pre_int':
            
            if payoff_type == 'digital':
                vals = vector_pre_int_digital(model, BATCH_SIZE, u_dir, Q_mat, mu_perp, 'RQMC' if 'RQMC' in label else 'MC')
            else:
                vals = vector_pre_int_arithmetic(model, BATCH_SIZE, u_dir, Q_mat, mu_perp, 'RQMC' if 'RQMC' in label else 'MC')

        # Update statistics
        batch_mean = np.mean(vals)
        history_means.append(batch_mean)
        
        mu = np.mean(history_means)
        
        # Check convergence (relative rmse)
        if len(history_means) > 1:
            # Standard error of the mean
            sem = np.std(history_means, ddof=1) / np.sqrt(len(history_means))
        else:
            sem = np.inf
        
        # Total samples used
        N = len(history_means) * BATCH_SIZE
        
        # Check stopping criteria
        if abs(mu) > 1e-12 and sem != np.inf:
            rel_rmse = sem / abs(mu)
            
            if rel_rmse <= tol:
                return N, mu
            
            if N > 2e6: # safety limit
                return N, mu

# ==========================================
# Execution Helpers (Per Question)
# ==========================================

def run_question_1(m, truth_dict):
    """Q1: CMC vs RQMC (No Pre-Integration) for K=100."""
    
    results = []   # Initialize results list
    K_val = 100   # Strike price
    
    # Create models for both options
    d_mod = create_model(S0, K_val, T, r, sigma, m)
    a_mod = create_model(S0, K_val, T, r, sigma, m)
    
    # Configuration for both options
    configs = [('Digital', d_mod, payoff_digital), 
               ('Arithmetic', a_mod, payoff_arithmetic)]
    
    # Loop over configurations
    for name, mod, func in configs:
        true_val = truth_dict[name + str(K_val)]
        
        # Loop over methods
        for method in ['MC', 'RQMC']:
            # Loop over sample sizes
            for N in SAMPLE_SIZES:
                errs = []   # Store estimates for RMSE calculation
                # Repeat estimation
                for _ in range(N_REPEATS):
                    est = standard_estimator(mod, N, func, method=method)
                    errs.append(est)
                
                # Compute RMSE
                rmse = np.sqrt(np.mean((np.array(errs) - true_val)**2))
                # Store results
                results.append({'m': m, 'Question': 'Q1', 'Option': name, 'K': K_val, 'Method': method, 'N': N, 'RMSE': rmse})
    
    return pd.DataFrame(results)


def run_question_2(m, truth_dict):
    """Q2: CMC vs RQMC with Pre-Integration for K=100."""

    results = []   # Initialize results list
    K_val = 100   # Strike price
    
    # Create models for both options
    d_mod = create_model(S0, K_val, T, r, sigma, m)
    a_mod = create_model(S0, K_val, T, r, sigma, m)
    
    # Compute active subspaces
    u_as_dig = get_active_subspace(d_mod, payoff_arithmetic) 
    Q_as_dig = householder_matrix(u_as_dig)
    
    u_as_ari = get_active_subspace(a_mod, payoff_arithmetic)
    Q_as_ari = householder_matrix(u_as_ari)
    
    # Configuration for both options
    configs = [
        ('Digital', d_mod, u_as_dig, Q_as_dig),
        ('Arithmetic', a_mod, u_as_ari, Q_as_ari)]
    
    # Loop over configurations
    for name, mod, u, Q in configs:
        true_val = truth_dict[name + str(K_val)]
        # Loop over methods and sample sizes
        for method in ['MC', 'RQMC']:
            label = f"Pre-Int {method}"
            # Loop over sample sizes
            for N in SAMPLE_SIZES:
                errs = []   # Store estimates for RMSE calculation
                # Repeat estimation
                for _ in range(N_REPEATS):
                    if name == 'Digital':
                        vals = vector_pre_int_digital(mod, N, u, Q, None, method)
                    else:
                        vals = vector_pre_int_arithmetic(mod, N, u, Q, None, method)
                    
                    est = np.mean(vals)
                    errs.append(est)
                
                # Compute RMSE
                rmse = np.sqrt(np.mean((np.array(errs) - true_val)**2))
                # Store results
                results.append({'m': m, 'Question': 'Q2', 'Option': name, 'K': K_val, 'Method': label, 'N': N, 'RMSE': rmse})
                
    return pd.DataFrame(results)


def run_question_3(m, truth_dict):
    """Q3: CMC + ODIS (Convergence) + Required N for K=120."""

    req_results = []   # Results for required N
    conv_results = []  # Results for convergence plots
    K_val = 120   # Strike price

    # Create models for both options
    d_mod = create_model(S0, K_val, T, r, sigma, m)
    a_mod = create_model(S0, K_val, T, r, sigma, m)
    
    # Compute ODIS shifts
    mu_dig = get_odis_shift_digital(d_mod)
    mu_ari = get_odis_shift_arithmetic(a_mod)
    
    # Configuration for both options
    configs = [('Digital', d_mod, mu_dig, payoff_digital), 
               ('Arithmetic', a_mod, mu_ari, payoff_arithmetic)]
    
    # Loop over configurations
    for name, mod, mu, func in configs:
        p_type = name.lower()
        true_val = truth_dict[name + str(K_val)]

        # Find Required N (CMC + ODIS)
        N_odis, val_odis = find_n_adaptive(mod, p_type, 'standard', mu, tol=0.01, label=f"{name} CMC+ODIS")
        req_results.append({
            'm': m, 'Question': 'Q3', 'Option': name, 'K': K_val, 
            'Method': 'CMC+ODIS', 'Req_N': N_odis, 'Price': val_odis
        })
        
        # Generate Convergence Data (CMC + ODIS)
        label = "CMC+ODIS"
        for N in SAMPLE_SIZES:
            errs = []
            for _ in range(N_REPEATS):
                est = standard_estimator(mod, N, func, method='MC', mu_shift=mu)
                errs.append(est)
            
            rmse = np.sqrt(np.mean((np.array(errs) - true_val)**2))
            conv_results.append({
                'm': m, 'Question': 'Q3', 'Option': name, 'K': K_val, 
                'Method': label, 'N': N, 'RMSE': rmse
            })
        
    return pd.DataFrame(req_results), pd.DataFrame(conv_results), mu_dig, mu_ari


def run_question_4(m, mu_dig, mu_ari, truth_dict):
    """Q4: RQMC+ODIS and RQMC+PreInt+ODIS (Convergence) + Required N for K=120."""

    req_results = []
    conv_results = []
    K_target = 120   # Strike price

    # Create models for both options
    d_mod = create_model(S0, K_target, T, r, sigma, m)
    a_mod = create_model(S0, K_target, T, r, sigma, m)
    
    # Compute active subspaces
    u_dig = get_active_subspace(d_mod, payoff_arithmetic)
    Q_dig = householder_matrix(u_dig)
    
    u_ari = get_active_subspace(a_mod, payoff_arithmetic)
    Q_ari = householder_matrix(u_ari)

    # Compute ODIS shifts for pre-integration (perpendicular)
    mu_perp_dig = (Q_dig.T @ mu_dig)[1:]
    mu_perp_ari = (Q_ari.T @ mu_ari)[1:]
    
    # Configuration for both options
    configs = [
        ('Digital', d_mod, mu_dig, u_dig, Q_dig, mu_perp_dig, payoff_digital),
        ('Arithmetic', a_mod, mu_ari, u_ari, Q_ari, mu_perp_ari, payoff_arithmetic)
    ]
    
    # Loop over configurations
    for name, mod, mu_opt, u, Q, mu_perp, func in configs:
        p_type = name.lower()
        true_val = truth_dict[name + str(K_target)]
        
        # Find Required N
        
        # RQMC + ODIS
        N_rqmc, val_rqmc = find_n_adaptive(mod, p_type, 'standard', mu_opt, tol=0.01, label=f"{name} RQMC+ODIS")
        req_results.append({
            'm': m, 'Question': 'Q4', 'Option': name, 'K': K_target, 
            'Method': 'RQMC+ODIS', 'Req_N': N_rqmc, 'Price': val_rqmc
        })
        
        # Pre-Int + ODIS
        N_pi, val_pi = find_n_adaptive(mod, p_type, 'pre_int', None, u, Q, mu_perp, tol=0.01, label=f"{name} PreInt+ODIS")
        req_results.append({
            'm': m, 'Question': 'Q4', 'Option': name, 'K': K_target, 
            'Method': 'PreInt+ODIS', 'Req_N': N_pi, 'Price': val_pi
        })
        
        # Generate Convergence Data
        
        # RQMC + ODIS (Standard Estimator)
        for N in SAMPLE_SIZES:
            errs = []
            for _ in range(N_REPEATS):
                est = standard_estimator(mod, N, func, method='RQMC', mu_shift=mu_opt)
                errs.append(est)
            rmse = np.sqrt(np.mean((np.array(errs) - true_val)**2))
            conv_results.append({
                'm': m, 'Question': 'Q4', 'Option': name, 'K': K_target, 
                'Method': 'RQMC+ODIS', 'N': N, 'RMSE': rmse
            })
            
        # PreInt + ODIS (Vector PreInt Estimator)
        # Note: Prompy asked for "rqmc with both odis and pre-int"
        for N in SAMPLE_SIZES:
            errs = []
            for _ in range(N_REPEATS):
                if name == 'Digital':
                    vals = vector_pre_int_digital(mod, N, u, Q, mu_perp, 'RQMC')
                else:
                    vals = vector_pre_int_arithmetic(mod, N, u, Q, mu_perp, 'RQMC')
                est = np.mean(vals)
                errs.append(est)
            rmse = np.sqrt(np.mean((np.array(errs) - true_val)**2))
            conv_results.append({
                'm': m, 'Question': 'Q4', 'Option': name, 'K': K_target, 
                'Method': 'PreInt+ODIS', 'N': N, 'RMSE': rmse
            })

    return pd.DataFrame(req_results), pd.DataFrame(conv_results)


# ==========================================
# Plots
# ==========================================

def plot_k100_results(df_q1, df_q2, m):
    """
    Plots K=100 results for a specific m.
    Combines CMC, RQMC, CMC+PreInt, RQMC+PreInt.
    Aligned reference lines.
    """
    df = pd.concat([df_q1, df_q2])
    options = df['Option'].unique()
    
    for opt in options:
        sub = df[df['Option'] == opt]
        if sub.empty: continue
        
        plt.figure(figsize=(10, 7))
        
        # Methods to plot: MC, RQMC, Pre-Int MC, Pre-Int RQMC
        # Colors: Blue for MC, Red for RQMC. 
        # Markers: Circle for Standard, Square for PreInt
        
        methods = sub['Method'].unique()
        
        for meth in methods:
            d = sub[sub['Method'] == meth].sort_values('N')
            if d.empty: continue
            
            slope = get_convergence_rate(d['N'], d['RMSE'])
            
            # Style Logic
            color = 'blue' if 'MC' in meth and 'RQMC' not in meth else 'red'
            marker = 'o' if 'Pre-Int' not in meth else 's'
            linestyle = '--' if 'Pre-Int' not in meth else '-'
            
            plt.loglog(d['N'], d['RMSE'], marker=marker, linestyle=linestyle, color=color, 
                       label=f"{meth} (Slope={slope:.2f})", base=2)
            
        # Reference Lines
        Ns = np.sort(sub['N'].unique())
        
        # O(N^-0.5): Anchor to Standard MC
        mc_sub = sub[sub['Method'] == 'MC']
        if not mc_sub.empty:
            rmse_0 = mc_sub['RMSE'].values[0]
            n_0 = mc_sub['N'].values[0]
            ref_05 = Ns**(-0.5) * (rmse_0 * n_0**0.5)
            plt.loglog(Ns, ref_05, 'k--', alpha=0.5, label='$O(N^{-0.5})$', base=2)
            
        # O(N^-1.0): Anchor to Standard RQMC
        rqmc_sub = sub[sub['Method'] == 'RQMC']
        if not rqmc_sub.empty:
            rmse_0 = rqmc_sub['RMSE'].values[0]
            n_0 = rqmc_sub['N'].values[0]
            ref_10 = Ns**(-1.0) * (rmse_0 * n_0**1.0)
            plt.loglog(Ns, ref_10, 'k:', alpha=0.5, label='$O(N^{-1.0})$', base=2)
            
        plt.title(f"K=100 Comparison: {opt} Option (m={m})", fontsize=14)
        plt.xlabel("Sample Size N (log2)", fontsize=12)
        plt.ylabel("RMSE", fontsize=12)
        plt.grid(True, which="both", alpha=0.2)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"plots_project_solutions/K100_{opt}_m{m}.png", dpi=300)
        plt.close()


def plot_k120_results(df_q3_conv, df_q4_conv, m):
    """
    Plots K=120 results for a specific m.
    Combines CMC+ODIS, RQMC+ODIS, RQMC+PreInt+ODIS.
    Aligned reference lines.
    """
    df = pd.concat([df_q3_conv, df_q4_conv])
    options = df['Option'].unique()
    
    for opt in options:
        sub = df[df['Option'] == opt]
        if sub.empty: continue
        
        plt.figure(figsize=(10, 7))
        
        methods = sub['Method'].unique()
        
        for meth in methods:
            d = sub[sub['Method'] == meth].sort_values('N')
            if d.empty: continue
            
            slope = get_convergence_rate(d['N'], d['RMSE'])
            
            # Style Logic
            if 'CMC+ODIS' in meth:
                color = 'blue'; marker = 'D'; ls = '-'
            elif 'PreInt' in meth:
                color = 'green'; marker = '*'; ls = '-'
            else: # RQMC+ODIS
                color = 'red'; marker = 'D'; ls = '-'
            
            plt.loglog(d['N'], d['RMSE'], marker=marker, linestyle=ls, color=color, 
                       label=f"{meth} (Slope={slope:.2f})", base=2)
            
        # Reference Lines (Anchored)
        Ns = np.sort(sub['N'].unique())
        
        # O(N^-0.5): Anchor to CMC+ODIS
        cmc_odis = sub[sub['Method'] == 'CMC+ODIS']
        if not cmc_odis.empty:
            rmse_0 = cmc_odis['RMSE'].values[0]
            n_0 = cmc_odis['N'].values[0]
            ref_05 = Ns**(-0.5) * (rmse_0 * n_0**0.5)
            plt.loglog(Ns, ref_05, 'k--', alpha=0.5, label='$O(N^{-0.5})$', base=2)
            
        # O(N^-1.0): Anchor to RQMC+ODIS
        rqmc_odis = sub[sub['Method'] == 'RQMC+ODIS']
        if not rqmc_odis.empty:
            rmse_0 = rqmc_odis['RMSE'].values[0]
            n_0 = rqmc_odis['N'].values[0]
            ref_10 = Ns**(-1.0) * (rmse_0 * n_0**1.0)
            plt.loglog(Ns, ref_10, 'k:', alpha=0.5, label='$O(N^{-1.0})$', base=2)
            
        plt.title(f"K=120 Comparison (ODIS): {opt} Option (m={m})", fontsize=14)
        plt.xlabel("Sample Size N (log2)", fontsize=12)
        plt.ylabel("RMSE", fontsize=12)
        plt.grid(True, which="both", alpha=0.2)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"plots_project_solutions/K120_{opt}_m{m}.png", dpi=300)
        plt.close()


# ==========================================
# Main Execution Loop
# ==========================================

if __name__ == "__main__":
    
    # Text file for output
    txt_file = open("required_N_results.txt", "w")
    txt_file.write(f"{'m':<5} | {'Option':<12} | {'Method':<20} | {'Req N':<10} | {'Price':<10}\n")
    txt_file.write("-" * 70 + "\n")
    
    # Loop over discretization parameters
    for m in M_VALUES:
        print(f"\n{'#'*60}")
        print(f"Processing m = {m}")
        print(f"{'#'*60}")
        
        # Generate ground truths (Need K=100 and K=120)
        print(f"  -> Generating Ground Truths...")
        truth = {}
        
        # K=100 Truths (for Q1/Q2)
        gt_d100 = create_model(S0, 100, T, r, sigma, m)
        u_d100 = get_active_subspace(gt_d100, payoff_arithmetic)
        Q_d100 = householder_matrix(u_d100)
        truth['Digital100'] = np.mean(vector_pre_int_digital(gt_d100, 2**17, u_d100, Q_d100, None, 'RQMC'))
        
        gt_a100 = create_model(S0, 100, T, r, sigma, m)
        u_a100 = get_active_subspace(gt_a100, payoff_arithmetic)
        Q_a100 = householder_matrix(u_a100)
        truth['Arithmetic100'] = np.mean(vector_pre_int_arithmetic(gt_a100, 2**17, u_a100, Q_a100, None, 'RQMC'))
        
        # K=120 Truths (for Q3/Q4)
        gt_d120 = create_model(S0, 120, T, r, sigma, m)
        u_d120 = get_active_subspace(gt_d120, payoff_arithmetic)
        Q_d120 = householder_matrix(u_d120)
        truth['Digital120'] = np.mean(vector_pre_int_digital(gt_d120, 2**17, u_d120, Q_d120, None, 'RQMC'))

        gt_a120 = create_model(S0, 120, T, r, sigma, m)
        u_a120 = get_active_subspace(gt_a120, payoff_arithmetic)
        Q_a120 = householder_matrix(u_a120)
        truth['Arithmetic120'] = np.mean(vector_pre_int_arithmetic(gt_a120, 2**17, u_a120, Q_a120, None, 'RQMC'))
        
        print(f"    Truths: {truth}")

        # Q1 & Q2: K=100 Analysis
        df_q1 = run_question_1(m, truth)
        df_q2 = run_question_2(m, truth)
        
        # Plot K=100 (One plot per option type per m)
        plot_k100_results(df_q1, df_q2, m)
        
        # Q3 & Q4: K=120 Analysis
        df_q3_req, df_q3_conv, mu_dig, mu_ari = run_question_3(m, truth)
        df_q4_req, df_q4_conv = run_question_4(m, mu_dig, mu_ari, truth)
        
        # Plot K=120
        plot_k120_results(df_q3_conv, df_q4_conv, m)
        
        # Record Required N
        all_req = pd.concat([df_q3_req, df_q4_req])
        for _, row in all_req.iterrows():
            txt_file.write(f"{row['m']:<5} | {row['Option']:<12} | {row['Method']:<20} | {row['Req_N']:<10} | {row['Price']:<.4f}\n")
            print(f"    K=120 {row['Option']} {row['Method']}: Req N = {row['Req_N']}")

    txt_file.close()
    
    print("\n" + "="*60)
    print("Full Analysis Complete.")
    print("Plots saved to 'plots_project_solutions/'")
    print("Required N results saved to 'required_N_results.txt'")