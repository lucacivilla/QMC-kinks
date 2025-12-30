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

# ==========================================
# 0. Setup and Configuration
# ==========================================

if not os.path.exists('plots_arithmetic_asian'):
    os.makedirs('plots_arithmetic_asian')

# Model Parameters
S0 = 100
T = 1.0
r = 0.1
sigma = 0.1
d = 32

# Simulation Parameters
# N as powers of 2
POWERS = np.arange(7, 14) # 7 to 13
SAMPLE_SIZES = 2**POWERS
N_REPEATS = 100 # Repetitions for stable RMSE

# ==========================================
#           Adaptive N Finder
# ==========================================

def find_N_specific_arithmetic(model, tol=0.01):
    """
    Sequential search starting at N=128.
    Uses specific recursive update formulas for Mean and Variance.
    """
    z_score = 1.96  # 95% Confidence
    
    # 1. Start with N = 128
    N = 128
    # Generate initial batch
    z_init = np.random.standard_normal((N, model.d))
    # Z represents the random variable (Payoff)
    Z_values = model.payoff(z_init) 
    
    # Initial Statistics
    mu = np.mean(Z_values)
    var = np.var(Z_values, ddof=1)
    
    print(f"\n--- Specific Recursive Search (Arithmetic, Start N={N}) ---")
    print(f"{'N':<10} | {'Mean':<10} | {'Var':<10} | {'RelErr':<10}")
    print("-" * 55)
    
    while True:
        # -------------------------------------------------------
        # 2. Generate new single Z_(N+1)
        # We generate a new path z (standard normal vector) and get its payoff Z
        # -------------------------------------------------------
        z_new = np.random.standard_normal((1, model.d))
        Z_next = model.payoff(z_new)[0]
        
        # Save previous mean for variance update
        mu_old = mu
        
        # -------------------------------------------------------
        # 3. Update Statistics using your specific formulas
        # -------------------------------------------------------
        
        # Update Mean: mu_new = (N / N+1) * mu_old + Z_next / (N+1)
        mu = (N / (N + 1)) * mu + Z_next / (N + 1)
        
        # Update Variance: sigma^2_new = ((N-1)/N)*sigma^2_old + (Z_next - mu_old)^2 / (N+1)
        # Note: We use the *old* mean (mu_old) inside the squared term as per standard Welford-like logic
        # matching the structure you provided.
        var = ((N - 1) / N) * var + (Z_next - mu_old)**2 / (N + 1)
        
        # 4. Increment N
        N += 1
        
        # 5. Check Convergence
        sigma = np.sqrt(max(var, 0.0))
        
        if abs(mu) > 1e-12 and sigma > 0:
            half_width = z_score * sigma / np.sqrt(N)
            rel_err = half_width / abs(mu)
            
            if rel_err <= tol:
                print("-" * 55)
                print(f"Converged at N = {N}")
                print(f"Final Mean: {mu:.6f}")
                print(f"Final Var:  {var:.6f}")
                return N, mu
            
            if N % 5000 == 0:
                print(f"{N:<10} | {mu:<10.4f} | {var:<10.4f} | {rel_err:<10.2%}")

# ==========================================
# 1. Model: Asian Option with Cholesky Path
# ==========================================

class AsianOptionCholesky:
    def __init__(self, S0=100, K=100, T=1.0, r=0.1, sigma=0.1, d=32):
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.d = d
        self.dt = T / d
        
        # 1. Covariance Matrix
        times = np.linspace(self.dt, T, d)
        C = np.minimum(times[:, None], times[None, :])
        
        # 2. Cholesky Decomposition (L @ L.T = C)
        self.A = np.linalg.cholesky(C)
        
        # Precompute drift
        self.drift_term = (self.r - 0.5 * self.sigma**2) * times

    def payoff(self, z):
        if z.ndim == 1: z = z.reshape(1, -1)
        B = z @ self.A.T
        S = self.S0 * np.exp(self.drift_term + self.sigma * B)
        S_mean = np.mean(S, axis=1)
        return np.exp(-self.r * self.T) * np.maximum(S_mean - self.K, 0)

# ==========================================
# 2. Directions & Shifts
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
    """Estimate dominant gradient direction (Active Subspace)"""
    sampler = qmc.Sobol(d=model.d, scramble=True)
    z_pilot = stats.norm.ppf(sampler.random(M))
    
    grads = []
    for i in range(M):
        g = get_gradient(model.payoff, z_pilot[i])
        grads.append(g)
    grads = np.array(grads)
    
    C_hat = (grads.T @ grads) / M
    evals, evecs = np.linalg.eigh(C_hat)
    u_as = evecs[:, -1]
    if np.sum(u_as) < 0: u_as = -u_as
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
    """Optimal Drift for Importance Sampling"""
    def obj(z):
        p = model.payoff(z)[0]
        if p <= 1e-12: return 1e6
        return 0.5 * np.sum(z**2) - np.log(p)
    
    # Multi-start to avoid local minima in OTM cases
    best_res = None
    best_val = np.inf
    starts = [np.zeros(model.d), np.ones(model.d)*0.5, np.ones(model.d)*1.5]
    
    for x0 in starts:
        res = minimize(obj, x0, method='L-BFGS-B')
        if res.fun < best_val:
            best_val = res.fun
            best_res = res
    return best_res.x

# ==========================================
# 3. Estimators
# ==========================================

def standard_estimator(model, N, method='MC', mu_shift=None):
    if method == 'MC':
        z = np.random.standard_normal((N, model.d))
    elif method == 'RQMC':
        sampler = qmc.Sobol(d=model.d, scramble=True)
        z = stats.norm.ppf(sampler.random(N))
    
    weights = np.ones(N)
    if mu_shift is not None:
        # Importance Sampling Weight
        # We treat 'z' as the base noise. The sample is X = z + mu.
        # Weight = exp(-X.mu + 0.5*mu^2)
        X = z + mu_shift
        dot = np.sum(X * mu_shift, axis=1)
        mu_sq = 0.5 * np.sum(mu_shift**2)
        weights = np.exp(-dot + mu_sq)
        payoffs = model.payoff(X)
    else:
        payoffs = model.payoff(z)
        
    return np.mean(payoffs * weights)

def pre_int_estimator(model, N, u, Q, mu_perp=None, method='RQMC'):
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
    
    for i in range(N):
        alpha = np.exp(const_part + exponent_perps[i])
        def g(v): return np.mean(alpha * np.exp(beta * v)) - model.K
        
        try: v_star = brentq(g, -30, 30)
        except ValueError: v_star = -30 if g(0) > 0 else 30
            
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
# 4. Simulation Logic
# ==========================================

def run_experiment(K_target):
    print(f"\n--- Running Experiment for K = {K_target} ---")
    model = AsianOptionCholesky(S0=S0, K=K_target, T=T, r=r, sigma=sigma, d=d)
    
    # Directions
    u_z1 = get_z1_direction(d)
    Q_z1 = np.eye(d)
    u_as = get_active_subspace(model)
    Q_as = householder_matrix(u_as)
    
    # ODIS Shift (Only needed for K=120)
    mu_opt = None
    mu_perp_as = None
    if K_target == 120:
        mu_opt = get_odis_shift(model)
        mu_local = Q_as.T @ mu_opt 
        mu_perp_as = mu_local[1:] 

    # Ground Truth
    if K_target == 120:
        true_val = pre_int_estimator(model, 2**17, u_as, Q_as, mu_perp_as, 'RQMC')
    else:
        true_val = pre_int_estimator(model, 2**17, u_as, Q_as, None, 'RQMC')
    print(f"  -> Truth: {true_val:.6f}")

    methods = {}
    if K_target == 100:
        # Comparison: MC vs RQMC vs PreInt(z1) vs PreInt(AS) (No ODIS)
        methods['Crude MC'] = lambda n: standard_estimator(model, n, 'MC')
        methods['Plain RQMC'] = lambda n: standard_estimator(model, n, 'RQMC')
        methods['Pre-Int (z1)'] = lambda n: pre_int_estimator(model, n, u_z1, Q_z1, None, 'RQMC')
        methods['Pre-Int (AS)'] = lambda n: pre_int_estimator(model, n, u_as, Q_as, None, 'RQMC')
        
    elif K_target == 120:
        # Comprehensive List
        methods['Crude MC'] = lambda n: standard_estimator(model, n, 'MC')
        methods['MC + ODIS'] = lambda n: standard_estimator(model, n, 'MC', mu_opt)
        methods['Plain RQMC'] = lambda n: standard_estimator(model, n, 'RQMC')
        methods['RQMC + ODIS'] = lambda n: standard_estimator(model, n, 'RQMC', mu_opt)
        methods['Pre-Int (AS)'] = lambda n: pre_int_estimator(model, n, u_as, Q_as, None, 'RQMC')
        methods['Pre-Int (AS) + ODIS'] = lambda n: pre_int_estimator(model, n, u_as, Q_as, mu_perp_as, 'RQMC')

    results = []
    total_ops = len(SAMPLE_SIZES) * len(methods) * N_REPEATS
    with tqdm(total=total_ops) as pbar:
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
                results.append({'K': K_target, 'N': N, 'Method': name, 'RMSE': rmse, 'Time': avg_time})
                
    return pd.DataFrame(results)

# ==========================================
# 5. Plotting Functions
# ==========================================

def get_convergence_rate(N, RMSE):
    # Fit log(RMSE) = a + b * log(N)
    # Slope b is the convergence rate
    slope, intercept = np.polyfit(np.log(N), np.log(RMSE), 1)
    return slope

def plot_k100(df):
    """
    K=100 Plot: 
    Left: Convergence (Log2 N vs RMSE)
    Right: Cost (Time vs RMSE)
    """
    df = df[df['K'] == 100]
    if df.empty: return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Convergence Plot
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

    ax.set_title('K=100: Convergence Analysis (Smoothing)', fontsize=14)
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

    ax.set_title('K=100: Efficiency (Error vs Time)', fontsize=14)
    ax.set_xlabel('Avg Computation Time (s)', fontsize=12)
    ax.set_ylabel('RMSE', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", ls="-", alpha=0.2)

    plt.tight_layout()
    plt.savefig('plots_arithmetic_asian/Arithmetic_Asian_K100_Smoothing_Analysis.png', dpi=300)
    print("Saved Arithmetic_Asian_K100_Smoothing_Analysis.png")

def plot_k120(df):
    """
    K=120 Plots.
    Plot A: Comprehensive (All methods).
    Plot B: Variance Reduction Focus (Method vs Method+ODIS).
    """
    df = df[df['K'] == 120]
    if df.empty: return

    # --- Plot A: Comprehensive ---
    plt.figure(figsize=(10, 7))
    methods = df['Method'].unique()
    
    for m in methods:
        sub = df[df['Method'] == m]
        slope = get_convergence_rate(sub['N'], sub['RMSE'])
        label_str = f"{m} ($N^{{{slope:.2f}}}$)"
        plt.loglog(sub['N'], sub['RMSE'], marker='o', label=label_str, base=2)

    plt.title('K=120: Comprehensive Comparison', fontsize=14)
    plt.xlabel('Sample Size $N$ (log2)', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    plt.grid(True, which="both", alpha=0.2)
    plt.legend()
    plt.savefig('plots_arithmetic_asian/Arithmetic_Asian_K120_Comprehensive.png', dpi=300)
    print("Saved Arithmetic_Asian_K120_Comprehensive.png")
    plt.close()

    # --- Plot B: Variance Reduction Focus ---
    plt.figure(figsize=(10, 7))
    
    # Pairs to compare
    pairs = [
        ('Crude MC', 'MC + ODIS', 'red'),
        ('Plain RQMC', 'RQMC + ODIS', 'blue'),
        ('Pre-Int (AS)', 'Pre-Int (AS) + ODIS', 'green')
    ]
    
    for base, odis, color in pairs:
        # Plot Base
        sub_b = df[df['Method'] == base]
        if not sub_b.empty:
            plt.loglog(sub_b['N'], sub_b['RMSE'], color=color, linestyle='--', marker='o', label=base, base=2, alpha=0.5)
        
        # Plot ODIS
        sub_o = df[df['Method'] == odis]
        if not sub_o.empty:
            slope = get_convergence_rate(sub_o['N'], sub_o['RMSE'])
            label_str = f"{odis} ($N^{{{slope:.2f}}}$)"
            plt.loglog(sub_o['N'], sub_o['RMSE'], color=color, linestyle='-', marker='D', label=label_str, base=2)

    plt.title('K=120: Impact of Variance Reduction (ODIS)', fontsize=14)
    plt.xlabel('Sample Size $N$ (log2)', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    plt.grid(True, which="both", alpha=0.2)
    plt.legend()
    plt.savefig('plots_arithmetic_asian/Arithmetic_Asian_K120_Variance_Impact.png', dpi=300)
    print("Saved Arithmetic_Asian_K120_Variance_Impact.png")
    plt.close()

# ==========================================
# 6. Main Execution
# ==========================================

if __name__ == "__main__":
    # 1. Run Experiments
    df100 = run_experiment(100)
    df120 = run_experiment(120)
    
    full_df = pd.concat([df100, df120])
    full_df.to_csv('arithmetic_asian_results.csv', index=False)
    print("\nResults saved to CSV.")
    
    # 2. Generate Plots
    plot_k100(full_df)
    plot_k120(full_df)

    print("\n" + "="*50)
    print("Find N (Arithmetic)")
    print("="*50)

    model_test = AsianOptionCholesky(S0=S0, K=100, T=T, r=r, sigma=sigma, d=d)
    
    final_N, final_price = find_N_specific_arithmetic(model_test, tol=0.01)
    
    with open('required_N_arithmetic.txt', 'w') as f:
        f.write(str(final_N))
    print(f"Saved N={final_N} to 'required_N_arithmetic.txt'")
    
    print("\nProject Complete.")
    