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

# ==========================================
# Setup and Configuration
# ==========================================

# Model Parameters
S0 = 100
T = 1.0
r = 0.1
sigma = 0.1

# Dimensions to test
DIMENSIONS = [32, 64, 128, 256, 512]

# Simulation Parameters
POWERS = np.arange(7, 14) # 7 to 13
SAMPLE_SIZES = 2**POWERS
N_REPEATS = 50 # Repetitions for stable RMSE

def find_N_specific_digital(model, d, tol=0.01):
    """
    Sequential search starting at N=128.
    Uses specific recursive update formulas for Mean and Variance.
    """
    z_score = 1.96  # 95% Confidence
   
    N = 128
    z_init = np.random.standard_normal((N, model.d))
    Z_values = model.payoff_digital(z_init)
   
    # Initial Statistics
    mu = np.mean(Z_values)
    var = np.var(Z_values, ddof=1)
   
    print(f"\n--- Specific Recursive Search (Digital, d={d}, Start N={N}) ---")
    print(f"{'N':<10} | {'Mean':<10} | {'Var':<10} | {'RelErr':<10}")
    print("-" * 55)
   
    while True:
        # Generate new single Z_(N+1)
        z_new = np.random.standard_normal((1, model.d))
        Z_next = model.payoff_digital(z_new)[0]
       
        mu_old = mu
       
        # Update Statistics
        mu = (N / (N + 1)) * mu + Z_next / (N + 1)
        var = ((N - 1) / N) * var + (Z_next - mu_old)**2 / (N + 1)
       
        N += 1
       
        # Check Convergence
        sigma_est = np.sqrt(max(var, 0.0))
       
        if abs(mu) > 1e-12 and sigma_est > 0:
            half_width = z_score * sigma_est / np.sqrt(N)
            rel_err = half_width / abs(mu)
           
            if rel_err <= tol:
                print("-" * 55)
                print(f"Converged at N = {N} for d={d}")
                print(f"Final Mean: {mu:.6f}")
                print(f"Final Var:  {var:.6f}")
                return N, mu
           
            if N % 5000 == 0:
                print(f"{N:<10} | {mu:<10.4f} | {var:<10.4f} | {rel_err:<10.2%}")

# ==========================================
# Model: Digital Asian Option
# ==========================================

class DigitalAsianOption:
    def __init__(self, S0=100, K=100, T=1.0, r=0.1, sigma=0.1, d=32):
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.d = d
        self.dt = T / d
       
        # 1. Covariance Matrix construction
        times = np.linspace(self.dt, T, d)
        C = np.minimum(times[:, None], times[None, :])
       
        # Cholesky Decomposition (L @ L.T = C)
        self.A = np.linalg.cholesky(C)
       
        # Precompute drift for simulation
        self.drift_term = (self.r - 0.5 * self.sigma**2) * times

    def get_S_mean(self, z):
        """Helper to get arithmetic mean of prices given z."""
        if z.ndim == 1: z = z.reshape(1, -1)
        B = z @ self.A.T
        S = self.S0 * np.exp(self.drift_term + self.sigma * B)
        return np.mean(S, axis=1)

    def payoff_digital(self, z):
        """Digital Payoff: 1 if S_mean > K else 0."""
        S_mean = self.get_S_mean(z)
        p = np.where(S_mean > self.K, 1.0, 0.0)
        return np.exp(-self.r * self.T) * p
   
    def payoff_arithmetic(self, z):
        """Arithmetic Payoff (proxy for AS gradient): max(S_mean - K, 0)."""
        S_mean = self.get_S_mean(z)
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
    Estimate dominant gradient direction using the Arithmetic payoff as a proxy.
    """
    sampler = qmc.Sobol(d=model.d, scramble=True)
    z_pilot = stats.norm.ppf(sampler.random(M))
   
    grads = []
    # Use Arithmetic payoff for gradient estimation
    for i in range(M):
        g = get_gradient(model.payoff_arithmetic, z_pilot[i])
        grads.append(g)
    grads = np.array(grads)
   
    C_hat = (grads.T @ grads) / M
    evals, evecs = np.linalg.eigh(C_hat)
    u_as = evecs[:, -1]
   
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
    Optimal Drift for Importance Sampling (Digital Option).
    """
    def constr(z):
        return model.get_S_mean(z)[0] - model.K
   
    def obj(z):
        return 0.5 * np.sum(z**2)

    x0 = np.ones(model.d) * 0.1
    cons = ({'type': 'eq', 'fun': constr})
    res = minimize(obj, x0, method='SLSQP', constraints=cons, tol=1e-4)
   
    if not res.success:
        return np.zeros(model.d)
       
    return res.x

# ==========================================
# Estimators
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
        X = z + mu_shift
        dot = np.sum(X * mu_shift, axis=1)
        mu_sq = 0.5 * np.sum(mu_shift**2)
        weights = np.exp(-dot + mu_sq)
        payoffs = model.payoff_digital(X)
    else:
        payoffs = model.payoff_digital(z)
       
    return np.mean(payoffs * weights)

def pre_int_estimator_closed_form(model, N, u, Q, mu_perp=None, method='RQMC'):
    """
    Pre-integration estimator using Closed-Form solution.
    P(Exercise | z_perp) = Phi(-v_star)
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

    U_perp = Q[:, 1:]
    Au = model.A @ u
    AU_perp_z_perp = (model.A @ U_perp) @ z_perp.T
   
    beta = model.sigma * Au
    const_log_S = np.log(model.S0) + model.drift_term
   
    estimates = np.zeros(N)
   
    # We search for root in [-30, 30] to cover the relevant probability mass
    BOUND = 30.0
   
    for i in range(N):
        perp_contribution = model.sigma * AU_perp_z_perp[:, i]
        alpha = np.exp(const_log_S + perp_contribution)
       
        def g(v):
            return np.mean(alpha * np.exp(beta * v)) - model.K
       
        # Root Finding
        try:
            if g(-BOUND) * g(BOUND) < 0:
                v_star = brentq(g, -BOUND, BOUND)
            else:
                v_star = -BOUND if g(0) > 0 else BOUND
        except ValueError:
             v_star = -BOUND if g(0) > 0 else BOUND
       
        # Closed-Form Integration
        val = np.exp(-model.r * model.T) * stats.norm.cdf(-v_star)
        estimates[i] = val * weights[i]
       
    return np.mean(estimates)

# ==========================================
# Simulation Logic
# ==========================================

def run_experiment(K_target, d_val):
    print(f"\n--- Running Experiment for K = {K_target}, d = {d_val} (Digital) ---")
    model = DigitalAsianOption(S0=S0, K=K_target, T=T, r=r, sigma=sigma, d=d_val)
   
    # Directions
    u_z1 = get_z1_direction(d_val)
    Q_z1 = np.eye(d_val)
    u_as = get_active_subspace(model)
    Q_as = householder_matrix(u_as)
   
    # ODIS Shift (Only needed for K=120)
    mu_opt = None
    mu_perp_as = None
    if K_target == 120:
        print("  -> Computing ODIS shift...")
        mu_opt = get_odis_shift(model)
        mu_local = Q_as.T @ mu_opt
        mu_perp_as = mu_local[1:]

    # Ground Truth Generation
    print("  -> Computing Ground Truth...")
    if K_target == 120:
        true_val = pre_int_estimator_closed_form(model, 2**17, u_as, Q_as, mu_perp_as, 'RQMC')
    else:
        true_val = pre_int_estimator_closed_form(model, 2**17, u_as, Q_as, None, 'RQMC')
    print(f"  -> Truth: {true_val:.6f}")

    methods = {}
    if K_target == 100:
        methods['Crude MC'] = lambda n: standard_estimator(model, n, 'MC')
        methods['Plain RQMC'] = lambda n: standard_estimator(model, n, 'RQMC')
        methods['Pre-Int (z1)'] = lambda n: pre_int_estimator_closed_form(model, n, u_z1, Q_z1, None, 'RQMC')
        methods['Pre-Int (AS)'] = lambda n: pre_int_estimator_closed_form(model, n, u_as, Q_as, None, 'RQMC')
       
    elif K_target == 120:
        methods['Crude MC'] = lambda n: standard_estimator(model, n, 'MC')
        methods['MC + ODIS'] = lambda n: standard_estimator(model, n, 'MC', mu_opt)
        methods['Plain RQMC'] = lambda n: standard_estimator(model, n, 'RQMC')
        methods['RQMC + ODIS'] = lambda n: standard_estimator(model, n, 'RQMC', mu_opt)
        methods['Pre-Int (AS)'] = lambda n: pre_int_estimator_closed_form(model, n, u_as, Q_as, None, 'RQMC')
        methods['Pre-Int (AS) + ODIS'] = lambda n: pre_int_estimator_closed_form(model, n, u_as, Q_as, mu_perp_as, 'RQMC')

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
    slope, intercept = np.polyfit(np.log(N), np.log(RMSE), 1)
    return slope

def plot_k100(df, d_val, save_dir):
    """
    K=100 Plot:
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
    K=120 Plots.
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