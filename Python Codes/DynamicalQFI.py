import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import residue
from scipy.linalg import matmul_toeplitz
from tqdm import tqdm

# ==========================================
# 1. HELPER FUNCTIONS & INVERSE LAPLACE
# ==========================================
def coth(z):
    return np.cosh(z) / np.sinh(z)

def csch2(z):
    s = np.sinh(z)
    return 1.0 / (s * s)

def inverse_laplace_rational(B, A, enforce_real=True):
    B = np.atleast_1d(np.array(B, dtype=np.complex128))
    A = np.atleast_1d(np.array(A, dtype=np.complex128))
    r, p, k = residue(B, A)

    info = {"residues": r, "poles": p, "direct_poly": k}
    return info

def derivatives_from_residues(t, residues, poles, enforce_real=True):
    t = np.asarray(t, dtype=float)
    G     = np.zeros_like(t, dtype=np.complex128)
    Gdot  = np.zeros_like(t, dtype=np.complex128)
    
    for r, p in zip(residues, poles):
        ep = np.exp(p * t)
        G    += r * ep
        Gdot += r * p * ep
        
    if enforce_real:
        G    = np.real_if_close(G)
        Gdot = np.real_if_close(Gdot)
        
    return G, Gdot

# ==========================================
# 2. NOISE KERNEL & TEMPERATURE DERIVATIVE
# ==========================================
def noise_kernel_residue(t, gamma, alpha1, alpha2, Temp, Nmats=20000, delta_eps=1e-10):
    scalar_input = np.isscalar(t)
    t = np.atleast_1d(np.asarray(t, dtype=np.float64))
    t = np.abs(t)
    
    Delta = 4.0 * alpha2 * gamma - gamma**2
    beta = 1.0 / Temp
    a = 0.5 * beta

    if abs(Delta) < delta_eps:
        w0 = 0.5j * gamma
        exp0 = np.exp(1j * w0 * t)
        coth0 = coth(a * w0)
        fprime = (a * (-csch2(a * w0)) + 1j * t * coth0) * exp0
        nu_sys = (np.pi * gamma * alpha1 * alpha2 / 2.0) * np.real(fprime)
    else:
        sqrtDelta = np.sqrt(Delta + 0j)  
        w_plus  = 0.5j * gamma + 0.5 * sqrtDelta
        w_minus = 0.5j * gamma - 0.5 * sqrtDelta
        term_plus  = coth(a * w_plus)  * np.exp(1j * w_plus  * t)
        term_minus = coth(a * w_minus) * np.exp(1j * w_minus * t)
        pref = ( gamma * alpha1 * alpha2) / (2.0 * sqrtDelta)
        nu_sys = np.real(pref * (term_plus - term_minus))

    n = np.arange(1, Nmats+1, dtype=np.float64)
    nu_n = (2.0 * np.pi / beta) * n
    denom = (alpha2 * gamma + nu_n**2)**2 - (gamma * nu_n)**2
    expo = np.exp(-nu_n[:, None] * t[None, :])
    mats_sum = (nu_n / denom)[:, None] * expo
    nu_M = -(2.0  * gamma**2 * alpha1 * alpha2 / beta) * mats_sum.sum(axis=0)

    out = (nu_sys + nu_M)
    return float(out[0]) if scalar_input else out

def dnu_dT_residue(t, gamma, alpha1, alpha2, Temp, Nmats=20000, delta_eps=1e-10):
    scalar_input = np.isscalar(t)
    t = np.atleast_1d(np.asarray(t, dtype=np.float64))
    t = np.abs(t)
    
    Delta = 4.0 * alpha2 * gamma - gamma**2
    beta = 1.0 / Temp
    T2 = Temp**2

    if abs(Delta) < delta_eps:
        w0 = 0.5j * gamma
        a = 0.5 * beta
        exp0 = np.exp(1j * w0 * t)
        nu_sys_deriv = (np.pi * gamma * alpha1 * alpha2 / (4.0 * T2)) * np.real(w0 * csch2(a * w0) * exp0) 
    else:
        sqrtDelta = np.sqrt(Delta + 0j)  
        w_plus  = 0.5j * gamma + 0.5 * sqrtDelta
        w_minus = 0.5j * gamma - 0.5 * sqrtDelta
        term_plus  = w_plus * csch2(0.5 * beta * w_plus) * np.exp(1j * w_plus * t)
        term_minus = w_minus * csch2(0.5 * beta * w_minus) * np.exp(1j * w_minus * t)
        pref = -(gamma * alpha1 * alpha2) / (4.0 * T2 * sqrtDelta)
        nu_sys_deriv = np.real(pref * (term_plus - term_minus))

    k = np.arange(1, Nmats + 1, dtype=np.float64)
    fk = (2.0 * np.pi / beta) * k
    
    denom = (alpha2 * gamma + fk**2)**2 - (gamma * fk)**2
    expo = np.exp(-fk[:, None] * t[None, :])
    
    term_A_sum = (fk[:, None] / denom[:, None]) * expo
    term_A = - (2.0 * gamma**2 * alpha1 * alpha2) * term_A_sum.sum(axis=0)
    
    num_part1 = expo * (1 - t[None, :] * fk[:, None]) * denom[:, None]

    # Calculate the purely k-dependent polynomial in 1D first
    inner_k_term = 4 * fk * (alpha2 * gamma + fk**2) - 2 * gamma**2 * fk

    # Then broadcast it against the time-dependent expo matrix
    num_part2 = fk[:, None] * expo * inner_k_term[:, None]
    term_B_sum = k[:, None] * ((num_part1 - num_part2) / (denom[:, None]**2))
    term_B = - (4.0 * np.pi * gamma**2 * alpha1 * alpha2 / beta) * term_B_sum.sum(axis=0)

    nu_M_deriv = term_A + term_B
    out = nu_sys_deriv + nu_M_deriv
    
    return float(out[0]) if scalar_input else out

# ==========================================
# 3. DYNAMICAL EVOLUTION & FISHER INFORMATION
# ==========================================
def evolve_FI_dynamics(t_array, sigma0, sigma_M, alpha1, alpha2, Temp, gamma, G, Gdot, mass=1.0):
    dt = t_array[1] - t_array[0]
    N = len(t_array)
    
    G_r = G.real
    Gdot_r = Gdot.real
    
    # System contribution to Covariance (Constant T-derivative = 0)
    sx0, sp0, sxp0 = sigma0[0, 0], sigma0[1, 1], sigma0[0, 1]
    var_x_sys = (Gdot_r**2) * sx0 + (G_r / mass)**2 * sp0 + 2 * Gdot_r * (G_r / mass) * sxp0
    a, b = mass * Gdot_r, Gdot_r 
    var_p_sys = (a**2) * sx0 + (b**2) * sp0 + 2 * a * b * sxp0
    cov_xp_sys = Gdot_r * a * sx0 + (G_r/mass) * b * sp0 + (Gdot_r * b + (G_r/mass)*a)*sxp0

    # Precompute noise kernel and its Temperature derivative
    nu_grid = np.array([noise_kernel_residue(t, gamma, alpha1, alpha2, Temp) for t in t_array]) / mass
    dnu_grid = dnu_dT_residue(t_array, gamma, alpha1, alpha2, Temp) / mass 

    fisher_info_t = np.zeros(N)
    var_x_total = np.zeros(N)
    var_p_total = np.zeros(N)
    QFI = np.zeros(N)

    #QFI
    Omega_symp = np.array([[0, 1], [-1, 0]])
    kron_omega = np.kron(Omega_symp, Omega_symp)
    
    for i in tqdm(range(1, N), desc="Simulating Dynamics & FI"):
        G_slice = G_r[:i+1][::-1] 
        Gd_slice = Gdot_r[:i+1][::-1] 
        
        nu_seg = nu_grid[:i+1]  
        dnu_seg = dnu_grid[:i+1]
        
        # Standard Covariance Integrals (Toeplitz)
        yG = matmul_toeplitz((nu_seg, nu_seg), G_slice)
        yGd = matmul_toeplitz((nu_seg, nu_seg), Gd_slice)
        
        var_x_bath = np.dot(G_slice, yG) * dt**2
        var_p_bath = np.dot(Gd_slice, yGd) * dt**2
        cov_xp_bath = np.dot(G_slice, yGd) * dt**2
        
        sigma_t = np.array([
            [var_x_sys[i] + var_x_bath, cov_xp_sys[i] + cov_xp_bath],
            [cov_xp_sys[i] + cov_xp_bath, var_p_sys[i] + var_p_bath]
        ])
        
        var_x_total[i] = sigma_t[0, 0]
        var_p_total[i] = sigma_t[1, 1]
        
        # Derivative Covariance Integrals (Toeplitz)
        dyG = matmul_toeplitz((dnu_seg, dnu_seg), G_slice)
        dyGd = matmul_toeplitz((dnu_seg, dnu_seg), Gd_slice)
        
        dvar_x_bath = np.dot(G_slice, dyG) * dt**2
        dvar_p_bath = np.dot(Gd_slice, dyGd) * dt**2
        dcov_xp_bath = np.dot(G_slice, dyGd) * dt**2
        
        dsigma_t = np.array([
            [dvar_x_bath, dcov_xp_bath],
            [dcov_xp_bath, dvar_p_bath]
        ])
        
        # Calculate Fisher Information Term
        inverse_term = np.linalg.inv(sigma_t + sigma_M)
        matrix_prod = np.dot(inverse_term, dsigma_t)
        fisher_info_t[i] = 0.5 * np.trace(np.dot(matrix_prod, matrix_prod))

        # Calculate QFI with a Truncated Spectral Inverse
        kron_sigma = np.kron(sigma_t, sigma_t)
        M = kron_sigma - kron_omega
        
        # 1. Get exact eigenvalues and eigenvectors (M is symmetric)
        evals, evecs = np.linalg.eigh(M)
        
        # 2. Filter out unphysical negative values and dangerously small positive ones.
        # Any eigenvalue < 1e-7 is numerical noise crossing the pure state boundary.
        threshold = 1e-7
        evals_inv = np.where(evals > threshold, 1.0 / evals, 0.0)
        
        # 3. Reconstruct the perfectly stable inverse matrix
        M_inv = evecs @ np.diag(evals_inv) @ evecs.T
        
        dsigma_vec = dsigma_t.flatten(order='F') 
        QFI[i] = 0.5 * np.dot(dsigma_vec.T, np.dot(M_inv, dsigma_vec))
            
    return {"time": t_array, "var_x": var_x_total, "var_p": var_p_total, "FI": fisher_info_t, "QFI": QFI}


# ==========================================
# 4. EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    # --- Physical Parameters (Replace with your actual values) ---
    gamma = 1.0
    alpha1 = 0.1
    alpha2 = 2.0
    Omega_S_bare = 1.0
    delta_Omega_s = np.sqrt(alpha1)
    Omega_S = np.sqrt(Omega_S_bare**2 + delta_Omega_s**2)
    temp = 1
    mass = 1.0
    
    # Simulation Time Grid
    times = np.linspace(0, 100, 500)
    
    # Laplace Polynomials for the Green's Function
    B = [1, gamma, alpha2 * gamma]
    A = [1, gamma, alpha2 * gamma + Omega_S**2, gamma * Omega_S**2, alpha2 * gamma * Omega_S**2 - gamma * alpha1 * alpha2]
    
    # Extract Green's Function and Derivative
    info = inverse_laplace_rational(B, A)
    G, Gdot = derivatives_from_residues(times, info["residues"], info["poles"])
    
    sigma0 = np.array([[2.0 / Omega_S_bare, 0], 
                       [0, Omega_S_bare / 2.0]])
    
    # ---------------------------------------------------------
    # DEFINE YOUR MEASUREMENT NOISE MATRIX HERE
    # Example 1: Homodyne (x-quadrature measurement) 
    sigma_M = np.array([[1e6, 0.0], [0.0, 0.0]])  # Large noise on p
    # Example 2: Heterodyne (symmetric measurement)
    #sigma_M = np.array([[0.5, 0.0], [0.0, 0.5]])
    # ---------------------------------------------------------

    # Run Simulation
    sol = evolve_FI_dynamics(times, sigma0, sigma_M, alpha1, alpha2, temp, gamma, G, Gdot, mass)

   # ==========================================
    # 5. PLOT RESULTS WITH ASYMPTOTES
    thermal_QFI = 0.25 * (Omega_S**2/temp**4) * csch2(Omega_S/(2*temp)) # Valid for HO
    
    
    plt.figure(figsize=(10, 6))
    plt.plot(sol["time"], sol["QFI"], color='black', label=r"$QFI$")
    #plt.axhline(thermal_QFI, color='black', linestyle='--', alpha=0.7, label=r"Thermal Asymptote $\mathcal{F}_{th}$")
    plt.xlabel("t")
    plt.ylabel(r"$\mathcal{F}_T$")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()