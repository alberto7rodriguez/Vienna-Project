import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import residue
from scipy.linalg import matmul_toeplitz
from tqdm import tqdm

plt.style.use(["science"])
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams.update({
    "axes.labelsize": 18,    # x- and y-label size
    "axes.titlesize": 18,    # title size
    "xtick.labelsize": 15,   # x-tick label size
    "ytick.labelsize": 15,   # y-tick label size
    "legend.fontsize": 34,   # legend text size
})

# ==========================================
# 1. HELPER FUNCTIONS & INVERSE LAPLACE
# ==========================================
def coth(z):
    z = np.asarray(z, dtype=np.complex128)
    out = np.empty_like(z)

    r = np.real(z)

    pos = r > 20
    neg = r < -20
    mid = ~(pos | neg)

    out[pos] = 1.0 + 2*np.exp(-2*z[pos])
    out[neg] = -1.0 - 2*np.exp(2*z[neg])
    out[mid] = np.cosh(z[mid])/np.sinh(z[mid])

    return out.item() if out.ndim == 0 else out


def csch2(z):
    z = np.asarray(z, dtype=np.complex128)
    out = np.empty_like(z)

    r = np.real(z)

    large = np.abs(r) > 20
    small = ~large

    out[large] = 4*np.exp(-2*np.abs(r[large]))*np.exp(-2j*np.imag(z[large])*0)

    s = np.sinh(z[small])
    out[small] = 1/(s*s)

    return out.item() if out.ndim == 0 else out

def inverse_laplace_rational(B, A, enforce_real=True):
    B = np.atleast_1d(np.array(B, dtype=np.complex128))
    A = np.atleast_1d(np.array(A, dtype=np.complex128))
    r, p, k = residue(B, A)

    info = {"residues": r, "poles": p, "direct_poly": k}
    return info

def derivatives_from_residues(t, residues, poles, enforce_real=True):
    t = np.asarray(t, dtype=float)

    G    = np.zeros_like(t, dtype=np.complex128)
    Gdot = np.zeros_like(t, dtype=np.complex128)
    Gddot= np.zeros_like(t, dtype=np.complex128)

    for r, p in zip(residues, poles):
        ep = np.exp(p * t)
        G     += r * ep
        Gdot  += r * p * ep
        Gddot += r * p * p * ep

    if enforce_real:
        G     = np.real_if_close(G)
        Gdot  = np.real_if_close(Gdot)
        Gddot = np.real_if_close(Gddot)

    return G, Gdot, Gddot

# ==========================================
# 2. NOISE KERNEL & TEMPERATURE DERIVATIVE
# ==========================================
def noise_kernel_residue(t, gamma, alpha1, alpha2, Temp, Nmats=50000, delta_eps=1e-10):
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

def dnu_dT_residue(t, gamma, alpha1, alpha2, Temp, Nmats=50000, delta_eps=1e-10):
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

def evolve_all_FI_dynamics(t_array, x0, p0, sigma0, alpha1, alpha2, Temp, gamma, G, Gdot, Gddot, mass=1.0):
    dt = t_array[1] - t_array[0]
    N = len(t_array)
    
    G_r = G.real
    Gdot_r = Gdot.real
    Gddot_r = Gddot.real
    
    # System contribution to Covariance (Constant T-derivative = 0)
    sx0, sp0, sxp0 = sigma0[0, 0], sigma0[1, 1], sigma0[0, 1]
    var_x_sys = (Gdot_r**2) * sx0 + (G_r / mass)**2 * sp0 + 2 * Gdot_r * (G_r / mass) * sxp0
    a, b = mass * Gddot_r, Gdot_r 
    var_p_sys = (a**2) * sx0 + (b**2) * sp0 + 2 * a * b * sxp0
    cov_xp_sys = Gdot_r * a * sx0 + (G_r/mass) * b * sp0 + (Gdot_r * b + (G_r/mass)*a)*sxp0

    # Precompute noise kernel and its Temperature derivative
    nu_grid = np.array([noise_kernel_residue(t, gamma, alpha1, alpha2, Temp) for t in t_array]) / mass
    dnu_grid = dnu_dT_residue(t_array, gamma, alpha1, alpha2, Temp) / mass 

    if not np.all(np.isfinite(nu_grid)):
        raise ValueError("nu_grid contains NaNs or infs")

    if not np.all(np.isfinite(dnu_grid)):
        raise ValueError("dnu_grid contains NaNs or infs")
    
    # Arrays to store tracking variables (add FI_rpd)
    # Arrays to store tracking variables (added FI_rpd_real)
    FI_het, FI_hom_x, FI_hom_p, QFI, FI_rpd, FI_rpd_real, covariance = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
    
    # Define Measurement Matrices
    sig_M_het = np.array([[0.5, 0.0], [0.0, 0.5]])
    sig_M_hom_x = np.array([[1e-6, 0.0], [0.0, 1e6]]) # Infinite noise on p
    sig_M_hom_p = np.array([[1e6, 0.0], [0.0, 1e-6]]) # Infinite noise on x

    # Thermal values for the FI
    FI_th_het = 0.5 * ((1/(Temp+sig_M_het[0,0]*Omega_S)**2)+(1/(Temp+sig_M_het[1,1])**2))
    FI_th_hom_x = 0.5 * ((1/(Temp + sig_M_hom_x[0,0]*Omega_S)**2)+(1/(Temp + sig_M_hom_x[1,1])**2))
    FI_th_hom_p = 0.5 * ((1/(Temp + sig_M_hom_p[0,0]*Omega_S)**2)+(1/(Temp + sig_M_hom_p[1,1])**2))

    
    def calc_classical_CFI(sig, dsig, sig_M):
        inv_term = np.linalg.inv(sig + sig_M)
        prod = np.dot(inv_term, dsig)
        return 0.5 * np.trace(np.dot(prod, prod))

    for i in tqdm(range(1, N), desc="Simulating Dynamics & FIs"):
        G_slice = G_r[:i+1][::-1] 
        Gd_slice = Gdot_r[:i+1][::-1] 
        nu_seg = nu_grid[:i+1]  
        dnu_seg = dnu_grid[:i+1]
        t_current = t_array[i]
        # Integrals (Toeplitz)
        yG = matmul_toeplitz((nu_seg, nu_seg), G_slice)
        yGd = matmul_toeplitz((nu_seg, nu_seg), Gd_slice)
        var_x_bath = np.dot(G_slice, yG) * dt**2
        var_p_bath = np.dot(Gd_slice, yGd) * dt**2
        cov_xp_bath = np.dot(G_slice, yGd) * dt**2
        
        sigma_t = np.array([
            [var_x_sys[i] + var_x_bath, cov_xp_sys[i] + cov_xp_bath],
            [cov_xp_sys[i] + cov_xp_bath, var_p_sys[i] + var_p_bath]
        ])
        
        dyG = matmul_toeplitz((dnu_seg, dnu_seg), G_slice)
        dyGd = matmul_toeplitz((dnu_seg, dnu_seg), Gd_slice)
        dvar_x_bath = np.dot(G_slice, dyG) * dt**2
        dvar_p_bath = np.dot(Gd_slice, dyGd) * dt**2
        dcov_xp_bath = np.dot(G_slice, dyGd) * dt**2
        
        dsigma_t = np.array([
            [dvar_x_bath, dcov_xp_bath],
            [dcov_xp_bath, dvar_p_bath]
        ])
        
        covariance[i] = cov_xp_sys[i] + cov_xp_bath

        # 1. Classical FIs
        FI_het[i] = (calc_classical_CFI(sigma_t, dsigma_t, sig_M_het))/t_current
        FI_hom_x[i] = (calc_classical_CFI(sigma_t, dsigma_t, sig_M_hom_x))/t_current
        FI_hom_p[i] = (calc_classical_CFI(sigma_t, dsigma_t, sig_M_hom_p))/t_current
        
        # --- On/Off Photodetection (RPD) FI ---
        A_rpd = sigma_t + sig_M_het
        det_A = np.linalg.det(A_rpd)
        inv_A = np.linalg.inv(A_rpd)
        trace_term = np.trace(np.dot(inv_A, dsigma_t))
        
        # Base vacuum probability without dark counts
        P_0_base = 1.0 / np.sqrt(det_A)
        
        # 1. Ideal RPD (P_D = 0.0)
        dP_0_ideal = -0.5 * P_0_base * trace_term
        FI_rpd[i] = ((dP_0_ideal**2) / (P_0_base * (1.0 - P_0_base) + 1e-15))/t_current
        
        # 2. Realistic RPD (P_D = 0.02)
        P_D_val = 0.07
        P_0_real = (1.0 - P_D_val) * P_0_base
        dP_0_real = -0.5 * P_0_real * trace_term
        FI_rpd_real[i] = ((dP_0_real**2) / (P_0_real * (1.0 - P_0_real) + 1e-15))/t_current
        
        # --- 3. Quantum FI (Using Analytical Formula from Appendix A) ---
        det_sigma = sigma_t[0,0]*sigma_t[1,1] - sigma_t[0,1]**2
        det_dsigma = dsigma_t[0,0]*dsigma_t[1,1] - dsigma_t[0,1]**2
        d_det_sigma = dsigma_t[0,0]*sigma_t[1,1] + sigma_t[0,0]*dsigma_t[1,1] - 2*sigma_t[0,1]*dsigma_t[0,1]
        
        epsilon = 1e-16 
        purity_term = np.maximum(det_sigma - 0.25, epsilon)
        
        term1 = 0.5 * (d_det_sigma**2) / (purity_term * (det_sigma + 0.25))
        term2 = det_dsigma / (det_sigma + 0.25)
        
        QFI[i] = (term1 - term2)/t_current
        
    return {
        "time": t_array, 
        "FI_het": FI_het, 
        "FI_hom_x": FI_hom_x, 
        "FI_hom_p": FI_hom_p, 
        "FI_rpd": FI_rpd,
        "FI_rpd_real": FI_rpd_real,  # <-- Add this line
        "QFI": QFI, 
        "FI_th_het": FI_th_het, 
        "FI_th_hom_x": FI_th_hom_x, 
        "FI_th_hom_p": FI_th_hom_p,
        "covariance" : covariance
    }


# ==========================================
# 4. EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    # --- Physical Parameters ---
    gamma = 1.0
    alpha1 = 1
    alpha2 = 10.0
    Omega_S_bare = 1.0
    delta_Omega_s = np.sqrt(alpha1)
    Omega_S = np.sqrt(Omega_S_bare**2 + delta_Omega_s**2)
    mass = 1.0
    
    # Simulation Time Grid
    # Simulation Time Grid (Main)
    times = np.linspace(0, 20, 5000)
    
    # --- NUEVO: Simulation Time Grid (Short for Inset) ---
    times_short = np.linspace(0, 0.2, 5000) 

    # Laplace Polynomials for Green's Function
    B = [1, gamma, alpha2 * gamma]
    A = [1, gamma, alpha2 * gamma + Omega_S**2, gamma * Omega_S**2, alpha2 * gamma * Omega_S**2 - gamma * alpha1 * alpha2]
    
    # Extract Green's Function (Main)
    info = inverse_laplace_rational(B, A)
    G, Gprime_of_t, Gddot = derivatives_from_residues(times, info["residues"], info["poles"])
    Gdot = Gprime_of_t
    
    # --- NUEVO: Extract Green's Function (Short) ---
    G_short, Gprime_short, Gddot_short = derivatives_from_residues(times_short, info["residues"], info["poles"])
    Gdot_short = Gprime_short
    
    # Initial State Covariance Matrix (Fixed grouping bug)
    x0 = 0.0
    p0 = 0.0
    sigma0 = np.array([[1.0 / (2.0 * Omega_S_bare), 0.0], 
                       [0.0, Omega_S_bare / 2.0]])

    # List of temperatures to plot
    temperatures = [0.01, 0.1, 1, 10]
    #temperatures = [30]
    
    # --- Setup 2x2 Subplot Grid ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()  # Flatten 2D array to 1D for easy iteration

    
    for ax, temp in zip(axes, temperatures):
        print(f"\nSimulating Dynamics & FIs for T = {temp}...")
        
        # Thermal QFI for a single HO
        QFI_th_HO = 0.25 * (Omega_S**2 / temp**4) * csch2(Omega_S / (2.0 * temp))
        
        # 1. Run Dynamic Simulation
        sol = evolve_all_FI_dynamics(times, x0, p0, sigma0, alpha1, alpha2, temp, gamma, G, Gdot, Gddot, mass)
        
        # 2. Plot Dynamical Curves
        ax.plot(sol["time"][1:], sol["QFI"][1:], color='black', linewidth=2.2, label=r"Quantum FI ($\mathcal{F}^Q$)")
        ax.plot(sol["time"][1:], sol["FI_het"][1:], color='blue', linewidth=1.5, label="Heterodyne")
        ax.plot(sol["time"][1:], sol["FI_hom_x"][1:], color='red', linewidth=1.5, label="Homodyne-x")
        ax.plot(sol["time"][1:], sol["FI_hom_p"][1:], color='green', linewidth=1.5, label="Homodyne-p")
        ax.plot(sol["time"][1:], sol["FI_rpd"][1:], color='purple', linewidth=1.5, linestyle='-.', label="Ideal RPD ($P_D=0$)")
        ax.plot(sol["time"][1:], sol["FI_rpd_real"][1:], color='darkorange', linewidth=1.5, linestyle=':', label="Realistic RPD ($P_D=0.07$)")
        
        # 4. Subplot Formatting
        ax.set_title(rf"$T/\Omega_S = {temp}$", fontsize=15)
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$\tilde{\mathcal{F}}$")
        ax.grid(True, alpha=0.3)
        #ax.legend(loc='best', fontsize=8)
        #ax.set_xlim(1e-1,10)
        #ax.set_xscale('log')


        # --- 3. Short-Time Inset (Alta Resolución) ---
        inset_ax = ax.inset_axes([0.52, 0.47, 0.48, 0.53]) 

        sol_short = evolve_all_FI_dynamics(times_short, x0, p0, sigma0, alpha1, alpha2, temp, gamma, G_short, Gdot_short, Gddot_short, mass)

        # 1. SALTAMOS los primeros 50 puntos para evitar el ruido numérico en t~0
        skip = 10
        t_s = times_short[skip:] 

        # Calculate noise kernel parameters at time zero
        nu_0 = noise_kernel_residue(0.0, gamma, alpha1, alpha2, temp) / mass
        dnu_0 = dnu_dT_residue(0.0, gamma, alpha1, alpha2, temp) / mass

        x2_0 = sigma0[0, 0]
        a_0 = sigma0[0, 0] 
        b_0 = sigma0[1, 1]
        P_D = 0.07

        # Analytical Rates
        qfi_analyt = x2_0 * (dnu_0**2 / nu_0) * t_s
        hom_x_analyt = (dnu_0**2 / (32 * a_0**2)) * t_s**7
        hom_p_analyt = (dnu_0**2 / (2 * b_0**2)) * t_s**3
        rpd_ideal_analyt = 0.5 * (dnu_0**2 / nu_0) * t_s
        rpd_real_analyt = ((1 - P_D) * dnu_0**2 / (4 * P_D)) * t_s**3

        # 2. Aplicamos el 'skip' a los datos exactos también
        inset_ax.plot(t_s, sol_short["QFI"][skip:], color='black', linewidth=1.5)
        inset_ax.plot(t_s, sol_short["FI_hom_x"][skip:], color='red', linewidth=1.5)
        inset_ax.plot(t_s, sol_short["FI_hom_p"][skip:], color='green', linewidth=1.5)
        inset_ax.plot(t_s, sol_short["FI_rpd"][skip:], color='purple', linewidth=1.5)
        inset_ax.plot(t_s, sol_short["FI_rpd_real"][skip:], color='darkorange', linewidth=1.5)

        # Plot analytical approximations
        inset_ax.plot(t_s, qfi_analyt, color='black', linestyle=':', linewidth=2.5)
        inset_ax.plot(t_s, hom_x_analyt, color='red', linestyle=':', linewidth=2.5)
        inset_ax.plot(t_s, hom_p_analyt, color='green', linestyle=':', linewidth=2.5)
        inset_ax.plot(t_s, rpd_ideal_analyt, color='purple', linestyle=':', linewidth=2.5)
        inset_ax.plot(t_s, rpd_real_analyt, color='darkorange', linestyle=':', linewidth=2.5)

        # Inset formatting 
        inset_ax.set_xscale('log')
        inset_ax.set_yscale('log')
        inset_ax.tick_params(axis='both', which='both', labelsize=13)
        
        
        # Opcional: Descomenta esto para forzar que el plot principal no se deforme
        # ax.set_ylim(0.0, 0.08)

        # --- AÑADIR LEYENDA DE ESTILOS AL INSET ---
        # 1. Creamos líneas vacías con los estilos correspondientes
        inset_ax.plot([], [], color='gray', linestyle='-', linewidth=1.5, label='Exact solution')
        inset_ax.plot([], [], color='gray', linestyle=':', linewidth=2.5, label='Short-time approx.')
        
        # 2. Mostramos la leyenda solo para este inset
        inset_ax.legend(loc='lower right', fontsize=9, frameon=False)
        
        limite_temporal_inset = times_short[-1] 
        
        ax.axvspan(0, limite_temporal_inset, color='gray', alpha=0.15)

    handles, labels = ax.get_legend_handles_labels()

    # Creamos la leyenda global anclada a la Figura (no al subplot)
    # ncol=3 pondrá las 6 curvas en 2 filas muy ordenadas
    fig.legend(handles, labels, 
               loc='upper center', 
               bbox_to_anchor=(0.5, 1.08), 
               ncol=3, 
               fontsize=18, 
               frameon=False) # Quitar el frame suele quedar más elegante en SciencePlots

    # Usamos tight_layout pero reservamos un 10% de espacio arriba (top=0.9) 
    # para que los títulos de los plots no choquen con la leyenda
    plt.tight_layout(rect=[0, 0, 1, 0.98]) 
    
    plt.show()