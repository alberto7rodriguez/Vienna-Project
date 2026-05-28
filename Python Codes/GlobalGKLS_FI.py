import numpy as np

def csch2(z):
    s = np.sinh(z)
    return 1.0 / (s * s)

def GKLS_thermal_FI(Omega_S, delta_Omega_S, Omega_R, g, Temp, sigma_M):
    """
    Calculates the exact asymptotic thermal state and Fisher Information 
    for the S-RC unit under the Global GKLS master equation.
    """
    # 1. Define normal mode frequencies and mixing angles (Eq. C39 - C41)
    Omega_1_sq = Omega_S**2 + delta_Omega_S**2
    Omega_2_sq = Omega_R**2
    
    delta_sq = Omega_1_sq - Omega_2_sq
    cross_term = np.sqrt(4 * g**2 + delta_sq**2)
    
    Omega_plus = np.sqrt(0.5 * (Omega_1_sq + Omega_2_sq + cross_term))
    Omega_minus = np.sqrt(0.5 * (Omega_1_sq + Omega_2_sq - cross_term))
    
    cos2_theta = (delta_sq + cross_term) / (2 * cross_term)
    sin2_theta = 1.0 - cos2_theta
    
    # 2. Calculate the exact analytical variances
    coth_plus = 1.0 / np.tanh(Omega_plus / (2 * Temp))
    coth_minus = 1.0 / np.tanh(Omega_minus / (2 * Temp))
    
    var_x = cos2_theta * (coth_plus / (2 * Omega_plus)) + sin2_theta * (coth_minus / (2 * Omega_minus))
    var_p = cos2_theta * (Omega_plus * coth_plus / 2) + sin2_theta * (Omega_minus * coth_minus / 2)
    
    sigma_S = np.array([[var_x, 0], 
                        [0, var_p]])
    
    # 3. Calculate Temperature derivatives of the variances
    T2 = Temp**2
    d_coth_plus = (Omega_plus / (2 * T2)) * csch2(Omega_plus / (2 * Temp))
    d_coth_minus = (Omega_minus / (2 * T2)) * csch2(Omega_minus / (2 * Temp))
    
    d_var_x = cos2_theta * (d_coth_plus / (2 * Omega_plus)) + sin2_theta * (d_coth_minus / (2 * Omega_minus))
    d_var_p = cos2_theta * (Omega_plus * d_coth_plus / 2) + sin2_theta * (Omega_minus * d_coth_minus / 2)
    
    dsigma_S = np.array([[d_var_x, 0], 
                         [0, d_var_p]])
    
    # 4. Calculate Fisher Information (Eq 7)
    # F = (1/2) Tr[ ( (sigma_S + sigma_M)^-1 * dsigma_S )^2 ]
    inverse_term = np.linalg.inv(sigma_S + sigma_M)
    matrix_prod = np.dot(inverse_term, dsigma_S)
    fisher_info = 0.5 * np.trace(np.dot(matrix_prod, matrix_prod))
    
    return {"FI": fisher_info, "sigma": sigma_S, "dsigma_dT": dsigma_S}

# ==========================================
# Example usage to add to your plotting block:
# ==========================================
# (You will need to define your g, Omega_R, and delta_Omega_S based on your RC parameters)
#
# g = np.sqrt(gamma * alpha1 * alpha2)
# Omega_R = np.sqrt(alpha2 * gamma)
# delta_Omega_S = np.sqrt( (2 * gamma * alpha1 * alpha2) / (np.pi * Omega_R) ) # (or your explicit integral)
# 
# gkls_results = GKLS_thermal_FI(Omega_S, delta_Omega_S, Omega_R, g, temp, sigma_M)
#
# ax2.axhline(gkls_results["FI"], color='blue', linestyle='-.', alpha=0.8, label="GKLS FI Asymptote")
# ==========================================