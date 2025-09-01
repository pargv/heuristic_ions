import itertools
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as sconst
import importlib.resources

from scipy.linalg import eigh, eigvalsh
from scipy.optimize import minimize
from scipy import special

def coulomb_gessian_matrix(eq_pos):
    N = eq_pos.size
    gessian = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            if i != j:
                gessian[i, j] = -1/np.abs(eq_pos[i] - eq_pos[j])**3

    for i in range(N):
        gessian[i,i] = np.sum(1/np.abs(eq_pos[:i] - eq_pos[i])**3) + np.sum(1/np.abs(eq_pos[i+1:] - eq_pos[i])**3)

    return gessian


def get_dimensionless_axial_normal_modes(mass_ratios, eq_pos_dm):
    n_ions = len(mass_ratios)
#    eq_pos = get_ions_dimensionless_equilibrium_positions(n_ions)
    G = coulomb_gessian_matrix(eq_pos_dm)
    A = np.diag(np.ones(eq_pos_dm.size)) + 2*G
    B = np.diag(mass_ratios)
    lambdas, eigenvectors = eigh(A, B)
    return np.sqrt(lambdas), eigenvectors


class UnstableChainError(ValueError):
    pass


def get_dimensionless_radial_normal_modes(mass_ratios, rad_ax_ratios, eq_pos_dm):
    n_ions = len(mass_ratios)
    G = coulomb_gessian_matrix(eq_pos_dm)
    
    A = np.diag(mass_ratios*rad_ax_ratios**2) - G
    B = np.diag(mass_ratios)
    lambdas, eigenvectors = eigh(A, B)

    if np.any(lambdas < 0):
        raise UnstableChainError('Try lowering chain axial frequency to get a stable linear chain')

    return np.sqrt(lambdas), eigenvectors


def get_axial_normal_modes(omega_ax, omega_rad_arr, eq_pos_dm, mass_ratios):
    omega_ratios, normal_modes = get_dimensionless_axial_normal_modes(mass_ratios, eq_pos_dm)
    omegas = omega_ratios*omega_ax
    return omegas, normal_modes


def get_radial_normal_modes(omega_ax, omega_rad_arr, eq_pos_dm, mass_ratios):
    rad_ax_ratios = omega_rad_arr/omega_ax
    omega_ratios, normal_modes = get_dimensionless_radial_normal_modes(mass_ratios, rad_ax_ratios, eq_pos_dm)
    omegas = omega_ratios*omega_ax
    
    return omegas, normal_modes


def minimal_rad_ax_ratio(n_ions):
    with importlib.resources.path('timslib.ion_crystals', 'linear_chain_eq_pos_data.npz') as p:
        eq_pos_data = np.load(p)
    gessian_max_vals = eq_pos_data['gessian_max_vals']
    return math.sqrt(gessian_max_vals[n_ions-2])


def nu_ax_from_nu_rad_min(nu_rad_min, nu_rad, n_ions):
    epsilon = nu_rad_min/nu_rad
    nu_ax = math.sqrt(1 - epsilon**2)*nu_rad/minimal_rad_ax_ratio(n_ions)
    return nu_ax


def equidistant_force_deriv(n_ions):
    n_plus = (n_ions+1) / 2
    eq_pos_dm = np.arange(-(n_ions-1)/2, (n_ions-1)/2 + 0.1)
    z_plus  = n_plus + eq_pos_dm
    z_minus = n_plus - eq_pos_dm

    force_deriv =  - (special.polygamma(2, z_plus) + special.polygamma(2, z_minus))
    return force_deriv


def get_axial_equidistant_normal_modes(omega_dx, n_ions):
    eq_pos_dm = np.arange(-(n_ions-1)/2, (n_ions-1)/2 + 0.1)

    force_deriv = equidistant_force_deriv(n_ions)
    G = coulomb_gessian_matrix(eq_pos_dm)
    A = np.diag(force_deriv) + 2*G

    lambdas, eigenvectors = eigh(A) 
    omegas = omega_dx*np.sqrt(lambdas)

    return omegas, eigenvectors


def get_radial_equidistant_normal_modes(omega_dx, omega_rad, n_ions):
    eq_pos_dm = np.arange(-(n_ions-1)/2, (n_ions-1)/2 + 0.1)

    omega_ratios, normal_modes = get_dimensionless_radial_normal_modes(np.ones(n_ions), omega_rad/omega_dx, eq_pos_dm)
    omegas = omega_dx*omega_ratios

    return omegas, normal_modes
