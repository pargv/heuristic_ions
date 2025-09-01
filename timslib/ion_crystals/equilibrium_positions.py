import math
import numpy as np
import importlib.resources

from scipy.optimize import minimize


def regularized_coulomb(x, epsilon):
    if epsilon > 0:
        return np.where(np.abs(x) > epsilon, 1/np.abs(x), 3/(2*epsilon) - x**2/epsilon**3)
    else:
        return 1/np.abs(x)


def regularized_force(x, epsilon):
    if epsilon > 0:
        return np.where(np.abs(x) > epsilon, -np.sign(x)/x**2, -2*x/epsilon**3)
    else:
        return -np.sign(x)/x**2


def ions_energy(xi, epsilon=0):
    N = xi.size

    potential_energy = 1/2*np.sum(xi**2)

    diff_xi = xi[:, np.newaxis] - xi[np.newaxis, :]
    np.fill_diagonal(diff_xi, 1) 
    coulomb_energies = np.triu(regularized_coulomb(diff_xi, epsilon))

    return potential_energy + np.sum(coulomb_energies)


def ions_energy_deriv(xi, epsilon):
    N = xi.size

    deriv = np.zeros(N) + xi

    diff_xi = xi[:, np.newaxis] - xi[np.newaxis, :]
    np.fill_diagonal(diff_xi, 1) 
    pairwise_forces = regularized_force(diff_xi, epsilon)
    np.fill_diagonal(pairwise_forces, 0) 

    deriv -= np.sum(pairwise_forces, axis=0)

    return deriv

def calculate_dimensionless_equilibrium_positions(N, init=None, epsilon=1e-4):
    if N == 1:
        return np.zeros(1)

    if init is None:
        dxi_0 = (12*math.log(N+1)/N**2)**(1/3)
        xi_0 = (np.arange(0,N) - (N-1)/2)*dxi_0
    else:
        xi_0 = init

    def xi_from_half_xi(half_xi):
        if divmod(N, 2)[1] == 0:
            return np.concatenate((-np.flip(half_xi), half_xi))
        else:
            return np.concatenate((-np.flip(half_xi), np.array([0]), half_xi))

    def half_xi_from_xi(xi):
        return xi[int((xi.size+1)/2):]


    res = minimize(lambda half_xi: ions_energy(xi_from_half_xi(half_xi), epsilon), half_xi_from_xi(xi_0), method='BFGS', 
                  jac=lambda half_xi: 2*half_xi_from_xi(ions_energy_deriv(xi_from_half_xi(half_xi), epsilon)))
    ions_pos = xi_from_half_xi(res.x)
    return ions_pos

def get_dimensionless_equilibrium_positions(N, init=None, epsilon=1e-4):
    with importlib.resources.path('timslib.ion_crystals', 'linear_chain_eq_pos_data.npz') as p:
        eq_pos_data = np.load(p)
    n_ions_range = eq_pos_data['n_ions_range']
    if N <= n_ions_range[-1]:
        ions_eq_pos = eq_pos_data['ions_eq_pos']
        return ions_eq_pos[N-1, :N]
    else:
        return calculate_dimensionless_equilibrium_positions(N, init, epsilon)
