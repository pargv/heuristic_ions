import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import sys

rc = {"font.family" : "Times New Roman", 
      "mathtext.fontset" : "stix"}
plt.rcParams.update(rc)

from hamiltonians import *
from heuristics import *

import warnings
warnings.filterwarnings("ignore")

# number of qubits
n = int(sys.argv[1])
# index of SLURM array
index = int(sys.argv[2])

# read maximal Rabi frequency
Omega_max = float(input())

# ions chain parameters
nu_rad = 1e6 # in Hz
nu_ax = 0.15e6 # in Hz
mu = 2*np.pi*(nu_rad + 10e3) # in Hz
Omega_max = 2*np.pi*Omega_max     # in Hz

# calculate the phonons contribution to the Ising coupling coefficients
coupling_mat = radial_coupling_matrix(n, nu_rad, nu_ax, mu, Omega_max) # in kHz

# generate random Sherrington-Kirkpatrick instance
np.random.seed(100*index)
n_edges = n*(n-1)//2
weights = list(np.random.normal(size=n_edges))

# read simulation parameters
n_iter = int(input())
max_restarts = int(input())
eps = float(input())
tol_lvl = float(input())
path = input() + f'/{n}q/'
fname = f'{n}q_nsk_' + str(index) + '.txt'

np.random.seed(123*index)
training(path,fname,n,weights,coupling_mat,n_iter,tol_lvl,max_restarts,eps)