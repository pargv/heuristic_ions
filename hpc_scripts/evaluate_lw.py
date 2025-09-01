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

p_max = int(input())
n_runs = int(input())
path = input() + f'/{n}q/'
fdata = path + f'{n}q_ev_' + str(index) + '.txt'
fname = path + f'{n}q_lw_' + str(index) + '.txt'

np.random.seed(123*index)

layerwise_evaluation(fname,fdata,n,coupling_mat,p_max,n_runs)