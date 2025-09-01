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

p_max = int(input())
n_runs = int(input())
path = input()
path_data = path + f'/{n}q/'
path_out = path + f'/{n}q_qaoa/'
fdata = path_data + f'{n}q_ev_' + str(index) + '.txt'
fname = path_out + f'{n}q_qaoa_' + str(index) + '.txt'

np.random.seed(123*index)

layerwise_evaluation_qaoa(fname,fdata,n,p_max,n_runs)