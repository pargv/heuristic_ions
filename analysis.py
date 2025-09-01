import numpy as np
import matplotlib.pyplot as plt

from QAOA import QAOA
from hamiltonians import ion_native_hamiltonian

from tqdm import tqdm


def get_landscape(Q,k=101):
    # инициализация ионно-совместимого анзаца
    
    cost = np.zeros((k,k))
    beta = np.linspace(0,0.5*np.pi,k)
    gamma = np.linspace(0.0,2.0*np.pi,k)
    
    for i in range(k):
        for j in range(k):
            cost[i][j] = Q.expectation(angles=[gamma[i],beta[j]])
    return cost