import numpy as np
from IPython.display import clear_output
from scipy.stats import entropy
import csv

from hamiltonians import ion_native_hamiltonian
from QAOA import QAOA


# =============================================================================
#                        Generating Data for Fidelities
# =============================================================================


def file_dump(line,name, format='a'):
    with open(name,format) as f:
        w=csv.writer(f,delimiter=',')
        w.writerow(line)
        
def generate_fidelities_data(n, p_max, n_samples, A, coupling_mat, fname, path):
    
    H1 = ion_native_hamiltonian(n,A,coupling_mat)
    Q = QAOA(1, H1, H1)
    
    data = Q.sample_fidelities(p_max,n_samples)
    
    file_dump(A, f"{path}{fname}.csv", format='w')
    
    for p in range(p_max):
        file_dump(data[p,:], f"{path}{fname}.csv")
    
        
def generate_fidelities_data_random_A(n, p_max, n_samples, n_A, coupling_mat, path):
     
    A = np.random.uniform(-1, 1, n)
    H1 = ion_native_hamiltonian(n,A,coupling_mat)
    Q = QAOA(1, H1, H1)

    for i in range(n_A):
        
        A = np.random.uniform(-1,1,size=n)
        H1 = ion_native_hamiltonian(n,A,coupling_mat)
        Q.H1 = H1
        
        data = Q.sample_fidelities(p_max,n_samples)
    
        file_dump(A, f"{path}data_A_{i}.csv", format='w')
        for p in range(p_max):
            file_dump(data[p,:], f"{path}data_A_{i}.csv")


# =============================================================================
#                             Expressibility evaluation
# =============================================================================

def read_fidelities(path,fname,p_max):
    
    with open(f'{path}{fname}.csv', mode='r') as file:
        database = csv.reader(file)
        A = np.array(next(database), dtype=float)
        fidelities = []
        
        for p in range(p_max):
            fidelities.append(np.array(next(database), dtype=float))
            
    return A, fidelities

def get_expressibility(n_bins,n,F,half_dim=1):

    pdf, x = np.histogram(F,bins=n_bins,density=True,range=(0,1))
    x = (x[1:]+x[:-1])/2.0
    
    N = 2**(n - half_dim)

    f_Haar = lambda x: (N-1)*(1.0-x)**(N-2)
    
    pdf_Haar = f_Haar(x)
    
    expr = entropy(pdf,qk=pdf_Haar)
    
    return expr

def get_layerwise_expressibility(path,fname,p_max,n,n_bins,half_dim=1):

    expr = np.zeros(p_max)
    L = np.array(range(1,p_max+1))

    A, fidelities = read_fidelities(path,fname,p_max)
    
    for p in range(p_max):
        F = fidelities[p]
        expr[p] = get_expressibility(n_bins,n,F,half_dim)
        
    return L, expr

def get_layerwise_expressibility_specific_A(A,coupling_mat,p_max,n,n_bins,n_samples,half_dim=1):

    expr = np.zeros(p_max)
    L = np.array(range(1,p_max+1))

    H1 = ion_native_hamiltonian(n,A,coupling_mat)
    Q = QAOA(1,H1,H1)   

    fidelities = Q.sample_fidelities(p_max,n_samples)

    for p in range(p_max):
        F = fidelities[p]
        expr[p] = get_expressibility(n_bins,n,F,half_dim)
        
    return L, expr
