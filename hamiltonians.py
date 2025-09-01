import numpy as np
from scipy.optimize import minimize
import networkx as nx 
import matplotlib.pyplot as plt 
from timslib.ion_crystals import IonChain
import networkx as nx

# =============================================================================
#                              Гамильтонианы
# =============================================================================

def radial_coupling_matrix(n, nu_rad, nu_ax, mu, Omega_max):
    """
    This function generates a specific ions chain and calculates its 
    phonon contribution to the effective Ising coupling matrix that 
    describes a chain of ions under the applied laser field. 

    Args:
        n (int): number of ions (qubits)
        nu_rad (float): radial frequency of ions (in Hz)
        nu_ax (float): axial frequency of ions (in Hz)
        mu (float): laser detuning (in Hz)

    Returns:
        coupling_mat: phonons contribution to the Ising coupling matrix (in kHz)
        Omega_max: maximal Rabi frequency (in Hz) for normalization of coupling to ~1.5 kHz
    """
    
    chain = IonChain(nu_rad=nu_rad, nu_ax=nu_ax, n_ions=n, ion_type='Ca40')
    
    eta = chain.eta_rad(np.pi/2)
    omegas_rad = chain.omegas_rad
    
    coupling_mat = eta @ np.diag(2*omegas_rad/(mu**2 - omegas_rad**2)) @ eta.T 
    coupling_mat = Omega_max**2*coupling_mat
    coupling_mat = coupling_mat/1e3 # convert to kHz

    return coupling_mat

def tensor(k):
    t = k[0]
    i = 1
    
    while i < len(k):
        t = np.kron(t,k[i])
        i+=1
    return t

def get_weighted_graph(n,weights):
    """
    This function generates a complete graph with 
    weighted edges based on the specific list of weights.

    Args:
        n (int): number of nodes
        weights (list): list of weights

    Returns:
        G: a complete graph with weighted edges represented as a networkx graph object.
    """
    
    assert len(weights) == n*(n-1)//2, 'Number of weights is not equal to the number of edges'
    
    w = weights.copy()
    G = nx.complete_graph(n)
    for (i,j) in G.edges():
        G[i][j]["weight"] = w.pop(0)
        
    return G

def get_hamiltonian(n,weights):
    """
    Generate the Sherrington-Kirkpatric Hamiltonian 
    using a graph representation of the problem. 

    Args:
        G (nx.graph): a networkx graph representing a S-K model

    Returns:
        H: diagonal elements of the S-K Hamiltonian, 
    """
    
    Z, Id, H = [1,-1], [1,1], np.zeros(2**n) 
    complete_graph = []
    
    w = weights.copy()

    for i in range(n):
        for j in range(i,n):
            if i!=j: complete_graph.append((i,j,w.pop(0))) 
    
    for x in complete_graph:
        tensor_array = [Id]*n
        tensor_array[x[0]] = Z
        tensor_array[x[1]] = Z
        H+= x[2]*tensor(tensor_array)    
    
    return H/np.sqrt(n)

def ion_native_hamiltonian(n,A,coupling_mat):
    """
    This functions calculates the effective Ising Hamiltonian
    that describes a chain of ions interacting with a laser field.

    Args:
        n (int): number of qubits
        A (numpy array): dimensionless controllable parameters (Rabi frequencies)
        coupling_mat (numpy ndarray): the phonons contribution to the Ising coupling matrix

    Returns:
        H: diagonal elements of the effective Ising Hamiltonian for 
           a chain of ions under a laser field.
    """
    vA = A.reshape([n,1])
    
    ising_coupling_mat = (vA @ vA.T) * coupling_mat
    
    Z, Id, H = [1,-1], [1,1], np.zeros(2**n) 
    complete_graph = []

    for i in range(n):
        for j in range(i,n):
            if i!=j: complete_graph.append((i,j,ising_coupling_mat[i][j])) 
    
    for x in complete_graph:
        tensor_array = [Id]*n 
        tensor_array[x[0]] = Z
        tensor_array[x[1]] = Z
        H+= x[2]*tensor(tensor_array)
        
    return H
