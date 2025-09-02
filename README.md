## Heuristic ansatz design for ion-native circuits

In this repo, we share the code and data for reproducing numerical results presented in our paper:

**Heuristic ansatz design for trainable ion-native digital-analog quantum circuits** (https://www.arxiv.org/abs/2505.15898)

## Files

Below, we provide a brief description of the code and data present in the repository.

### Modules

* `QAOA.py` : class for implementing a QAOA-like parameterized quantum circuit (PQC). Its methods allow to calculate the expectation value for a given diagonal problem Hamiltonian, optimize variational parameters of the circuit using the layerwise training heuristic, etc.

* `hamiltonians.py` : functions for calculating matrices of the trapped ions $H_1$ and problem $H_2$ Hamiltonians. The first Hamiltonian $H_1$ is used to construct the ion-native ansatz, while the second one $H_2$ is minimized in the QAOA-like algorithm.

* `heuristics.py` : functions for implementing our heuristic that identifies problem-specific hyperparameters (ion-ion interactions) of the ion native ansatz and evaluating the algorithmic performance of the ion native and standard QAOA.

* `expressibility.py` : functions for calculating the expressibility of the QAOA-like circuits based on the Kullback-Leibler divergence. 

* `analysis.py` : functions for calculating characteristics of the single-layered cost landscape. 

* `visualization.py` : functions for visualizing data. 

* `timslib` : this folder contains the code for analytical calculations of the quantum dynamics of trapped-ion chains illuminated by bichromatic amplitude-shaped laser pulses taken from [this repo](https://github.com/EvgAnikin/fast_molmer_sorensen_w_carrier/). This code is used to calculate the phonon contribution to the ion-ion interaction. 


### Demos

* `main.ipynb` : this notebook provides an example of the complete training pipeline for a random instance of the Sherrington-Kirkpatrick model for $n=6$ qubits (see Fig. 2 in our [paper](https://www.arxiv.org/abs/2505.15898)). This pipeline consists of the following steps:
  + calculating the ion-ion couplings
  + generating a random SK instance
  + calculaing a matrix of the problem Hamiltonian
  + identifying problem-specific hyperparameters of the ion native ansatz using the proposed heuristic
  + rescaling hyperparameters to eliminate a narrow gorge on the cost landscape
  + evaluating the performance of the ion native QAOA using the trained and asymmetric configurations of hyperparameters

* `interplay.ipynb` : this notebook provides an example of calculating the expressbility of the ion native ansatz. It shows the interplay between the circuit expressibility and trainability when using problem-agnostic and problem-specific configurations of hyperparameters (see Fig. 3 in our [paper](https://www.arxiv.org/abs/2505.15898)).

* `statistics.ipynb` : this notebook provides a statistical analysis of the ion native QAOA performance. This analysis is based on the data obtained in our HPC simulations. For each system size, $n=5$-$10$ and $15$ qubits, 100 random instances of the SK model were sampled. For each instance, the problem-specific ansatz hyperparameters were obtained followed by the evaluation of the ion native QAOA performance in the range of circuit depths up to $p=10$ layers (see Figs. 4 and 5 in our [paper](https://www.arxiv.org/abs/2505.15898)).

### Data files

* `example_data` : output data files generated in demos above.

* `data_cycle_{i}` : data obtained after the `i`-th cycle of training in our HPC simulations. For each number of qubits $n$, 100 random instances of the SK model were sampled. Each cycle consists of identifying hyperparameters for every specific instance, followed by the Hamiltonian minimization using the obtained ansatzes. After each cycle, the fraction of solved instances was evaluated using a $p = n$ layer circuit. Each subsequent cycle was run only for the remaining unsolved instances. The identified problem-specific hyperpameters are stored in the files `{n}q/{n}q_nsk_{j}.txt`, while the  performance evaluation - in the files `{n}q/{n}q_ev_{j}.txt`, where `n` is the number of qubits and `j` is the index of random SK instance ($j=$ 1-100). 

* `data` : final data obtained after 4 cycles of training followed by the layerwise evaluation of the ion native QAOA performance in the range of circuit depths up to $p=10$ layers. The results of this evaluation are stored in the files `{n}q/{n}q_lw_{j}.txt`. Also, this folder contains the data for standard QAOA for $n=6$ and $8$ qubits (folders `6q_qaoa/` and `8q_qaoa/`).

### HPC Scripts

For gathering statistics, we perfomed simulations on our HPC cluster. For each system size $n$, we run in parallel 100 jobs for 100 random instances using `slurm array jobs`. A typical bash script for running the code on a cluster looks as follows:

```
#!/bin/bash
#SBATCH --job-name=5q_training
#SBATCH -p htc
#SBATCH --time=00:10:00
#SBATCH --mem=4G
#SBATCH --array=1-100
#SBATCH --output=out/%j.out
#SBATCH --error=err/%j.err

module load python/3.8
export OMP_NUM_THREADS=1

python3 ./train_hyperparameters.py 5 $((  {SLURM_ARRAY_TASK_ID} )) < input.txt

```

The input file ``input.txt`` contains all required simulation parameters. 

The following scripts were used to perform simulations:

* `train_hyperparameters.py` : code for identifying hyperparameters of the ion native ansatz well-suited for a specific problem instance.

* `evaluate_fixed_depth.py` : code for evaluating the ion native QAOA performance at the fixed circuit depth $p=n$.

* `evaluate_lw.py` : code for evaluating the ion native QAOA performance in the range of circuit depths up to $p_{\max}$ layers.

* `evaluate_lw_qaoa.py` : code for evaluating the standard QAOA performance in the range of circuit depths up to $p_{\max}$ layers.
