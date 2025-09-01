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
    """
    Генерация выборок перекрытий, порожденных ионно-совместимым анзацем
    с заданной конфигурацией контролируемых параметров. Выборки вычисляются
    для каждой глубины квантовой цепи от 1 до p_max слоев. 
    Результаты записываются в заданный файл. 
    """
    
    # инициализация ионно-совместимого анзаца
    H1 = ion_native_hamiltonian(n,A,coupling_mat)
    Q = QAOA(1, H1, H1)
    
    # генерация выборок перекрытий
    data = Q.sample_fidelities(p_max,n_samples)
    
    # запись результатов в файл
    file_dump(A, f"{path}{fname}.csv", format='w')
    
    for p in range(p_max):
        file_dump(data[p,:], f"{path}{fname}.csv")
    
        
def generate_fidelities_data_random_A(n, p_max, n_samples, n_A, coupling_mat, path):
    """
    Генерация выборок перекрытий, порожденных ионно-совместимыми анзацами 
    со случайными конфигурациями контролируемых параметров. 
    Для каждого случайной конфигурации параметров, выборки перекрытий 
    генерируются для каждой глубины соответствующего анзаца от 1 до p_max слоев.
    Результаты записыватся в отдельные файлы для каждой случайной
    конфигурации omegas в заданный путь path. 
    """
     
    # инициализация ионно-совместимого анзаца
    A = np.random.uniform(-1, 1, n) # случайная конфигурация контролируемых параметров
    H1 = ion_native_hamiltonian(n,A,coupling_mat)
    Q = QAOA(1, H1, H1)

    # цикл по выборке случайных конфигураций omegas 
    for i in range(n_A):
        
        # инициализация "случайного" анзаца
        A = np.random.uniform(-1,1,size=n)
        H1 = ion_native_hamiltonian(n,A,coupling_mat)
        Q.H1 = H1
        
        # генерация выборки перекрытий
        data = Q.sample_fidelities(p_max,n_samples)
    
        # запись результатов в файл
        file_dump(A, f"{path}data_A_{i}.csv", format='w')
        for p in range(p_max):
            file_dump(data[p,:], f"{path}data_A_{i}.csv")


# =============================================================================
#                             Оценка экспрессивности
# =============================================================================

def read_fidelities(path,fname,p_max):
    """
    Считывание данных из заданного файла с указанным путем к папке.
    Файл должен содержать выборки перекрытий для каждой глубины 
    QAOA-подобного анзаца от 1 до p_max слоев. 

    Args:
        path (str): путь к файлу данных с выборками перекрытий
        fname (str): файл данных с выборками перекрытий
        max_p (int): максимальная глубина QAOA-подобного анзаца

    Returns:
        tuple: кортеж с массивом конфигурации контролируемых параметров omegas и списком 
               с массивами, содержащими выборки перекрытий для каждой глубины анзаца
    """
    with open(f'{path}{fname}.csv', mode='r') as file:
        database = csv.reader(file)
        A = np.array(next(database), dtype=float)
        fidelities = []
        
        for p in range(p_max):
            fidelities.append(np.array(next(database), dtype=float))
            
    return A, fidelities

def get_expressibility(n_bins,n,F,half_dim=1):
    """
    Вычисление экспрессивности ионно-совместимого анзаца
    по расхождению Кульбака-Лейблера между распределение перекрытий,
    порожденных квантовой цепью, и распределением в ансамбле Хаара.

    Args:
        n_bins (int): число столбцов для построения гистограммы с 
                      распределением перекрытий
        n (int): число кубитов
        F (numpy array): массив с выборкой перекрытий
        half_dim (int, optional): учет поправки на симметрию QAOA-подобного
                                  анзаца, уменьшающей размерность гильбертова 
                                  пространства состояний. По умолчанию, 1.

    Returns:
        float: дексриптор, оценивающий экспрессивность анзаца
    """
    
    # расчет плотности распределения перекрытий по гистограмме
    pdf, x = np.histogram(F,bins=n_bins,density=True,range=(0,1))
    x = (x[1:]+x[:-1])/2.0
    
    # размерность гильбертова пространства состояний
    N = 2**(n - half_dim)
    
    # плотность распределения в ансамбле Хаара
    f_Haar = lambda x: (N-1)*(1.0-x)**(N-2)
    
    # расчет плотности распределения в ансамбле Хаара
    pdf_Haar = f_Haar(x)
    
    # оценка экспрессивности по расхождению Кульбака-Лейблера
    expr = entropy(pdf,qk=pdf_Haar) # DKL это относительная энтропия
    
    return expr

def get_layerwise_expressibility(path,fname,p_max,n,n_bins,half_dim=1):
    """
    Вычисление экспрессивности ионно-совместимого анзаца как функции
    глубины квантовой цепи на основе выборки перекрытий из заданного файла.

    Args:
        path (str): путь к папке с заданному файлу данных
        fname (str): файл данных с выборками перекрытий
        p_max (int): максимальная глубина QAOA-подобного анзаца
        n (int): число кубитов
        n_bins (int): число столбцов для построения гистограммы с 
                      распределением перекрытий
        half_dim (int, optional): учет поправки на симметрию QAOA-подобного
                                  анзаца, уменьшающей размерность гильбертова 
                                  пространства состояний. По умолчанию, 1.

    Returns:
       tuple: кортеж с массивами числа слоев анзаца L и расчетной экспрессивности по слоям expr
    """

    # инициализация массивов для хранения данных
    expr = np.zeros(p_max)
    L = np.array(range(1,p_max+1))
    
    # считывание выборки перекрытий из файла
    A, fidelities = read_fidelities(path,fname,p_max)
    
    # цикл по числу слоев анзаца
    for p in range(p_max):
        F = fidelities[p]
        expr[p] = get_expressibility(n_bins,n,F,half_dim)
        
    return L, expr

def get_layerwise_expressibility_specific_A(A,coupling_mat,p_max,n,n_bins,n_samples,half_dim=1):
    """
    Вычисление экспрессивности ионно-совместимого анзаца как функции
    глубины квантовой цепи для заданной конфигурации контролируемых параметров
    (выборки перекрытий не считываются из файла, а генерируются непосредственно в процессе).
    """

    # инициализация массивов для хранения данных
    expr = np.zeros(p_max)
    L = np.array(range(1,p_max+1))
    
    # инициализация ионно-совместимого анзаца
    H1 = ion_native_hamiltonian(n,A,coupling_mat)
    Q = QAOA(1,H1,H1)   
    
    # генерация выборок перекрытий для глубины анзаца от 1 до p_max слоев
    fidelities = Q.sample_fidelities(p_max,n_samples)
    
    # цикл по числу слоев анзаца
    for p in range(p_max):
        F = fidelities[p]
        expr[p] = get_expressibility(n_bins,n,F,half_dim)
        
    return L, expr
