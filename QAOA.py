import numpy as np
from scipy.optimize import minimize
from IPython.display import clear_output
import time

from hamiltonians import ion_native_hamiltonian

# =============================================================================
#                                 Класс QAOA
# =============================================================================

class QAOA:
    
    def __init__(self,depth,H1,H2):   
        """                                                 
        Конструктор класса QAOA (инициализация).
        
        Args:
            depth (int): глубина QAOA-подобного анзаца
            H1 (numpy array): диагональный гамильтониан, определяющий оператор изменения фазы (пропагатор)
            H2 (numpy array): диагональный целевой гамильтониан для минимизации
        """                                   
        
        self.H1 = H1
        self.H2 = H2
        self.n = int(np.log2(int(len(self.H1)))) # Расчет числа кубитов.
        
        #______________________________________________________________________________________________________
        self.X = self.new_mixerX()              # Выполняет последовательность преобразований над массивами,
                                                # реализующих эффект применения оператора \sum \sigma_x, к
                                                # смешивающего кубиты, к вектору квантового состояния,

        #______________________________________________________________________________________________________
        
        self.min = min(self.H2)                  # Минимальное собственное значение целевого гамильтониана 
                                                 # (энергия основного состояния)
        
        self.deg = len(self.H2[self.H2 == self.min]) 
        self.p = depth                           # Стандартная глубина QAOA в обозначении p
        
        self.heruistic_LW_seed1 = 10
        self.heruistic_LW_seed2 = 20
        
        #______________________________________________________________________________________________________   
    
    def new_mixerX(self):
        """
        Создает последовательность преобразований над вектором квантового 
        состояния под действием оператора, смешивающего кубиты, в виде 
        списка перестановок индексов кубитов.
        
        Returns:
            list: список перестановок индексов кубитов
        """

        def split(x,k):
            return x.reshape((2**k,-1))
        def sym_swap(x):
            return np.asarray([x[-1],x[-2],x[1],x[0]])
        
        n = self.n
        x_list = []
        t1 = np.asarray([np.arange(2**(n-1),2**n),np.arange(0,2**(n-1))])
        t1 = t1.flatten()
        x_list.append(t1.flatten())
        t2 = t1.reshape(4,-1)
        t3 = sym_swap(t2)
        t1 = t3.flatten()
        x_list.append(t1)
        
        
        k = 1
        while k < (n-1):
            t2 = split(t1,k)
            t2 = np.asarray(t2)
            t1=[]
            for y in t2:
                t3 = y.reshape((4,-1))
                t4 = sym_swap(t3)
                t1.append(t4.flatten())
            t1 = np.asarray(t1)
            t1 = t1.flatten()
            x_list.append(t1)
            k+=1        
        
        return x_list
    #__________________________________________________________________________________________________________   
        
    def U_gamma(self,angle,state):
        """
        Применить оператор изменения фазы exp{-i*gamma*H} к вектору квантового состояния.
        
        Args:
            angle (float): вариационный параметр
            state (numpy array): вектор квантового состояния
        
        Returns:
            numpy array: преобразованный вектор квантового состояния
        """       
        t = -1j*angle
        state = state*np.exp(t*self.H1.reshape(2**self.n,1))
        
        return state
    
    def V_beta(self,angle,state):       
        """
         Применить оператор смешения кубитов exp{-i*beta*H_x} к вектору квантового состояния.
        
        Args:
            angle (float): вариационный параметр
            state (numpy array): вектор квантового состояния
        
        Returns:
            numpy array: преобразованный вектор квантового состояния
        """
        c = np.cos(angle)
        s = np.sin(angle)
        
        for i in range(self.n):
            t = self.X[i]
            st = state[t]
            state = c*state + (-1j*s*st)
            
        return state

    #__________________________________________________________________________________________________________
    
    def qaoa_ansatz(self, angles):
        """
        Вычисление вектора квантового состояния, приготовленного QAOA-подобным анзацем, т.е.
        последовательностью операторов изменения фазы exp{-i*gamma*H} и смешения кубитов exp{-i*beta*H_x}
        для квантовой цепи глубины p с вектором вариационных параметров 
        [gamma_1,gamma_2,...,gamma_p,beta_1,beta2,....beta_p], примененной к начальному состоянию |+>.
        
        Args:
            angles (numpy array): вектор вариационных параметров 
                                  [gamma_1,gamma_2,...,gamma_p,beta_1,beta2,....beta_p]
        
        Returns:
            numpy array: вектор квантового состояния после применения QAOA-подобного анзаца
        """
        
        state = np.ones((2**self.n,1),dtype = 'complex128')*(1/np.sqrt(2**self.n))
        p = int(len(angles)/2)
        for i in range(p):
            state = self.U_gamma(angles[i],state)
            state = self.V_beta(angles[p + i],state)
        
        return state 
    
    #__________________________________________________________________________________________________________ 
    
    def apply_ansatz(self, angles, state):
        """
        Применение последоватльности операторов QAOA-подобного анзаца к заданному квантовому состоянию state. 
        
        Args:
            angle (numpy array): вектор вариационных параметров 
                                 [gamma_1,gamma_2,...,gamma_p,beta_1,beta2,....beta_p]
            state (numpy array): вектор квантового состояния
        
        Returns:
            numpy array: преобразованный вектор квантового состояния
        """
        p = int(len(angles)/2)
        for i in range(p):
            state = self.U_gamma(angles[i],state)
            state = self.V_beta(angles[p + i],state)
        
        return state
    
    #__________________________________________________________________________________________________________ 
    
    def expectation(self,angles): 
        """
        Вычисление математического ожидаемого энергии целевого гамильтониана.
        
        Args:
            angles (numpy array): вектор вариационных параметров
        
        Returns:
            float: математическое ожидаемое энергии
        """ 
        state = self.qaoa_ansatz(angles)
        
        ex = np.vdot(state,state*(self.H2).reshape((2**self.n,1)))
        
        return np.real(ex)
            
    
    def overlap(self,state):
        """
        Вычисление перекрытия заданного состояния с основным состояниям целевого гамильтониана.
        
        Args:
            state (numpy array): вектор квантового состояния
        
        Returns:
            float: величина перекрытия
        """       
        g_ener = min(self.H2)
        olap = 0
        for i in range(len(self.H2)):
            if self.H2[i] == g_ener:
                olap+= np.absolute(state[i])**2
        
        return olap
    
   #__________________________________________________________________________________________________________ 
    
    def run_heuristic_LW(self):
        """
        Симуляция варационного QAOA-подобного квантового алгоритма с
        использованием эвристического метода оптимизации квантовой цепи 
        по слоям (layer-wise, LW).
        """
        initial_guess = lambda x: ([np.random.uniform(0,2*np.pi) for _ in range(x) ] 
                                   + [np.random.uniform(0,0.5*np.pi) for _ in range(x)])
        bds = lambda x: [(0.0,2*np.pi)]*x + [(0.0,0.5*np.pi)]*x
        
        def combine(a,b):

            a = list(a)
            b = list(b)
            a1 = a[0:int(len(a)/2)]
            a2 = a[int(len(a)/2)::]
            b1 = b[0:int(len(b)/2)]
            b2 = b[int(len(b)/2)::]
            a = a1+b1
            b = a2+b2
            
            return a + b 
        
        temp = [] 
        t_start = time.time()
        
        for _ in range(self.heruistic_LW_seed1):
            initial_guess_p1 = initial_guess(1)
            res = minimize(self.expectation,initial_guess_p1,method='L-BFGS-B',\
                           jac=None, bounds=bds(1), options={'maxfun': 150000})
            temp.append([res.fun, res.x])
            
        temp = np.asarray(temp,dtype=object)
        idx = np.argmin(temp[:,0])
        opt_angles = temp[idx][1]
       
        
        t_state = np.ones((2**self.n,1),dtype = 'complex128')*(1/np.sqrt(2**self.n))
        
        while len(opt_angles) < 2*self.p:
            ts1 = time.time()
            t_state = self.qaoa_ansatz(opt_angles)
            
            
            ex = lambda x : np.real(np.vdot(self.apply_ansatz(x,t_state),\
                                            self.apply_ansatz(x,t_state)*(self.H2).reshape((2**self.n,1))))
            temp = [] 
            
            for _ in range(self.heruistic_LW_seed2):
                
                res = minimize(ex,initial_guess(1),method='L-BFGS-B', jac=None, bounds=bds(1), \
                               options={'maxfun': 150000})
                temp.append([res.fun, res.x])
            temp = np.asarray(temp,dtype=object)
            idx = np.argmin(temp[:,0])
            lw_angles = temp[idx][1]
            opt_angles = combine(opt_angles,lw_angles)
            
            res = minimize(self.expectation,opt_angles,method='L-BFGS-B', jac=None, \
                           bounds=bds(int(len(opt_angles)/2)), options={'maxfun': 150000})    
            opt_angles = res.x
        self.opt_angles = opt_angles    
            
        t_end = time.time()
        self.exe_time = float(t_end - t_start)
        self.opt_iter = float(res.nfev)
        self.q_energy = self.expectation(self.opt_angles)
        self.q_error = self.q_energy - self.min
        self.f_state = self.qaoa_ansatz(self.opt_angles)
        self.olap = self.overlap(self.f_state)[0]
    
    #__________________________________________________________________________________________________________ 
    
    
    def sample_fidelities_fixed_depth(self, p, n_samples):
        """
        Генерация выборки перекрытий, порожденных анзацем, 
        для заданной глубины анзаца p.
        
        Args:
            p (int): глубина QAOA-подобного анзаца
            n_samples (int): размер выборки перекрытий
        
        Returns:
            numpy array: сгенерированная выборка перекрытий
        """
         
        F = np.zeros(n_samples)
        
        for s in range(n_samples):
            pars = np.random.uniform(0, 2*np.pi, 2*p)
            psi1 = self.qaoa_ansatz(pars)
                
            pars = np.random.uniform(0, 2*np.pi, 2*p)
            psi2 = self.qaoa_ansatz(pars)
                
            F[s] = (np.abs(np.dot(psi1.conj().T, psi2).item())**2)
                
        return F
    
    #__________________________________________________________________________________________________________ 
    
    def sample_fidelities(self, p_max, n_samples):
        """
        Генерация выборки перекрытий для каждой глубины анзаца в диапазоне,
        не превышающем p_max слоев. 
        
        Args:
            p_max (int): максимальная глубина QAOA-подобного анзаца
            n_samples (int): размер выборки перекрытий
        
        Returns:
            list: список массивов сгенерированный выборок перекрытий для каждой глубины 
            анзаца в диапазоне от 1 до p_max слоев
        """
        fidelities = np.zeros((p_max, n_samples))
        
        for p in range(1, p_max + 1):
            self.p = p
            fidelities[p-1,:] = self.sample_fidelities_fixed_depth(p,n_samples)
                
        return fidelities
    
    
def run_QAOA_for_fixed_p(H1,H2,p,n_runs):
        
    # инициализация ионно-совместимого анзаца
    Q = QAOA(p,H1,H2)
        
    # инициализация массивов для хранения результатов запусков
    energies = np.zeros(n_runs)
    ovlp = np.zeros(n_runs)
    angles = np.zeros((n_runs,2*p))
            
    # несколько запусков минимизации с помощью эвристического метода
    # оптимизации квантовой цепи по слоям в попытке достичь глобального минимума
    for i in range(n_runs):
        Q.run_heuristic_LW()
        energies[i] = Q.q_energy
        angles[i,:] = Q.opt_angles
        ovlp[i] = Q.olap
            
    # извлечение лучшего результата с наименьшей энергией
    imin = np.argmin(energies)
    
    return energies[imin], angles[imin,:], ovlp[imin]

def run_QAOA(H1,H2,p_max,n_runs):
    
    # инициализация массивов для хранения результатов запусков
    energies = np.zeros(p_max)
    ovlp = np.zeros(p_max)
    
    # цикл глубине анзаца
    for p in range(1,p_max+1):
        
        energies[p-1], angles, ovlp[p-1] = run_QAOA_for_fixed_p(H1,H2,p,n_runs)

    return energies, angles, ovlp