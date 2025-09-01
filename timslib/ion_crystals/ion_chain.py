import math
import numpy as np
import scipy.constants as sconst
import pandas as pd
import importlib.resources

from timslib.ion_crystals.equilibrium_positions import get_dimensionless_equilibrium_positions
from timslib.ion_crystals.normal_modes import get_axial_normal_modes, get_radial_normal_modes, get_axial_equidistant_normal_modes, get_radial_equidistant_normal_modes


class Ion:
    @classmethod
    def species_list(cls):
        with importlib.resources.path('timslib.ion_crystals', 'ions_data.ods') as p:
            df = pd.read_excel(p, index_col=0)
        return df.indices()

    def __init__(self, ion_type):
        atomic_mass_unit = sconst.physical_constants['atomic mass constant'][0]

        with importlib.resources.path('timslib.ion_crystals', 'ions_data.ods') as p:
            df = pd.read_excel(p, index_col=0)

        self.m = df.loc[ion_type, 'mass (a.u.)']*atomic_mass_unit
        self.qubit_lambda = df.loc[ion_type, 'qubit wavelength (nm)']*1e-9

class IonChain:
    def __init__(self, ion_type, n_ions, nu_ax, nu_rad):
        self.n_ions = n_ions
        self.nu_ax  = nu_ax
        self.nu_rad = nu_rad
        self.omega_ax  = 2*math.pi*nu_ax
        self.omega_rad = 2*math.pi*nu_rad
        self.ion = Ion(ion_type)
        self.x0 = math.sqrt(sconst.hbar/(4*math.pi*self.ion.m*nu_ax))
        self.dx0 = (sconst.e**2/(4*math.pi*sconst.epsilon_0*self.ion.m*self.omega_ax**2))**(1/3)

        self.eq_pos_dm = get_dimensionless_equilibrium_positions(n_ions)
        self.eq_pos    = self.dx0*self.eq_pos_dm

        self.omegas_ax,  self.normal_modes_ax  = get_axial_normal_modes(self.omega_ax, self.omega_rad*np.ones(n_ions), self.eq_pos_dm, np.ones(n_ions))
        self.omegas_rad, self.normal_modes_rad = get_radial_normal_modes(self.omega_ax, self.omega_rad*np.ones(n_ions), self.eq_pos_dm, np.ones(n_ions))

    def eta_ax(self, angle):
        ''' Matrix of Lamb-Dicke parameters for axial modes'''
        k0 = 2*math.pi/self.ion.qubit_lambda
        return k0*math.cos(angle)*np.sqrt(sconst.hbar/(2*self.ion.m*self.omegas_ax[np.newaxis, :])) *self.normal_modes_ax

    def eta_rad(self, angle):
        ''' Matrix of Lamb-Dicke parameters for radial modes'''
        k0 = 2*math.pi/self.ion.qubit_lambda
        return k0*math.sin(angle)*np.sqrt(sconst.hbar/(2*self.ion.m*self.omegas_rad[np.newaxis, :])) *self.normal_modes_rad
