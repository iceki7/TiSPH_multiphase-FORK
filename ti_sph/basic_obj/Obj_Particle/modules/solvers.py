import taichi as ti
from ....basic_solvers.Solver_adv import Adv_slover
from ....basic_solvers.Solver_df import DF_solver
from ....basic_solvers.Solver_sph import SPH_solver
from ....basic_solvers.Solver_wcsph import WCSPH_solver
from ....basic_solvers.Solver_ism import Implicit_mixture_solver
from ....basic_solvers.Solver_JL21 import JL21_mixture_solver
from ....basic_solvers.Solver_isph import ISPH_Elastic_solver
from ....basic_solvers.Solver_porous import Porous_multi_solver

def add_solver_adv(self):
    self.m_solver_adv = Adv_slover(self)

def add_solver_sph(self):
    self.m_solver_sph = SPH_solver(self)

def add_solver_df(self, incompressible_threshold: ti.f32 = 2e-4, div_free_threshold: ti.f32 = 1e-3, incompressible_iter_max: ti.i32 = 100, div_free_iter_max: ti.i32 = 50, incomp_warm_start: bool = False, div_warm_start: bool = False):
    self.m_solver_df = DF_solver(self, incompressible_threshold, div_free_threshold, incompressible_iter_max, div_free_iter_max, incomp_warm_start, div_warm_start)

def add_solver_wcsph(self, gamma: ti.f32 = 7, stiffness: ti.f32 = 1000):
    self.m_solver_wcsph = WCSPH_solver(self, gamma, stiffness)

def add_solver_ism(self, Cd, Cf, k_vis_inter, k_vis_inner):
    self.m_solver_ism = Implicit_mixture_solver(self, Cd, Cf, k_vis_inter, k_vis_inner, self.m_world)

def add_solver_JL21(self, kd, Cf, k_vis):
    self.m_solver_JL21 = JL21_mixture_solver(self, kd, Cf, k_vis, self.m_world)

def add_solver_isph(self,K=1e5, G=1e5):
    self.m_solver_isph = ISPH_Elastic_solver(self,self.m_world,K, G)

def add_solver_porous(self,elastic_part, absorption_constant, porosity, rest_pore_pressure, permeability, capillary, pore_parm,kinematic_viscosity_fluid):
    self.m_solver_porous = Porous_multi_solver(self,elastic_part, absorption_constant, porosity, rest_pore_pressure, permeability, capillary, pore_parm,kinematic_viscosity_fluid,self.m_world)