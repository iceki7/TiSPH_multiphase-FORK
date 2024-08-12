import taichi as ti
from .sph_funcs import *
from .Solver_sph import SPH_solver
from ..basic_op.type import *
from ..basic_obj.Obj_Particle import Particle


@ti.data_oriented
class Porous_multi_solver(SPH_solver):
    def __init__(self, fluid_part: Particle, elastic_part, absorption_constant, porosity, rest_pore_pressure, permeability, capillary, pore_parm,kinematic_viscosity_fluid,world):        
        super().__init__(fluid_part)
        self.world = world
        self.solid = elastic_part
        self.fluid = fluid_part
        # self.Nf = ti.field(float, self.fluid.ti_get_stack_top()[None])
        
        self.absorption_constant = vec3_f(absorption_constant)
        self.porosity = porosity
        self.rest_pore_pressure = rest_pore_pressure
        self.permeability = vec3_f(permeability)
        self.capillary = capillary
        self.pore_parm = pore_parm
        self.kinematic_viscosity_fluid=kinematic_viscosity_fluid    
        self.capillary_ratio = 2e-2
        self.stay_ratio=2e-2
        self.fluid_factor=ti.Vector.field(3, ti.f32, ())
        self.fluid_factor[None]=vec3f(1,0,0)
    

    @ti.kernel
    def clear_attr(self,attr:ti.template(),obj:ti.template()):
        for part_id in range(obj.ti_get_stack_top()[None]):            
            attr[part_id] *= 0  


    def clear_attrs(self):        
        self.clear_attr(self.fluid.porous_multi_fluid.Nf,self.fluid)
        self.clear_attr(self.solid.porous_multi_fluid.solid_beta,self.solid)  
        self.clear_attr(self.solid.porous_multi_fluid.solid_pore_pressure,self.solid)  
        self.clear_attr(self.fluid.porous_multi_fluid.Darcy_flux,self.fluid)
        self.clear_attr(self.fluid.porous_multi_fluid.grad_Darcy_flux,self.fluid)
        self.clear_attr(self.fluid.porous_multi_fluid.solid_strain,self.fluid)
        self.clear_attr(self.fluid.porous_multi_fluid.grad_pore_pressure,self.fluid)
        
        
        
    # 计算流体外部孔压
    @ti.kernel
    def compute_outer_pore_pressure(self):
        for part_id in range(self.fluid.ti_get_stack_top()[None]):
            for k in range(self.world.g_phase_num[None]):
                self.fluid.porous_multi_fluid.outer_pore_pressure[part_id][k] = self.rest_pore_pressure * \
                    self.porosity * self.absorption_constant[None][k]

    #计算流体粒子附近的固体粒子对它的贡献
    @ti.func
    def inloop_compute_Nf(self, part_id: ti.i32, neighb_part_id: ti.i32, neighb_part_shift: ti.i32, neighb_pool:ti.template(),neighb_obj:ti.template()):
        cached_dist = neighb_pool.cached_neighb_attributes[neighb_part_shift].dist
        cached_W = neighb_pool.cached_neighb_attributes[neighb_part_shift].W
        if bigger_than_zero(cached_dist): 
            self.fluid.porous_multi_fluid.Nf[part_id] += cached_W


    #计算固体粒子的虚相
    @ti.func
    def inloop_compute_solid_beta(self, part_id: ti.i32, neighb_part_id: ti.i32, neighb_part_shift: ti.i32, neighb_pool:ti.template(),neighb_obj:ti.template()):
        cached_dist = neighb_pool.cached_neighb_attributes[neighb_part_shift].dist
        cached_W = neighb_pool.cached_neighb_attributes[neighb_part_shift].W
        if bigger_than_zero(cached_dist): 
            if bigger_than_zero(self.fluid.porous_multi_fluid.Nf[neighb_part_id]):       
                for k in range(self.world.g_phase_num[None]):                             
                    tmp1 = self.fluid.porous_multi_fluid.fluid_beta[neighb_part_id][k] * \
                                        self.fluid.mass[neighb_part_id]                    
                    tmp2 = cached_W
                    tmp3 = self.world.g_phase_rest_density[None][k] / (
                        self.fluid.rest_density[neighb_part_id] * self.fluid.porous_multi_fluid.Nf[neighb_part_id])
                    self.solid.porous_multi_fluid.solid_beta[part_id][k] += tmp1 * tmp2 * tmp3/(self.solid.volume[part_id]*self.world.g_phase_rest_density[None][k])
                    

    # @ti.kernel
    # def sign_saturated_solid(self):
    #     for part_id in range(self.solid.ti_get_stack_top()[None]):            
    #         for k in range(self.world.g_phase_num[None]):    
    #             if self.solid.porous_multi_fluid.saturated[part_id][k]==0:
    #                 if self.solid.porous_multi_fluid.solid_beta[part_id][k]>0.1:
    #                     self.solid.porous_multi_fluid.saturated[part_id][k]=1        

    # @ti.kernel
    # def compute_solid_saturation(self):
    #     for part_id in range(self.solid.ti_get_stack_top()[None]):            
    #         for k in range(self.world.g_phase_num[None]):    
    #             if self.solid.porous_multi_fluid.saturated[part_id][k]==0:
    #                 if self.solid.porous_multi_fluid.solid_beta[part_id][k]>0.1:
    #                     self.solid.porous_multi_fluid.saturated[part_id][k]=1       

    #计算固体孔压
    @ti.kernel
    def compute_solid_pore_pressure(self):
        for part_id in range(self.solid.ti_get_stack_top()[None]):
            volumetriStrain = self.solid.elastic_sph.eps[part_id].trace()
            # volumetriStrain = self.solid.peridynamics.dilatation[part_id]
            # if volumetriStrain>0.1:
            #     print(volumetriStrain)
            sum_solid_beta = 0.0
            for k in range(self.world.g_phase_num[None]):
                sum_solid_beta += self.solid.porous_multi_fluid.solid_beta[part_id][k]
            self.solid.porous_multi_fluid.Sr[part_id]=sum_solid_beta
            # if sum_solid_beta>0.9:
            #     self.solid.rgb[part_id]=vec3f(0.9,0.1,0.1)   
            # else:
            #     self.solid.rgb[part_id] = vec3f(0.9,0.9,0.05)         
            #计算溶解指数
            #     self.solid.elastic_sph[part_id].dissolve+=self.solid.porous_multi_fluid.solid_beta[part_id][k]*self.fluid_factor[None][k]
            # #没达到溶解阈值的饱和度前不能溶解
            # if self.solid.elastic_sph[part_id].dissolve<=0.9:
            #     self.solid.elastic_sph[part_id].dissolve=0
            self.solid.porous_multi_fluid.solid_pore_pressure[part_id] = self.rest_pore_pressure * \
            sum_solid_beta -  volumetriStrain
                       


    #计算流体的达西通量
    @ti.func
    def inloop_compute_Darcy_flux(self, part_id: ti.i32, neighb_part_id: ti.i32, neighb_part_shift: ti.i32, neighb_pool:ti.template(),neighb_obj:ti.template()):
        cached_dist = neighb_pool.cached_neighb_attributes[neighb_part_shift].dist
        cached_grad_W = neighb_pool.cached_neighb_attributes[neighb_part_shift].grad_W
        if bigger_than_zero(cached_dist):
            grad_W_vec = cached_grad_W
            tmp21 = self.solid.mass[neighb_part_id] /  self.solid.rest_density[neighb_part_id]
            for k in range(self.world.g_phase_num[None]):
                tmp1 = self.capillary * self.permeability[None][k]
                tmp22 = self.fluid.porous_multi_fluid.fluid_alpha[part_id][k] +  self.solid.porous_multi_fluid.solid_beta[neighb_part_id][k]
                tmp3 = (self.fluid.porous_multi_fluid.outer_pore_pressure[part_id][k] - self.solid.porous_multi_fluid.solid_pore_pressure[
                    neighb_part_id]) * grad_W_vec / self.kinematic_viscosity_fluid   
                tmp_res=   tmp1 * tmp21 * tmp22 * tmp3
                for dim_num in range(self.world.g_dim[None]):
                    self.fluid.porous_multi_fluid.Darcy_flux[part_id][k,dim_num] += tmp_res[dim_num]
                    # self.fluid.porous_multi_fluid.grad_pore_pressure[part_id][k,dim_num]+=tmp3[dim_num]

     #计算流体的达西通量梯度
    @ti.func
    def inloop_compute_grad_Darcy_flux(self, part_id: ti.i32, neighb_part_id: ti.i32, neighb_part_shift: ti.i32, neighb_pool:ti.template(),neighb_obj:ti.template()):
        cached_dist = neighb_pool.cached_neighb_attributes[neighb_part_shift].dist
        cached_grad_W = neighb_pool.cached_neighb_attributes[neighb_part_shift].grad_W
        if bigger_than_zero(cached_dist):
            grad_W_vec = cached_grad_W
            tmp1 = self.solid.mass[neighb_part_id] /self.solid.rest_density[neighb_part_id]            
            for k in range(self.world.g_phase_num[None]):
                tmp2 =0.0
                for dim_num in range(self.world.g_dim[None]):
                    tmp2 += self.fluid.porous_multi_fluid.Darcy_flux[part_id][k,dim_num]*grad_W_vec[dim_num]
                self.fluid.porous_multi_fluid.grad_Darcy_flux[part_id][k] += tmp1 * tmp2 

    #计算毛细加速度
    @ti.func
    def inloop_compute_capillary_force(self, part_id: ti.i32, neighb_part_id: ti.i32, neighb_part_shift: ti.i32, neighb_pool:ti.template(),neighb_obj:ti.template()):
        cached_dist = neighb_pool.cached_neighb_attributes[neighb_part_shift].dist
        cached_grad_W = neighb_pool.cached_neighb_attributes[neighb_part_shift].grad_W
        if bigger_than_zero(cached_dist):
            grad_W_vec = cached_grad_W
            for k in range(self.world.g_phase_num[None]):
                self.fluid.phase.acc[part_id, k]+=self.capillary_ratio * self.fluid.porous_multi_fluid.fluid_beta[part_id][k] * self.solid.mass[
                                neighb_part_id] * grad_W_vec                

    # 计算虚相变化
    @ti.kernel
    def compute_virtual_fraction_change(self):
        for part_id in range(self.fluid.ti_get_stack_top()[None]):
            sum = 0.0  # 计算各种分数的sum
            for k in range(self.world.g_phase_num[None]):
                self.fluid.porous_multi_fluid.grad_Darcy_flux[part_id][k] *= self.dt[None]*1e-5
                
                if self.fluid.porous_multi_fluid.grad_Darcy_flux[part_id][k] > 0:
                    if self.fluid.porous_multi_fluid.fluid_alpha[part_id][k] - self.fluid.porous_multi_fluid.grad_Darcy_flux[part_id][k] < 0.001:
                        self.fluid.porous_multi_fluid.grad_Darcy_flux[part_id][k] = self.fluid.porous_multi_fluid.fluid_alpha[part_id][k]
                else:                              
                    if self.fluid.porous_multi_fluid.fluid_beta[part_id][k] + self.fluid.porous_multi_fluid.grad_Darcy_flux[part_id][k] < 0.001:
                        self.fluid.porous_multi_fluid.grad_Darcy_flux[part_id][k] = -self.fluid.porous_multi_fluid.fluid_beta[part_id][k]
                self.fluid.porous_multi_fluid.fluid_alpha[part_id][k] -= self.fluid.porous_multi_fluid.grad_Darcy_flux[part_id][k]
                self.fluid.porous_multi_fluid.fluid_beta[part_id][k] += self.fluid.porous_multi_fluid.grad_Darcy_flux[part_id][k]
                sum += self.fluid.porous_multi_fluid.fluid_alpha[part_id][k] + self.fluid.porous_multi_fluid.fluid_beta[part_id][k]
            # print('sum==============',sum)
            self.fluid.porous_multi_fluid.fluid_alpha[part_id] /= sum
            self.fluid.porous_multi_fluid.fluid_beta[part_id] /= sum
            for k in range(self.world.g_phase_num[None]):
                self.fluid.porous_multi_fluid.volume_frac_temp[part_id][k]= self.fluid.phase.val_frac[part_id, k]

    @ti.kernel
    def reset_alpha_and_beta(self):
        for part_id in range(self.fluid.ti_get_stack_top()[None]):
            self.fluid.porous_multi_fluid.alpha_sum[part_id] *= 0
            #增加挤水效果
            # if self.fluid.porous_multi_fluid.Nf[part_id] < 1e-6 or self.fluid.porous_multi_fluid.solid_strain[part_id]<-5e-2:
            #     for k in range(self.world.g_phase_num[None]):
            #         self.fluid.porous_multi_fluid.fluid_alpha[part_id][k] = self.fluid.phase.val_frac[part_id,k]
            #     self.fluid.porous_multi_fluid.fluid_beta[part_id] *= 0
            # else:                    
            for k in range(self.world.g_phase_num[None]):
                if self.fluid.porous_multi_fluid.volume_frac_temp[part_id][k]==0:
                    self.fluid.porous_multi_fluid.fluid_alpha[part_id][k]*=0
                    self.fluid.porous_multi_fluid.fluid_beta[part_id][k]*=0
                else:
                    self.fluid.porous_multi_fluid.fluid_alpha[part_id][k]/=self.fluid.porous_multi_fluid.volume_frac_temp[part_id][k]
                    self.fluid.porous_multi_fluid.fluid_alpha[part_id][k]*=self.fluid.phase.val_frac[part_id, k]
                    self.fluid.porous_multi_fluid.fluid_beta[part_id][k]=self.fluid.phase.val_frac[part_id, k]-self.fluid.porous_multi_fluid.fluid_alpha[part_id][k]                        
            for k in range(self.world.g_phase_num[None]):
                self.fluid.porous_multi_fluid.alpha_sum[part_id] += self.fluid.porous_multi_fluid.fluid_alpha[part_id][k]/(self.permeability[None][k]+1)

    @ti.kernel
    def reset_alpha_and_beta_compare(self):
        for part_id in range(self.fluid.ti_get_stack_top()[None]):
            self.fluid.porous_multi_fluid.alpha_sum[part_id] *= 0
            #增加挤水效果
            # if self.fluid.porous_multi_fluid.Nf[part_id] < 1e-6 or self.fluid.porous_multi_fluid.solid_strain[part_id]<-5e-2:
            #     # for k in range(self.world.g_phase_num[None]):
            #     #     self.fluid.porous_multi_fluid.fluid_alpha[part_id][k] = self.fluid.phase.val_frac[part_id,k]
            #     # self.fluid.porous_multi_fluid.fluid_beta[part_id] *= 0
            #     pass
            # else:                    
            for k in range(self.world.g_phase_num[None]):
                if self.fluid.porous_multi_fluid.volume_frac_temp[part_id][k]==0:
                    self.fluid.porous_multi_fluid.fluid_alpha[part_id][k]*=0
                    self.fluid.porous_multi_fluid.fluid_beta[part_id][k]*=0
                else:
                    self.fluid.porous_multi_fluid.fluid_alpha[part_id][k]/=self.fluid.porous_multi_fluid.volume_frac_temp[part_id][k]
                    self.fluid.porous_multi_fluid.fluid_alpha[part_id][k]*=self.fluid.phase.val_frac[part_id, k]
                    self.fluid.porous_multi_fluid.fluid_beta[part_id][k]=self.fluid.phase.val_frac[part_id, k]-self.fluid.porous_multi_fluid.fluid_alpha[part_id][k]                        
            for k in range(self.world.g_phase_num[None]):
                self.fluid.porous_multi_fluid.alpha_sum[part_id] += self.fluid.porous_multi_fluid.fluid_alpha[part_id][k]

    @ti.kernel      
    def set_alpha_and_beta(self):
        for part_id in range(self.fluid.ti_get_stack_top()[None]):
                for k in range(self.world.g_phase_num[None]):
                    self.fluid.porous_multi_fluid.fluid_alpha[part_id][k] = self.fluid.phase.val_frac[part_id, k]
                    self.fluid.porous_multi_fluid.fluid_beta[part_id][k] = 0
                    self.fluid.porous_multi_fluid.volume_frac_temp[part_id][k]= self.fluid.phase.val_frac[part_id, k]
                    self.fluid.porous_multi_fluid.alpha_sum[part_id] += self.fluid.porous_multi_fluid.fluid_alpha[part_id][k]
    
    @ti.kernel   
    def change_fluid_volume(self):
        for pid in range(self.fluid.ti_get_stack_top()[None]):
            self.fluid.volume[pid] *= self.fluid.porous_multi_fluid.alpha_sum[pid]    
            # self.fluid.volume[pid] = 0.0
            # self.fluid.rgb[pid]=vec3f(0,0.5,self.fluid.porous_multi_fluid.alpha_sum[pid])       
            # 
    @ti.kernel   
    def delete_internal_particle(self):
        for pid in range(self.fluid.ti_get_stack_top()[None]):
            # self.fluid.volume[pid] *= self.fluid.porous_multi_fluid.alpha_sum[pid] 
            if self.fluid.porous_multi_fluid.alpha_sum[pid]<0.5 and self.fluid.phase.val_frac[pid, 2]>0.9:
                self.fluid.phase.val_frac[pid, 0]=1.0
                self.fluid.phase.val_frac[pid, 2]*=0
                # self.fluid.pos[pid] = ti.Vector([-8, -8, -8])
                # self.fluid.vel[pid] *= 0
                # self.fluid.acc[pid] *= 0
                # self.fluid.volume[pid] *= 0
                # self.fluid.state[pid] = 1

    @ti.kernel   
    def recover_fluid_volume(self):
        for pid in range(self.fluid.ti_get_stack_top()[None]):
            self.fluid.volume[pid] =  self.fluid.m_part_size[None]**self.world.g_dim[None]
            
            
    @ti.kernel   
    def compute_acc_pore(self):
        for pid in range(self.fluid.ti_get_stack_top()[None]):
            ρ_βm = 0.0
            sum_beta = 0.0
            for k in range(self.world.g_phase_num[None]):
                sum_beta += self.fluid.porous_multi_fluid.fluid_beta[pid][k]
            if sum_beta > 0:
                for k in range(self.world.g_phase_num[None]):
                    ρ_βm += self.fluid.porous_multi_fluid.fluid_beta[pid][k] * \
                        self.world.g_phase_rest_density[None][k] / sum_beta
            for k in range(self.world.g_phase_num[None]):
                tmp1 =self.fluid.porous_multi_fluid.fluid_beta[pid][k] / \
                    (self.fluid.rest_density[pid] * self.dt[None])
                tmp22 = ρ_βm * self.fluid.vel[pid]
                # tmp2 = self.fluid.porous_multi_fluid.solid_vel[pid]- tmp22
                tmp2 = - tmp22
                self.fluid.phase.acc[pid, k]+=tmp1 * tmp2*self.stay_ratio

    @ti.func
    def inloop_compute_solid_vel_to_fluid(self, part_id: ti.i32, neighb_part_id: ti.i32, neighb_part_shift: ti.i32, neighb_pool:ti.template(),neighb_obj:ti.template()):
        cached_dist = neighb_pool.cached_neighb_attributes[neighb_part_shift].dist
        cached_W = neighb_pool.cached_neighb_attributes[neighb_part_shift].W
        if bigger_than_zero(cached_dist):
            # volumetriStrain = neighb_obj.peridynamics.dilatation[neighb_part_id]*1e2
            volumetriStrain = neighb_obj.elastic_sph.eps[neighb_part_id].trace()
            self.fluid.porous_multi_fluid.solid_strain[part_id]+=volumetriStrain*(cached_W/self.fluid.porous_multi_fluid.Nf[part_id])    

        
    # @ti.kernel
    # def regularize_compression_ratio_by_alpha(self):
    #     for part_id in range(self.obj.ti_get_stack_top()[None]):   
    #             # self.obj.sph[part_id].compression_ratio*= self.fluid.porous_multi_fluid.alpha_sum[part_id]    
    #             self.obj.sph[part_id].compression_ratio=1