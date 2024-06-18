import taichi as ti
import math
from .sph_funcs import *
from .Solver_sph import SPH_solver
from .Neighb_looper import Neighb_looper
from ..basic_op.type import *
from ..basic_obj.Obj_Particle import Particle

from typing import List
import numpy as np


@ti.data_oriented
class ISPH_Elastic_solver(SPH_solver):
    def __init__(self, obj: Particle, world,K=1e5, G=1e5):
        
        super().__init__(obj)
        self.K = ti.field(ti.f32, ())
        self.G = ti.field(ti.f32, ())
        self.K[None] = float(K)
        self.G[None] = float(G)
        self.world = world

    
    @ti.kernel
    def clear_force(self):
        for part_id in range(self.obj.ti_get_stack_top()[None]):            
            self.obj.elastic_sph.force[part_id] *= 0

    @ti.kernel
    def clear_F(self):
        for part_id in range(self.obj.ti_get_stack_top()[None]):            
            self.obj.elastic_sph.F[part_id] *= 0

    @ti.kernel
    def clear_L(self):
        for part_id in range(self.obj.ti_get_stack_top()[None]):            
            self.obj.elastic_sph.L[part_id] *= 0        

    @ti.kernel
    def clear_link_num(self):
        for part_id in range(self.obj.ti_get_stack_top()[None]):            
            self.obj.elastic_sph.link_num[part_id] *= 0      

    @ti.kernel
    def set_link_num_0(self):
        for part_id in range(self.obj.ti_get_stack_top()[None]):            
            self.obj.elastic_sph.link_num_0[part_id] = self.obj.elastic_sph.link_num[part_id]             
                      

    def copy_neighbor_0(self,neighb_pool_0:ti.template(),neighb_pool:ti.template()):
       neighb_pool_0.neighb_obj_pointer.copy_from(neighb_pool.neighb_obj_pointer)
       neighb_pool_0.neighb_pool_container.copy_from(neighb_pool.neighb_pool_container)
       neighb_pool_0.cached_neighb_attributes.copy_from(neighb_pool.cached_neighb_attributes)
       

    @ti.kernel
    def init_neighbor_flag(self):
        for part_id in range(self.obj.ti_get_stack_top()[None]):            
            self.obj.elastic_sph[part_id].flag=1 

    @ti.func
    def inloop_set_neighbor_flag(self, part_id: ti.i32, neighb_part_id: ti.i32, neighb_part_shift: ti.i32, neighb_pool:ti.template(),neighb_obj:ti.template()):
        if self.obj.m_id!=neighb_obj.m_id:
            self.obj.elastic_sph[part_id].flag=0

    @ti.kernel
    def regularize_compression_ratio(self):
        for part_id in range(self.obj.ti_get_stack_top()[None]):   
            if self.obj.elastic_sph[part_id].flag==1:
                self.obj.sph[part_id].compression_ratio=1      

    
    @ti.func
    def inloop_compute_F(self, part_id: ti.i32, neighb_part_id: ti.i32, neighb_part_shift: ti.i32, neighb_pool_0:ti.template(),neighb_obj:ti.template()):
        if not (self.obj.elastic_sph[part_id].dissolve>0 or neighb_obj.elastic_sph[neighb_part_id].dissolve>0):
            cached_dist_0 = neighb_pool_0.cached_neighb_attributes[neighb_part_shift].dist
            cached_grad_W_0 = neighb_pool_0.cached_neighb_attributes[neighb_part_shift].grad_W
            if bigger_than_zero(cached_dist_0):            
                x_ij = self.obj.pos[part_id] - neighb_obj.pos[neighb_part_id]            
                grad_W_vec=self.obj.elastic_sph[part_id].L @ cached_grad_W_0
                self.obj.elastic_sph[part_id].F += neighb_obj.volume[neighb_part_id] * (-x_ij).outer_product(grad_W_vec)

    def compute_L_inv(self,par_num):
        L = self.obj.elastic_sph.L.to_numpy()
        inv = np.linalg.pinv(L[: par_num])
        L[: par_num] = inv
        self.obj.elastic_sph.L.from_numpy(L)

    def compute_F_svd(self,batch_size):
        F_temp = self.obj.elastic_sph.F.to_numpy()
        U, sigma, V_T = np.linalg.svd(F_temp, full_matrices=True)        
        self.obj.elastic_sph.U.from_numpy(U)       
        self.obj.elastic_sph.V_T.from_numpy(V_T)       
        sigma_diag = np.zeros((batch_size, 3, 3))
        for i in range(batch_size):
            sigma_diag[i] = np.diag(sigma[i])
        self.obj.elastic_sph.sigma.from_numpy(sigma_diag)

    @ti.func
    def inloop_compute_L_ker(self, part_id: ti.i32, neighb_part_id: ti.i32, neighb_part_shift: ti.i32, neighb_pool_0:ti.template(),neighb_obj:ti.template()):
        if not (self.obj.elastic_sph[part_id].dissolve>0 or neighb_obj.elastic_sph[neighb_part_id].dissolve>0):
            self.obj.rgb[part_id] = vec3f(0.9,0.9,0.05)
            cached_dist_0 = neighb_pool_0.cached_neighb_attributes[neighb_part_shift].dist
            cached_grad_W_0 = neighb_pool_0.cached_neighb_attributes[neighb_part_shift].grad_W
            x_ij_0 = self.obj.elastic_sph[part_id].pos_0 - self.obj.elastic_sph[neighb_part_id].pos_0
            if bigger_than_zero(cached_dist_0):
                self.obj.elastic_sph[part_id].L += neighb_obj.volume[neighb_part_id] *cached_grad_W_0.outer_product(-x_ij_0)
                self.obj.elastic_sph[part_id].link_num += 1

    @ti.kernel
    def compute_R_pd(self):
        for part_id in range(self.obj.ti_get_stack_top()[None]):    
            self.obj.elastic_sph[part_id].R=ti.polar_decompose(self.obj.elastic_sph[part_id].F)[0]


    @ti.kernel
    def compute_Z_projection(self):
        for part_id in range(self.obj.ti_get_stack_top()[None]):    
            U     = self.obj.elastic_sph[part_id].U
            sigma = self.obj.elastic_sph[part_id].sigma
            V_T   = self.obj.elastic_sph[part_id].V_T
            alpha_p=0.5
            d=self.obj.m_world.g_dim[None]
            I = ti.Matrix.identity(dt=ti.f32, n=self.obj.elastic_sph[0].F.n)
            epsilon=I*ti.log(sigma)  
            # print("epsilon:",epsilon)          
            epsilon_hat=epsilon-(epsilon.trace()/d)*I
            miu0=self.G[None]
            lambda0=self.K[None]-2*miu0/3
            delta_gamma=epsilon_hat.norm()+((d*lambda0+2*miu0)/(2*miu0))*epsilon.trace()*alpha_p
            # print("delta_gamma",delta_gamma)
            # print("delta_gamma",delta_gamma,"epsilon_hat:", epsilon_hat, "lambda0:", lambda0, "miu0:", miu0, "epsilon:", epsilon, "alpha_p:", alpha_p)
            # case1:delta_gamma<=0应力已经在屈服面内，直接返回F
            if delta_gamma<=0:
                # print("case1")
                self.obj.rgb[part_id]=vec3f(0.1,0.9,0.1)
                continue
            # case2:epsilon_hat.norm()=0 or epsilon.trace()>0 应力在圆锥顶点，需要投影到顶点
            elif epsilon_hat.norm()<1e-5 or epsilon.trace()>0:
                # print("F0=",self.obj.elastic_sph[part_id].F,",U=",U,",sigma=",sigma,",V_T=",V_T, "F'=",U @ sigma @ V_T)
                self.obj.elastic_sph[part_id].F=U @ V_T
                self.obj.rgb[part_id]=vec3f(0.9,0.1,0.1)
                # print("case2")                
            # case3:其他情况，应力在圆锥外，需要投影到圆锥面
            else:
                H=epsilon-delta_gamma*(epsilon_hat/epsilon_hat.norm())
                e_H=ti.exp(H)*I
                self.obj.elastic_sph[part_id].F=U @ e_H @ V_T
                self.obj.rgb[part_id]=vec3f(0.1,0.1,0.9)
                # print("case3"," H:",H,"e_H",e_H)

    @ti.kernel
    def clamp_singular(self):
        for part_id in range(self.obj.ti_get_stack_top()[None]):    
            sigma = self.obj.elastic_sph[part_id].sigma
            V_T   = self.obj.elastic_sph[part_id].V_T
            theta_c=2.5e-2
            theta_s=7.5e-3         
            for i in ti.static(range(3)):
                        if sigma[i, i] < 1 - theta_c:
                            sigma[i, i] = 1 - theta_c
                            self.obj.rgb[part_id]=vec3f(0.9,0.1,0.1)
                        elif sigma[i, i] > 1 + theta_s:
                            sigma[i, i] = 1 + theta_s
                            self.obj.rgb[part_id]=vec3f(0.1,0.1,0.9)
                        else:
                            self.obj.rgb[part_id]=vec3f(0.1,0.9,0.1)
            self.obj.elastic_sph[part_id].F=V_T.transpose() @ sigma @ V_T
                

    @ti.kernel
    def compute_friction(self):
        for part_id in range(self.obj.ti_get_stack_top()[None]):
            delta_v=self.obj.vel[part_id]-self.obj.elastic_sph[part_id].vel_0
            j=self.obj.mass[part_id]*delta_v
            n=j/j.norm()
            vin=n.dot(self.obj.vel[part_id])
            vit=self.obj.vel[part_id]-n*vin
            t=vit/vit.norm()
            mu=0.5

            if vit.norm()<=(mu/self.obj.mass[part_id])*j.norm():
                self.obj.vel[part_id]=vin*n
            else:
                self.obj.vel[part_id]=self.obj.vel[part_id]-(mu/self.obj.mass[part_id])*j.norm()*t


            
    @ti.kernel
    def copy_last_vel(self):
        for part_id in range(self.obj.ti_get_stack_top()[None]): 
            self.obj.elastic_sph[part_id].vel_0=self.obj.vel[part_id]        

    # Eqn.5
    @ti.func
    def inloop_compute_F_star(self, part_id: ti.i32, neighb_part_id: ti.i32, neighb_part_shift: ti.i32, neighb_pool_0:ti.template(),neighb_obj:ti.template()):
        if not (self.obj.elastic_sph[part_id].dissolve>0 or neighb_obj.elastic_sph[neighb_part_id].dissolve>0):
            cached_dist_0 = neighb_pool_0.cached_neighb_attributes[neighb_part_shift].dist
            cached_grad_W_0 = neighb_pool_0.cached_neighb_attributes[neighb_part_shift].grad_W
            if bigger_than_zero(cached_dist_0):            
                x_ij = self.obj.pos[part_id] - neighb_obj.pos[neighb_part_id]
                x_ij_0 = self.obj.elastic_sph[part_id].pos_0 - self.obj.elastic_sph[neighb_part_id].pos_0
                grad_W_vec=self.obj.elastic_sph[part_id].R @ self.obj.elastic_sph[part_id].L @ cached_grad_W_0
                self.obj.elastic_sph[part_id].F += neighb_obj.volume[neighb_part_id] *(-x_ij-(self.obj.elastic_sph[part_id].R @ (-x_ij_0))).outer_product(grad_W_vec)


    @ti.kernel
    def compute_F_star_add_I(self):
        I = ti.Matrix.identity(dt=ti.f32, n=self.obj.elastic_sph[0].F.n)
        for part_id in range(self.obj.ti_get_stack_top()[None]):            
            self.obj.elastic_sph[part_id].F += I

    @ti.kernel
    def compute_eps(self):
        I = ti.Matrix.identity(dt=ti.f32, n=self.obj.elastic_sph[0].F.n)
        for part_id in range(self.obj.ti_get_stack_top()[None]):            
            self.obj.elastic_sph[part_id].eps = (self.obj.elastic_sph[part_id].F+self.obj.elastic_sph[part_id].F.transpose())*0.5-I
  
    @ti.kernel
    def compute_P(self):
        I = ti.Matrix.identity(dt=ti.f32, n=self.obj.elastic_sph[0].F.n)
        for part_id in range(self.obj.ti_get_stack_top()[None]):        
            self.obj.elastic_sph[part_id].P = (2 * self.G[None] * self.obj.elastic_sph[part_id].eps) + (
                (self.K[None] - (2 / 3 * self.G[None])) * self.obj.elastic_sph[part_id].eps.trace() * I
            )
            #G = miu, K=lambda+2*miu/3
            #miu=G, lambda=k-2*miu/3

    @ti.func
    def inloop_compute_force(self, part_id: ti.i32, neighb_part_id: ti.i32, neighb_part_shift: ti.i32, neighb_pool_0:ti.template(),neighb_obj:ti.template()):
        if not (self.obj.elastic_sph[part_id].dissolve>0 or neighb_obj.elastic_sph[neighb_part_id].dissolve>0):
            cached_dist_0 = neighb_pool_0.cached_neighb_attributes[neighb_part_shift].dist
            cached_grad_W_0 = neighb_pool_0.cached_neighb_attributes[neighb_part_shift].grad_W        
            if bigger_than_zero(cached_dist_0):
                grad_W_vec_i=self.obj.elastic_sph[part_id].R @ self.obj.elastic_sph[part_id].L @ cached_grad_W_0
                grad_W_vec_j=self.obj.elastic_sph[neighb_part_id].R @ self.obj.elastic_sph[neighb_part_id].L @ (-cached_grad_W_0 )         
                self.obj.elastic_sph[part_id].force += self.obj.volume[part_id]*neighb_obj.volume[neighb_part_id] * ((self.obj.elastic_sph[part_id].P @ grad_W_vec_i)-(self.obj.elastic_sph[neighb_part_id].P @ grad_W_vec_j))

    @ti.kernel
    def update_acc_from_force(self):
        for part_id in range(self.obj.ti_get_stack_top()[None]): 
            self.obj.acc[part_id] += self.obj.elastic_sph[part_id].force / self.obj.mass[part_id]      

    @ti.func
    def inloop_compute_vis(self, part_id: ti.i32, neighb_part_id: ti.i32, neighb_part_shift: ti.i32, neighb_pool:ti.template(), neighb_obj:ti.template()):
        cached_dist = neighb_pool.cached_neighb_attributes[neighb_part_shift].dist
        cached_grad_W = neighb_pool.cached_neighb_attributes[neighb_part_shift].grad_W
        if bigger_than_zero(cached_dist):
            A_ij = self.obj.vel[part_id] - neighb_obj.vel[neighb_part_id]
            x_ij = self.obj.pos[part_id] - neighb_obj.pos[neighb_part_id]            
            self.obj.acc[part_id] += self.obj.k_vis[part_id]*2*(2+self.obj.m_world.g_dim[None])*neighb_obj.volume[neighb_part_id] * cached_grad_W*x_ij*A_ij.dot(x_ij)/(cached_dist**3)

    @ti.kernel
    def add_acc_gravity(self):
        for part_id in range(self.obj.ti_get_stack_top()[None]): 
            self.obj.acc[part_id] += self.world.g_gravity[None]    

    @ti.kernel
    def clear_array(
        self,
        obj_attr: ti.template(),
        list: ti.types.ndarray(),  # 分段赋值
    ):
        for i in range(list.shape[0]):
            obj_attr[list[i]] *= 0

    @ti.kernel
    def attr_add_seq(
        self,
        obj_attr: ti.template(),
        val: ti.template(),
        dt: ti.template(),
        list: ti.types.ndarray(),  # 分段赋值
    ):
        for i in range(list.shape[0]):
            obj_attr[list[i]] += val * dt

    @ti.kernel
    def pos_rotate_seq(
        self,
        obj_attr: ti.template(),
        center: ti.template(),
        angle: ti.template(),
        list: ti.types.ndarray(),  # 分段赋值
    ):
        for i in range(list.shape[0]):
            rotation_matrix = ti.Matrix([[1,0,  0],
                            [0, ti.cos(angle), -ti.sin(angle)],
                            [0,ti.sin(angle), ti.cos(angle)]
                            ])
            relative_coords = obj_attr[list[i]] - center
            rotated_coords = relative_coords@rotation_matrix
            rotated_points = rotated_coords + center
            obj_attr[list[i]] = rotated_points