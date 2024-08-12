import taichi as ti
from ti_sph import *
from template_part import part_template
import time
import sys
import numpy as np
import csv
import copy
from ply_util import *
np.set_printoptions(threshold=sys.maxsize)

''' TAICHI SETTINGS '''
ti.init(arch=ti.gpu, device_memory_fraction=0.9) 


''' SOLVER SETTINGS '''
SOLVER_ISM = 0
SOLVER_JL21 = 1
solver = SOLVER_ISM # choose the solver                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  

''' SETTINGS OUTPUT DATA '''
# output fps
fps = 60
# max output frame number
output_frame_num = 2000

#####porous start#####
absorption_constant = vec3f(5, 5,5)
porosity = 0.9
rest_pore_pressure = 1000
permeability = vec3f(5,0.5,5)
capillary = 0.5
pore_parm = 1e-1
#####porous end#####

''' SETTINGS SIMULATION '''
world_dim=3
part_size = 0.014
run_ply=True
phase_num = 3 
if solver == SOLVER_ISM:
    max_time_step = part_size/100
elif solver == SOLVER_JL21:
    max_time_step = part_size/100
#  diffusion amount: Cf = 0 yields no diffusion
Cf = 0.0 
#  drift amount (for ism): Cd = 0 yields free driftand Cd = 1 yields no drift
Cd = 0.3 
# drag coefficient (for JL21): kd = 0 yields maximum drift 
kd = 0.0
flag_strat_drift = True
# kinematic viscosity of fluid
kinematic_viscosity_fluid = 2e-2

''' INIT SIMULATION WORLD '''
world = World(dim=world_dim)
world.set_part_size(part_size)
world.set_dt(max_time_step) 
# set up the multiphase. The first argument is the number of phases. The second argument is the color of each phase (RGB). The third argument is the rest density of each phase.
world.set_multiphase(phase_num,[vec3f(0.5,0.2,0),vec3f(0,0.1,0.6),vec3f(0,0.5,1)],[500,500,1000]) 

''' DATA SETTINGS FOR FLUID PARTICLE '''
# fluid_cube_data = Cube_data(type=Cube_data.FIXED_CELL_SIZE, lb=vec3f(-8.5, -8.8,8.0), rt=vec3f(-8.5, -8.8,8.0), span=world.g_part_size[None]*1.001)
boundary_box_data = Plane_data(lb=vec3f(-3, -2.45,-0.5), rt=vec3f(3, -0.5,5), span=world.g_part_size[None]*1.001,layers=2)
elastic_cube_data=Cube_data(type=Cube_data.FIXED_CELL_SIZE, lb=vec3f(-0.6, -2.4,2.0), rt=vec3f(0.6, -1.0,3.0), span=world.g_part_size[None]*1.001)

# particle number of fluid/boundary
# fluid_part_num = fluid_cube_data.num
bound_part_num = boundary_box_data.num
elastic_part_num=elastic_cube_data.num
# print("fluid_part_num", fluid_part_num)
print("bound_part_num", bound_part_num)
print("elastic_part_num", elastic_part_num)

'''INIT AN FLUID PARTICLE OBJECT'''
# create a fluid particle object. first argument is the number of particles. second argument is the size of the particle. third argument is whether the particle is dynamic or not.
# fluid_part = world.add_part_obj(part_num=fluid_part_num, size=world.g_part_size, is_dynamic=True,type=1)
# fluid_part.instantiate_from_template(part_template, world)

# ''' FEED DATA TO THE FLUID PARTICLE OBJECT '''
# fluid_part.open_stack(val_i(fluid_part_num)) # open the stack to feed data
# fluid_part.fill_open_stack_with_nparr(fluid_part.pos, fluid_cube_data.pos) # feed the position data
# fluid_part.fill_open_stack_with_val(fluid_part.size, fluid_part.get_part_size()) # feed the particle size
# fluid_part.fill_open_stack_with_val(fluid_part.volume, val_f(fluid_part.get_part_size()[None]**world.g_dim[None])) # feed the particle volume
# val_frac = ti.field(dtype=ti.f32, shape=phase_num) # create a field to store the volume fraction
# val_frac[0], val_frac[1], val_frac[2] = 0.0,0.0,1.0 # set up the volume fraction
# fluid_part.fill_open_stack_with_vals(fluid_part.phase.val_frac, val_frac) # feed the volume fraction
# fluid_part.close_stack() # close the stack


''' INIT A ELASTIC PARTICLE OBJECT '''
elastic_part=world.add_part_obj(part_num=elastic_part_num,size=world.g_part_size, is_dynamic=True,type=2)
elastic_part.instantiate_from_template(part_template, world)
''' FEED DATA TO THE ELASTIC PARTICLE OBJECT '''
elastic_part.open_stack(val_i(elastic_part_num)) # open the stack to feed data
elastic_part.fill_open_stack_with_nparr(elastic_part.pos, elastic_cube_data.pos) # feed the position data
elastic_part.fill_open_stack_with_nparr(elastic_part.elastic_sph.pos_0, elastic_cube_data.pos) # feed the position data
elastic_part.fill_open_stack_with_val(elastic_part.size, elastic_part.get_part_size()) # feed the particle size
elastic_part.fill_open_stack_with_val(elastic_part.volume, val_f(elastic_part.get_part_size()[None]**world.g_dim[None])) # feed the particle volume
elastic_part.fill_open_stack_with_val(elastic_part.mass, val_f(1000*elastic_part.get_part_size()[None]**world.g_dim[None]))
elastic_part.fill_open_stack_with_val(elastic_part.rest_density, val_f(1000))
init_vel = np.zeros((elastic_cube_data.num,world.g_dim[None]))
init_vel[..., 1] = 0.0
elastic_part.fill_open_stack_with_nparr(elastic_part.vel, init_vel) # feed the particle velocity
elastic_part.close_stack() # close the stack


''' INIT A BOUNDARY PARTICLE OBJECT '''
bound_part = world.add_part_obj(part_num=bound_part_num, size=world.g_part_size, is_dynamic=False,type=0)
bound_part.instantiate_from_template(part_template, world)

''' FEED DATA TO THE BOUNDARY PARTICLE OBJECT '''
bound_part.open_stack(val_i(bound_part_num))
bound_part.fill_open_stack_with_nparr(bound_part.pos, boundary_box_data.pos)
bound_part.fill_open_stack_with_val(bound_part.size, bound_part.get_part_size())
bound_part.fill_open_stack_with_val(bound_part.volume, val_f(bound_part.get_part_size()[None]**world.g_dim[None]))
bound_part.fill_open_stack_with_val(bound_part.mass, val_f(1000*bound_part.get_part_size()[None]**world.g_dim[None]))
bound_part.fill_open_stack_with_val(bound_part.rest_density, val_f(1000))
bound_part.close_stack()


'''INIT NEIGHBOR SEARCH OBJECTS'''
neighb_list=[elastic_part, bound_part]
# fluid_part.add_module_neighb_search()
bound_part.add_module_neighb_search()
elastic_part.add_module_neighb_search()

# fluid_part.add_neighb_objs(neighb_list)
bound_part.add_neighb_objs(neighb_list)
elastic_part.add_neighb_objs(neighb_list)


# fluid_part.add_solver_adv()
# fluid_part.add_solver_sph()
# fluid_part.add_solver_porous(elastic_part, absorption_constant, porosity, rest_pore_pressure, permeability, capillary, pore_parm,kinematic_viscosity_fluid)

# if solver == SOLVER_ISM:
#     fluid_part.add_solver_df(div_free_threshold=2e-4, incomp_warm_start=False, div_warm_start=False)
#     fluid_part.add_solver_ism(Cd=Cd, Cf=Cf, k_vis_inter=kinematic_viscosity_fluid, k_vis_inner=kinematic_viscosity_fluid)
# elif solver == SOLVER_JL21:
#     fluid_part.add_solver_wcsph()
#     fluid_part.add_solver_JL21(kd=kd,Cf=Cf,k_vis=kinematic_viscosity_fluid)

bound_part.add_solver_sph()
if solver == SOLVER_ISM:
    bound_part.add_solver_df(div_free_threshold=2e-4)
    bound_part.add_solver_ism(Cd=Cd, Cf=Cf, k_vis_inter=kinematic_viscosity_fluid, k_vis_inner=kinematic_viscosity_fluid)
elif solver == SOLVER_JL21:
    bound_part.add_solver_wcsph()
    bound_part.add_solver_JL21(kd=kd,Cf=Cf,k_vis=kinematic_viscosity_fluid)

elastic_part.add_solver_adv()
elastic_part.add_solver_sph()
elastic_part.add_solver_isph(K=8e4,G=8e4)
elastic_part.add_solver_df(div_free_threshold=2e-4, incomp_warm_start=False, div_warm_start=False)


''' INIT ALL SOLVERS '''
world.init_modules()

''' DATA PREPERATIONS '''
def prep_ism():
    world.neighb_search() # perform the neighbor search  
    
    elastic_part.m_solver_isph.copy_neighbor_0(elastic_part.m_neighb_search.neighb_pool_0,elastic_part.m_neighb_search.neighb_pool)
    # fluid_part.m_solver_ism.update_rest_density_and_mass()
    # fluid_part.m_solver_ism.update_color() # update the color
    # fluid_part.m_solver_ism.recover_phase_vel_from_mixture() # recover the phase velocity from the mixture velocity
    
    elastic_part.m_solver_isph.clear_F()
    elastic_part.m_solver_isph.clear_link_num()
    elastic_part.m_solver_isph.loop_neighb(elastic_part.m_neighb_search.neighb_pool_0,elastic_part, elastic_part.m_solver_isph.inloop_compute_L_ker)
    elastic_part.m_solver_isph.compute_L_inv(elastic_part_num)    
    elastic_part.m_solver_isph.set_link_num_0()
    
    # fluid_part.m_solver_porous.set_alpha_and_beta()
    # fluid_part.m_solver_porous.clear_attr(fluid_part.porous_multi_fluid.outer_pore_pressure,fluid_part)
    # fluid_part.m_solver_porous.compute_outer_pore_pressure()
    
def prep_JL21():
    world.neighb_search() # perform the neighbor search
    # fluid_part.m_solver_JL21.update_rest_density_and_mass()
    # fluid_part.m_solver_JL21.update_color() # update the color
    # fluid_part.m_solver_JL21.recover_phase_vel_from_mixture() # recover the phase velocity from the mixture velocity


''' SIMULATION LOOPS '''
def loop_ism():       
    ''' color '''
    # fluid_part.m_solver_ism.update_color()

    ''' neighb search'''
    world.neighb_search()

    ''' sph pre-computation '''
    # fluid_part.m_solver_porous.change_fluid_volume()
    world.step_sph_compute_compression_ratio()
    world.step_df_compute_beta()    
    # print(elastic_part.sph.compression_ratio)

    elastic_part.m_solver_isph.init_neighbor_flag()
    world.eliminate_interior_df()
    elastic_part.m_solver_isph.regularize_compression_ratio()

    # fluid_part.m_solver_porous.regularize_compression_ratio_by_alpha()

    ''' pressure accleration (divergence-free) '''
    world.step_vfsph_div(update_vel=False)
    # fluid_part.m_solver_porous.recover_fluid_volume()
    #print('div_free iter:', fluid_part.m_solver_df.div_free_iter[None])
    
    ''' [ISM] distribute pressure acc to phase acc and update phase vel '''
    # fluid_part.m_solver_df.get_acc_pressure()
    # fluid_part.m_solver_ism.clear_phase_acc()
    # fluid_part.m_solver_ism.ditribute_acc_pressure_2_phase()
    # fluid_part.m_solver_ism.phase_acc_2_phase_vel()
    # fluid_part.m_solver_ism.update_vel_from_phase_vel()
      
    ''' viscosity & gravity accleration and update phase vel '''
    # fluid_part.m_solver_ism.clear_phase_acc()
    # fluid_part.m_solver_ism.add_phase_acc_gravity()
    # fluid_part.m_solver_ism.loop_neighb(fluid_part.m_neighb_search.neighb_pool, fluid_part, fluid_part.m_solver_ism.inloop_add_phase_acc_vis)
    
    # fluid_part.m_solver_ism.phase_acc_2_phase_vel() 
    # fluid_part.m_solver_ism.update_vel_from_phase_vel()

    #porous
    # fluid_part.m_solver_porous.reset_alpha_and_beta()    
    # fluid_part.m_solver_porous.clear_attrs()
    # fluid_part.m_solver_porous.loop_neighb(fluid_part.m_neighb_search.neighb_pool, elastic_part, fluid_part.m_solver_porous.inloop_compute_Nf)
    # # print("Nf: ",fluid_part.porous_multi_fluid.Nf)    
    # elastic_part.m_solver_isph.loop_neighb(elastic_part.m_neighb_search.neighb_pool, fluid_part, fluid_part.m_solver_porous.inloop_compute_solid_beta)
    # fluid_part.m_solver_porous.compute_solid_pore_pressure()
    # fluid_part.m_solver_porous.loop_neighb(fluid_part.m_neighb_search.neighb_pool, elastic_part, fluid_part.m_solver_porous.inloop_compute_Darcy_flux)
    # fluid_part.m_solver_porous.loop_neighb(fluid_part.m_neighb_search.neighb_pool, elastic_part, fluid_part.m_solver_porous.inloop_compute_grad_Darcy_flux)
    # fluid_part.m_solver_porous.compute_virtual_fraction_change()
    # # print("grad_Darcy_flux: ",fluid_part.porous_multi_fluid.grad_Darcy_flux)
    # # print("Sr: ",elastic_part.porous_multi_fluid.Sr)
    # fluid_part.m_solver_porous.loop_neighb(fluid_part.m_neighb_search.neighb_pool, elastic_part, fluid_part.m_solver_porous.inloop_compute_capillary_force)
    # fluid_part.m_solver_porous.loop_neighb(fluid_part.m_neighb_search.neighb_pool, elastic_part, fluid_part.m_solver_porous.inloop_compute_solid_vel_to_fluid)
    # fluid_part.m_solver_porous.compute_acc_pore()
            
    # fluid_part.m_solver_ism.phase_acc_2_phase_vel() 
    # fluid_part.m_solver_ism.update_vel_from_phase_vel()

    '''elastic compute'''
    elastic_part.m_solver_adv.clear_acc()
    elastic_part.m_solver_isph.clear_force()
    elastic_part.m_solver_isph.copy_last_vel()
    elastic_part.m_solver_isph.clear_F()
    elastic_part.m_solver_isph.loop_neighb(elastic_part.m_neighb_search.neighb_pool_0,elastic_part, elastic_part.m_solver_isph.inloop_compute_F)
    elastic_part.m_solver_isph.compute_R_pd()
    # compute F_star with F cleared first
    elastic_part.m_solver_isph.clear_F()
    elastic_part.m_solver_isph.loop_neighb(elastic_part.m_neighb_search.neighb_pool_0,elastic_part, elastic_part.m_solver_isph.inloop_compute_F_star)
    elastic_part.m_solver_isph.compute_F_star_add_I()
    elastic_part.m_solver_isph.compute_F_svd(elastic_part_num)
    elastic_part.m_solver_isph.compute_Z_projection()
    # print("G:",elastic_part.elastic_sph.G[1000])
    # print("K:",elastic_part.elastic_sph.K[1000])
    # elastic_part.m_solver_isph.clamp_singular()
    elastic_part.m_solver_isph.compute_eps()
    elastic_part.m_solver_isph.compute_P()    
    elastic_part.m_solver_isph.loop_neighb(elastic_part.m_neighb_search.neighb_pool_0,elastic_part, elastic_part.m_solver_isph.inloop_compute_force)    
    elastic_part.m_solver_isph.update_acc_from_force()
    # elastic_part.m_solver_isph.loop_neighb(elastic_part.m_neighb_search.neighb_pool_0,elastic_part, elastic_part.m_solver_isph.inloop_compute_vis)
    
    elastic_part.m_solver_adv.add_gravity_acc()
    elastic_part.m_solver_adv.acc2vel()

    # print("alpha_sum: ",fluid_part.porous_multi_fluid.alpha_sum)
    
    # fluid_part.m_solver_porous.change_fluid_volume()
    # world.step_sph_compute_compression_ratio()
    # world.step_df_compute_beta()    
    # print(elastic_part.sph.compression_ratio)

    # elastic_part.m_solver_isph.init_neighbor_flag()
    # world.eliminate_interior_df()
    # elastic_part.m_solver_isph.regularize_compression_ratio()
    ''' pressure accleration (divergence-free) '''
    world.step_vfsph_incomp(update_vel=False)
    # fluid_part.m_solver_porous.recover_fluid_volume()
    #print('incomp iter:', fluid_part.m_solver_df.incompressible_iter[None])

    ''' distribute pressure acc to phase acc and update phase vel '''
    # fluid_part.m_solver_df.get_acc_pressure()
    # fluid_part.m_solver_ism.clear_phase_acc()
    # fluid_part.m_solver_ism.ditribute_acc_pressure_2_phase()
    # fluid_part.m_solver_ism.phase_acc_2_phase_vel()
    # fluid_part.m_solver_ism.update_vel_from_phase_vel()

    elastic_part.m_solver_df.update_vel(elastic_part.vel)
    elastic_part.m_solver_isph.compute_friction()

    # world.enhance_fluid_boundary_coupling_start(elastic_part=elastic_part)
    # world.step_sph_compute_compression_ratio()
    # world.step_df_compute_beta()    
    # world.step_vfsph_incomp(update_vel=False)
    # world.enhance_fluid_boundary_coupling_end(elastic_part=elastic_part)  

    # fluid_part.m_solver_df.get_acc_pressure()
    # fluid_part.m_solver_ism.clear_phase_acc()
    # fluid_part.m_solver_ism.ditribute_acc_pressure_2_phase()
    # fluid_part.m_solver_ism.phase_acc_2_phase_vel()
    # fluid_part.m_solver_ism.update_vel_from_phase_vel()

    ''' update particle position from velocity '''
    world.update_pos_from_vel()

    ''' phase change '''
    # fluid_part.m_solver_ism.update_val_frac()
    # fluid_part.m_solver_ism.update_vel_from_phase_vel()

    ''' update mass and velocity '''
    # fluid_part.m_solver_ism.regularize_val_frac()
    # fluid_part.m_solver_ism.update_rest_density_and_mass()
    # fluid_part.m_solver_ism.update_vel_from_phase_vel()

    ''' cfl condition update'''
    # world.cfl_dt(0.4, max_time_step)

    ''' statistical info '''
    # #print(' ')
    # fluid_part.m_solver_ism.statistics_linear_momentum_and_kinetic_energy()
    # fluid_part.m_solver_ism.statistics_angular_momentum()
    # fluid_part.m_solver_ism.debug_check_val_frac()

def loop_JL21():
    pass

''' Viusalization and run '''
def vis_run(loop):
    global flag_strat_drift
    inv_fps = 1/fps
    timer = 0
    sim_time = 0
    loop_count = 0
    flag_write_img = False

    gui = Gui3d()
    gui.op_system_run=True
    while gui.window.running:

        gui.monitor_listen()

        if gui.op_system_run:
            loop()
            loop_count += 1       
            sim_time += world.g_dt[None]
            
            if(sim_time > timer*inv_fps):
                if gui.op_write_file:
                    pass
                timer += 1
                flag_write_img = True

            # gui.op_system_run=False
        if gui.op_refresh_window:
            gui.scene_setup()
            # gui.scene_add_parts_colorful(obj_pos=fluid_part.pos, obj_color=fluid_part.rgb,index_count=fluid_part.get_stack_top()[None],size=world.g_part_size[None]/2)
            gui.scene_add_parts_colorful(obj_pos=elastic_part.pos, obj_color=elastic_part.rgb,index_count=elastic_part.get_stack_top()[None],size=world.g_part_size[None]/2)
            gui.scene_add_parts(obj_pos=bound_part.pos, obj_color=(0.86,0.86,0.86),index_count=bound_part.get_stack_top()[None],size=world.g_part_size[None]/12)
            gui.canvas.scene(gui.scene)  # Render the scene

            if gui.op_save_img and flag_write_img:
                gui.window.save_image('output/'+str(timer)+'.png')
                flag_write_img = False

            gui.window.show()
        
        if timer > output_frame_num:
            break

''' no Viusalization and run '''
def run(loop):
    inv_fps = 1/fps
    timer = 0
    sim_time = 0
    loop_count = 0

    while(True):
        loop()
        loop_count += 1
        sim_time += world.g_dt[None]
        
        if(sim_time > timer*inv_fps):
            timer += 1
            if run_ply:
                write_ply(path='c0/sand', 
                        frame_num=timer, 
                        dim=world.g_dim[None], 
                        num=elastic_part.m_part_num[None], 
                        pos=elastic_part.pos.to_numpy())
                
        if timer > output_frame_num:
            exit()


if run_ply==False:
    ''' RUN THE SIMULATION '''
    if solver == SOLVER_ISM:
        prep_ism()
        vis_run(loop_ism)
else:
    ''' RUN THE SIMULATION '''
    if solver == SOLVER_ISM:
        prep_ism()
        run(loop_ism)        








