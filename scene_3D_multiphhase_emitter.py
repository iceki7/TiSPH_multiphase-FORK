import math
import taichi as ti
from ti_sph import *
from template_part import part_template
import time
import sys
import numpy as np
import csv

from ti_sph.basic_data_generator.PLY_data import PLY_data
from ti_sph.basic_data_generator.ply_util import write_ply


#prm_
# rigidname=r"D:\CODE\dataProcessing\rigidx.ply"
rigidname=r"d:\CODE\bunny_0.05.ply"
# rigidname=r"D:\CODE\dataProcessing\propeller.ply"



prm_large=0
prm_vis=1
prm_rigidmodel=0

prm_exportPath=r"./output/"
prm_export=0
# prm_exportPath=r"/w/TiSPH_multiphase/output/"


model1=PLY_data(
    ply_filename=rigidname,
    offset=ti.Vector([0,0,0]))

print("[model part num]\t"+str(model1.pos.shape))


if(prm_rigidmodel):
    rigid1=model1
    rigid_pos=rigid1.pos



np.set_printoptions(threshold=sys.maxsize)

''' TAICHI SETTINGS '''
# Use GPU, comment the below command to run this programme on CPU
if(prm_large):
    ti.init(arch=ti.cuda, device_memory_fraction=0.9) 
else:
    ti.init(arch=ti.cuda, device_memory_GB=3) 


# Use CPU, uncomment the below command to run this programme if you don't have GPU
# ti.init(arch=ti.cpu) 

''' SOLVER SETTINGS '''
SOLVER_ISM = 0  # proposed method
SOLVER_JL21 = 1 # baseline method
solver = SOLVER_ISM # choose the solver

''' SETTINGS OUTPUT DATA '''
# output fps
fps = 60

#prm_
# max output frame number
output_frame_num = 100

''' SETTINGS SIMULATION '''
# size of the particle
part_size = 0.1 
#prm_

# number of phases
phase_num = 3 
# max time step size
if solver == SOLVER_ISM:
    max_time_step = part_size/100
elif solver == SOLVER_JL21:
    max_time_step = part_size/100
#  diffusion amount: Cf = 0 yields no diffusion
Cf = 0.0 
#  drift amount (for ism): Cd = 0 yields free driftand Cd = 1 yields no drift
Cd = 0.0 
# drag coefficient (for JL21): kd = 0 yields maximum drift 
kd = 0.0
flag_strat_drift = True
# kinematic viscosity of fluid
kinematic_viscosity_fluid = 1e-4

#发射粒子设置
emit_counter=0 #计数器：判断当前loop是否该发射
emit_num=0 # 统计发射次数
emit_vel=ti.Vector([10, 0, 0]) #发射速度
emit_threshold=math.ceil((part_size*1.001)/(max_time_step*emit_vel.norm())) #当loop次数到达阈值时发射一次
print("emit_threshold",emit_threshold)

''' INIT SIMULATION WORLD '''
# create a 2D world
world = World(dim=3) 
# set the particle diameter
world.set_part_size(part_size) 
# set the max time step size
world.set_dt(max_time_step) 
# set up the multiphase. The first argument is the number of phases. The second argument is the color of each phase (RGB). The third argument is the rest density of each phase.
world.set_multiphase(phase_num,[vec3f(0.8,0.2,0),vec3f(0,0.8,0.2),vec3f(0,0,1)],[500,500,1000]) 

''' DATA SETTINGS FOR FLUID PARTICLE '''
# generate the fluid particle data as a hollowed sphere, rotating irrotationally
fluid_cube_data = Cube_data(type=Cube_data.FIXED_CELL_SIZE, lb=vec3f(-10.2,0.5,-0.5), rt=vec3f(-1.2, 0.9,0.5), span=world.g_part_size[None]*1.001)
# particle number of fluid/boundary
fluid_part_num = fluid_cube_data.num


#start:找到流体块的最外侧的一层粒子，作为粒子发射位置
direction_emit_init_pos=0 #0 means left-right,1 means high-low,2 means front-back
fluid_pos = fluid_cube_data.pos
xmax = np.max(fluid_pos[:fluid_cube_data.num, direction_emit_init_pos])
bottom_points=[]
for i, p in enumerate(fluid_pos):
    if abs(p[direction_emit_init_pos]-xmax)<part_size/4:
        bottom_points.append(i)
bottom_positions = fluid_pos[bottom_points]
data_ti_container = ti.Vector.field(bottom_positions.shape[1], ti.f32, bottom_positions.shape[0])
data_ti_container.from_numpy(bottom_positions)
emit_num_max=fluid_part_num/bottom_positions.shape[0]
#end

boundary_box_data = Box_data(lb=vec3f(-1.3,-1,-1), rt=vec3f(1.3, 1,1), span=world.g_part_size[None]*1.001,layers=2)
boundary_init_emit_pos = bottom_positions
bound_part_num = boundary_box_data.num
bound_part_num_emit=bottom_positions.shape[0]
if(prm_rigidmodel):
    rigid_part_num=rigid1.num
print("fluid_part_num", fluid_part_num)
# position info of fluid/boundary (as numpy arrays)



# print(fluid_part_pos.dtype)
# print(fluid_part_pos.shape)#fluidpartnum x 3
# exit(0)

if(prm_rigidmodel):
    rigid_part_pos=rigid_pos
# initial velocity info of fluid

'''INIT AN FLUID PARTICLE OBJECT'''
# create a fluid particle object. first argument is the number of particles. second argument is the size of the particle. third argument is whether the particle is dynamic or not.
fluid_part = world.add_part_obj(part_num=fluid_part_num, size=world.g_part_size, is_dynamic=True)
fluid_part.instantiate_from_template(part_template, world)

''' FEED DATA TO THE FLUID PARTICLE OBJECT '''
fluid_part.open_stack(val_i(fluid_part_num)) # open the stack to feed data
# fluid_part.open_stack(val_i(rigid_part_num)) # open the stack to feed data

fluid_part.fill_open_stack_with_nparr(fluid_part.pos, fluid_cube_data.pos) # feed the position data
fluid_part.fill_open_stack_with_val(fluid_part.size, fluid_part.get_part_size()) # feed the particle size
fluid_part.fill_open_stack_with_val(fluid_part.volume, val_f(fluid_part.get_part_size()[None]**world.g_dim[None])) # feed the particle volume
val_frac = ti.field(dtype=ti.f32, shape=phase_num) # create a field to store the volume fraction
val_frac[0], val_frac[1], val_frac[2] = 0.5,0.0,0.5 # set up the volume fraction
fluid_part.fill_open_stack_with_vals(fluid_part.phase.val_frac, val_frac) # feed the volume fraction
init_state= val_i(1)
fluid_part.fill_open_stack_with_val(fluid_part.state, init_state)#set dead
fluid_part.close_stack() # close the stack

''' INIT A BOUNDARY PARTICLE OBJECT '''
bound_part = world.add_part_obj(part_num=bound_part_num+bound_part_num_emit, size=world.g_part_size, is_dynamic=False)
bound_part.instantiate_from_template(part_template, world)
if(prm_rigidmodel):
    rigid_part = world.add_part_obj(part_num=rigid_part_num, size=world.g_part_size, is_dynamic=False)
    rigid_part.instantiate_from_template(part_template, world)


''' FEED DATA TO THE BOUNDARY PARTICLE OBJECT '''
bound_part.open_stack(val_i(bound_part_num))
bound_part.fill_open_stack_with_nparr(bound_part.pos, boundary_box_data.pos)
bound_part.fill_open_stack_with_val(bound_part.size, bound_part.get_part_size())
bound_part.fill_open_stack_with_val(bound_part.volume, val_f(bound_part.get_part_size()[None]**world.g_dim[None]))
bound_part.fill_open_stack_with_val(bound_part.mass, val_f(1000*bound_part.get_part_size()[None]**world.g_dim[None]))
bound_part.fill_open_stack_with_val(bound_part.rest_density, val_f(1000))
bound_part.close_stack()

bound_part.open_stack(val_i(bound_part_num_emit))
bound_part.fill_open_stack_with_nparr(bound_part.pos, boundary_init_emit_pos)
bound_part.fill_open_stack_with_val(bound_part.size, bound_part.get_part_size())
bound_part.fill_open_stack_with_val(bound_part.volume, val_f(bound_part.get_part_size()[None]**world.g_dim[None]))
bound_part.fill_open_stack_with_val(bound_part.mass, val_f(1000*bound_part.get_part_size()[None]**world.g_dim[None]))
bound_part.fill_open_stack_with_val(bound_part.rest_density, val_f(1000))
bound_part.close_stack()

if(prm_rigidmodel):
    rigid_part.open_stack(val_i(rigid_part_num))
    rigid_part.fill_open_stack_with_nparr(rigid_part.pos,rigid_part_pos)
    rigid_part.fill_open_stack_with_val(rigid_part.size, rigid_part.get_part_size())
    rigid_part.fill_open_stack_with_val(rigid_part.volume, val_f(rigid_part.get_part_size()[None]**world.g_dim[None]))
    rigid_part.fill_open_stack_with_val(rigid_part.mass, val_f(1000*rigid_part.get_part_size()[None]**world.g_dim[None]))
    rigid_part.fill_open_stack_with_val(rigid_part.rest_density, val_f(1000))
    rigid_part.close_stack()


'''INIT NEIGHBOR SEARCH OBJECTS'''
neighb_list=[fluid_part, bound_part]
fluid_part.add_module_neighb_search()
bound_part.add_module_neighb_search()
if(prm_rigidmodel):
    rigid_part.add_module_neighb_search()
    rigid_part.add_neighb_objs(neighb_list)

fluid_part.add_neighb_objs(neighb_list)
bound_part.add_neighb_objs(neighb_list)

fluid_part.add_solver_adv()
fluid_part.add_solver_sph()
if solver == SOLVER_ISM:
    fluid_part.add_solver_df(div_free_threshold=2e-4, incomp_warm_start=False, div_warm_start=False)
    fluid_part.add_solver_ism(Cd=Cd, Cf=Cf, k_vis_inter=kinematic_viscosity_fluid, k_vis_inner=kinematic_viscosity_fluid)
elif solver == SOLVER_JL21:
    fluid_part.add_solver_wcsph()
    fluid_part.add_solver_JL21(kd=kd,Cf=Cf,k_vis=kinematic_viscosity_fluid)

bound_part.add_solver_sph()
if(prm_rigidmodel):
    rigid_part.add_solver_sph()
if solver == SOLVER_ISM:
    bound_part.add_solver_df(div_free_threshold=2e-4)
    bound_part.add_solver_ism(Cd=Cd, Cf=Cf, k_vis_inter=kinematic_viscosity_fluid, k_vis_inner=kinematic_viscosity_fluid)
    if(prm_rigidmodel):
        rigid_part.add_solver_df(div_free_threshold=2e-4)
        rigid_part.add_solver_ism(Cd=Cd, Cf=Cf, k_vis_inter=kinematic_viscosity_fluid, k_vis_inner=kinematic_viscosity_fluid)
elif solver == SOLVER_JL21:
    bound_part.add_solver_wcsph()
    bound_part.add_solver_JL21(kd=kd,Cf=Cf,k_vis=kinematic_viscosity_fluid)
    if(prm_rigidmodel):
        rigid_part.add_solver_wcsph()
        rigid_part.add_solver_JL21(kd=kd,Cf=Cf,k_vis=kinematic_viscosity_fluid)

''' INIT ALL SOLVERS '''
world.init_modules()

''' DATA PREPERATIONS '''
def prep_ism():
    world.neighb_search() # perform the neighbor search
    fluid_part.m_solver_ism.update_rest_density_and_mass()
    fluid_part.m_solver_ism.update_color() # update the color
    fluid_part.m_solver_ism.recover_phase_vel_from_mixture() # recover the phase velocity from the mixture velocity
    fluid_part.m_solver_sph.set_dead_particle()

def prep_JL21():
    world.neighb_search() # perform the neighbor search
    fluid_part.m_solver_JL21.update_rest_density_and_mass()
    fluid_part.m_solver_JL21.update_color() # update the color
    fluid_part.m_solver_JL21.recover_phase_vel_from_mixture() # recover the phase velocity from the mixture velocity

''' SIMULATION LOOPS '''
def loop_ism():
    ''' color '''
    fluid_part.m_solver_ism.update_color()
    #发射粒子
    global emit_counter
    global emit_num
    emit_counter+=1
    fluid_part.m_solver_sph.adjust_emitted_particle(emit_threshold)

    if emit_counter>=emit_threshold and emit_num<=emit_num_max:
        emit_indices_py = [i + bottom_positions.shape[0] * emit_num for i in range(bottom_positions.shape[0])]
        emit_indices=np.array(emit_indices_py)
        fluid_part.m_solver_sph.emit_particle_from_array(data_ti_container, emit_vel, emit_indices)
        emit_num+=1
        emit_counter=0
    
    ''' neighb search'''
    ''' [TIME START] neighb_search '''
    world.neighb_search()
    ''' [TIME END] neighb_search '''

    ''' sph pre-computation '''
    ''' [TIME START] DFSPH Part 1 '''
    world.step_sph_compute_compression_ratio()
    world.step_df_compute_beta()
    ''' [TIME START] DFSPH Part 1 '''

    ''' pressure accleration (divergence-free) '''
    ''' [TIME START] DFSPH Part 2 '''
    world.step_vfsph_div(update_vel=False)
    ''' [TIME END] DFSPH Part 2 '''
    print('div_free iter:', fluid_part.m_solver_df.div_free_iter[None])

    ''' [ISM] distribute pressure acc to phase acc and update phase vel '''
    '''  [TIME START] ISM Part 1 '''
    fluid_part.m_solver_df.get_acc_pressure()
    fluid_part.m_solver_ism.clear_phase_acc()
    fluid_part.m_solver_ism.ditribute_acc_pressure_2_phase()
    fluid_part.m_solver_ism.phase_acc_2_phase_vel()
    fluid_part.m_solver_ism.update_vel_from_phase_vel()
    '''  [TIME END] ISM Part 1 '''
    

    ''' viscosity & gravity (not requird in this scene)  accleration and update phase vel '''
    '''  [TIME START] ISM Part 2 '''
    fluid_part.m_solver_ism.clear_phase_acc()
    fluid_part.m_solver_ism.add_phase_acc_gravity()
    fluid_part.m_solver_ism.loop_neighb(fluid_part.m_neighb_search.neighb_pool, fluid_part, fluid_part.m_solver_ism.inloop_add_phase_acc_vis)
    fluid_part.m_solver_ism.phase_acc_2_phase_vel() 
    fluid_part.m_solver_ism.update_vel_from_phase_vel()
    '''  [TIME END] ISM Part 2 '''

    ''' pressure accleration (divergence-free) '''
    '''  [TIME START] DFSPH Part 3 '''
    world.step_vfsph_incomp(update_vel=False)
    '''  [TIME START] DFSPH Part 3 '''
    print('incomp iter:', fluid_part.m_solver_df.incompressible_iter[None])

    ''' distribute pressure acc to phase acc and update phase vel '''
    '''  [TIME START] ISM Part 3 '''
    fluid_part.m_solver_df.get_acc_pressure()
    fluid_part.m_solver_ism.clear_phase_acc()
    fluid_part.m_solver_ism.ditribute_acc_pressure_2_phase()
    fluid_part.m_solver_ism.phase_acc_2_phase_vel()
    fluid_part.m_solver_ism.update_vel_from_phase_vel()
    '''  [TIME END] ISM Part 3 '''

    ''' update particle position from velocity '''
    '''  [TIME START] DFSPH Part 4 '''
    world.update_pos_from_vel()
    '''  [TIME START] DFSPH Part 4 '''

    ''' phase change '''
    '''  [TIME START] ISM Part 4 '''
    fluid_part.m_solver_ism.update_val_frac()
    fluid_part.m_solver_ism.update_vel_from_phase_vel()

    ''' update mass and velocity '''
    fluid_part.m_solver_ism.regularize_val_frac()
    fluid_part.m_solver_ism.update_rest_density_and_mass()
    fluid_part.m_solver_ism.update_vel_from_phase_vel()
    '''  [TIME END] ISM Part 4 '''

    ''' cfl condition update'''
    '''  [TIME START] CFL '''
    # world.cfl_dt(0.4, max_time_step)
    '''  [TIME END] CFL '''

    ''' statistical info '''
    print(' ')
    fluid_part.m_solver_ism.statistics_linear_momentum_and_kinetic_energy()
    fluid_part.m_solver_ism.statistics_angular_momentum()
    fluid_part.m_solver_ism.debug_check_val_frac()

def loop_JL21():
    ''' update color based on the volume fraction '''
    fluid_part.m_solver_JL21.update_color()

    ''' neighb search'''
    ''' [TIME START] neighb_search '''
    world.neighb_search()
    ''' [TIME END] neighb_search '''

    ''' compute number density '''
    ''' [TIME START] WCSPH Part 1 '''
    world.step_sph_compute_number_density()
    ''' [TIME END] WCSPH Part 1 '''

    ''' gravity (not requird in this scene) accleration '''
    ''' [TIME START] WCSPH Part 2 '''
    world.clear_acc()
    world.add_acc_gravity()

    ''' viscosity force '''
    fluid_part.m_solver_JL21.clear_vis_force()
    fluid_part.m_solver_JL21.loop_neighb(fluid_part.m_neighb_search.neighb_pool, fluid_part, fluid_part.m_solver_JL21.inloop_add_force_vis)
    fluid_part.m_solver_JL21.loop_neighb(fluid_part.m_neighb_search.neighb_pool, bound_part, fluid_part.m_solver_JL21.inloop_add_force_vis)

    ''' pressure force '''
    fluid_part.m_solver_JL21.clear_pressure_force()
    world.step_wcsph_add_acc_number_density_pressure()
    fluid_part.m_solver_JL21.loop_neighb(fluid_part.m_neighb_search.neighb_pool, fluid_part, fluid_part.m_solver_JL21.inloop_add_force_pressure)
    fluid_part.m_solver_JL21.loop_neighb(fluid_part.m_neighb_search.neighb_pool, bound_part, fluid_part.m_solver_JL21.inloop_add_force_pressure)
    ''' [TIME END] WCSPH Part 2 '''

    ''' update phase vel (from all accelerations) '''
    if flag_strat_drift:
        ''' [TIME START] JL21 Part 1 '''
        fluid_part.m_solver_JL21.update_phase_vel()
        ''' [TIME END] JL21 Part 1 '''
    else:
        fluid_part.m_solver_JL21.vis_force_2_acc()
        fluid_part.m_solver_JL21.pressure_force_2_acc()
        fluid_part.m_solver_JL21.acc_2_vel()

    ''' update particle position from velocity '''
    ''' [TIME START] WCSPH Part 3 '''
    world.update_pos_from_vel()
    ''' [TIME END] WCSPH Part 3 '''

    ''' phase change (spacial care with lambda scheme) '''
    ''' [TIME START] JL21 Part 2 '''
    fluid_part.m_solver_JL21.update_val_frac_lamb()
    ''' [TIME END] JL21 Part 2 '''    

    ''' statistical info '''
    print(' ')
    fluid_part.m_solver_JL21.statistics_linear_momentum_and_kinetic_energy()
    fluid_part.m_solver_JL21.statistics_angular_momentum()
    fluid_part.m_solver_JL21.debug_check_val_frac()

    # world.cfl_dt(0.4, max_time_step) 

''' Viusalization and run '''
def novis_run(loop):
    global flag_strat_drift
    inv_fps = 1/fps
    timer = 0
    sim_time = 0
    loop_count = 0
    flag_write_img = False

    # gui = Gui3d()
    while True:#gui.window.running:

        #gui.monitor_listen()

        if True:#gui.op_system_run:
            loop()
            loop_count += 1
            sim_time += world.g_dt[None]
            
            if(sim_time > timer*inv_fps):#为满足FPS，定时输出，flag为真时即要输出
                # if gui.op_write_file:
                #     pass
                timer += 1
                flag_write_img = True
        if True:#gui.op_refresh_window:
            # gui.scene_setup()
            # gui.scene_add_parts_colorful(obj_pos=fluid_part.pos, obj_color=fluid_part.rgb,index_count=fluid_part.get_stack_top()[None],size=world.g_part_size[None])
            # gui.scene_add_parts(obj_pos=bound_part.pos, obj_color=(0,0.5,1),index_count=bound_part.get_stack_top()[None],size=world.g_part_size[None])
            # if(prm_rigidmodel):
            #     gui.scene_add_parts(obj_pos=rigid_part.pos, obj_color=(0,0.5,1),index_count=bound_part.get_stack_top()[None],size=world.g_part_size[None])
            # gui.canvas.scene(gui.scene)  # Render the scenet

            # if gui.op_save_img and flag_write_img:
            if prm_export and flag_write_img:
                # gui.window.save_image(prm_exportPath+str(timer)+'.png')
                flag_write_img = False
                write_ply(
                    path=prm_exportPath,
                    frame_num=timer,
                    type="fluid",
                    dim=3,
                    num=fluid_part_num,
                    pos=world.part_obj_list[0].pos.to_numpy(),
                    phase_num=3,
                    volume_frac=world.part_obj_list[0].phase.val_frac
                    # solid_beta=np.zeros_like(rigid_pos)
                    )
                
                # write_ply(path=r"D:\CODE\dataProcessing\\rigid_pos",
                #         frame_num=999,
                #         type="solid",
                #         dim=3,
                #         num=rigid1.num,
                #         pos=rigid_pos,
                #         phase_num=1,
                #         solid_beta=np.zeros_like(rigid_pos)
                #         )

            # print(world.part_obj_list[0].pos.shape)#fluid
            # print(world.part_obj_list[1].pos.shape)#rigid_pos

            # gui.window.show()
        
        if timer > output_frame_num:
            break

def vis_run(loop):
    global flag_strat_drift
    inv_fps = 1/fps
    timer = 0
    sim_time = 0
    loop_count = 0
    flag_write_img = False

    gui = Gui3d()
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
        if gui.op_refresh_window:
            gui.scene_setup()
            gui.scene_add_parts_colorful(obj_pos=fluid_part.pos, obj_color=fluid_part.rgb,index_count=fluid_part.get_stack_top()[None],size=world.g_part_size[None]/2)
            gui.scene_add_parts(obj_pos=bound_part.pos, obj_color=(0.86,0.86,0.86),index_count=bound_part.get_stack_top()[None],size=world.g_part_size[None]/12)
            if(prm_rigidmodel):
                gui.scene_add_parts(obj_pos=rigid_part.pos, obj_color=(0,0.5,1),index_count=bound_part.get_stack_top()[None],size=world.g_part_size[None])
            gui.canvas.scene(gui.scene)  # Render the scene

            # if gui.op_save_img and flag_write_img:
            if prm_export and  flag_write_img:
                gui.window.save_image(prm_exportPath+str(timer)+'.png')
                flag_write_img = False
                write_ply(
                    path=prm_exportPath,
                    frame_num=timer,
                    type="fluid",
                    dim=3,
                    num=fluid_part_num,
                    pos=world.part_obj_list[0].pos.to_numpy(),
                    phase_num=3,
                    volume_frac=world.part_obj_list[0].phase.val_frac
                    # solid_beta=np.zeros_like(rigid_pos)
                    )
                
                # write_ply(path=r"D:\CODE\dataProcessing\\rigid_pos",
                #         frame_num=999,
                #         type="solid",
                #         dim=3,
                #         num=rigid1.num,
                #         pos=rigid_pos,
                #         phase_num=1,
                #         solid_beta=np.zeros_like(rigid_pos)
                #         )

            # print(world.part_obj_list[0].pos.shape)#fluid
            # print(world.part_obj_list[1].pos.shape)#rigid_pos

            gui.window.show()
        
        if timer > output_frame_num:
            break

''' RUN THE SIMULATION '''
if solver == SOLVER_ISM:
    prep_ism()
    if(prm_vis):
        vis_run(loop_ism)
    else:
        novis_run(loop_ism)

elif solver == SOLVER_JL21:
    prep_JL21()
    if(prm_vis):
        vis_run(loop_JL21)
    else:
        novis_run(loop_JL21)








