import taichi as ti
from ti_sph.basic_op import *
from ti_sph import *
import numpy as np

def part_template(part_obj, world, verbose=False):

    ''' Enssential arrays'''
    ''' encouraged to add for any particle system'''
    part_obj.add_array("pos", vecxf(part_obj.m_world.g_dim[None]).field())
    part_obj.add_array("vel", vecxf(part_obj.m_world.g_dim[None]).field())
    part_obj.add_array("vel_adv", vecxf(part_obj.m_world.g_dim[None]).field())
    part_obj.add_array("mass", ti.field(ti.f32))
    part_obj.add_array("size", ti.field(ti.f32))
    part_obj.add_array("volume", ti.field(ti.f32))
    part_obj.add_array("rest_density", ti.field(ti.f32))
    part_obj.add_array("pressure", ti.field(ti.f32))
    part_obj.add_array("k_vis", ti.field(ti.f32))
    part_obj.add_array("acc", vecxf(part_obj.m_world.g_dim[None]).field())
    part_obj.add_array("rgb", vecxf(3).field())
    part_obj.add_array("state", ti.field(ti.i32))#0 means dead,1 means alive,2 or more means emitting
    
    part_obj.add_attr("statistics_linear_momentum", vecx_f(part_obj.m_world.g_dim[None]))
    part_obj.add_attr("statistics_angular_momentum", vecx_f(3))
    part_obj.add_attr("statistics_kinetic_energy", val_f(0))

    ''' Optional arrays'''
    # part_obj.add_array("volume_fraction", ti.field(ti.f32), bundle=2)
    ## example only, not used in this test
    # fluid_part.add_array("volume_fraction", [ti.field(ti.f32), ti.field(ti.f32)])

    # fluid_phase = ti.types.struct(
    # val_frac=ti.f32,
    # phase_vel=vecxf(part_obj.m_world.g_dim[None]),
    # phase_acc=vecxf(part_obj.m_world.g_dim[None]),
    # phase_force=vecxf(part_obj.m_world.g_dim[None]),
    # )

    sph = ti.types.struct(
        h=ti.f32,
        sig=ti.f32,
        sig_inv_h=ti.f32,
        density=ti.f32,
        compression_ratio=ti.f32,
        pressure=ti.f32,
        pressure_force=vecxf(part_obj.m_world.g_dim[None]),
        viscosity_force=vecxf(part_obj.m_world.g_dim[None]),
        gravity_force=vecxf(part_obj.m_world.g_dim[None]),
    )

    sph_df = ti.types.struct(
        alpha_1=vecxf(part_obj.m_world.g_dim[None]),
        alpha_2=ti.f32,
        alpha=ti.f32,
        kappa_incomp=ti.f32,
        kappa_div=ti.f32,
        delta_density=ti.f32,
        delta_compression_ratio=ti.f32,
        vel_adv=vecxf(part_obj.m_world.g_dim[None]),
    )
    sph_wc = ti.types.struct(
        B=ti.f32,
    )    

    phase = ti.types.struct(
        val_frac=ti.f32,
        val_frac_in=ti.f32,
        val_frac_out=ti.f32,
        vel=vecxf(part_obj.m_world.g_dim[None]),
        drift_vel=vecxf(part_obj.m_world.g_dim[None]),
        acc=vecxf(part_obj.m_world.g_dim[None]),
    )
    mixture = ti.types.struct(
        lamb = ti.f32,
        flag_negative_val_frac = ti.i32,
        acc_pressure=vecxf(part_obj.m_world.g_dim[None]),
    )

    elastic_sph = ti.types.struct(
        force=vecxf(part_obj.m_world.g_dim[None]),
        pos_0=vecxf(part_obj.m_world.g_dim[None]),
        F=matxf(part_obj.m_world.g_dim[None]),
        L=matxf(part_obj.m_world.g_dim[None]),
        R=matxf(part_obj.m_world.g_dim[None]),
        eps=matxf(part_obj.m_world.g_dim[None]),
        P=matxf(part_obj.m_world.g_dim[None]),
        link_num=ti.i32,
        link_num_0=ti.i32,
        dissolve= ti.f32,
        U=matxf(part_obj.m_world.g_dim[None]),
        sigma=matxf(part_obj.m_world.g_dim[None]),
        V_T=matxf(part_obj.m_world.g_dim[None]),
        vel_0=vecxf(part_obj.m_world.g_dim[None]),
        flag=ti.i32,
        K=ti.f32,
        G=ti.f32,
    )

    porous_multi_fluid=ti.types.struct(
        outer_pore_pressure=vecxf(part_obj.m_world.g_phase_num[None]),#外孔压             
        Nf=ti.f32,
        SumNWf=ti.f32,
        delta_volume=ti.f32,
        alpha_last=ti.f32,
        solid_beta=vecxf(part_obj.m_world.g_phase_num[None]),#固体粒子的虚相（类似于饱和度）
        Sr=ti.f32,
        fluid_alpha=vecxf(part_obj.m_world.g_phase_num[None]),
        fluid_beta=vecxf(part_obj.m_world.g_phase_num[None]),
        solid_pore_pressure=ti.f32, 
        Darcy_flux=ti.types.matrix(part_obj.m_world.g_phase_num[None], part_obj.m_world.g_dim[None], ti.f32),# 相k在固体内外的体积分数      
        grad_Darcy_flux=vecxf(part_obj.m_world.g_phase_num[None]),
        volume_frac_temp=vecxf(part_obj.m_world.g_phase_num[None]),
        alpha_sum=ti.f32, 
        solid_strain=ti.f32,
        grad_pore_pressure=ti.types.matrix(part_obj.m_world.g_phase_num[None], part_obj.m_world.g_dim[None], ti.f32),# 相k在固体内外的体积分数      
    )

    part_obj.add_struct("sph", sph)
    part_obj.add_struct("sph_df", sph_df)
    part_obj.add_struct("sph_wc", sph_wc)
    if part_obj.m_type==2:# 是固体
        part_obj.add_struct("elastic_sph", elastic_sph)
    if hasattr(world, 'g_phase_num'):
        if part_obj.m_type==1:# 是流体
            part_obj.add_struct("phase", phase, bundle=world.g_phase_num[None])
            part_obj.add_struct("porous_multi_fluid", porous_multi_fluid)     
            part_obj.add_struct("mixture", mixture)

    if verbose:
        part_obj.verbose_attrs("fluid_part")
        part_obj.verbose_arrays("fluid_part")
        part_obj.verbose_structs("fluid_part")

    return part_obj
