from plyfile import *
import numpy as np

def write_ply(path, frame_num, type,dim, num, pos, phase_num, volume_frac=None,solid_beta=None):
    if dim == 3:
        list_pos = []
        for i in range(num):
            pos_tmp = [pos[i, 0], pos[i, 1], pos[i, 2]]
            for j in range(phase_num):
                if type=="fluid":
                    pos_tmp.append(volume_frac[i, j])
                if type=="solid":
                    pos_tmp.append(solid_beta[i][j])#zxc extra attr
            list_pos.append(tuple(pos_tmp))
    elif dim == 2:
        list_pos = [(pos[i, 0], pos[i, 1], 0) for i in range(num)]
    else:
        print('write_ply(): dim exceeds default values')
        return
    data_type = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    for k in range(phase_num):
        if type=="fluid":
            data_type.append(('f'+str(k+1),'f4'))
        if type=="solid":
            data_type.append(('solid_beta'+str(k+1),'f4'))
    np_pos = np.array(list_pos, dtype=data_type)
    el_pos = PlyElement.describe(np_pos, 'vertex')
    PlyData([el_pos]).write(str(path) + '_' + str(frame_num) + '.ply')

