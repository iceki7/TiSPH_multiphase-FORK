import taichi as ti
import numpy as np

from .Data_generator import Data_generator

@ti.data_oriented
class Plane_data(Data_generator):
    def __init__(self, lb: ti.Vector, rt: ti.Vector, span: ti.f32, layers: ti.i32):
        self.lb = lb
        self.rt = rt
        self.span = span
        self.layers = layers
        self.dim = ti.static(lb.n)

        self.num = self.calculate_num_particles()
        pos_data = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.num)
        self.generate_pos(pos_data)
        self.pos = pos_data.to_numpy()


    def calculate_num_particles(self) -> int:
        """ 计算底部平面上的粒子数量 """
        span = self.span
        pushed_node_seq = np.ceil((self.rt - self.lb) / span).astype(int)
        for i in range(self.dim):
            if pushed_node_seq[i] == 0:
                pushed_node_seq[i] = 1  # 至少要有一个粒子
        num_particles = pushed_node_seq[0] * pushed_node_seq[2] * self.layers  # 计算底部层数
        return num_particles

    @ti.kernel
    def generate_pos(self, _data: ti.template()):
        span = self.span
        current_node_num = 0
        pushed_node_seq_coder = ti.Vector([0, 0, 0])

        pushed_node_seq = int(ti.ceil((self.rt - self.lb) / span))
        pushed_node_seq_offset = int(ti.ceil((self.rt - self.lb) / span)) + (self.layers * 2)

        for i in ti.static(range(self.dim)):
            if pushed_node_seq[i] == 0:
                pushed_node_seq[i] = 1  # 至少要有一个粒子

        tmp = 1
        for i in ti.static(range(self.dim)):
            pushed_node_seq_coder[i] = tmp
            tmp *= pushed_node_seq_offset[i]

        inc = ti.Vector([current_node_num])
        for i in range(pushed_node_seq[0]):
            for j in range(self.layers):  # 生成底部多层
                for k in range(pushed_node_seq[2]):  # 这里是z坐标
                    index = ti.atomic_add(inc[0], 1)
                    _data[index][0] = (i - self.layers) * span + self.lb[0]
                    _data[index][1] = (j - self.layers) * span + self.lb[1]  # 保持y坐标为底部
                    _data[index][2] = (k - self.layers) * span + self.lb[2]