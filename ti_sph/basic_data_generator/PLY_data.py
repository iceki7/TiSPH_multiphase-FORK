import numpy as np
import taichi as ti
from plyfile import PlyData

from .Data_generator import Data_generator

@ti.data_oriented
class PLY_data(Data_generator):
    def __init__(self, ply_filename: str,offset: ti.Vector = None):
        print("Loading particle data from PLY file...")

        # 从PLY文件加载粒子位置
        self.positions = self.load_ply_positions(ply_filename)

        # 输出数据
        self.pos = None
        self.num = None

        # 生成数据
        self.generate_data()
        self.translate(offset)

        print('Done!')

    def load_ply_positions(self, filename: str) -> np.ndarray:
        """从PLY文件加载粒子位置"""
        plydata = PlyData.read(filename)
        # 假设粒子位置存储在vertex元素的'x', 'y', 'z'属性中
        x = plydata['vertex']['x']
        y = plydata['vertex']['y']
        z = plydata['vertex']['z']
        positions = np.stack((x, y, z), axis=-1)
        return positions

    def generate_data(self):
        """生成数据"""
        self.pos = self.positions
        self.num = self.pos.shape[0]

    def translate(self, offset: ti.Vector):
        self.pos += offset.to_numpy()
        return self    

