import os
import numpy as np
from pyevtk.hl import gridToVTK

x,y,z = [np.arange(0,121,1).astype(np.int16) for _ in range(3)]

# +
source_folder = "../data/raw/regimeA/"
vtk_folder = "../data/visualization/regimeA"
name = '112747_986016'

data = np.load(os.path.join(source_folder, name + '.npy'))

# +
save_directory = os.path.join(vtk_folder, name)

if not os.path.exists(save_directory):
    os.makedirs(save_directory)   
    
for i, d in enumerate(data):
    
    _savename = os.path.join(save_directory, f'{i:04d}' + '.vtk')#   f'{save_directory}{i:04d}' + '.vtk'
    gridToVTK(_savename, x, y, z, cellData = {'u': d})