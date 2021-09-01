# +
import numpy as np
from pyevtk.hl import gridToVTK
import torch
import torch.nn as nn

from omegaconf import OmegaConf
import os, sys, inspect, yaml

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

import src.modules as modules

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Found', torch.cuda.device_count(), 'GPUs')
# -

cfg = dict(model_file='/home/roland/Projekte/FromSurface2Depth/models/regimeA/STLSTM_t32_d32',
           source_folder="../data/raw/regimeA",
           name = '656492_27189',
           vtk_folder = "../data/visualization/regimeA",
           depth=32,
           time_steps=32)
cfg = OmegaConf.create(cfg)


def getModel(model_file):   
    model = nn.DataParallel(modules.STLSTM(1,64))
    model.load_state_dict(torch.load(model_file, map_location=device), strict=True)
    return model



model = getModel('/home/roland/Projekte/FromSurface2Depth/models/regimeA/STLSTM_t32_d32')

# +
#depth = 32
#time_steps = 32

#source_folder = "../data/raw/regimeA"
#name = '188040_843316'

#vtk_folder = "../data/visualization/regimeA"

transform = lambda data:(data.float()+127)/255.
data = np.load(os.path.join(cfg.source_folder, cfg.name + '.npy'))
data = transform(torch.tensor(data))
X = data[:cfg.time_steps,:1].unsqueeze(0)
Y = data[:1,:cfg.depth,:,:].detach().cpu().numpy()
# -

with torch.no_grad():
    Y_pred = model(X, max_depth=cfg.depth)[0].detach().cpu().numpy()

# +
save_directory = os.path.join(cfg.vtk_folder, cfg.name)

if not os.path.exists(save_directory):
    os.makedirs(save_directory)   

# +
shape = Y.shape
x,y,z = [np.arange(0,shape[i]+1,1).astype(np.int16) for i in range(1,4)]

for i, d in enumerate(Y):
    _savename = os.path.join(save_directory, f'{i:04d}' + '_true.vtk')#   f'{save_directory}{i:04d}' + '.vtk'
    gridToVTK(_savename, x, y, z, cellData = {'u': d})

# +
shape = Y_pred.shape
x,y,z = [np.arange(0,shape[i]+1,1).astype(np.int16) for i in range(1,4)]

for i, d in enumerate(Y_pred):
    _savename = os.path.join(save_directory, f'{i:04d}' + '_pred.vtk')#   f'{save_directory}{i:04d}' + '.vtk'
    gridToVTK(_savename, x, y, z, cellData = {'u': d})
# -


