# %%
import os, sys
import numpy as np
import torch
import torch.nn as nn
import string

import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
import matplotlib as mpl

#import tikzplotlib

working_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), '')
sys.path.append(working_path)

from src.modules import STLSTM
from src.datasets import TestDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Found', torch.cuda.device_count(), 'GPUs')

# %%
# Das hier sind die files die man auch im data.bmp-Ordner findet, (habe Sie auch lokal bei mir abgespeichert)
fileX ='/home/roland/Projekte/FromSurface2Depth/data/processed/regimeA/X_test.npy'
fileY = '/home/roland/Projekte/FromSurface2Depth/data/processed/regimeA/Y_test.npy'

# Model-file für Vorhersagen
filesSTLSTM = ['/home/roland/Projekte/FromSurface2Depth/models/regimeA/STLSTM_t1_d32',
               '/home/roland/Projekte/FromSurface2Depth/models/regimeA/STLSTM_t8_d32',
               #'/home/roland/Projekte/FromSurface2Depth/models/regimeA/STLSTM_t20_d32',
               '/home/roland/Projekte/FromSurface2Depth/models/regimeA/STLSTM_t32_d32']
           

# %%
# Das hier sind die files die man auch im data.bmp-Ordner findet, (habe Sie auch lokal bei mir abgespeichert)
"""
fileX ='/home/roland/Projekte/FromSurface2Depth/data/processed/regimeB/X_test.npy'
fileY = '/home/roland/Projekte/FromSurface2Depth/data/processed/regimeB/Y_test.npy'

# Model-file für Vorhersagen
filesSTLSTM = ['/home/roland/Projekte/FromSurface2Depth/models/regimeB/STLSTM_t1_d0-32,1',
               '/home/roland/Projekte/FromSurface2Depth/models/regimeB/STLSTM_t8_d0-32,1',
               '/home/roland/Projekte/FromSurface2Depth/models/regimeB/STLSTM_t32_d0-32,1']
   
"""
Ts = [1,8,32]
# Welche Tiefe soll vorhergesagt werden
depth = 31
num_examples = 3

# %%
################ Teil für ST-LSTM ################
dataset = TestDataset(fileX, fileY)

# %%
models = [nn.DataParallel(STLSTM(1,64)) for _ in range(len(filesSTLSTM))]
[models[i].load_state_dict(torch.load(filesSTLSTM[i], map_location=device), strict=True) for i in range(len(models))]

# %%

numbering = False


# %%
### Gridplot ###
N = 6
data = dataset[N]
depths = np.array([0,3,6,9,12,15,18,21,24,27,30])

#with torch.no_grad():
#    y_pred = model(data[0].unsqueeze(0), max_depth=max(depths)+1).cpu().detach().numpy()[0,0, depths]
    
y_plot = data[1][0,depths].cpu().detach().numpy()
#X_plot = data[0].cpu().detach().numpy()

num_rows = len(models) + 1
num_cols = len(depths)

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
font = {'size':28}
mpl.rc('font', **font)

mpl.rcParams.update({'font.size': 28})

fig, axs = plt.subplots(num_rows, num_cols, sharex=True, sharey=True, figsize=(num_cols*2-3,num_rows*2))
plt.subplots_adjust(left=0.022, right=0.985, top=0.95, bottom=0.155)


for col in range(num_cols):
    axs[0,col].imshow(y_plot[col])
    
for row in range(num_rows-1):
    
#    for i in range(len(models)):
    with torch.no_grad():
        y_pred = models[row](data[0][:Ts[row]].unsqueeze(0),
                           max_depth=max(depths)+1).cpu().detach().numpy()[0,0, depths]
        print(y_pred.shape)
        
        axs[row+1,0].set_ylabel(f'$T=${Ts[row]}')


    for col in range(num_cols):
        im = axs[row+1, col].imshow(y_pred[col], vmin=0, vmax=1)

 
#            axs[1,col].imshow(y_pred[col])

    for row in range(num_rows):
        axs[row, col].set_xticks([])
        axs[row, col].set_yticks([])


for i, d in enumerate(depths):
    axs[0,i].set_title(f'Depth: {d}', fontsize=28)
            
axs[0,0].set_ylabel('Ground truth')


#plt.draw()
p0 = axs[-1,0].get_position().get_points().flatten()
p1 = axs[-1,-1].get_position().get_points().flatten()
ax_cbar = fig.add_axes([p0[0], 0.07, p1[2]-p0[0], 0.05])
plt.colorbar(im, cax=ax_cbar, orientation='horizontal')

#plt.show()


plt.savefig('regimeA_T_comparison.pdf')







