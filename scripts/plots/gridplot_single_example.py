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
#fileX ='/home/roland/Projekte/FromSurface2Depth/data/processed/regimeB/X_test.npy'
#fileY = '/home/roland/Projekte/FromSurface2Depth/data/processed/regimeB/Y_test.npy'

# Model-file für Vorhersagen
#fileSTLSTM = '/home/roland/Projekte/FromSurface2Depth/scripts/plots/STLSTM_t32_d0-32,1'


# Das hier sind die files die man auch im data.bmp-Ordner findet, (habe Sie auch lokal bei mir abgespeichert)
fileX ='/home/roland/Projekte/FromSurface2Depth/data/processed/regimeA/X_test.npy'
fileY = '/home/roland/Projekte/FromSurface2Depth/data/processed/regimeA/Y_test.npy'

# Model-file für Vorhersagen
fileSTLSTM = '/home/roland/Projekte/FromSurface2Depth/models/regimeA/STLSTM_t32_d32'


# Welche Tiefe soll vorhergesagt werden
depth = 31
num_examples = 3

# %%
################ Teil für ST-LSTM ################
dataset = TestDataset(fileX, fileY)

# %%
model = nn.DataParallel(STLSTM(1,64))
model.load_state_dict(torch.load(fileSTLSTM, map_location=device), strict=True)


numbering = False


# %%
### Gridplot ###
N = 14
data = dataset[N]
depths = np.array([0,3,6,9,12,15,18,21,24,27,30])

with torch.no_grad():
    y_pred = model(data[0].unsqueeze(0), max_depth=max(depths)+1).cpu().detach().numpy()[0,0, depths]
    
y_plot = data[1][0,depths].cpu().detach().numpy()
#X_plot = data[0].cpu().detach().numpy()

# %%
num_rows = 3
num_cols = len(depths)

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
font = {'size':26}
mpl.rc('font', **font)

mpl.rcParams.update({'font.size': 26})

fig, axs = plt.subplots(num_rows, num_cols, sharex=True, sharey=True, figsize=(num_cols*2+1,num_rows*2))
plt.subplots_adjust(left=0.0175, right=1.0, top=0.95, bottom=0.003)


if numbering:
    letters = list(string.ascii_uppercase)[:num_cols*num_rows]

for col in range(num_cols):
    axs[0,col].imshow(y_plot[col])
    axs[1,col].imshow(y_pred[col])
    axs[2,col].imshow(y_plot[col]-y_pred[col], cmap='seismic', vmin=-1, vmax=1)
    
    for row in range(num_rows):
        axs[row, col].set_xticks([])
        axs[row, col].set_yticks([])

        if numbering:
            idx = num_cols*row + col
            letter = letters[idx]
            rect = mpatch.Rectangle((0.0, 0.0), 23, 23, linewidth=2, edgecolor='black', facecolor='white')
            axs[row,col].add_artist(rect)
            rx, ry = rect.get_xy()
            cx = rx + rect.get_width()/2.0
            cy = ry + rect.get_height()/2.0
            axs[row,col].annotate(letter, (cx, cy), color='black', weight='bold', 
                                  ha='center', va='center', **font)


for i, d in enumerate(depths):
    axs[0,i].set_title(f'Depth: {d}', fontsize=26)
            
axs[0,0].set_ylabel('Ground truth')
axs[1,0].set_ylabel('Prediction')
axs[2,0].set_ylabel('Difference')



plt.savefig('regimeA_single_example.pdf')
