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
fileX ='/home/roland/Projekte/FromSurface2Depth/data/processed/regimeB/X_test.npy'
fileY = '/home/roland/Projekte/FromSurface2Depth/data/processed/regimeB/Y_test.npy'

# Model-file für Vorhersagen
fileSTLSTM = '/home/roland/Projekte/FromSurface2Depth/scripts/plots/STLSTM_t32_d0-32,1'

# Welche Tiefe soll vorhergesagt werden
depth = 31
num_examples = 3

# %%
################ Teil für ST-LSTM ################
dataset = TestDataset(fileX, fileY)

# %%
model = nn.DataParallel(STLSTM(1,64))
model.load_state_dict(torch.load(fileSTLSTM, map_location=device), strict=True)

# %%

data = dataset[:num_examples]

with torch.no_grad():
    y_pred = model(data[0], max_depth=depth).cpu().detach().numpy()[:,0,-1]
y_plot = data[1][:, 0,depth].cpu().detach().numpy()
X_plot = data[0][:,-1,0].cpu().detach().numpy()

# %%

# plot:
#            | input | true | pred1 | diff1 | pred2 | diff2 |
# example 1: |  xx   |  xx  |  xx   |  xx   |  xx   |  xx   |
# example 2: |  xx   |  xx  |  xx   |  xx   |  xx   |  xx   |
# example 3: |  xx   |  xx  |  xx   |  xx   |  xx   |  xx   |

num_rows = num_examples
num_cols = 6


mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
font = {'size':16}
mpl.rc('font', **font)


# %%

fig, axs = plt.subplots(num_rows, num_cols, sharex=True, sharey=True, figsize=(num_cols*2,num_rows*2))
plt.subplots_adjust(left=0.0, right=1.0, top=0.95, bottom=0.01)


letters = list(string.ascii_uppercase)[:num_cols*num_rows]

for row in range(num_rows):
    im1 = axs[row,0].imshow(X_plot[row])
    axs[row,1].imshow(y_plot[row])
    axs[row,2].imshow(y_pred[row])
    im2 = axs[row,3].imshow(y_plot[row]-y_pred[row], cmap='seismic', vmin=-1, vmax=1)
    
    
    ######## Hier fängt dein Teil an ########
    axs[row,4].imshow(y_pred[row]) # Sebastians prediction, shape y_pred: [num_examples, 120,120]
    axs[row,5].imshow(y_plot[row]-y_pred[row], cmap='seismic', vmin=-1, vmax=1)
    
    for col in range(num_cols):
        axs[row, col].axis('off')
        
        idx = num_cols*row + col
        letter = letters[idx]
        
        #axs[row,col].text(0.03, 0.97, letter, transform=axs[row,col].transAxes, fontsize=16, fontweight='bold', 
        #                  va='top', ha='left')
        
        rect = mpatch.Rectangle((0.0, 0.0), 23, 23, linewidth=2, edgecolor='black', facecolor='white')
        axs[row,col].add_artist(rect)
        rx, ry = rect.get_xy()
        cx = rx + rect.get_width()/2.0
        cy = ry + rect.get_height()/2.0
        axs[row,col].annotate(letter, (cx, cy), color='black', weight='bold', 
                              ha='center', va='center', **font)
        
        
        #axs[row,col].add_patch(rect)
        
            
axs[0,0].set_title('Input')
axs[0,1].set_title('Ground truth')
axs[0,2].set_title('ST-LSTM')
axs[0,3].set_title('Diff. ST-LSTM')
axs[0,4].set_title('CRF')
axs[0,5].set_title('Diff. CRF')

#plt.savefig('comparison_grid.pdf')

#tikzplotlib.save('comparison_grid.tex')









# %%
### Anderer Gridplot ###
N = 10
data = dataset[10]
depths = np.array([0,6,12,18,24,31])

with torch.no_grad():
    y_pred = model(data[0].unsqueeze(0), max_depth=max(depths)+1).cpu().detach().numpy()[0,0, depths]
    
y_plot = data[1][0,depths].cpu().detach().numpy()
#X_plot = data[0].cpu().detach().numpy()

# %%
num_rows = 3
num_cols = len(depths)

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
font = {'size':16}
mpl.rc('font', **font)


fig, axs = plt.subplots(num_rows, num_cols, sharex=True, sharey=True, figsize=(num_cols*2+1,num_rows*2))
plt.subplots_adjust(left=0.0175, right=1.0, top=0.95, bottom=0.003)


letters = list(string.ascii_uppercase)[:num_cols*num_rows]

for col in range(num_cols):
    axs[0,col].imshow(y_plot[col])
    axs[1,col].imshow(y_pred[col])
    axs[2,col].imshow(y_plot[col]-y_pred[col], cmap='seismic', vmin=-1, vmax=1)
    
    for row in range(num_rows):
        #axs[row, col].axis('off')
        axs[row, col].set_xticks([])
        axs[row, col].set_yticks([])

        idx = num_cols*row + col
        letter = letters[idx]
        
        #axs[row,col].text(0.03, 0.97, letter, transform=axs[row,col].transAxes, fontsize=16, fontweight='bold', 
        #                  va='top', ha='left')
        
        rect = mpatch.Rectangle((0.0, 0.0), 23, 23, linewidth=2, edgecolor='black', facecolor='white')
        axs[row,col].add_artist(rect)
        rx, ry = rect.get_xy()
        cx = rx + rect.get_width()/2.0
        cy = ry + rect.get_height()/2.0
        axs[row,col].annotate(letter, (cx, cy), color='black', weight='bold', 
                              ha='center', va='center', **font)


for i, d in enumerate(depths):
    axs[0,i].set_title(f'Depth: {d}')
            
axs[0,0].set_ylabel('Input')
axs[1,0].set_ylabel('Prediction')
axs[2,0].set_ylabel('Difference')



plt.savefig('time_series_pred.pdf')

#axs[0,1].set_title('Ground truth')
#axs[0,2].set_title('ST-LSTM')
#axs[0,3].set_title('Diff. ST-LSTM')
#axs[0,4].set_title('CRF')
#axs[0,5].set_title('Diff. CRF')
