# %%
import numpy as np
import torch


import matplotlib.pyplot as plt

import matplotlib as mpl

import tikzplotlib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Found', torch.cuda.device_count(), 'GPUs')

# %%
fileLongtermA ='/home/roland/Projekte/FromSurface2Depth/data/visualization/regimeA/longterm/000000_000000.npy'
fileLongtermB ='/home/roland/Projekte/FromSurface2Depth/data/visualization/regimeB/longterm/000000_000000.npy'

# %%
data = np.load(fileLongtermA)
voxel_timelineA = (data[:,80,100,0].astype(np.float32)+127)/255.

data = np.load(fileLongtermB)
voxel_timelineB = (data[:,80,100,0].astype(np.float32)+127)/255.

# %%

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4,0.8), sharey=True)

plt.subplots_adjust(left=0.0, right=1.0, top=0.95, bottom=0.01)

ax1.plot(voxel_timelineA)
ax1.set_ylabel(r'$u$')
ax1.set_xlabel('Time step in $\Delta t$')
ax1.set_title('Regime A')

plt.sca(ax1)
plt.xticks(np.linspace(0,512,5), np.linspace(0,512*16,5).astype(int))

ax2.plot(voxel_timelineB)
ax2.set_xlabel('Time step in $\Delta t$')
ax2.set_title('Regime B')

plt.sca(ax2)
plt.xticks(np.linspace(0,512,5), np.linspace(0,512*16,5).astype(int))

tikzplotlib.save('longterms_voxel.tex', axis_height='5cm', axis_width='9cm')