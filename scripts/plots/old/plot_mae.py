# %%
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib
from omegaconf import OmegaConf

# %%
lambda_c = 16.2
T_c = 74.93

dataset = 'regimeB'
folder = f'/home/roland/Projekte/FromSurface2Depth/log/{dataset}/validation'

name = f'mae_globalmodels.npy'
file = os.path.join(folder, name)
losses = np.load(file)

average = f'mae_average.npy'
avg_file = os.path.join(folder, average)
avg_reg = np.load(avg_file)

subnetworks = f'mae_subnetworks.npy'
subnetworks_file = os.path.join(folder, subnetworks)
subnetworks_losses = np.load(subnetworks_file, allow_pickle=True)
depths_per_model = [np.arange(0,32,2).astype(np.int8), 
                    np.array([0,3,6,9,12,15,18,21,25,28,31]),
                    np.array([0,4,8,12,16,20,24,28,31]),
                    np.array([0,5,10,15,20,25,30]),
                    np.array([0,6,12,18,24,30]),
                    np.array([0,8,16,14,31])]

layermodel_losses = np.load('../log/regimeB/validation/mae_layermodels.npy')

# %%
### Global optimized with different T, comparison###

# %matplotlib inline
ts = [1,4,12,20,28,32] #[16,8,4,1]

plt.style.use('default')

fig = plt.figure()
ax1 = fig.add_subplot(111)
plt.subplots_adjust(left=0.10, right=0.98, top=0.83, bottom=0.12)

ax2 = ax1.twiny()

inds = [[1,2,4,8,12,16,20,25,28,30,32].index(i) for i in ts]

for i,t in enumerate(inds):
    ax1.plot(losses[t], label=f'{ts[i]}, {np.round(4*ts[i]/T_c,2)}$T_c$')

ax1.set_xlabel("Depth in $\Delta$s")
plt.title("Error per layer till depth of 32$\cdot\Delta$s; regime A; $ST$.")

ticks = np.arange(0,34,2)
ax1.set_xticks(ticks)

avg, = ax1.plot(avg_reg[0])

ax2.set_xlim(ax1.get_xlim())
tick_locations = np.arange(0,34,6)
ax2.set_xticks(tick_locations)
tick_function = lambda x: np.round(x/lambda_c,2)
ax2.set_xticklabels(tick_function(tick_locations))
ax2.set_xlabel(r"Depth in $\lambda_c$")

leg = plt.legend([avg], ["Average regressor"], loc=4)
plt.gca().add_artist(leg)
ax1.legend(title='Time steps $T$', loc=0)
#plt.savefig(f'STLSTM_layer_global.pdf', dpi=600)

import tikzplotlib
tikzplotlib.save(f'STLSTM_layer_global.tex')

# %%
### Comparison, layer-optimizer vs. global optimized ###

# %matplotlib notebook
T = 32

plt.style.use('default')

fig = plt.figure()
ax1 = fig.add_subplot(111)
plt.subplots_adjust(left=0.10, right=0.98, top=0.83, bottom=0.12)

ax2 = ax1.twiny()

#losses = mae_STLSTM_periodic.copy()

ax1.plot(losses[-1], label=f'{32}, {np.round(4*32/T_c,2)}$T_c$')
ax1.scatter(range(0,32),losses[-1], c='blue', s=12)

ax1.set_xlabel("Depth in $\Delta$s")

ax1.scatter(range(0,32),layermodel_losses, label=f'{32}, {np.round(4*32/T_c,2)}$T_c$', c='red', s=12)

plt.title("Error per layer till depth of 32$\cdot\Delta$s; chaotic; $ST$.")

ticks = np.arange(0,34,2)
ax1.set_xticks(ticks)

#avg, = ax1.plot((0,31),(0.3137254901960784, 0.3137254901960784))
avg, = ax1.plot(avg_reg[0])

ax2.set_xlim(ax1.get_xlim())
tick_locations = np.arange(0,34,6)
ax2.set_xticks(tick_locations)
tick_function = lambda x: np.round(x/lambda_c,2)
ax2.set_xticklabels(tick_function(tick_locations))
ax2.set_xlabel(r"Depth in $\lambda_c$")

leg = plt.legend([avg], ["Average regressor"], loc=6)
plt.gca().add_artist(leg)
ax1.legend(title='Time steps $T$', loc=0)
plt.savefig(f'STLSTM_global_layer.pdf', dpi=600)

# %%
### Comparison, layer-optimizer vs. global optimized ###

# %matplotlib auto
T = 32

plt.style.use('default')

fig = plt.figure()
ax1 = fig.add_subplot(111)
plt.subplots_adjust(left=0.10, right=0.98, top=0.83, bottom=0.12)

ax2 = ax1.twiny()

#losses = mae_STLSTM_periodic.copy()

ax1.plot(losses[-1], label=f'{32}, {np.round(4*32/T_c,2)}$T_c$')
ax1.scatter(range(0,32),losses[-1], c='blue', s=12)

ax1.set_xlabel("Depth in $\Delta$s")

#ax1.plot(depths_per_model[0], subnetworks_losses[0], label=0)
#ax1.scatter(depths_per_model[0], subnetworks_losses[0], s=12, c='orange')

i = 2

ax1.plot(depths_per_model[i], subnetworks_losses[i])
ax1.scatter(depths_per_model[i], subnetworks_losses[i], s=12, c='orange')

i = -2
ax1.plot(depths_per_model[i], subnetworks_losses[i])
ax1.scatter(depths_per_model[i], subnetworks_losses[i], s=24, c='green')

ax1.scatter(range(0,32),layermodel_losses, c='red', s=12)


#ax1.scatter(range(0,32),layermodel_losses, label=f'{32}, {np.round(4*32/T_c,2)}$T_c$', c='red', s=12)

plt.title("Error per layer till depth of 32$\cdot\Delta$s; chaotic; $ST$. Time steps: 32")

ticks = np.arange(0,34,2)
ax1.set_xticks(ticks)

#avg, = ax1.plot((0,31),(0.3137254901960784, 0.3137254901960784))
avg, = ax1.plot(avg_reg[0])

ax2.set_xlim(ax1.get_xlim())
tick_locations = np.arange(0,34,6)
ax2.set_xticks(tick_locations)
tick_function = lambda x: np.round(x/lambda_c,2)
ax2.set_xticklabels(tick_function(tick_locations))
ax2.set_xlabel(r"Depth in $\lambda_c$")

leg = plt.legend([avg], ["Average regressor"], loc=6)
plt.gca().add_artist(leg)
#ax1.legend(title='Time steps $T$', loc=0)
plt.savefig(f'STLSTM_global_layer_sub.pdf', dpi=600)

# %%
