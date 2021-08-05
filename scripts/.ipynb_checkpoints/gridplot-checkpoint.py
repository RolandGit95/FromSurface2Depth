# %%
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#from matplotlib.patches import ConnectionPatch
from omegaconf import OmegaConf
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader

working_path = os.path.join(os.path.dirname(os.getcwd()), 'src/')
sys.path.append(working_path)
#os.chdir(working_path)

from datasets import BarkleyDataset, TestDataset
from modules import STLSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Found', torch.cuda.device_count(), 'GPUs')


# %%
def getModel(cfg, suffix):   
    model = nn.DataParallel(STLSTM(1,64))
    
    prefix = 'STLSTM_'
    model_file = prefix + suffix
    folder = f'../models/{cfg.dataset}'
    #model_file = f'STLSTM_t{cfg.time_step}_d{cfg.depth}'
    file = os.path.join(folder, model_file)
    print(file)
      
    model.load_state_dict(torch.load(file, map_location=device), strict=True)
    return model

def getModels(cfg, suffixes):
    models = []
    for suffix in suffixes:
        model = nn.DataParallel(STLSTM(1,64))

        folder = f'../models/{cfg.dataset}'
        folder = '/home/roland/Projekte/FromSurface2Depth/models/regimeB'
        model_file = f'STLSTM_{suffix}'
        file = os.path.join(folder, model_file)
        print(file)

        model.load_state_dict(torch.load(file, map_location=device), strict=True)
        
        models.append(model)
    return models    
    
@torch.no_grad()
def plotGridOld(models, dataset, N, cfg):
    plt.rcParams.update({'font.size': 18})
    plt.subplots_adjust(left=0.03, bottom=0.1, right=1-0.03, top=1, wspace=None, hspace=0.1)

    fig, axs = plt.subplots(len(cfg.Ts)+1, cfg.max_depth,
                            sharex=True, sharey=True,
                            figsize=(cfg.max_depth*2,(len(cfg.Ts)+1)*2))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    plt.xticks([]), plt.yticks([])

    cmap = 'viridis'
    #axs[0,0].set_title(f'Depth: {0}')
    axs[0,0].set_ylabel(f'True')
    
    y_true = dataset[N][1][0].cpu().detach().numpy()
    for i in range(cfg.max_depth):
        #axs[0,i].set_title(f'Depth: {i}')
        _y_true = y_true[i]
        axs[0,i].imshow(_y_true, cmap=cmap, vmin=0, vmax=1)

    for j, model in enumerate(models):
        with torch.no_grad():
            y_pred = model(dataset[N][0].unsqueeze(0), max_depth=cfg.max_depth).cpu().detach().numpy()
            print(y_pred.shape)
        axs[j+1,0].set_ylabel(f"$T$={cfg.Ts[j]}")
    
        for i in range(cfg.max_depth):         
            _y_pred = y_pred[0,0,i,:,:]
            axs[j+1,i].imshow(_y_pred, cmap=cmap, vmin=0, vmax=1)
            
    transFigure = fig.transFigure.inverted()
    coord1 = transFigure.transform(axs[1,0].transData.transform([60,60]))
    coord2 = transFigure.transform(axs[1,1].transData.transform([60,60]))

    
    line = matplotlib.lines.Line2D((coord1[0],coord2[0]),(coord1[1],coord2[1]),
                               transform=fig.transFigure, lw=2)
    fig.lines = line,
        
    plt.savefig(f'multiplot_p{N}_{cfg.dataset}', dpi=600)
    
@torch.no_grad()
def plotGridDiff(model, dataset, N, cfg):
    plt.rcParams.update({'font.size': 18})
    plt.subplots_adjust(left=0.03, bottom=0.1, right=1-0.03, top=1, wspace=None, hspace=0.1)

    fig, axs = plt.subplots(3, cfg.max_depth,
                            sharex=True, sharey=True,
                            figsize=(cfg.max_depth*2,6))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    plt.xticks([]), plt.yticks([])

    cmap = 'viridis'
    #axs[0,0].set_title(f'Depth: {0}')
    axs[0,0].set_ylabel(f'True')
    
    y_true = dataset[N][1][0].cpu().detach().numpy()
    for i in range(cfg.max_depth):
        #axs[0,i].set_title(f'Depth: {i}')
        _y_true = y_true[i]
        axs[0,i].imshow(_y_true, cmap=cmap, vmin=0, vmax=1)

    with torch.no_grad():
        y_pred = model(dataset[N][0].unsqueeze(0), max_depth=cfg.max_depth).cpu().detach().numpy()
        print(y_pred.shape)
    axs[1,0].set_ylabel(f"$T$={cfg.time_step}")
    
    for i in range(cfg.max_depth):
        _y_pred = y_pred[0,0,i,:,:]
        axs[1,i].imshow(_y_pred, cmap=cmap, vmin=0, vmax=1)
        
    for i in range(cfg.max_depth):
        _y_pred = y_pred[0,0,i,:,:]
        _y_true = y_true[i]
        axs[2,i].imshow(_y_pred-_y_true, cmap='RdBu', vmin=-1, vmax=1)

    plt.savefig(f'multiplot_p{N}_{cfg.dataset}', dpi=200)
    plt.show()


# %%
@torch.no_grad()
def plotGrid(models, depths_per_model, time_steps, dataset, N, cfg):
    plt.rcParams.update({'font.size': 18})
    plt.subplots_adjust(left=0.03, bottom=0.1, right=1-0.03, top=1, wspace=0, hspace=0.1)

    fig, axs = plt.subplots(len(models)+1, cfg.max_depth,
                            sharex=True, sharey=True,
                            figsize=(cfg.max_depth*2,(len(models)+1)*2))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)


    plt.xticks([]), plt.yticks([])

    transform = lambda data:(data.float().cpu().detach().numpy()+127)/255.
    
    cmap = 'viridis'
    #axs[0,0].set_title(f'Depth: {0}')
    axs[0,0].set_ylabel(f'Ground truth')
    
    y_true = transform(dataset[N][1][0])#.cpu().detach().numpy())
    
    
    for i in range(cfg.max_depth):
        #axs[0,i].set_title(f'Depth: {i}')
        _y_true = y_true[i]
        axs[0,i].imshow(_y_true, cmap=cmap, vmin=0, vmax=1)
            
          
    
    for j, (model, depths, time_step) in enumerate(zip(models, depths_per_model, time_steps)):
        with torch.no_grad():
            
            X = (dataset[N][0][:time_step].unsqueeze(0).float()+127)/255.
            
            y_pred = model(X, max_depth=len(depths)).cpu().detach().numpy()
            print(y_pred.shape)
        #axs[j+1,0].set_ylabel(f"$T$={time_steps}")
    
    
        for i in range(cfg.max_depth):
            #axs[j+1,i].imshow(np.zeros((120,120)), cmap=cmap, vmin=0, vmax=1)
            axs[j+1, i].axis('off')

        for i, depth in enumerate(depths): # [0,5,10,...]
            if depth<cfg.max_depth:
                _y_pred = y_pred[0,0,i,:,:]
                axs[j+1,depth].imshow(_y_pred, cmap=cmap, vmin=0, vmax=1)
                
    
    
    
    plt.savefig(f'../media/multiplot_p{N}.pdf', dpi=200, bbox_inches = 'tight', pad_inches = 0)

# %%
suffixes = ["[ 0  2  4  6  8 10 12 14 15 16 18 20 22 24 26 28 30]",
            #"[ 0  3  6  9 12 15 18 21 25 28 31]",
            "[ 0  4  8 12 16 20 24 28 31]",
            #"[ 0  5 10 15 20 25 30]",
            #"[ 0  6 12 18 24 30]",
            "[ 0  8 16 24 31]",
            "[0 2 4 6 8]",
            "[ 4  6  8 10 12]",
            "[ 8 10 12 14 16]",
            "[12 14 16 18 20]"]

depths_per_model = [np.arange(0,32,2).astype(np.int8), 
                    #np.array([0,3,6,9,12,15,18,21,25,28,31]),
                    np.array([0,4,8,12,16,20,24,28,31]),
                    #np.array([0,5,10,15,20,25,30]),
                    #np.array([0,6,12,18,24,30]),
                    np.array([0,8,16,24,31]),
                    np.array([0,2,4,6,8]),
                    np.array([4,6,8,10,12]),
                    np.array([8,10,12,14,16]),
                    np.array([12,14,16,18,20])]

time_steps = [32 for _ in range(len(depths_per_model))]

cfg = dict(dataset='regimeB', # 'chaotic', 'two_spirals'
           time_step=32,
           depth=32,
           max_depth=32,
           train=False,
           Ts=[32])

cfg = OmegaConf.create(cfg)

# %%
suffixes = [f"t{sfx}_d32" for sfx in ["32"]]

depths_per_model = [np.arange(0,32,1).astype(np.int8),
                    np.arange(0,32,1).astype(np.int8),
                    np.arange(0,32,1).astype(np.int8),
                    np.arange(0,32,1).astype(np.int8),
                    np.arange(0,32,1).astype(np.int8)]

time_steps = [32 for _ in range(len(depths_per_model))]

#dataset = 'regimeB'
cfg = dict(regime='B', # 'chaotic', 'two_spirals'
           #root=f'/home/roland/Projekte/FromSurface2Depth/data/processed/{dataset}',
           time_step=32,
           depth=32,
           max_depth=32,
           train=False,
           Ts=[32])

cfg = OmegaConf.create(cfg)

# %%
#dataset = getDataset(cfg)
dataset = TestDataset(metadata='../metadata_own.yaml', regime = 'B')

# %%
y_true = dataset[30][1][0].cpu().detach().numpy()
plt.imshow(y_true[0])

# %%
for i in range(20,21):
    plotGrid(getModels(cfg, suffixes=suffixes), depths_per_model, time_steps, dataset, i, cfg)


# %%
@torch.no_grad()
def plotGrid(models, dataset, N, cfg):
    plt.rcParams.update({'font.size': 18})
    plt.subplots_adjust(left=0.03, bottom=0.1, right=1-0.03, top=1, wspace=None, hspace=0.1)

    fig, axs = plt.subplots(len(cfg.Ts)+1, cfg.max_depth,
                            sharex=True, sharey=True,
                            figsize=(cfg.max_depth*2,(len(cfg.Ts)+1)*2))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    plt.xticks([]), plt.yticks([])

    cmap = 'viridis'
    #axs[0,0].set_title(f'Depth: {0}')
    axs[0,0].set_ylabel(f'True')
    
    y_true = dataset[N][1][0].cpu().detach().numpy()
    for i in range(cfg.max_depth):
        #axs[0,i].set_title(f'Depth: {i}')
        _y_true = y_true[i]
        axs[0,i].imshow(_y_true, cmap=cmap, vmin=0, vmax=1)

    for j, model in enumerate(models):
        with torch.no_grad():
            y_pred = model(dataset[N][0].unsqueeze(0), max_depth=cfg.max_depth).cpu().detach().numpy()
            print(y_pred.shape)
        axs[j+1,0].set_ylabel(f"$T$={cfg.Ts[j]}")
    
        for i in range(cfg.max_depth):         
            _y_pred = y_pred[0,0,i,:,:]
            axs[j+1,i].imshow(_y_pred, cmap=cmap, vmin=0, vmax=1)
            
    transFigure = fig.transFigure.inverted()
    coord1 = transFigure.transform(axs[1,0].transData.transform([60,60]))
    coord2 = transFigure.transform(axs[1,1].transData.transform([60,60]))

    
    line = matplotlib.lines.Line2D((coord1[0],coord2[0]),(coord1[1],coord2[1]),
                               transform=fig.transFigure, lw=2)
    fig.lines = line,
        
    plt.savefig(f'multiplot_p{N}_{cfg.dataset}', dpi=600)

# %%
cfg = dict(dataset='chaotic', # 'chaotic', 'two_spirals'
           root='/home/roland/Projekte/FromSurface2Depth/data/datasets/chaotic/processed',
           time_step=32,
           depth=32,
           max_depth=16,
           train=False,
           Ts=[32])

cfg = OmegaConf.create(cfg)

# %%
plotGrid(getModels(cfg), dataset, 0, cfg)

# %%
plotGridDiff(getModel(cfg), dataset, 1, cfg)

# %%
for i in range(2,34):
    plotGridDiff(getModel(cfg), dataset, i, cfg)

# %%
