#! python
# %%
import sys, os, argparse
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error as mae
import torch
import torch.nn as nn
import numpy as np
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import random_split

# %%
#sys.path.append('../src/training')

working_path = os.path.join(os.path.dirname(os.getcwd()), 'src/')
sys.path.append(working_path)

from modules import STLSTM
from datasets import BarkleyDataset, TestDataset

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Found', torch.cuda.device_count(), 'GPUs')


# %%
def getTestDataset(cfg):    
    root = f'../data/datasets/processed/regime{cfg.regime}'
    dataset = TestDataset(metadata='../metadata_own.yaml', regime = cfg.regime)

    return dataset


# %%
def getModel(cfg, **kwargs):   
    model = nn.DataParallel(STLSTM(1,64))
    
    suffix = kwargs.pop('suffix', "[0]")
    
    folder = f'../models/regime{cfg.regime}'
    model_file = f'{cfg.prefix}{suffix}'
    #model_file = f'STLSTM_t{cfg.time_step}_d{cfg.depth}'
    file = os.path.join(folder, model_file)
      
    model.load_state_dict(torch.load(file, map_location=device), strict=True)
    return model

# %%
def mae_b(img1, img2):
    return mae(img1.reshape((-1,)), img2.reshape((-1,)))


# %%
@torch.no_grad()
def getYTruePredPairs(model, dataset, cfg, **kwargs):
    dataloader = DataLoader(dataset, cfg.batch_size, drop_last=True)
    
    depths = kwargs.pop('depths', [0])
    suffixes = kwargs.pop('suffix', "[0]")
    time_step = kwargs.pop('time_step', 32)
                          
    
    y_trues, y_preds = [], []
    for i, (X,y) in tqdm(enumerate(dataloader), total=len(dataloader)):
        X = X[:,:time_step].to(device) # [bs,t,d,120,120]
        y = y[:,:,np.array(depths)].cpu().detach().numpy()
        
        y_pred = model(X, max_depth=len(depths)).cpu().detach().numpy()
        
        y_preds.append(y_pred)
        y_trues.append(y)
    y_preds = np.concatenate(y_preds, 0)
    y_trues = np.concatenate(y_trues, 0) 
        
    return y_trues, y_preds


# %%
def getLossPerDepth(y_trues, y_preds, config, criterion=mae_b, num_examples=None, **kwargs): 
    losses = []
    
    for i in range(y_trues.shape[2]):
        #print(f'Depth: {config.depth}')
        loss = criterion(y_trues[:,0,i], y_preds[:,0,i])
        losses.append(loss)
        
    return losses


# %%
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Training of Neural Networks, the Barkley Diver')
    parser.add_argument('-metadata', '--metadata', type=str, help='Main folder structure', default='../metadata.yaml')

    _args = parser.parse_args()
    args = vars(_args)

    try:
        with open(args['metadata']) as config_file:
            metadata_args = yaml.load(config_file, Loader=yaml.FullLoader)
            args.update(metadata_args)
    except FileNotFoundError:
        print('Metadata-file not found, use default values')
        assert('Metadata-file not found, use default values')   
          
    pprint.pprint(args)
    cfg = OmegaConf.create(args)

# %%
suffixes = ["20-30,2"]

depths_per_model = [np.array([0,8,16,24,31])]
                    
time_steps = [32 for _ in range(1)]

cfg = dict(model='STLSTM',
           dataset_dir='../data/datasets/chaotic/processed',
           #dataset_dir = '/data.bmp/heart/SimulationData/2020_3DExMedSurfaceToDepth/2021-04-18_STLSTM_paper/datasets/chaotic/processed/',
           train=False,
           regime='B',
           datatype='chaotic',
           prefix='STLSTM_t32_d',
           batch_size=16
)
cfg = OmegaConf.create(cfg)

# %%
dataset = getTestDataset(cfg)
#dataset, _ = random_split(dataset, [64, len(dataset)-64])

# %%
len(dataset)

# %%
losses_mae = []
with torch.no_grad():
    for suffix, depths, time_step in zip(suffixes, depths_per_model, time_steps):
        #cfg_temp = OmegaConf.merge(cfg, dict(suffix=suffix, depths=list(depths), time_step=time_step))
        model = getModel(cfg, suffix=suffix,).to(device)
        
        y_trues, y_preds = getYTruePredPairs(model, 
                                             dataset, 
                                             cfg,
                                             suffix=suffix, 
                                             depths=depths, 
                                             time_step=time_step)
            
        loss_mae = getLossPerDepth(y_trues, y_preds, cfg)
        losses_mae.append(loss_mae)
        
    np.save(f'mae_{suffix}_{cfg.dataset}.npy', losses_mae)

# %%
