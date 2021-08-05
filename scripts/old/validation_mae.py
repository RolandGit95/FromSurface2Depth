# %%
import os,sys,inspect
from sklearn.metrics import mean_absolute_error as mae

import torch
import torch.nn as nn
from torch.utils.data import random_split

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from src.modules import STLSTM
from src.datasets import BarkleyDataset, TestDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Found', torch.cuda.device_count(), 'GPUs')


# %%
def getDataset(cfg):
    dataset = BarkleyDataset(root=cfg.dataset_dir,
                             train=cfg.train, 
                             depth=32, 
                             time_steps=cfg.time_step)

    n_train = int(len(dataset)*0.90+0.5)
    n_val = int(len(dataset)*0.10+0.5)

    if cfg.train:
        torch.manual_seed(42)
        dataset, _ = random_split(dataset, [n_train, n_val])
    else:
        torch.manual_seed(42)
        _, dataset = random_split(dataset, [n_train, n_val])
        dataset.train = False
        
    return dataset

def getTestDataset(cfg):    
    root = f'../data/datasets/{cfg.dataset}/processed'
    dataset = TestDataset(mode = cfg.dataset)

    return dataset

def getModel(cfg, **kwargs):   
    model = nn.DataParallel(STLSTM(1,64))
    
    suffix = kwargs.pop('suffix', "[0]")
    
    folder = f'../models/{cfg.datatype}'
    model_file = f'{cfg.prefix}{suffix}'
    #model_file = f'STLSTM_t{cfg.time_step}_d{cfg.depth}'
    file = os.path.join(folder, model_file)
      
    model.load_state_dict(torch.load(file, map_location=device), strict=True)
    return model


def mae_b(img1, img2):
    return mae(img1.reshape((-1,)), img2.reshape((-1,)))

