# %%
import sys, os
from tqdm import tqdm
import math 
import numpy as np
import torch
from torch.utils.data import DataLoader


import matplotlib.pyplot as plt

# %%

def getModel(model_architecture, model_file, device, strict=True):         
    model_architecture.load_state_dict(torch.load(model_file, map_location=device), strict=strict)
    return model_architecture


@torch.no_grad()
def getYTruePredPairs(model, dataset, device, time_steps=np.arange(0,32,1), depths=np.arange(0,32,1)):
    dataloader = DataLoader(dataset, batch_size=8, drop_last=True, shuffle=False)
    
    y_trues, y_preds = [], []
    for i, (X,y) in tqdm(enumerate(dataloader), total=len(dataloader)):
        X = X[:,time_steps].to(device) # [bs,t,d,120,120]
        y = y[:,:,depths].cpu().detach().numpy()
        
        y_pred = model(X, max_depth=len(depths)).cpu().detach().numpy()
        
        y_preds.append(y_pred)
        y_trues.append(y)
    y_preds = np.concatenate(y_preds, 0)
    y_trues = np.concatenate(y_trues, 0) 
        
    return y_trues, y_preds