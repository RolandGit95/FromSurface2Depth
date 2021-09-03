#! python
# %%
import sys, os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error as mae

#import numpy as np
#from omegaconf import OmegaConf
#import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
#from torch.utils.data import random_split

import torch
import torch.nn as nn
from torchvision.datasets import VisionDataset



# %%

working_path = os.path.join(os.path.dirname(os.getcwd()), '')
sys.path.append(working_path)

from src.modules import STLSTM
from src.datasets import TestDataset
from src.utils import getModel, getYTruePredPairs

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Found', torch.cuda.device_count(), 'GPUs')

# %%
fileX ='/home/roland/Projekte/FromSurface2Depth/data/processed/regimeA/X_test.npy'
fileY = '/home/roland/Projekte/FromSurface2Depth/data/processed/regimeA/Y_test.npy'
fileSTLSTM = '/home/roland/Projekte/FromSurface2Depth/models/regimeA/STLSTM_t32_d32'

    
# %%
dataset = TestDataset(fileX, fileY)

# %%
model_architecture = nn.DataParallel(STLSTM(1,64))
model = getModel(model_architecture, fileSTLSTM, device)

# %%
def mae_b(img1, img2):
    return mae(img1.reshape((-1,)), img2.reshape((-1,)))





# %%

if __name__=='__main__':
    y_trues, y_pred = getYTruePredPairs(model, 
                                        dataset, 
                                        device, 
                                        time_steps=np.arange(0,32,1), 
                                        depths=np.arange(0,32,1))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    