#! python
# %%
# name
#$ -N multicondition

# execute from current directory
#$ -cwd

# Preserve environment variables
#$ -V

# Provide path to python executable
#$ -S /home/stenger/smaxxhome/anaconda3/envs/gpu/bin/python

# Merge error and out
#$ -j yes

# Path for output
#$ -o /home/stenger/smaxxhome/outputs

# Limit memory to <64G/16
#$ -hard -l h_vmem=7.0G

# serial queue
#$ -q grannus.q

# job array of length 1
#$ -t 300:301

# %%
import glob, re, os, json, time, yaml
from tqdm import tqdm

import numpy as np
import torch
import matplotlib.pyplot as plt
from omegaconf import OmegaConf


#dirname = os.path.dirname(os.path.realpath(__file__)) + '/'
#os.chdir(dirname)

# %%
def processData():
    max_depth = 32
    
    files = glob.glob(os.path.join(cfg.raw_folder, '*.npy'))
    
    a = np.arange(0, min(len(files), cfg.maxdata), cfg.size, dtype=np.int)
    
    X, y = [], []
    #print(a, len(files), min(len(files), maxdata))
    for i, (start, stop) in enumerate(zip(a,a+cfg.size)):
        if stop>len(files):
            stop=len(files)
        
        #print(start, stop)
    
        X, y = [],[]
        for file in tqdm(files[start:stop]):
            d = np.load(file)[:32]
            #print(d.shape)
            
            data = []
            data.append(np.swapaxes(d,1,1)[:,:max_depth,:,:])
            data.append(np.swapaxes(d,1,2)[:,:max_depth,:,:])
            data.append(np.swapaxes(d,1,3)[:,:max_depth,:,:])
                            
            data.append(np.flip(np.swapaxes(d,1,1), axis=1)[:,:max_depth,:,:])
            data.append(np.flip(np.swapaxes(d,1,2), axis=1)[:,:max_depth,:,:])
            data.append(np.flip(np.swapaxes(d,1,3), axis=1)[:,:max_depth,:,:])
                        
            for j in range(len(data)):
                k = np.random.randint(0,4)
                data[j] = np.rot90(data[j], k=k, axes=(2,3))
                
            _X = np.array(data)[:,:,:1]
            _y = np.array(data)[:,:1]
            
            #print(_X.shape, _y.shape)         
            
            X.append(_X)
            y.append(_y)
        
        X = np.concatenate(X, axis=0)
        X = np.array(X)
            
        y = np.concatenate(y, axis=0)
        y = np.array(y)
        
        print(X.shape, y.shape)
            
        savefileX = os.path.join(cfg.processed_folder, f'subsetX{i}')
        savefileY = os.path.join(cfg.processed_folder, f'subsetY{i}')

        np.save(savefileX, X)
        np.save(savefileY, y)
            
    return X, y

# %% Merging after processing
def mergeData():
    filesX = glob.glob(os.path.join(cfg.processed_folder, 'subsetX*'))
    filesY = glob.glob(os.path.join(cfg.processed_folder, 'subsetY*'))
    
    filesX.sort()
    filesY.sort()
    
    X = []
    for fileX in filesX:
        _X = np.load(fileX)

        X.append(_X)
        
    X = np.concatenate(X, axis=0)
    X = np.array(X)
    
    np.save(os.path.join(cfg.processed_folder, 'X'), X)

    
    y = []
    for fileY in filesY:
        _y = np.load(fileY)
        #print(_y.shape)
        y.append(_y)
        
    y = np.concatenate(y, axis=0)
    y = np.array(y)
    np.save(os.path.join(cfg.processed_folder, 'Y'), y)
    
    for fileX, fileY in zip(filesX, filesY):
        os.remove(fileX)
        os.remove(fileY)
    return X,y


# %%
if __name__=='__main__':      
    cfg = dict(metadata='../metadata_own.yaml',
              maxdata=513,
              size=64)
    
    with open(cfg['metadata']) as config_file:
        metadata_args = yaml.load(config_file, Loader=yaml.FullLoader)
        cfg.update(metadata_args)
        
    cfg = OmegaConf.create(cfg)
    
    cfg.raw_folder = cfg.data_folder_regimeB
    cfg.processed_folder = cfg.data_folder_raw_regimeB
            
    processData()
    mergeData()

# %%
cfg

# %%
