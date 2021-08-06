# -*- coding: utf-8 -*-
# %%
# execute with qsub main.py -epochs 16 -batch_size 2 -config config/training_two_spirals.yaml -time_steps 1 2 4 8 16 20 25 28 30 32 -depth 32
# this loops multiple trainings with different time-steps. See in the config file for further specifications
# ! python

# name
#$ -N training

# execute from current directory
#$ -cwd

# Preserve environment variables
#$ -V

# Provide path to python executable, has to provide the packages from requirements.txt
#$ -S /home/stenger/smaxxhome/anaconda3/envs/gpu/bin/python

# Merge error and out
#$ -j yes

# serial queue
#$ -q taranis-gpu1.q

# Path for output
#$ -o /home/stenger/smaxxhome/outputs

# %%
import os, sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split

import matplotlib.pyplot as plt
import numpy as np

from torch.optim.lr_scheduler import ReduceLROnPlateau

from tqdm import tqdm
import yaml
import pprint
import re

import wandb

working_path = os.path.join(os.path.dirname(os.getcwd()), '')
sys.path.append(working_path)

from src.modules import STLSTM, Conv2D
from src.datasets import BarkleyDataset

import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Found', torch.cuda.device_count(), 'GPUs')

wandb.login()

os.environ["WANDB_MODE"] = "dryrun"

torch.manual_seed(42)


# %%
def make(config):  
    #model = nn.DataParallel(STLSTM(1,config['hidden_size'])).to(device)
    model = nn.DataParallel(Conv2D()).to(device)
        
    print(config['depths'])
    
    if config['dataset']=='regimeA':
        root = config['data_folder_regimeA']
    elif config['dataset']=='regimeB':
        root = config['data_folder_regimeB']
        
    #train_dataset = BarkleyDataset(root=root,
    #                               train=True, 
    #                               depths=config['depths'],
    #                               time_steps=config['time_steps'])
    
    #n_train = int(len(train_dataset)*0.90+0.5)
    #n_val = int(len(train_dataset)*0.10+0.5)
        
    #torch.manual_seed(42)
    #train_dataset, test_dataset = random_split(train_dataset, [n_train, n_val])  
    #test_dataset.train = False

    #train_loader = DataLoader(train_dataset, config['batch_size'], shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    #test_loader = DataLoader(test_dataset, 2, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    
    return model, criterion, optimizer

# %%
def getDataLoader(dataset_idx, config):
    print(f'Load dataset number {dataset_idx}')
    if config['dataset']=='regimeA':
        root = config['data_folder_regimeA']
    elif config['dataset']=='regimeB':
        root = config['data_folder_regimeB']
        
    #print('Hier', config['depths'], type(config['depths']), type(config['depths'][0]))
    train_dataset = BarkleyDataset(root=root,
                                   train=True, 
                                   depths=config['depths'],
                                   time_steps=config['time_steps'])
    
    n_train = int(len(train_dataset)*0.90+0.5)
    n_val = int(len(train_dataset)*0.10+0.5)
        
    torch.manual_seed(42)
    train_dataset, val_dataset = random_split(train_dataset, [n_train, n_val])  
    val_dataset.train = False

    train_loader = DataLoader(train_dataset, config['batch_size'], shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, 2, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
 
    return train_loader, val_loader


# %%
def train(model, criterion, optimizer, config, val_fn=nn.MSELoss()):
    #torch.save(model.state_dict(), 'model')
    def get_lr():
        for param_group in optimizer.param_groups:
            return param_group['lr']
    
    lrp = ReduceLROnPlateau(optimizer, patience=512, factor=0.3, min_lr=1e-7, verbose=True)
        
    #test_dataloader_iter = iter(test_dataloader)
    
    if config['dataset']=='regimeA':
        config['save_dir'] = config['save_folder_models_regimeA']
    elif config['dataset']=='regimeB':
        config['save_dir'] = config['save_folder_models_regimeB']
        
        
    min_val_loss = 10000      
    val_losses = []
    
    depths = config['depths']
        
    print(config['save_name'] )

    for epoch in range(config['epochs']): 
        print(f'Epoch number {epoch}')
        for dataset_idx in range(config['num_datasets']):
            train_loader, test_loader = getDataLoader(dataset_idx=dataset_idx, config=config)
            test_loader_iter = iter(test_loader)
            
            for i, (X,y) in tqdm(enumerate(train_loader), total=len(train_loader)):
                model.zero_grad()
                optimizer.zero_grad()

                X = X.to(device)
                y = y.to(device)

                outputs = model(X, max_depth=len(depths))

                loss = 0.0
                loss += criterion(y, outputs) # [depths,batch,features=1,:,:]

                outputs = outputs.detach()

                loss.backward()
                optimizer.step()      

                if i%10==0:
                    try:
                        X_val, y_val = next(test_loader_iter)
                    except StopIteration:
                        test_loader_iter = iter(test_loader)
                        X_val, y_val = next(test_loader_iter)
                    X_val = X_val.to(device)
                    y_val = y_val.to(device)

                    with torch.no_grad():
                        val_outputs = model(X_val, max_depth=len(depths))
                        val_loss = val_fn(y_val, val_outputs)
                        val_losses.append(val_loss.cpu().detach().numpy())
                    lrp.step(val_loss)

                    wandb.log({"loss": loss, "val_loss":val_loss})
            if val_loss < min_val_loss:
                min_val_loss = val_loss

                name = config['save_name'] + '_t' + str(config['time_steps_savename']) + '_d' + config['depths_savename']

                try:
                    os.makedirs(config['save_dir'])
                except FileExistsError:
                    pass

                savename = os.path.join(config['save_dir'], name)
                print('Save model under:', savename)
                torch.save(model.state_dict(), savename)    

# %%
def pipeline(config):   
    if config['dataset']=='regimeA':
        data_folder = config['data_folder_regimeA']
        log_folder = config['log_folder_regimeA']
        save_folder_models = config['save_folder_models_regimeA']
    elif config['dataset']=='regimeB':
        data_folder = config['data_folder_regimeB']
        log_folder = config['log_folder_regimeB']
        save_folder_models = config['save_folder_models_regimeB']
        
    #if config['wandb']:
        
    name = config['save_name'] + '_t' + str(config['time_steps_savename']) + '_d' + config['depths_savename']
        
    wandb.init(project=config['project_name'], 
                name=name, dir=log_folder, 
                config=config, 
                reinit=True)
        
    #config = wandb.config
        
    model, criterion, optimizer = make(config)
    #if config['wandb']:
    wandb.watch(model, criterion, log="all", log_freq=32)
    
    train(model, criterion, optimizer, config, val_fn=nn.L1Loss())

    #run.finish()
    return model

# %%
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Training of Neural Networks, the Barkley Diver')

    ### Names ###
    parser.add_argument('-project_name', '--project_name', type=str)
    parser.add_argument('-name', '--name', type=str, help='')
    
    ### Save and load dataset/weights etc. ###
    parser.add_argument('-dataset', '--dataset', type=str)
    parser.add_argument('-save_name', '--save_name', type=str)
    
    ### Model ###
    parser.add_argument('-architecture', '--architecture', type=str)
    parser.add_argument('-hidden_size', '--hidden_size', type=int)
    
    ### Training process ###
    parser.add_argument('-lr', '--lr', type=int)
    parser.add_argument('-batch_size', '--batch_size', type=int)
    
    ### Experiment specific quantities ###
    parser.add_argument('-epochs', '--epochs', type=int)
    parser.add_argument('-time_steps', '--time_steps', type=list, nargs='+', 
                        help='Time steps given as input, for example like: 0-31,2; or as a list: 0 1 2 3,')
    parser.add_argument('-depths', '--depths', type=list, nargs='+', 
                        help='Depths given as target for prediction, for example like: 0-31,2; or as a list: 0 1 2 3,')
    
    # Config files
    parser.add_argument('-config', '--config', type=str, help='Place of config file')
    parser.add_argument('-metadata', '--metadata', type=str, help='Main folder structure')
    parser.add_argument('-offline', '--offline', type=str, help='Use wandb-logger online?')
    parser.add_argument('-num_datasets', '--num_datasets', type=int, help='How many dataset-files should be used?...increases the training-size')

    _args = parser.parse_args()
    args = vars(_args)
    args_parser = {k: v for k, v in args.items() if v is not None}
    #args_parser = vars(parser.parse_args())
    #print(args_parser)

    metadata = args['metadata']
    
    if not isinstance(args['config'], type(None)):
        try:
            with open(args['config']) as config_file:
                config_args = yaml.load(config_file, Loader=yaml.FullLoader)
                args.update(config_args)
        except FileNotFoundError:
            print('Config-file not found, use default values')
            assert('Config-file not found, use default values')   
            
    if metadata is not None:
        args['metadata'] = metadata
        
    if not isinstance(args['metadata'], type(None)):
        #print(args['metadata'])
        try:
            with open(args['metadata']) as config_file:
                metadata_args = yaml.load(config_file, Loader=yaml.FullLoader)
                args.update(metadata_args)
        except FileNotFoundError:
            print('Metadata-file not found, use default values')
            assert('Metadata-file not found, use default values')   

    
    #pipeline(args)
    #print(args, '\n', args_parser)
    
    args.update(args_parser)
    
    if len(args['depths'])>1:
        depths = [int(''.join(depth)) for depth in args['depths']]

    elif len(args['depths'])==1:
        s = args['depths'][0]
        s = ''.join(s)
        ints = [int(r) for r in re.split(',|-', s)]
        #print('Länge ints', ints)
        if len(ints)==1:
            depths = ints
            #print(ints)
        else:
            depths = np.arange(*ints)
        
    args['depths_savename'] = s
    args['depths'] = depths
    
    
    
    if len(args['time_steps'])>1:
        time_steps = [int(''.join(time_step)) for time_step in args['time_steps']]

    elif len(args['time_steps'])==1:
        s = args['time_steps'][0]
        
        s = ''.join(s)
        ints = [int(r) for r in re.split(',|-', s)]
        if len(ints)==1:
            time_steps = ints
            #print(ints)
        else:
            time_steps = np.arange(*ints)
        
    args['time_steps_savename'] = s
    args['time_steps'] = time_steps
    
    pprint.pprint(args)

    m = pipeline(args)
        
        
    """
    args_config.update(specified_config)

    if int(args.offline)==True:
        print('No internet')
        os.environ['WANDB_MODE'] = 'dryrun'
        WANDB_MODE="dryrun"

    if not isinstance(args.depths, type(None)):
        print(args.depths)
        depths = args.depths
        print(depths)
        ds = [int(''.join(depth)) for depth in depths]

        args_config['depths'] = ds
        if not isinstance(args.time_steps, type(None)):
            for t in args.time_steps:
                t_int = int(''.join(t))
                args_config['time_step'] = t_int

                for key, value in args_config.items():
                    print(key + ':', value)

                m = pipeline(args_config)
        else:
            for key, value in args_config.items():
                print(key + ':', value)

                m = pipeline(args_config)
                
    elif not isinstance(args.time_steps, type(None)):
        for t in args.time_steps:
            t_int = int(''.join(t))
            args_config['time_step'] = t_int
            for key, value in args_config.items():
                print(key + ':', value)
            m = pipeline(args_config)
    else:
        for key, value in args_config.items():
            print(key + ':', value)
        m = pipeline(args_config)
        
    """
