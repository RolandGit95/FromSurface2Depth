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
import sys, os
import numpy as np
#import matplotlib.pyplot as plt

import yaml
import argparse
import pprint

sys.path.append('../simulation/')
#from BarkleySimulation import BarkleySimluation3D

#from omegaconf import OmegaConf
from scipy import ndimage


# %%
class BarkleySimluation3D:
    def __init__(self, a=0.6, b=0.01, epsilon=0.02, deltaT=0.01, deltaX=0.1, D=0.02, alpha=1, boundary_mode='noflux'):
        self.a = a
        self.b = b
        self.epsilon = epsilon
        self.boundary_mode = boundary_mode # Neumann
        self.deltaT = deltaT
        self.deltaX = deltaX
        self.alpha = alpha
        self.h = D/self.deltaX**2
                    
    def set_boundaries(self, oldFields):
        if self.boundary_mode == "noflux":
            for (field, oldField) in zip((self.u, self.v), oldFields):
                field[:,:,0] = oldField[:,:,1]
                field[:,:,-1] = oldField[:,:,-2]
                
                field[:,0,:] = oldField[:,1,:]
                field[:,-1,:] = oldField[:,-2,:]
                
                field[0,:,:] = oldField[1,:,:]
                field[-1,:,:] = oldField[-2,:,:]
        
    def explicit_step(self):
        uOld = self.u.copy()
        vOld = self.v.copy()

        f = 1/self.epsilon * self.u * (1 - self.u) * (self.u - (self.v+self.b)/self.a)
        
        laplace = -6*self.u.copy()

        laplace += np.roll(self.u, +1, axis=0)
        laplace += np.roll(self.u, -1, axis=0)
        laplace += np.roll(self.u, +1, axis=1)
        laplace += np.roll(self.u, -1, axis=1)
        laplace += np.roll(self.u, +1, axis=2)
        laplace += np.roll(self.u, -1, axis=2)

        self.u = self.u + self.deltaT * (f + self.h * laplace)
        self.v = self.v + self.deltaT * (np.power(uOld, self.alpha) - self.v)

        self.set_boundaries((uOld, vOld))


# %%
def barkleySimulation(u0, v0, args, savename=''):
    s=BarkleySimluation3D(a=args['a'], b=args['b'], epsilon=args['epsilon'],deltaT=args['dt'], deltaX=args['ds'], D=args['D'], alpha=args['alpha'])

    s.u = u0
    s.v = v0
    
    U = []

    transform = lambda data: (np.array(data)*255-128).astype(np.int8)

    init_phase = np.random.randint(3000,3500)
    
    for i in range(100000):
        s.explicit_step()

        #if i%64==0:
        #    print(i)
        #    plt.imshow(s.u[:,:,0], vmin=0, vmax=1)
        #    plt.show()

        if i>=init_phase:
            if i==init_phase: print('Start recording')
            if i%args['dSave']==0:
                U.append(s.u)
                if len(U)>=args['max_save_length']:                    
                    _savename = f'{savename}'
     
                    print(_savename)

                    np.save(_savename, transform(U)) 
                    del U
                    return

# %%
def get_starting_condition_chaotic(args, seed=42):
    def initialize_random(n_boxes=(20,20,20), size=(120,120,120), seed=None):
        np.random.seed(seed)
        tmp = np.random.rand(*n_boxes)
        
        rpt = size[0]//n_boxes[0], size[1]//n_boxes[1], size[2]//n_boxes[2]
        
        tmp = np.repeat(tmp, np.ones(n_boxes[0], dtype=int)*rpt[0], axis=0)
        tmp = np.repeat(tmp, np.ones(n_boxes[1], dtype=int)*rpt[1], axis=1)
        tmp = np.repeat(tmp, np.ones(n_boxes[2], dtype=int)*rpt[2], axis=2)
        
        U = tmp
        
        V = U.copy()
        V[V<0.4] = 0.0
        V[V>0.4] = 1.0
        
        return U, V
    
    U, V = initialize_random(size=args['size'], seed=seed)
    return U, V
        
    
        
def get_starting_condition_two_spirals(args, seeds=[0,1]):
    def getRotatedSpiral(size=(60,60,60), seed=None):
        np.random.seed(seed)

        rot_size = np.array(size)*5
        u, v = np.zeros((2, *rot_size))

        spiralPosX = np.random.randint(size[0]*0.25, size[0]*0.75)
        spiralPosY = np.random.randint(size[1]*0.25, size[1]*0.75)

        x, y = spiralPosX+int(2*size[0]), spiralPosY+int(2*size[1])

        u[x:,:,:] = 1.
        v[:,:y,:] = 0.5

        p1,p2,p3 = np.random.randint(-90,90,3)
        u = ndimage.rotate(u, p1, axes=(0,1), reshape=0)
        u = ndimage.rotate(u, p2, axes=(0,2), reshape=0)
        u = ndimage.rotate(u, p3, axes=(1,2), reshape=0)

        v = ndimage.rotate(v, p1, axes=(0,1), reshape=0)
        v = ndimage.rotate(v, p2, axes=(0,2), reshape=0)
        v = ndimage.rotate(v, p3, axes=(1,2), reshape=0)

        s = u.shape
        u = u[s[0]//2-size[0]//2:s[0]//2+size[0]//2,s[1]//2-size[1]//2:s[1]//2+size[1]//2,s[2]//2-size[2]//2:s[2]//2+size[2]//2]
        v = v[s[0]//2-size[0]//2:s[0]//2+size[0]//2,s[1]//2-size[1]//2:s[1]//2+size[1]//2,s[2]//2-size[2]//2:s[2]//2+size[2]//2]

        u[u>0.5]=1
        u[u<=0.5]=0
        v[v>0.4]=0.5
        v[v<=0.4]=0.0

        return u, v
    
    size = (np.array(args['size'])/2).astype(int)
    
    _u, _v = getRotatedSpiral(size=size, seed=seeds[0])
    _u2, _v2 = getRotatedSpiral(size=size, seed=seeds[1])
    
    U = np.concatenate([_u, np.zeros(size)], axis=2)
    U = np.concatenate([U, np.zeros(size*[1,1,2])], axis=1)
    U2 = np.concatenate([np.zeros(size), _u2], axis=2)
    U2 = np.concatenate([np.zeros(size*[1,1,2]), U2], axis=1)
    U = np.concatenate([U, U2], axis=0)
    
    V = np.concatenate([_v, np.zeros(size)], axis=2)
    V = np.concatenate([V, np.zeros(size*[1,1,2])], axis=1)
    V2 = np.concatenate([np.zeros(size), _v2], axis=2)
    V2 = np.concatenate([np.zeros(size*[1,1,2]), V2], axis=1)
    V = np.concatenate([V, V2], axis=0)
    
    return U, V

# %%
def simulation(args, seeds=(0,1), sim_num=0):    
    if args['starting_condition']=='two_spirals':
        U, V = get_starting_condition_two_spirals(args, seeds=seeds)
    elif args['starting_condition']=='chaotic':
        U, V = get_starting_condition_chaotic(args)
    
    if args['dataset']=='regimeA':
        save_folder = args['data_folder_raw_regimeA']
    elif args['dataset']=='regimeB':
        save_folder = args['data_folder_raw_regimeB']
    savename = f'{seeds[0]}_{seeds[1]}'
    savename = os.path.join(save_folder, savename)
    
    U = barkleySimulation(U, V, args, savename=savename)
    
    return U

# %%
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Training of Neural Networks, the Barkley Diver')
    parser.add_argument('-metadata', '--metadata', type=str, help='Main folder structure', default='../metadata.yaml')
    parser.add_argument('-config', '--config', type=str, default='../config/simulation/regimeA.yaml')
    
    _args = parser.parse_args()
    args = vars(_args)
    
    if not isinstance(args['config'], type(None)):
        try:
            with open(args['config']) as config_file:
                config_args = yaml.load(config_file, Loader=yaml.FullLoader)
                args.update(config_args)
        except FileNotFoundError:
            print('Config-file not found, use default values')
            assert('Config-file not found, use default values')  
    else:
        print('No config file given')
        assert('No config file given')
        

    try:
        with open(args['metadata']) as config_file:
            metadata_args = yaml.load(config_file, Loader=yaml.FullLoader)
            args.update(metadata_args)
    except FileNotFoundError:
        print('Metadata-file not found, use default values')
        assert('Metadata-file not found, use default values')   
            
    
    #cfg = OmegaConf.load('../config/simulation/regimeB.yaml')
    
    pprint.pprint(args)
    seeds = np.random.randint(0,1000000, size=(args['num_sims'],2))
    
    for i, (s1, s2) in enumerate(seeds):
        U = simulation(args, seeds=(s1,s2), sim_num=i)


# %%
