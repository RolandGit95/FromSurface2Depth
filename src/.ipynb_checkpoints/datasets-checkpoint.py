# %%
import os, yaml, glob

import torch
from torchvision.datasets import VisionDataset

import numpy as np

import matplotlib.pyplot as plt

# %%
class BarkleyDataset(VisionDataset):
    # filenames of the input (X) and target data (y)
    dataX = 'X*.npy'
    dataY = 'Y*.npy'
    
    def __init__(self, root: str, train: bool = True, depths=[0,1,2], time_steps=[0,1,2], max_length=-1, dataset_idx=0) -> None:
        """
        Parameters
        ----------
        root : str
            folder in which the data is stored. In this file there have to be 
            the files X.npy and Y.npy for training and validation.
        train : bool, optional
            dataset for training or testing? If training,
            the input and target get rotated randomly in __getitem__
            to increase the diversity of the data for training. In the 
            validation mode this rotation is not applied.
            The default is True.
        depths : int, optional
            Layers in depth, which the network should predict. 
            The default is [0,1,2].
        time_steps : int, optional
            Number of time steps the input should have.
            The default is 32.
        max_length: int, optional
            Maximal number of example in the dataset, if -1 the original size will be used.
            The default is -1.
        num_datasets: int, optional
            Number of sub-datasets should be used for training, because of the size of the files they are splitted,
            while one contains 2048*6 examples.

        Returns
        -------
        None.

        """
        super(BarkleyDataset, self).__init__(root)
        
        self.train = train  # training set or test set
        #self.depth = depth
        
        self.depths = np.array(depths)
        max_depth = max(depths)
        
        print(self.depths)
        
        self.time_steps = np.array(time_steps)
        #max_time_steps = max(time_steps)

        #self.time_steps = time_steps
        
        self.root = root
        
        # This transformation function will be applied on the data if it is called in __getitem__
        self.transform = lambda data:(data.float()+127)/255.
        self.target_transform = lambda data:(data.float()+127)/255.
        
        #if not self._check_exists():
        #    raise RuntimeError('Dataset not found.')
            
        xfiles = glob.glob(os.path.join(self.root, self.dataX))
        yfiles = glob.glob(os.path.join(self.root, self.dataY))
        
        xfiles.sort(), yfiles.sort()
        #print(xfiles, yfiles)
        
        xfile = xfiles[dataset_idx]
        yfile = yfiles[dataset_idx]
        
        self.X = torch.tensor(np.load(xfile)[:max_length])[:,self.time_steps]
        self.y = torch.tensor(np.load(yfile)[:max_length])[:,:,self.depths]

    def __getitem__(self, idx: int):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (time-series at the surface, dynamic at time=0 till depth=self.depth)
                shapes: ([N,T,1,120,120], [N,1,D,120,120]), T and D are choosen in __init__, 
                The value for N depends if it is the training- or validaton-set.
        """
        
        # transform data of type int8 to float32 only at execution time to save memory
        X, y = self.transform(self.X[idx]), self.target_transform(self.y[idx])
        
        # Training data augmentation (random rotation of 0,90,180 or 270 degree)
        if self.train:
            k = np.random.randint(0,4)
            X = torch.rot90(X, k=k, dims=[2,3])
            y = torch.rot90(y, k=k, dims=[2,3])

        return X, y

    def __len__(self):
        return len(self.X)
    
    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        if self.root is not None:
            body.append("Root location: {}".format(self.root))
        body.append("Max. depth: {}".format(max(self.depths)))
        body.append("Number of time-steps: {}".format(self.time_steps))
        body += self.extra_repr().splitlines()
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)


    def setMode(self, train=True):
        """
        Parameters
        ----------
        train : bool, optional
            dataset for training or testing? If train=True,
            the input and target get rotated randomly in __getitem__
            to increase the diversity of the data for training. In the 
            validation mode (train=False) this rotation is not applied.
            The default is True.
        """
        self.train = train
        
    @property
    def folder(self):
        return self.root

    def _check_exists(self):       
        #print(os.path.join(self.root, self.dataX))
        return (os.path.exists(os.path.join(self.folder, self.dataX)) and
                os.path.exists(os.path.join(self.folder, self.dataY)))

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")       


# %%
class TestDataset(VisionDataset):
    
    def __init__(self, metadata='../metadata.yaml', regime='B'):
        super(TestDataset, self).__init__(root='') 
        
        self.transform = lambda data:(data.float()+127)/255.
        
        try:
            with open(metadata) as config_file:
                args = yaml.load(config_file, Loader=yaml.FullLoader)
        except FileNotFoundError:
            print('Metadata-file not found, use default values')
            assert('Metadata-file not found, use default values')   
            
        if regime=='A':
            dataX = os.path.join(args['data_folder_regimeA'], 'X_test.npy')
            dataY = os.path.join(args['data_folder_regimeA'], 'Y_test.npy')

            self.X = np.load(dataX)
            self.Y = np.load(dataY)
        elif regime=='B':
            dataX = os.path.join(args['data_folder_regimeB'], 'X_test.npy')
            dataY = os.path.join(args['data_folder_regimeB'], 'Y_test.npy')

            self.X = np.load(dataX)
            self.Y = np.load(dataY)
            
        self.X = torch.from_numpy(self.X)
        self.Y = torch.from_numpy(self.Y)

    def __getitem__(self, idx):
        return self.transform(self.X[idx]), self.transform(self.Y[idx])

    def __len__(self):
        return len(self.X)
    
    #def _check_exists(self):       
    #    return (os.path.exists(os.path.join(self.root, self.dataX)) and
    #            os.path.exists(os.path.join(self.root, self.dataY)))        
        
    

# %%
if __name__=='__main__':
    #pass
    dataset = BarkleyDataset('/home/roland/Projekte/FromSurface2Depth/data/processed/regimeB', dataset_idx=3)
