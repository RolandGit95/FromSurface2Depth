# %%
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision.datasets import VisionDataset
import string

import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
from matplotlib import rc
import matplotlib as mpl

import tikzplotlib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Found', torch.cuda.device_count(), 'GPUs')
# %%

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=None, name='Convolutional Encoder'):
        super(Encoder, self).__init__()
        
        self.id = 'encoder'
        
        if output_size==None:
            output_size=input_size
            
        self.encoder = nn.Sequential(
            nn.Conv2d(input_size,hidden_size, 3, padding=(1,1)),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_size),
            
            nn.Conv2d(hidden_size,hidden_size, 3, padding=(1,1), stride=(1,1)),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_size),
                                
            nn.Conv2d(hidden_size,output_size, 3, padding=(1,1), stride=(2,2)),
            nn.Sigmoid()
        )
        
    def forward(self, input):
        return self.encoder(input) 
    
class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=None, name='Convolutional Decoder'):
        super(Decoder, self).__init__()
        
        self.id = 'decoder'
        
        if output_size==None:
            output_size=input_size       
        
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(input_size,hidden_size, 3, padding=(1,1)),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_size),
            
            nn.Conv2d(hidden_size,hidden_size, 3, padding=(1,1)),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_size),
            
            nn.Conv2d(hidden_size,output_size, 3, padding=(1,1)),
            nn.Sigmoid()        
        )
        
    def forward(self, input):
        return self.decoder(input) 
 
class TimeDistributed(nn.Module):
    def __init__(self, module, name='Time Distributed'):
        super(TimeDistributed, self).__init__()
        self.module = module
        
    def forward(self, input):
        batch_or_time1, batch_or_time2 = input.size(0), input.size(1)
        
        new_shape = list([batch_or_time1*batch_or_time2]) + list(input.shape[2:])
        input = self.module(input.reshape(new_shape))
        
        output_shape = list([batch_or_time1, batch_or_time2]) + list(input.shape[1:])
        return input.reshape(output_shape)  
    
class STLSTM_cell(nn.Module):
    def __init__(self, input_size, hidden_size, name='STLSTM-Cell'):
        super(STLSTM_cell, self).__init__() 
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.Wx = nn.Conv2d(input_size, hidden_size*7, (3,3), stride=1, padding=1)
        self.Wh = nn.Conv2d(hidden_size, hidden_size*4, (3,3), stride=1, padding=1)
        self.Wm = nn.Conv2d(hidden_size, hidden_size*3, (3,3), stride=1, padding=1)
        
        self.Wco = nn.Conv2d(hidden_size, hidden_size, (3,3), stride=1, padding=1)
        self.Wmo = nn.Conv2d(hidden_size, hidden_size, (3,3), stride=1, padding=1)
        
        self.W1 = nn.Conv2d(hidden_size*2, hidden_size, (1,1), stride=1, padding=0)
        
    def forward(self, input, H, C, M):   
        Wx = self.Wx(input) # [b,f*7,h,w]
        Wh = self.Wh(H) # [b,f*4,h,w]
        Wm = self.Wm(M) # [b,f*3,h,w]
        
        _gx, _ix, _fx, _ox, _gx2, _ix2, _fx2 = torch.split(Wx, self.hidden_size, dim=1)
        _gh, _ih, _fh, _oh = torch.split(Wh, self.hidden_size, dim=1)
        _gm, _im, _fm = torch.split(Wm, self.hidden_size, dim=1)
        
        g = torch.tanh(_gx + _gh)
        i = torch.sigmoid(_ix + _ih)
        f = torch.sigmoid(_fx + _fh)

        C_new = f*C + i*g
        
        g2 = torch.tanh(_gx2 + _gm)
        i2 = torch.sigmoid(_ix2 + _im)
        f2 = torch.sigmoid(_fx2 + _fm)
        
        M_new = f2*M + i2*g2
        
        _oc, _om = self.Wco(C_new), self.Wmo(M_new)
        o = torch.sigmoid(_ox + _oh + _oc + _om)
        H_new = o*torch.tanh(self.W1(torch.cat([C_new,M_new], dim=1)))
        
        return H_new, C_new, M_new
    
class STLSTM_layers(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, name='STLSTM-Layers'):
        super(STLSTM_layers, self).__init__() 
        
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.cells = []
        
        cell_0 = [STLSTM_cell(input_size, hidden_size)]
        self.cells = nn.ModuleList(cell_0 + [STLSTM_cell(hidden_size, hidden_size) for i in range(1,num_layers)])
        
    def getShapedTensor(self, input, repeat_input=False):
        if repeat_input:
            return torch.zeros(input.size(0), self.hidden_size, input.size(2), input.size(3), dtype=input.dtype, device=input.device)
        else:
            return torch.zeros(input.size(0), self.hidden_size, input.size(3), input.size(4), dtype=input.dtype, device=input.device)
        
    def forward(self, input, H=None, C=None, M=None, return_sequence=False, repeat_input=False, output_sequence_length=8, return_M=True):  
        # repeat input: input is not a sequence with dimensions [b,t,f,h,w], 
        # but only one step which is input for every recurrent loop [b,f,h,w]
             
        Hs = [self.getShapedTensor(input, repeat_input=repeat_input) for _ in range(len(self.cells))]
        Cs = [self.getShapedTensor(input, repeat_input=repeat_input) for _ in range(len(self.cells))]

        if not isinstance(H, type(None)): #H!=None:
            Hs[0] = H
        if not isinstance(C, type(None)): #C!=None:
            Cs[0] = C
        if isinstance(M, type(None)): # M==None:
            M = self.getShapedTensor(input, repeat_input=False)
        
        if return_sequence:
            Os = []      
            
        if repeat_input:  
            for t in range(output_sequence_length):
                Hs[0], Cs[0], M = self.cells[0](input, Hs[0], Cs[0], M)
    
                for l in range(1, len(self.cells)):
                    Hs[l], Cs[l], M = self.cells[l](Hs[l-1], Hs[l], Cs[l], M)
                
                if return_sequence:
                    Os.append(Hs[-1])
                    
        else:
            for t in range(input.size(1)):
                Hs[0], Cs[0], M = self.cells[0](input[:,t], Hs[0], Cs[0], M)
    
                for l in range(1, len(self.cells)):
                    Hs[l], Cs[l], M = self.cells[l](Hs[l-1], Hs[l], Cs[l], M)
                
                if return_sequence:
                    Os.append(Hs[-1]) 
                    
        # return output sequence or cell states (hidden-state, gate-state, m-state (H,C,M))
        if return_sequence:
            if return_M:
                return torch.stack(Os).transpose(0,1), M
            return torch.stack(Os).transpose(0,1) # [t,b,hd,:,:]
        return Hs[-1], Cs[-1], M # [b,hd,:,:], [b,hd,:,:], [b,hd,:,:]

class STLSTM(nn.Module):
    
    
    def __init__(self, input_size=1, 
                 encoder_hidden_size=64, 
                 decoder_hidden_size=64,
                 lstm_hidden_size=64,
                 output_size=1, 
                 name='STLSTM'):
        
        super(STLSTM, self).__init__()
        
        self.id = 'STLSTM'
        
        if output_size==None:
            output_size = input_size
            
        self.encoder = TimeDistributed(Encoder(input_size, encoder_hidden_size, output_size=lstm_hidden_size))
        self.decoder = TimeDistributed(Decoder(lstm_hidden_size, decoder_hidden_size, output_size=output_size))

        self.encoder_stlstm = STLSTM_layers(lstm_hidden_size, lstm_hidden_size, num_layers=3)
        self.decoder_stlstm = STLSTM_layers(lstm_hidden_size, lstm_hidden_size, num_layers=3)

    def forward(self, input, max_depth=10):
        input = self.encoder(input)
        H, C, M = self.encoder_stlstm(input.flip(1), return_sequence=False, return_M=True)   
        input = self.decoder_stlstm(H, H=H, C=C, M=M, return_sequence=True, repeat_input=True, output_sequence_length=max_depth, return_M=False)
        input = self.decoder(input)
        return input.transpose(1,2)
    
class TestDataset(VisionDataset):
    def __init__(self, fileX, fileY):
        super(TestDataset, self).__init__(root='') 
        
        self.transform = lambda data:(data.float()+127)/255.

        self.X = np.load(fileX)
        self.Y = np.load(fileY)

        self.X = torch.from_numpy(self.X)
        self.Y = torch.from_numpy(self.Y)

    def __getitem__(self, idx):
        return self.transform(self.X[idx]), self.transform(self.Y[idx])

    def __len__(self):
        return len(self.X)


# %%
fileX ='/home/roland/Projekte/FromSurface2Depth/data/processed/regimeB/X_test.npy'
fileY = '/home/roland/Projekte/FromSurface2Depth/data/processed/regimeB/Y_test.npy'
fileSTLSTM = '/home/roland/Projekte/FromSurface2Depth/scripts/plots/STLSTM_t32_d0-32,1'
depth = 31
num_examples = 3

# %%
dataset = TestDataset(fileX, fileY)

# %%
model = nn.DataParallel(STLSTM(1,64))
model.load_state_dict(torch.load(fileSTLSTM, map_location=device), strict=True)

# %%
data = dataset[:num_examples]

with torch.no_grad():
    y_pred = model(data[0], max_depth=depth).cpu().detach().numpy()[:,0,-1]
y_plot = data[1][:, 0,depth].cpu().detach().numpy()
X_plot = data[0][:,-1,0].cpu().detach().numpy()

# %%

# plot:
#            | input | true | pred1 | diff1 | pred2 | diff2 |
# example 1: |  xx   |  xx  |  xx   |  xx   |  xx   |  xx   |
# example 2: |  xx   |  xx  |  xx   |  xx   |  xx   |  xx   |
# example 3: |  xx   |  xx  |  xx   |  xx   |  xx   |  xx   |

num_rows = num_examples
num_cols = 6


mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

font = {'size':16}

mpl.rc('font', **font)

#matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')


fig, axs = plt.subplots(num_rows, num_cols, sharex=True, sharey=True, figsize=(num_cols*2,num_rows*2))
plt.subplots_adjust(left=0.0, right=1.0, top=0.95, bottom=0.01)


letters = list(string.ascii_uppercase)[:num_cols*num_rows]

for row in range(num_rows):
    im1 = axs[row,0].imshow(X_plot[row])
    axs[row,1].imshow(y_plot[row])
    axs[row,2].imshow(y_pred[row])
    im2 = axs[row,3].imshow(y_plot[row]-y_pred[row], cmap='seismic', vmin=-1, vmax=1)
    
    axs[row,4].imshow(y_pred[row]) # Sebastians prediction
    axs[row,5].imshow(y_plot[row]-y_pred[row], cmap='seismic', vmin=-1, vmax=1)
    
    
    for col in range(num_cols):
        axs[row, col].axis('off')
        
        idx = num_cols*row + col
        letter = letters[idx]
        
        #axs[row,col].text(0.03, 0.97, letter, transform=axs[row,col].transAxes, fontsize=16, fontweight='bold', 
        #                  va='top', ha='left')
        
        rect = mpatch.Rectangle((0.0, 0.0), 23, 23, linewidth=2, edgecolor='black', facecolor='white')
        axs[row,col].add_artist(rect)
        rx, ry = rect.get_xy()
        cx = rx + rect.get_width()/2.0
        cy = ry + rect.get_height()/2.0
        axs[row,col].annotate(letter, (cx, cy), color='black', weight='bold', 
                              ha='center', va='center', **font)
        
        
        #axs[row,col].add_patch(rect)
        
            
axs[0,0].set_title('Input')
axs[0,1].set_title('Ground truth')
axs[0,2].set_title('ST-LSTM')
axs[0,3].set_title('Diff. ST-LSTM')
axs[0,4].set_title('CRF')
axs[0,5].set_title('Diff. CRF')

plt.savefig('comparison_grid.pdf')

#tikzplotlib.save('comparison_grid.tex')
#fig.colorbar(im1, ax=axs.ravel().tolist(), aspect=50)
#fig.colorbar(im2, ax=axs.ravel().tolist(), aspect=50)

#fig.subplots_adjust(right=0.9)
#cbar_ax = fig.add_axes([0.9, 0.15, 0.05, 0.7])
#fig.colorbar(im1, cax=cbar_ax)

#cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
#fig.colorbar(im2, cax=cbar_ax)
