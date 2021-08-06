import os, glob
import numpy as np
from tqdm import tqdm
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import matplotlib as mpl

import tikzplotlib

dir_raw ='/home/roland/Projekte/FromSurface2Depth/data/visualization/regimeA/longterm'

files = glob.glob(os.path.join(dir_raw, "*.npy"))

dt = 0.1
    
# %%

# Characteristic length

_FFTS = []

timestep = 1.
freq = np.fft.fftfreq(120, d=timestep)
inds = np.where(freq>0)[0]
freq = freq[inds]


_FFTS = []
dir_raw ='/home/roland/Projekte/FromSurface2Depth/data/visualization/regimeA/longterm'
files = glob.glob(os.path.join(dir_raw, "*.npy"))

for file in tqdm(files):
    data = np.load(file) # first time step # [120,120,120]
    
    for ind in np.random.randint(0,512,64):
        
        for axes in [[0,0]]:#, [0,1], [0,2]]:
            d = data[ind]
            _d = np.swapaxes(d, *axes)
            _d = np.reshape(_d, (120,-1)).T[np.random.randint(0,120**2,100)]
        
            for __d in _d:
                fft = np.abs(np.real(np.fft.fft(__d)))[inds]
                _FFTS.append(fft)
    
FFTS_A = np.mean(np.array(_FFTS), axis=0)


_FFTS = []
dir_raw ='/home/roland/Projekte/FromSurface2Depth/data/visualization/regimeB/longterm'
files = glob.glob(os.path.join(dir_raw, "*.npy"))

for file in tqdm(files):
    data = np.load(file) # first time step # [120,120,120]
    
    for ind in np.random.randint(0,512,64):
        
        for axes in [[0,0]]:#, [0,1], [0,2]]:
            d = data[ind]
            _d = np.swapaxes(d, *axes)
            _d = np.reshape(_d, (120,-1)).T[np.random.randint(0,120**2,100)]
        
            for __d in _d:
                fft = np.abs(np.real(np.fft.fft(__d)))[inds]
                _FFTS.append(fft)
    
FFTS_B = np.mean(np.array(_FFTS), axis=0)

# %%

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4,2), sharey=True)

plt.subplots_adjust(left=0.0, right=1.0, top=0.95, bottom=0.01)

ax1.plot(freq, FFTS_A)
ax1.set_ylabel(r'Intensity')
ax1.set_xlabel(r'Frequency [$\Delta s^{-1}]$')
ax1.set_title('Regime A')

maximum = freq[6]
ax1.axvline(x=maximum, color='b', alpha=0.5, label=f'Max. at {np.round(maximum,4)}/$\Delta s$, $L_c=${np.round(1/maximum,2)}$\cdot\Delta s$')
ax1.legend()

ax2.plot(freq, FFTS_B)
ax2.set_xlabel(r'Frequency [$\Delta t^{-1}]$')
ax2.set_title('Regime B')

#maximum = freq[np.where(FFTS_B==max(FFTS_B))[0][0]]
#ax2.axvline(x=maximum, color='b', alpha=0.5, label=f'Max. at {np.round(maximum,4)}/$\Delta s$, $L_c=${np.round(1/maximum, 2)}$\cdot\Delta s$')
#ax2.legend()

plt.yticks([])
tikzplotlib.save('characteristic_length.tex')

# %%

# Characteristic time

_FFTS = []

timestep = 16.
freq = np.fft.fftfreq(512, d=timestep)
inds = np.where(freq>0)[0]
freq = freq[inds]


_FFTS = []
dir_raw ='/home/roland/Projekte/FromSurface2Depth/data/visualization/regimeA/longterm'
files = glob.glob(os.path.join(dir_raw, "*.npy"))

for file in tqdm(files):
    data = np.load(file)
    data = np.reshape(data, (512,-1)).T[np.random.randint(0,120**3,1000)]
    
    for d in data:
        fft = np.abs(np.real(np.fft.fft(d)))[inds]
        _FFTS.append(fft)
    
FFTS_A = np.mean(np.array(_FFTS), axis=0)


_FFTS = []
dir_raw ='/home/roland/Projekte/FromSurface2Depth/data/visualization/regimeB/longterm'
files = glob.glob(os.path.join(dir_raw, "*.npy"))

for file in tqdm(files):
    data = np.load(file)
    data = np.reshape(data, (512,-1)).T[np.random.randint(0,120**3,1000)]
    
    for d in data:
        fft = np.abs(np.real(np.fft.fft(d)))[inds]
        _FFTS.append(fft)
    
FFTS_B = np.mean(np.array(_FFTS), axis=0)

# %%

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4,2), sharey=True)

plt.subplots_adjust(left=0.0, right=1.0, top=0.95, bottom=0.01)

ax1.plot(freq, FFTS_A)
ax1.set_ylabel(r'Intensity')
ax1.set_xlabel(r'Frequency [$\Delta t^{-1}]$')
ax1.set_title('Regime A')

maximum = freq[np.where(FFTS_A==max(FFTS_A))[0][0]]
ax1.axvline(x=maximum, color='b', alpha=0.5, label=f'Max. at {np.round(maximum,4)}/$\Delta t$, $T_c=${int(1/maximum)}$\cdot\Delta t$')
ax1.legend()

ax2.plot(freq, FFTS_B)
ax2.set_xlabel(r'Frequency [$\Delta t^{-1}]$')
ax2.set_title('Regime B')

maximum = freq[np.where(FFTS_B==max(FFTS_B))[0][0]]
ax2.axvline(x=maximum, color='b', alpha=0.5, label=f'Max. at {np.round(maximum,4)}/$\Delta t$, $T_c=${int(1/maximum)}$\cdot\Delta t$')
ax2.legend()

plt.yticks([])
tikzplotlib.save('characteristic_time.tex')





















# %%

signal = data[0,0]
DPS = []

for file in tqdm(files):
    data = np.load(file)
    for d in data[0]:
        for _d in d:
            peaks, _ = find_peaks(_d, height=0)
            
            if len(peaks)>1:
                dp = np.diff(peaks)
                
                for _dp in dp:
                    DPS.append(_dp)
            
    
    #plt.plot(x)
    #plt.scatter(peaks, range(len(peaks)))
    #plt.show()
    
# %%



