#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 21 11:17:51 2026

@author: alberto-razo
"""
import time
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm


def VortexMask(size, ordmax=1, Sym=True, Max=None, Min=None):
    """
    Generates spiral phase mask exp(i n phi)

    Size: Size of the array side
    ordmax: Maximum order of the mask
    Sym: if True, generate mask considering [-n, n]
    Max: Cuts the values with r>Max
    Min: Cuts the values with r<Min
    """
    if Max is None:
        Max = 2
    if Min is None:
        Min = -1
    ax = np.linspace(-1, 1, size)
    axX, axY = np.meshgrid(ax, ax)

    r = np.sqrt(axX**2 + axY**2)
    theta = np.arctan2(axY, axX)
    
    if Sym:
        inds = np.arange(-ordmax, ordmax+1)
    else:
        inds = np.arange(0, ordmax+1)
    
    Masks = []
    for i in inds:
        phase = (i*theta+np.pi)%(2*np.pi)-np.pi

        mask = np.zeros([size, size], dtype='complex')
        indX, indY = np.where((r<=Max) & (r>=Min))
        mask[indX, indY] = np.exp(-1j*phase[indX, indY])

        Masks.append(mask.astype(complex))
    if ordmax == 0:
        return Masks[0]
    else:
        return Masks
    
    
def GetFarField(array, Npad=5, WinSize=5):
    """
    Computes the far field propagation of a given array

    array: the array
    Npad: the size of the padded figure used to compute the Fourier transform
    WinSize: the size of the final figure
    conj: if True shifts by one the pixel the final image
    """
    n = len(array)
    pad = (Npad-1)*n//2
    array = np.pad(array, pad, mode='constant')
    array = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(array)))
    return CropCenter(array, int(n*WinSize))


def GetFarField_Torch(array, Npad=5, WinSize=5):
    """
    Computes the far field propagation of a given tensor

    array: the array
    Npad: the size of the padded figure used to compute the Fourier transform
    WinSize: the size of the final figure
    conj: if True shifts by one the pixel the final image
    """
    n = array.shape[-1]
    pad = (Npad - 1) * n // 2
    # Pad (left, right, top, bottom) — note reversed order in F.pad for 2D
    array_padded = F.pad(array, (pad, pad, pad, pad), mode='constant', value=0)
    # Apply fftshift, fft2, then fftshift again
    shifted_input = torch.fft.fftshift(array_padded, dim=(-2, -1))
    fft_output = torch.fft.fft2(shifted_input)
    shifted_fft = torch.fft.fftshift(fft_output, dim=(-2, -1))
    # Crop center
    return CropCenter_Torch(shifted_fft, int(n*WinSize))


def CropCenter(array, crop):
    """
    Crops the central section of a given array

    array: the array
    crop: number of pixels size we will extract
    conj: if True shifts by one the pixel the final image
    """
    x, y = array.shape
    startx = x//2-(crop//2)
    starty = y//2-(crop//2)
    array = array[startx:startx+crop, starty:starty+crop]
    return array


def CropCenter_Torch(tensor, size):
    """
    Crops the central section of a given tensor

    array: the array
    crop: number of pixels size we will extract
    conj: if True shifts by one the pixel the final image
    """
    center = tensor.shape[-1] // 2
    half = size // 2

    tensor = tensor[..., center - half:center - half + size, center - half:center - half + size]
    return tensor


def L2_Norm(x, y):
    """
    Computes the norm in 2D between two different arrays
    
    x: the array 1
    y: the array 2
    """
    return np.sqrt(np.mean(np.abs(x - y)**2))


def L2_Norm_Torch(x, y):
    """
    Computes the norm in 2D between two different tensors
    
    x: the array 1
    y: the array 2
    """
    return torch.sqrt(torch.mean(torch.abs(x - y)**2))


class EarlyStopping:
    """
    Early stopping function for the torch optimizer

    Initialization: EarlyS = EarlyStopping(Patience=100, Mindelta=0)
    Usage: EarlyS(Error)

    Patience: Number of steps after the code stops
    Mindelta: Minimum value of decreasing of the error to consider the optimizer
              is not improving
    Error: Error to minimice by the optimizer
    """
    def __init__(self, Patience=500, Mindelta=0):
        self.Patience = Patience
        self.Mindelta = Mindelta
        self.Counter = 0
        self.BestLoss = None
        self.should_stop = False

    def __call__(self, Loss):
        if self.BestLoss is None or Loss < self.BestLoss-self.Mindelta:
            self.BestLoss = Loss
            self.Counter = 0
        else:
            self.Counter += 1
            if self.Counter >= self.Patience:
                self.should_stop = True


class IncreasingThreshold:
    """
    Function for the torch optimizer. Since it is considered a certain maximum of
    error to accept or not a diffuser, and that such maximum can be hardly achieve
    when considering systems with disorder.

    Initialization: Threshold = IncreasingThreshol(Patience=5, Increase=0.05)
    Usage: MaxLoss = Threshold(MaxLoss, Loss)

    Patience: Number of times the solver will restar before increasing the threshold
    Increase: Increase of the maximum threshold
    MaxLoss: Current threshold
    Loss: Error to minimice by the optimizer 
    """
    def __init__(self, Patience=5, Increase=0.05):
        self.Patience = Patience
        self.Increase = Increase
        self.Counter = 0

    def __call__(self, MaxLoss, Loss):
        self.Counter += 1
        if self.Counter >= self.Patience:
            self.Counter = 0
            print(f'Increasing maximum loss to {self.Increase + MaxLoss}, Loss={Loss}')
            return self.Increase + MaxLoss
        else:
            print(f'Restarting {self.Counter}/{self.Patience-1}, Loss={Loss}')
            return MaxLoss


class SavingBest:
    """
    Function for the torch optimizer. Keeps the best diffuser ever found if it is not
    good enough for the current threshold. If threshold becomes larger than the storaged
    diffuser, then this array is automatically taken.

    Initialization: Best = SavingBest()
    Usage: Best(Diff, Loss)

    Diff: Diffuser
    Loss: Error to minimice by the optimizer
    """
    def __init__(self):
        self.Best = None
        self.Loss = None

    def __call__(self, Diff, Loss):
        if ((self.Loss is None) or (Loss<self.Loss)):
            self.Loss = Loss
            self.Best = Diff
            print(f'Saving best diffuser with loss {self.Loss}')


def FindDiffusera0(Input, deph=0, LR_init=1, NPad=10, TryR=False, InitDiff=None, RMax=None, DiffSize=None, MaxSteps=None, MaxLoss=None):
    device = "cpu"
    EarlyS = EarlyStopping(Patience=100, Mindelta=0)
    Threshold = IncreasingThreshold(Patience=5, Increase=0.05)

    # Recieving and computing numpy arrays
    alpha = np.pi
    Nangle = (2*np.pi)/(alpha)
    cm1 = np.abs(np.sinc(1/Nangle))/(Nangle-1)*np.exp(-1j*deph)
    cp1 = np.abs(np.sinc(1/Nangle))/(Nangle-1)*np.exp(1j*deph)

    a02 = Input[0]
    S0 = Input[1]
    S1 = Input[2]
    a0 = Input[3]

    if DiffSize is None:
        DiffSize = a02.shape[0]

    if MaxLoss is None:
        MaxLoss = 0.3

    if MaxSteps is None:
        MaxSteps = int(1e5)

    VMasks = VortexMask(DiffSize, 1, Max=RMax)

    #Loading to torch
    a02_target = torch.from_numpy(a02).to(torch.complex64).to(device)
    S0_target = torch.from_numpy(S0).to(torch.complex64).to(device)
    S1_target = torch.from_numpy(S1).to(torch.complex64).to(device)
    a0_target = torch.from_numpy(a0).to(torch.complex64).to(device)

    a02_target = a02_target/torch.mean(torch.abs(a02_target))
    S0_target = S0_target/torch.mean(torch.abs(S0_target))
    S1_target = S1_target/torch.mean(torch.abs(S1_target))
    a0_target = a0_target/torch.mean(torch.abs(a0_target))

    V0 = torch.from_numpy(VMasks[1]).to(torch.complex64).to(device)
    Vp1 = torch.from_numpy(VMasks[2]).to(torch.complex64).to(device)
    Vm1 = torch.from_numpy(VMasks[0]).to(torch.complex64).to(device)

    if InitDiff is None:
        param_diff = 2*np.pi*(torch.rand(DiffSize, DiffSize, requires_grad=True) - 0.5)
        param_diff = param_diff.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([param_diff], lr=LR_init)
    else:
        param_diff = torch.from_numpy(InitDiff).clone().detach().to(torch.float32).to(device).requires_grad_(True)
        optimizer = torch.optim.Adam([param_diff], lr=LR_init)
        
    # Optimization
    TotLoss = np.zeros([MaxSteps])
    ElapTime = np.zeros([MaxSteps])
    EnergyLoss = np.zeros([MaxSteps])

    timei = time.time()
    for _step in tqdm(range(MaxSteps)):
        optimizer.zero_grad()

        diff = torch.exp(1j*param_diff)

        pred_a0 = GetFarField_Torch(diff*V0, NPad, a02.shape[0]/DiffSize)
        pred_ap1 = GetFarField_Torch(diff*Vp1, NPad, a02.shape[0]/DiffSize)
        pred_am1 = GetFarField_Torch(diff*Vm1, NPad, a02.shape[0]/DiffSize)

        pred_a0 = pred_a0/torch.mean(torch.abs(pred_a0))
        pred_ap1 = pred_ap1*cp1/torch.mean(torch.abs(pred_ap1))
        pred_am1 = pred_am1*cm1/torch.mean(torch.abs(pred_am1))

        S0_pred = torch.abs(pred_a0)**2 + torch.abs(pred_am1)**2 + torch.abs(pred_ap1)**2
        S1_pred = pred_a0*pred_am1.conj() + pred_ap1*pred_a0.conj()

        a02_pred = torch.abs(pred_a0)**2/torch.mean(torch.abs(pred_a0)**2)
        S0_pred = S0_pred/torch.mean(torch.abs(S0_pred))
        S1_pred = S1_pred/torch.mean(torch.abs(S1_pred))

        loss_a0 = L2_Norm_Torch(a02_pred, a02_target)
        loss_S0 = L2_Norm_Torch(S0_pred, S0_target)
        loss_S1 = L2_Norm_Torch(S1_pred, S1_target)

        loss_total = loss_a0 + loss_S0 + loss_S1

        loss_total.backward()
        optimizer.step()

        EnergyMap = torch.abs(torch.fft.fft2(torch.exp(1j*(torch.angle(a0_target) - torch.angle(pred_a0)))))

        TotLoss[_step] = loss_total.item()
        ElapTime[_step] = time.time() - timei
        EnergyLoss[_step] = (torch.max(EnergyMap)/DiffSize**2).item()

        EarlyS(loss_total.item())
        if TryR and (EarlyS.should_stop and loss_total.item()>MaxLoss):
            EarlyS = EarlyStopping(Patience=100, Mindelta=0)
            MaxLoss = Threshold(MaxLoss, loss_total.item())
            with torch.no_grad():
                param_diff = 2*np.pi*(torch.rand(DiffSize, DiffSize) - 0.5)
            param_diff = param_diff.clone().detach().requires_grad_(True)
            optimizer = torch.optim.Adam([param_diff], lr=LR_init)
        if EarlyS.should_stop:
            print('Found', loss_total.item())
            TotLoss = TotLoss[:_step+1]
            ElapTime = ElapTime[:_step+1]
            EnergyLoss = EnergyLoss[:_step+1]
            break

    Diff = param_diff.detach().numpy()
    return Diff, TotLoss, ElapTime, EnergyLoss


def FindBigDiffusera0(Input, deph=0, Div=2, LR_init=1, NPad=10, RMax=None, DiffSize=None, MaxSteps=None, MaxLoss=None):
    a02= Input[0]
    S0 = Input[1]
    S1 = Input[2]
    a0 = Input[3]

    if DiffSize is None:
        DiffSize = a02.shape[0]

    if MaxLoss is None:
        MaxLoss = 0.3

    for i in np.flip(range(Div+1)):
        _a02 = CropCenter(a02, DiffSize//(2**i))
        _S0 = CropCenter(S0, DiffSize//(2**i))
        _S1 = CropCenter(S1, DiffSize//(2**i))
        _a0 = CropCenter(a0, DiffSize//(2**i))

        if i == Div:
            Diff, _TotLoss, _ElapTime, _EnergyLoss = FindDiffusera0([_a02, _S0, _S1, _a0], deph, LR_init, NPad, True, RMax=RMax, DiffSize=DiffSize//(2**Div), MaxSteps=MaxSteps, MaxLoss=MaxLoss)
        else:
            Diff, _TotLoss, _ElapTime, _EnergyLoss = FindDiffusera0([_a02, _S0, _S1, _a0], deph, LR_init/2, NPad, False, InitDiff=Diff, RMax=RMax, MaxSteps=MaxSteps, MaxLoss=MaxLoss)

        if i != 0:
            Diff = np.repeat(np.repeat(Diff, 2, axis=0), 2, axis=1)

        try:
            TotLoss = np.append(TotLoss, _TotLoss)
            ElapTime = np.append(ElapTime, _ElapTime+ElapTime[-1])
            EnergyLoss = np.append(EnergyLoss, _EnergyLoss)
        except:
            TotLoss = _TotLoss
            ElapTime = _ElapTime
            EnergyLoss = _EnergyLoss

    Diff = np.exp(1j*Diff)
    return Diff, TotLoss, ElapTime, EnergyLoss


def FindDiffuser(Input, deph=0, LR_init=1, NPad=10, TryR=False, InitDiff=None, RMax=None, DiffSize=None, MaxSteps=None, MaxLoss=None):
    device = "cpu"
    EarlyS = EarlyStopping(Patience=100, Mindelta=0)
    Threshold = IncreasingThreshold(Patience=5, Increase=0.05)
    Best = SavingBest()

    # Recieving and computing numpy arrays
    alpha = np.pi
    Nangle = (2*np.pi)/(alpha)
    cm1 = np.abs(np.sinc(1/Nangle))/(Nangle-1)*np.exp(-1j*deph)
    cp1 = np.abs(np.sinc(1/Nangle))/(Nangle-1)*np.exp(1j*deph)

    a02 = Input[0]
    S0 = Input[1]
    S1 = Input[2]

    if DiffSize is None:
        DiffSize = a02.shape[0]

    if MaxLoss is None:
        MaxLoss = 0.3

    if MaxSteps is None:
        MaxSteps = int(1e5)

    VMasks = VortexMask(DiffSize, 1, Max=RMax)

    #Loading to torch
    a02_target = torch.from_numpy(a02).to(torch.complex64).to(device)
    S0_target = torch.from_numpy(S0).to(torch.complex64).to(device)
    S1_target = torch.from_numpy(S1).to(torch.complex64).to(device)

    a02_target = a02_target/torch.mean(torch.abs(a02_target))
    S0_target = S0_target/torch.mean(torch.abs(S0_target))
    S1_target = S1_target/torch.mean(torch.abs(S1_target))

    V0 = torch.from_numpy(VMasks[1]).to(torch.complex64).to(device)
    Vp1 = torch.from_numpy(VMasks[2]).to(torch.complex64).to(device)
    Vm1 = torch.from_numpy(VMasks[0]).to(torch.complex64).to(device)

    if InitDiff is None:
        param_diff = 2*np.pi*(torch.rand(DiffSize, DiffSize, requires_grad=True) - 0.5)
        param_diff = param_diff.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([param_diff], lr=LR_init)
    else:
        param_diff = torch.from_numpy(InitDiff).clone().detach().to(torch.float32).to(device).requires_grad_(True)
        optimizer = torch.optim.Adam([param_diff], lr=LR_init)

    # Optimization
    TotLoss = np.zeros([MaxSteps])
    ElapTime = np.zeros([MaxSteps])

    timei = time.time()
    for _step in tqdm(range(MaxSteps)):
        optimizer.zero_grad()

        diff = torch.exp(1j*param_diff)

        pred_a0 = GetFarField_Torch(diff*V0, NPad, a02.shape[0]/DiffSize)
        pred_ap1 = GetFarField_Torch(diff*Vp1, NPad, a02.shape[0]/DiffSize)
        pred_am1 = GetFarField_Torch(diff*Vm1, NPad, a02.shape[0]/DiffSize)

        pred_a0 = pred_a0/torch.mean(torch.abs(pred_a0))
        pred_ap1 = pred_ap1*cp1/torch.mean(torch.abs(pred_ap1))
        pred_am1 = pred_am1*cm1/torch.mean(torch.abs(pred_am1))

        S0_pred = torch.abs(pred_a0)**2 + torch.abs(pred_am1)**2 + torch.abs(pred_ap1)**2
        S1_pred = pred_a0*pred_am1.conj() + pred_ap1*pred_a0.conj()

        a02_pred = torch.abs(pred_a0)**2/torch.mean(torch.abs(pred_a0)**2)
        S0_pred = S0_pred/torch.mean(torch.abs(S0_pred))
        S1_pred = S1_pred/torch.mean(torch.abs(S1_pred))

        loss_a0 = L2_Norm_Torch(a02_pred, a02_target)
        loss_S0 = L2_Norm_Torch(S0_pred, S0_target)
        loss_S1 = L2_Norm_Torch(S1_pred, S1_target)

        loss_total = loss_a0 + loss_S0 + loss_S1

        loss_total.backward()
        optimizer.step()

        TotLoss[_step] = loss_total.item()
        ElapTime[_step] = time.time() - timei

        EarlyS(loss_total.item())
        if TryR and (EarlyS.should_stop and loss_total.item()>MaxLoss):
            MaxLoss = Threshold(MaxLoss, loss_total.item())
            Best(param_diff, loss_total.item())
            if Best.Loss < MaxLoss:
                print(f'Best diffuser with loss {Best.Loss} has been previously found')
                param_diff = Best.Best
            else:
                EarlyS = EarlyStopping(Patience=100, Mindelta=0)
                with torch.no_grad():
                    param_diff = 2*np.pi*(torch.rand(DiffSize, DiffSize) - 0.5)
                param_diff = param_diff.clone().detach().requires_grad_(True)
                optimizer = torch.optim.Adam([param_diff], lr=LR_init)
        if EarlyS.should_stop:
            print('Found', loss_total.item())
            TotLoss = TotLoss[:_step+1]
            ElapTime = ElapTime[:_step+1]
            break

    Diff = param_diff.detach().numpy()
    return Diff, TotLoss, ElapTime


def FindBigDiffuser(Input, deph=0, Div=2, LR_init=1, NPad=10, RMax=None, DiffSize=None, MaxSteps=None, MaxLoss=None):
    a02= Input[0]
    S0 = Input[1]
    S1 = Input[2]

    if DiffSize is None:
        DiffSize = a02.shape[0]

    if MaxLoss is None:
        MaxLoss = 0.3

    for i in np.flip(range(Div+1)):
        _a02 = CropCenter(a02, DiffSize//(2**i))
        _S0 = CropCenter(S0, DiffSize//(2**i))
        _S1 = CropCenter(S1, DiffSize//(2**i))

        if i == Div:
            print(f'Starting with a size of {len(_a02)}x{len(_a02)} pixels')
            Diff, _TotLoss, _ElapTime = FindDiffuser([_a02, _S0, _S1], deph, LR_init, NPad, True, RMax=RMax, DiffSize=DiffSize//(2**Div), MaxSteps=MaxSteps, MaxLoss=MaxLoss)
        else:
            print(f'Increasing the size of the system to {len(_a02)}x{len(_a02)} pixels')
            Diff, _TotLoss, _ElapTime = FindDiffuser([_a02, _S0, _S1], deph, LR_init/2, NPad, False, InitDiff=Diff, RMax=RMax, MaxSteps=MaxSteps, MaxLoss=MaxLoss)

        if i != 0:
            Diff = np.repeat(np.repeat(Diff, 2, axis=0), 2, axis=1)

        try:
            TotLoss = np.append(TotLoss, _TotLoss)
            ElapTime = np.append(ElapTime, _ElapTime+ElapTime[-1])
        except:
            TotLoss = _TotLoss
            ElapTime = _ElapTime

    Diff = np.exp(1j*Diff)
    return Diff, TotLoss, ElapTime