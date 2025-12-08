#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 14:58:15 2025

@author: alberto-razo
"""
import time
import torch
import numpy as np
import torch.nn.functional as F
import translib.functions as fun
from tqdm import tqdm
from scipy.special import jn
from scipy.optimize import curve_fit


def HarmComb(_Imgs, _Npad=1, _axis=0, _Nper=1):
    """
    Takes the array of figures modulated in time and perform the Fourier tranforms
    to the intensity comb and respective frequencies

    Imgs: Array of data with time dependence
    Npad: Number of periods after padding
    axis: axis of 'Imgs' that represents the time dependence 
    Nper: The number of time cycles in the original signal
    """
    if _Nper != 1:
        _Imgs = np.array_split(_Imgs, _Nper, axis=_axis)[0]
        
    _img = _Imgs[1:]
    for i in range(_Npad-1):
        _Imgs = np.append(_Imgs, _img, axis=_axis)
    del _img
    
    _harms = np.fft.fftshift(np.fft.fft(_Imgs, axis=_axis), axes=_axis)
    
    _amps = np.mean(np.abs(_harms), axis=(1,2))
    del _harms
    
    _Ntot = len(_amps)
    _frqs = np.linspace(0, _Npad, _Ntot)
    _frqs = _frqs[1] - _frqs[0]
    _frqs = np.linspace(-0.5/_frqs, 0.5/_frqs, _Ntot)
    return _amps, _frqs


def Harmonics(_Imgs, _Nharms=4, _Npad=1, _Sym=True, _axis=0, _Nper=1):
    """
    Takes the array of figures modulated in time and perform the Fourier tranforms
    to find the harmonics

    Imgs: Array of data with time dependence
    Nharm: Number of Harmonics that will be extracted from the data
    Npad: Number of periods after padding
    Sym: if True extract the harmonics [-n, n]
    axis: axis of 'Imgs' that represents the time dependence 
    Nper: The number of time cycles in the original signal
    """
    if _Nper != 1:
        _Imgs = np.array_split(_Imgs, _Nper, axis=_axis)[0]
        
    _img = _Imgs[1:]
    for i in range(_Npad-1):
        _Imgs = np.append(_Imgs, _img, axis=_axis)
    del _img
    
    _Ntot = len(_Imgs)
    if np.mod(_Ntot,2) == 0:
        _Imgs = np.append(_Imgs, _Imgs[0], axis=_axis)
        _Ntot = len(_Imgs)

    _harms = np.fft.fftshift(np.fft.fft(_Imgs, axis=_axis), axes=_axis)
    
    _frqs = np.linspace(0, _Npad, _Ntot)
    _frqs = _frqs[1] - _frqs[0]
    _frqs = np.linspace(-0.5/_frqs, 0.5/_frqs, _Ntot)

    _inds = []
    for i in range(-_Nharms, _Nharms+1):
        _inds = np.append(_inds, np.where(i==_frqs))

    _inds = _inds.astype(int)
    _harms = np.take(_harms, _inds, axis=_axis)
    
    if not _Sym:
        _harms = _harms[_Nharms:]
    return _harms


def PhaseDiffuser(_N, _ps=1, Seed=None):
    """
    Computes a random mask with values between -pi and pi.

    N: Size of the array side
    ps: Size of the pixels
    Seed: seed for the pseudorandom numbers
    """
    if Seed is None:
        Seed = np.random.randint(low=100)
    np.random.seed(Seed)
    _phasediffuser = 2*np.pi*(np.random.rand(_N//_ps, _N//_ps) - 0.5)
    _phasediffuser = np.repeat(np.repeat(_phasediffuser, _ps, axis=0), _ps, axis=1)
    return _phasediffuser


def ElipticMask(size, order=1, a=1, b=1, Max=None, Min=None):
    """
    Generates spiral eliptical phase mask

    size: Size of the array side
    ordet: Order of the spiral
    a, b: elongation of the spiral
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
    theta = np.arctan2(axY/b, axX/a)

    mask = np.zeros([size, size], dtype='complex')
    indX, indY = np.where((r<=Max) & (r>=Min))

    mask[indX, indY] = np.exp(1j*order*theta[indX, indY])
    return mask


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
    
    
def SectionMask(size, nfigs, angle, deph=0, clock=True, Max=None, Min=None):
    """
    Generates section turning mask

    Size: Size of the array side
    nfigs: Number of masks used to map a circle
    angle: angle of the apperture
    deph: dephasing of the apperture respect to the x-axis
    clock: if True, sections turn clockwise
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

    if angle > 2*np.pi: # Considers that if the apperture angle is larger than 2pi then it was given if degrees
        dtheta = np.pi*angle/180
    if np.abs(deph) > 2*np.pi: # Considers that if the dephase angle is larger than 2pi then it was given if degrees  
        deph = np.pi*deph/180
    cthetas = np.linspace(-np.pi, np.pi, nfigs) + deph

    if clock:
        c = 1
    else:
        c = -1

    Masks = []
    for ctheta in cthetas:
        mask = np.zeros([size, size], dtype='complex')
        mask[r<=Max] = 1
        mask[r<=Min] = 0
        _indx, _indy = np.where((np.abs(theta-c*ctheta-2*np.pi)<=dtheta/2) | 
                                (np.abs(theta-c*ctheta+2*np.pi)<=dtheta/2) | 
                                (np.abs(theta-c*ctheta)<=dtheta/2))
        
        mask[_indx, _indy] = 0

        Masks.append(mask.astype(complex))
    return Masks


def GetExperimentDMD(Diffuser, angle, nfigs, deph=0, NPad=10, WinSize=1, clock=True, Max=None, Min=None):
    """
    Generates a serie of speckle patterns as the once obtained from the experiment
    where only the intensity can be measured and using a turning section

    Diffuser: The diffuser
    angle: angle of the apperture
    nfigs: Number of masks used to map a circle
    deph: dephasing of the apperture respect to the x-axis
    Npad: the size of the padded figure used to compute the Fourier transform
    WinSize: the size of the final figure
    clock: if True, sections turn clockwise
    Max: Cuts the values with r>Max
    Min: Cuts the values with r<Min
    """
    N = Diffuser.shape[0]
    Masks = SectionMask(N, nfigs, angle, deph, clock, Max=Max, Min=Min)
    speckles = []
    for _mask in Masks:
        _out = GetFarField(_mask*Diffuser, NPad, WinSize)
        speckles.append(np.abs(_out)**2)
    return speckles


def AddNoise(Intensity, Percentage, Seed=None):
    """
    For a certain intensity map, adds random noise in a certain percentage of the 
    mean intensity

    Intensity: The intenstity map
    Percentage: the percentage of the noise
    Seed: The seed use to add the noise
    """
    Percentage = Percentage/100
    mean = np.mean(Intensity)
    if Seed is None:
        Seed = np.random.randint(low=100)
    np.random.seed(Seed)
    noise = np.random.normal(0, mean*Percentage, Intensity.shape)
    NewInt = Intensity+noise
    return NewInt+np.min(NewInt)


def DoubleSectionMask(size, nfigs, angle1, angle2, deph1, deph2, rInt=0.5, clock1=True, clock2=True, Max=None, Min=None):
    """
    Generates section turning mask composed by two section mask that can rotate with different
    angles, dephase and in different sense

    Size: Size of the array side
    nfigs: Number of masks used to map a circle
    angle1: angle of the internal apperture
    angle2: angle of the external apperture
    deph1: dephasing of the internal apperture respect to the x-axis
    deph2: dephasing of the external apperture respect to the x-axis
    rInt: the radius of the transition between rotating mask
    clock1: if True, intenal section turn clockwise
    clock2: if True, external section turn clockwise
    Max: Cuts the values with r>Max
    Min: Cuts the values with r<Min
    """
    masks1 = SectionMask(size, nfigs, angle1, deph1, clock1, Max=rInt, Min=Min)
    masks2 = SectionMask(size, nfigs, angle2, deph2, clock2, Max=Max, Min=rInt)

    Masks = []
    for i in range(nfigs):
        mask = np.zeros([size, size], dtype = complex)
        mask[(np.abs(masks1[i])>0.5) | (np.abs(masks2[i])>0.5)] = 1

        Masks.append(mask.astype(complex))
    if nfigs == 1:
        return Masks[0]
    else:
        return Masks
    

def GetFarField(array, Npad=5, WinSize=5, conj=False):
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
    return CropCenter(array, int(n*WinSize), conj)


def GetFarField_Torch(array, Npad=5, WinSize=5, conj=False):
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
    return CropCenter_Torch(shifted_fft, int(n*WinSize), conj)


def CropCenter(array, crop, conj=False):
    """
    Crops the central section of a given array

    array: the array
    crop: number of pixels size we will extract
    conj: if True shifts by one the pixel the final image
    """
    x, y = array.shape
    startx = x//2-(crop//2)
    starty = y//2-(crop//2)
    if conj and np.mod(x,2)==0 and (x-crop)>1 and (y-crop)>1:
        array = array[startx+1:startx+crop+1, starty+1:starty+crop+1]
    else:
        array = array[startx:startx+crop, starty:starty+crop]
    return array


def CropCenter_Torch(tensor, size, conj=False):
    """
    Crops the central section of a given tensor

    array: the array
    crop: number of pixels size we will extract
    conj: if True shifts by one the pixel the final image
    """
    center = tensor.shape[-1] // 2
    half = size // 2

    if conj:
        tensor = tensor[..., center - half+1:center - half + size+1, center - half+1:center - half + size+1]
    else:
        tensor = tensor[..., center - half:center - half + size, center - half:center - half + size]
    return tensor


def GetFarDiffuser(array, Npad=5, WinSize=5):
    """
    Computes the far diffuser of a given propagated array

    array: the array
    Npad: the size of the padded figure used to compute the Fourier transform
    WinSize: the size of the final figure
    """
    n = len(array)
    pad = (Npad-1)*n//2
    array = np.pad(array, pad, mode='constant')
    array = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(array)))
    return CropCenter(array, int(n*WinSize))


def GetFarDiffuser_torch(array, Npad=5, WinSize=5):
    """
    Simulate the far-field (FFT) of a complex 2D array with zero padding and cropping.
    """
    n = array.shape[-1]
    pad = (Npad - 1) * n // 2
    # Pad (left, right, top, bottom) — note reversed order in F.pad for 2D
    array_padded = F.pad(array, (pad, pad, pad, pad), mode='constant', value=0)
    # Apply fftshift, fft2, then fftshift again
    shifted_input = torch.fft.ifftshift(array_padded, dim=(-2, -1))
    fft_output = torch.fft.ifft2(shifted_input)
    shifted_fft = torch.fft.ifftshift(fft_output, dim=(-2, -1))
    # Crop center
    return CropCenter_Torch(shifted_fft, int(n*WinSize))


def Contrast(IntensityArray):
    """
    Computes the constrast of an intensity array
    """
    I = np.mean(IntensityArray, axis=(0,1))
    s = np.std(IntensityArray, axis=(0,1))
    return s/I


def Airy(_r, _a, _b):
    """
    Evaluates the Airy spot function
    
    r: Domain array
    a: the numerical apperture over/wavelenght
    b: the constant of proportionality
    """
    return (_b*jn(1, 2*np.pi*_r*_a)/_r)**2


def GetGrainSize(_IntensArray, _axis=0, _p00=0.1, _p01=1e1):
    """
    Given an intensity array, computes the grain size by comparing the autocorrelation
    with the Airy spot
    
    IntensArray: the intensity array
    axis: axis to fit the airy spot
    p00: first paraleter of the fitting (the numerical apperture over/wavelenght)
    p01: second parameter of the fitting (the constant of proportionality)
    """
    _IntensArray = np.abs(fun.Corr(_IntensArray, _IntensArray))
    _IntensArray = _IntensArray/np.max(_IntensArray)
    _len = _IntensArray.shape[_axis]
    _IntensArray = np.take(_IntensArray, indices=_len//2, axis=_axis)
    _dom = np.linspace(-_len//2, _len//2, _len)
    _opt, _cov = curve_fit(Airy, _dom, _IntensArray, p0=[_p00, _p01])
    return 1/(2*_opt[0])


def GetNumModes(_IntensArray):
    """
    Given an intensity array computes the average number of modes
    
    IntensArray: the intensity array
    """
    _gs = GetGrainSize(_IntensArray)
    _len = _IntensArray.shape[0]
    return _len**2/(np.pi*(_gs/2)**2)


class EarlyStopping:
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


def FindDiffuser(Input, LR_init=1, NPad=10, theta=180, TryR=False, InitDiff=None, RMax=None, DiffSize=None, MaxSteps=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Running on {device}')

    EarlyS = EarlyStopping(Patience=100, Mindelta=0)

    # Recieving and computing numpy arrays
    alpha = theta*2*np.pi/360
    Nangle = (2*np.pi)/(alpha)
    c = np.abs(np.sinc(1/Nangle))/(Nangle-1)

    a02 = Input[0]
    S0 = Input[1]
    S1 = Input[2]

    if DiffSize is None:
        DiffSize = a02.shape[0]

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

    if MaxSteps is None:
        MaxSteps = int(1e5)

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
        pred_ap1 = pred_ap1/torch.mean(torch.abs(pred_ap1))
        pred_am1 = pred_am1/torch.mean(torch.abs(pred_am1))

        S0_pred = torch.abs(pred_a0)**2 + torch.abs(c*pred_am1)**2 + torch.abs(c*pred_ap1)**2
        S1_pred = pred_a0*pred_am1.conj() + pred_ap1*pred_a0.conj()

        a02_pred = torch.abs(pred_a0)**2/torch.mean(torch.abs(pred_a0)**2)
        S0_pred = S0_pred/torch.mean(torch.abs(S0_pred))
        S1_pred = S1_pred/torch.mean(torch.abs(S1_pred))

        loss_a0 = fun.L2_Norm_Torch(a02_pred, a02_target)
        loss_S0 = fun.L2_Norm_Torch(S0_pred, S0_target)
        loss_S1 = fun.L2_Norm_Torch(S1_pred, S1_target)

        loss_total = loss_a0 + loss_S0 + loss_S1

        loss_total.backward()
        optimizer.step()

        TotLoss[_step] = loss_total.item()
        ElapTime[_step] = time.time() - timei

        EarlyS(loss_total.item())
        if TryR and (EarlyS.should_stop and loss_total.item()>3e-1):
            print('Restarting', loss_total.item())
            with torch.no_grad():
                param_diff = 2*np.pi*(torch.rand(DiffSize, DiffSize) - 0.5)
            param_diff = param_diff.clone().detach().requires_grad_(True)
            optimizer = torch.optim.Adam([param_diff], lr=LR_init)
            EarlyS = EarlyStopping(Patience=100, Mindelta=0)
        if EarlyS.should_stop:
            print('Found', loss_total.item())
            TotLoss = TotLoss[:_step+1]
            ElapTime = ElapTime[:_step+1]
            break

    Diff = param_diff.detach().numpy()
    return Diff, TotLoss, ElapTime


def FindBigDiffuser(Input, Div=2, LR_init=1, NPad=10, theta=180, RMax=None, DiffSize=None, MaxSteps=None):
    a02= Input[0]
    S0 = Input[1]
    S1 = Input[2]

    if DiffSize is None:
        DiffSize = a02.shape[0]

    for i in np.flip(range(Div+1)):
        _a02 = CropCenter(a02, DiffSize//(2**i))
        _S0 = CropCenter(S0, DiffSize//(2**i))
        _S1 = CropCenter(S1, DiffSize//(2**i))

        if i == Div:
            Diff, _TotLoss, _ElapTime = FindDiffuser([_a02, _S0, _S1], LR_init, NPad, theta, True, RMax=RMax, DiffSize=DiffSize//(2**Div), MaxSteps=MaxSteps)
        else:
            Diff, _TotLoss, _ElapTime = FindDiffuser([_a02, _S0, _S1], LR_init/2, NPad, theta, False, InitDiff=Diff, RMax=RMax, MaxSteps=MaxSteps)

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


def FindDiffusera0(Input, LR_init=1, NPad=10, theta=180, TryR=False, InitDiff=None, RMax=None, DiffSize=None, MaxSteps=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Running on {device}')

    EarlyS = EarlyStopping(Patience=100, Mindelta=0)

    # Recieving and computing numpy arrays
    alpha = theta*2*np.pi/360
    Nangle = (2*np.pi)/(alpha)
    c = np.abs(np.sinc(1/Nangle))/(Nangle-1)

    a02 = Input[0]
    S0 = Input[1]
    S1 = Input[2]
    a0 = Input[3]

    if DiffSize is None:
        DiffSize = a02.shape[0]

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

    if MaxSteps is None:
        MaxSteps = int(1e5)
        
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
        pred_ap1 = pred_ap1/torch.mean(torch.abs(pred_ap1))
        pred_am1 = pred_am1/torch.mean(torch.abs(pred_am1))

        S0_pred = torch.abs(pred_a0)**2 + torch.abs(c*pred_am1)**2 + torch.abs(c*pred_ap1)**2
        S1_pred = pred_a0*pred_am1.conj() + pred_ap1*pred_a0.conj()

        a02_pred = torch.abs(pred_a0)**2/torch.mean(torch.abs(pred_a0)**2)
        S0_pred = S0_pred/torch.mean(torch.abs(S0_pred))
        S1_pred = S1_pred/torch.mean(torch.abs(S1_pred))

        loss_a0 = fun.L2_Norm_Torch(a02_pred, a02_target)
        loss_S0 = fun.L2_Norm_Torch(S0_pred, S0_target)
        loss_S1 = fun.L2_Norm_Torch(S1_pred, S1_target)

        loss_total = loss_a0 + loss_S0 + loss_S1

        loss_total.backward()
        optimizer.step()

        EnergyMap = torch.abs(torch.fft.fft2(torch.exp(1j*(torch.angle(a0_target) - torch.angle(pred_a0)))))

        TotLoss[_step] = loss_total.item()
        ElapTime[_step] = time.time() - timei
        EnergyLoss[_step] = (torch.max(EnergyMap)/DiffSize**2).item()

        EarlyS(loss_total.item())
        if TryR and (EarlyS.should_stop and loss_total.item()>1):
            print('Restarting', loss_total.item())
            with torch.no_grad():
                param_diff = 2*np.pi*(torch.rand(DiffSize, DiffSize) - 0.5)
            param_diff = param_diff.clone().detach().requires_grad_(True)
            optimizer = torch.optim.Adam([param_diff], lr=LR_init)
            EarlyS = EarlyStopping(Patience=100, Mindelta=0)
        if EarlyS.should_stop:
            print('Found', loss_total.item())
            TotLoss = TotLoss[:_step+1]
            ElapTime = ElapTime[:_step+1]
            EnergyLoss = EnergyLoss[:_step+1]
            break

    Diff = param_diff.detach().numpy()
    return Diff, TotLoss, ElapTime, EnergyLoss


def FindBigDiffusera0(Input, Div=2, LR_init=1, NPad=10, theta=180, RMax=None, DiffSize=None, MaxSteps=None):
    a02= Input[0]
    S0 = Input[1]
    S1 = Input[2]
    a0 = Input[3]

    if DiffSize is None:
        DiffSize = a02.shape[0]

    for i in np.flip(range(Div+1)):
        _a02 = CropCenter(a02, DiffSize//(2**i))
        _S0 = CropCenter(S0, DiffSize//(2**i))
        _S1 = CropCenter(S1, DiffSize//(2**i))
        _a0 = CropCenter(a0, DiffSize//(2**i))

        if i == Div:
            Diff, _TotLoss, _ElapTime, _EnergyLoss = FindDiffusera0([_a02, _S0, _S1, _a0], LR_init, NPad, theta, True, RMax=RMax, DiffSize=DiffSize//(2**Div), MaxSteps=MaxSteps)
        else:
            Diff, _TotLoss, _ElapTime, _EnergyLoss = FindDiffusera0([_a02, _S0, _S1, _a0], LR_init/2, NPad, theta, False, InitDiff=Diff, RMax=RMax, MaxSteps=MaxSteps)

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
