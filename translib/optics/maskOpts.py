#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 14:58:15 2025

@author: alberto-razo
"""
import math
import numpy as np
from numba import njit, prange


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
        
    _img = _Imgs
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

    Imgs: Array of data with time dependence (Odd number)
    Nharm: Number of Harmonics that will be extracted from the data
    Npad: Number of periods after padding
    Sym: if True extract the harmonics [-n, n]
    axis: axis of 'Imgs' that represents the time dependence 
    Nper: The number of time cycles in the original signal
    """
    if np.mod(len(_Imgs), 2) == 0:
        print('Number of images is even, please try with an odd number')
    
    if _Nper != 1:
        _Imgs = np.array_split(_Imgs, _Nper, axis=_axis)[0]
        
    if np.mod(_Npad, 2) == 0:
        _Npad += 1
        
    _img = _Imgs
    for i in range(_Npad-1):
        _Imgs = np.append(_Imgs, _img, axis=_axis)
    del _img
    
    _Ntot = len(_Imgs)
    _harms = np.fft.fftshift(np.fft.fft(_Imgs, axis=_axis), axes=_axis)
    
    _frqs = np.linspace(0, _Npad, _Ntot)
    _frqs = _frqs[1] - _frqs[0]
    _frqs = np.linspace(-0.5/_frqs, 0.5/_frqs, _Ntot)

    _inds = []
    for i in range(-_Nharms, _Nharms+1):
        _inds = np.append(_inds, np.where(np.abs(i-_frqs)<1e-6))

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
    '''
    Generates spiral phase mask exp(i n phi)

    Parameters
    ----------
    Size : Integer
           Size of the array side
    ordmax : Integer
             Maximum order of the mask
    Sym : Boolean
          If True, generate mask considering [-n, n]
    Max : Float in [0, 2]
          Cuts the values with r>Max
    Min : Float in [0, 2]
          Cuts the values with r<Min

    Returns
    -------
    2D float : A stack of 2D arrays with an amplitude step function 
               defined by Min and Max, and with phases exp(i n phi)
    '''
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
    cthetas = np.linspace(-np.pi, np.pi, nfigs+1) + deph
    cthetas = cthetas[:-1]

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


@njit(parallel=True, fastmath=True, cache=True)
def get_lee_holo(complex_pattern, period, center, angle, nbits):
    """
    Compute a binary Lee hologram mask from a complex pattern.
    Assumes |complex_pattern| <= 1.
    
    Parameters:
      complex_pattern : 2D np.complex64 array.
      period          : grating period.
      center          : tuple (center_y, center_x).
      angle           : grating angle (radians).
      nbits           : bit-depth.
      
    Returns:
      2D np.uint8 mask.
    """
    height, width = complex_pattern.shape
    omega = 2.0 * math.pi / period
    mask = np.empty((height, width), dtype=np.uint8)
    max_val = (1 << nbits) - 1  # equivalent to 2**nbits - 1
    cy, cx = center
    sin_angle = math.sin(angle)
    cos_angle = math.cos(angle)
    
    for i in prange(height):
        y = i - cy
        for j in range(width):
            x = j - cx
            tilt_angle = -sin_angle * x + cos_angle * y
            amp = abs(complex_pattern[i, j]) + math.asin(math.cos(tilt_angle * omega)) / math.pi
            amplitude_term = 1 if amp > 0.5 else 0
            phase_pattern = math.atan2(complex_pattern[i, j].imag, complex_pattern[i, j].real)
            phase_arg = (cos_angle * x + sin_angle * y) * omega - phase_pattern
            phase_term = 1 if (1.0 + math.cos(phase_arg)) / 2.0 > 0.5 else 0
            mask[i, j] = max_val if amplitude_term * phase_term else 0
    return mask


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