#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 17:45:22 2026

@author: alberto-razo
"""
import numpy as np
import translib.functions as fun
import translib.optics.maskOpts as mo
from scipy.special import jn
from scipy.optimize import curve_fit
from scipy.ndimage import map_coordinates


def GetExperimentDMD(Diffuser, angle, nfigs, deph=0, NPad=5, clock=True, WinSize=None, Max=None, Min=None):
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
    if WinSize is None:
        WinSize = N
    Masks = mo.SectionMask(N, nfigs, angle, deph, clock, Max=Max, Min=Min)
    speckles = []
    for _mask in Masks:
        _out = GetFarField(_mask*Diffuser, NPad, WinSize)
        speckles.append(np.abs(_out)**2)
    return speckles


def GetFarField(array, Npad=5, WinSize=None, Scale=None):
    '''
    Computes the far field propagation of a given array

    Parameters
    ----------
    array : 2D complex float array
            The array to propagate
    NPad : Float
           The size of the padded (N*len(array)) figure used to compute 
           the Fourier transform
    WinSize : Integer
              the size of the final figure
    Scale : Float
            Scales used to resample the image

    Returns
    -------
    2D float : A 2D array given as the Fourier transform of array
    '''
    n = len(array)
    if WinSize is None:
        WinSize = n
    if Scale is None:
        Scale = 1
    pad = int((Npad-1)*n)//2
    array = np.pad(array, pad, mode='constant')
    array = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(array)))
    array = Resample(array, Scale)
    return CropCenter(array, WinSize)


def CropCenter(array, crop):
    '''
    Crops the central section of a given array

    Parameters
    ----------
    array : 2D complex float array
            The array to crop
    crop : Integer
           Number of pixels size we want extract

    Returns
    -------
    2D float array : A 2D array given as the center of the original
                     one with size crop*crop
    '''
    x, y = array.shape
    startx = x//2-(crop//2)
    starty = y//2-(crop//2)
    return array[startx:startx+crop, starty:starty+crop]


def Resample(Array, Scale):
    '''
    Resamples a given 2D array to certain scale

    Parameters
    ----------
    Array : 2D complex float array
            The array to resample
    Scale : Integer
            the value to rescale the array

    Returns
    -------
    2D float array : A 2D array of the same size of the original but
                     stretch according to Scale
    '''
    H, W = Array.shape

    # Build coordinate grid
    y = np.linspace(-1, 1, H)
    x = np.linspace(-1, 1, W)
    Y, X = np.meshgrid(y, x, indexing='ij')

    # Scale coordinates
    Xs = X/Scale
    Ys = Y/Scale

    # Convert normalized [-1,1] → pixel coordinates [0, H-1]
    Xp = (Xs + 1) * (W - 1) / 2
    Yp = (Ys + 1) * (H - 1) / 2
    coords = np.vstack([Yp.ravel(), Xp.ravel()])

    # Interpolate real and imaginary separately
    real_part = map_coordinates(
        Array.real, coords, order=1, mode='constant', cval=0).reshape(H, W)
    imag_part = map_coordinates(
        Array.imag, coords, order=1, mode='constant', cval=0).reshape(H, W)
    return real_part + 1j * imag_part


def GetFarDiffuser(array, Npad=5, WinSize=None, Scale=None):
    '''
    Computes the far field propagation of a given array

    Parameters
    ----------
    array : 2D complex float array
            The array to backpropagate
    NPad : Float
           The size of the padded (N*len(array)) figure used to compute 
           the Fourier transform
    WinSize : Integer
              the size of the final figure
    Scale : Float
            Scales used to resample the image

    Returns
    -------
    2D float : A 2D array given as the inverse Fourier transform of array
    '''
    n = len(array)
    if WinSize is None:
        WinSize = n
    if Scale is None:
        Scale = 1
    pad = int((Npad-1)*n)//2
    array = np.pad(array, pad, mode='constant')
    array = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(array)))
    array = Resample(array, Scale)
    return CropCenter(array, WinSize)


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