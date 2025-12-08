#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 18:20:48 2025

@author: alberto-razo
"""

import torch
import numpy as np
from scipy.signal import argrelmax


def Corr(_Arr1, _Arr2):
    """
    Computes correlation of two arrays
    
    Arr1: the array 1
    Arr2: the array 2
    """
    _Arr1 = _Arr1 - np.mean(_Arr1)
    _Arr2 = _Arr2 - np.mean(_Arr2)
    _corr = np.fft.ifft2(np.conjugate(np.fft.fft2(_Arr1))*np.fft.fft2(_Arr2))
    _corr = np.fft.fftshift(_corr)
    return _corr


def FindVortex(_PhaseArray):
    """
    Given a phase array, finds the vortex (points where the phase is not defined)
    
    PhaseArray: the phase array 
    """
    _lenX, _lenY = _PhaseArray.shape
    _VP = []
    _VM = []
    for i in range(_lenX-1):
        for j in range(_lenY-1):
            _dp = np.unwrap([_PhaseArray[i,j], _PhaseArray[i,j+1], _PhaseArray[i+1,j+1],
                             _PhaseArray[i+1,j], _PhaseArray[i,j]])
            if np.abs(np.abs(_dp[-1]-_dp[0])-2*np.pi)<1e-2 and (i!=0 and j!=0):
                if _dp[-1]-_dp[0] > 0:
                    _VM.append([j, i])
                else:
                    _VP.append([j, i])
    return _VP, _VM


def FindMax(_IntensArray, _radius=25):
    """
    Given a lanscape array, finds the maxima with a certain radius
    
    IntensArray: the lanscape array 
    radius: the exclusion radius
    """
    _len = _IntensArray.shape[0]
    _vec = np.arange(_len)
    _mesh = np.meshgrid(_vec, _vec)

    _Max = []
    for i in _vec:
        _max = argrelmax(_IntensArray[i, :], order=_radius-1)[0]
        for j in _max:
            _r = np.sqrt((_mesh[0]-j)**2 + (_mesh[1]-i)**2)
            _inds = np.where(_r < _radius)
            if np.max(_IntensArray[_inds]) <= _IntensArray[i, j]:
                _Max.append([j, i])
    return _Max


def FindMin(_arr, _radius=25):
    """
    Given a lanscape array, finds the minima with a certain radius
    
    IntensArray: the lanscape array 
    radius: the exclusion radius
    """
    return FindMax(1/_arr, _radius)


def FindDistances(_Pos1, _Pos2, _metric=None):
    """
    Given two set of positions, computes the distribution of minimum distances 
    between both sets, starting from the set 1
    
    Pos1: the array of positions 1
    Pos2: the array of positions 2
    metric: if given, weight the elements of the distribution by this value
    """
    _dist = []
    for _v in _Pos1:
        _dist.append(np.inf)
        for _m in _Pos2:
            _d = np.sqrt((_m[0]-_v[0])**2 + (_m[1]-_v[1])**2)
            if _d < _dist[-1]:
                _dist[-1] = _d

    if _metric != None:
        _dist = _dist/_metric
    return _dist


def RadialHistogram(_dist, _bins):
    """
    Given a distribution and and axis, computes the radial probability in that
    axis
    
    dist: the distribution
    bins: the axis
    """
    _hist, _edge = np.histogram(_dist, bins=_bins, density=True)
    _edge = (_edge[:-1] + _edge[1:])/2
    _hist = _hist/_edge
    _hist = _hist/np.trapezoid(_hist*2*np.pi*_edge, _edge)
    return _hist, _edge


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