#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 10:33:51 2026

@author: alberto-razo
"""

import numpy as np


def NormalizeParam(x, N, default=None):
    '''
    Auxiliar function that allows to fill the one D hamiltonian easily
    '''
    if x is None:
        return np.full(N, default)
    if np.isscalar(x):
        return np.full(N, x)
    x = np.asarray(x)
    if len(x) == 1:
        return np.full(N, x[0])
    if len(x) == N:
        return x
    raise ValueError(f"Parameter must be scalar, length 1, or length {N}")


def OneDFirstNeigh(N, Self=None, Coup=None, Periodic=False): 
    Self = NormalizeParam(Self, N, default=0)
    if Periodic:
        Coup = NormalizeParam(Coup, N, default=-1)
        _H = np.diag(Self) + np.diag(Coup[:-1], 1) + np.diag(Coup[:-1], -1)
        _H[0, -1] = Coup[-1]
        _H[-1, 0] = Coup[-1]
    else:    
        Coup = NormalizeParam(Coup, N-1, default=-1)
        _H = np.diag(Self) + np.diag(Coup, 1) + np.diag(Coup, -1)
    return _H


def ContCorrDis(x, sigma, V0, Seed=None):
    if Seed is None:
        Seed = np.random.randint(low=100)

    N = len(x)
    dx = x[1] - x[0]

    # Fourier wavevectors
    k = 2 * np.pi * np.fft.fftfreq(N, d=dx)

    # White Gaussian noise
    np.random.seed(Seed)
    white = np.random.normal(0, 1, N)

    # Fourier transform
    W = np.fft.fft(white)

    # Amplitude filter corresponding to
    # C(r) = V0^2 exp[-r^2/(2 sigma^2)]
    filter_k = np.exp(-0.25 * sigma**2 * k**2)

    # Apply filter
    V = np.fft.ifft(W * filter_k).real

    # Normalize to the desired RMS amplitude V0
    V -= np.mean(V)
    V *= V0 / np.std(V)
    return V