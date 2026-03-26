#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 10:33:51 2026

@author: alberto-razo
"""

import numpy as np


def NormalizeParam(x, N, default=None):
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
    Self = NormalizeParam(Self, N, default=-2)
    if Periodic:
        Coup = NormalizeParam(Coup, N, default=1)
        _H = np.diag(Self) + np.diag(Coup[:-1], 1) + np.diag(Coup[:-1], -1)
        _H[0, -1] = Coup[-1]
        _H[-1, 0] = Coup[-1]
    else:    
        Coup = NormalizeParam(Coup, N-1, default=1)
        _H = np.diag(Self) + np.diag(Coup, 1) + np.diag(Coup, -1)
    return _H