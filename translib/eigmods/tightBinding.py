#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 12:52:41 2026

@author: alberto-razo
"""

import numpy as np
from scipy.sparse.linalg import expm_multiply


def EigVec(H, mode=None):
    val, vec = np.linalg.eig(H)    
    _ind = np.argsort(np.real(val))
    val = val[_ind]
    vec = vec[:, _ind]
    return val, vec


def EigVecHerm(H):
    val, vec = np.linalg.eigh(H)
    _ind = np.argsort(np.real(val))
    val = val[_ind]
    vec = vec[:, _ind]
    return val, vec


def IPR(State, axis=0):
    return np.sum(np.abs(State)**4, axis=axis)/np.sum(np.abs(State)**2, axis=axis)**2


def BioIPR(StateL, StateR, axis=0):
    return np.sum(np.abs(StateL*StateR)**2, axis=axis)/(np.sum(np.abs(StateL)**2, axis=axis)*np.sum(np.abs(StateR)**2, axis=axis))


def ShannonEnt(State, axis=0):
    p = np.abs(State)**2
    mask = p > 0
    return -np.sum(p[mask]*np.log(p[mask]), axis=axis)


def BioShannonEnt(StateL, StateR, axis=0):
    p = np.abs(StateL*StateR)
    mask = p > 0
    return -np.sum(p[mask]*np.log(p[mask]), axis=axis)


def TimeEvolution(H, start, end, npoints, center, width):
    x = np.linspace(0, len(H)-1, len(H)) 
    IniState = np.exp(-(x-center)**2/(2*width**2))
    IniState /= np.linalg.norm(IniState)
    
    return expm_multiply(-1j*H, IniState, start, end, npoints)