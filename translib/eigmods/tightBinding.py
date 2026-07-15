#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 12:52:41 2026

@author: alberto-razo
"""

import numpy as np


def EigVec(H):
    val, vec = np.linalg.eig(H)
    _ind = np.argsort(np.real(val))
    val = val[_ind]
    vec = vec[:, _ind]
    
    return val, vec


def LocLengthPer(E, H):
    S = 0
    _v = np.random.randn(2) + 1j*np.random.randn(2)
    _v /= np.linalg.norm(_v)
    _TM = np.zeros([2,2], dtype='complex')
    for i in range(len(H)):
        if i == 0:
            _TM[0, 0] = (E-H[0, 0])/H[1, 0]
            _TM[0, 1] = -H[0, -1]/H[1, 0]
        elif i == len(H)-1:
            _TM[0, 0] = (E-H[-1, -1])/H[-1, 0]
            _TM[0, 1] = -H[-1, -2]/H[-1, 0]
        else:
            _TM[0, 0] = (E-H[i, i])/H[i+1, i]
            _TM[0, 1] = -H[i, i-1]/H[i+1, i]
        _TM[1, 0] = 1
        _TM[1, 1] = 0
        
        _v = np.matmul(_TM, _v)

        _r = np.linalg.norm(_v)
        _v = _v/_r
        
        S = S + np.log(_r)
    gamma = S/len(H)
    return 1/gamma


def LocLengthOpen(Es, On):
    S = np.zeros([len(Es)])
    _v = np.random.randn(2) + 1j*np.random.randn(2)
    _v /= np.linalg.norm(_v)
    _v = np.tile(_v, (len(Es), 1))
    for i in range(len(On)):
        _TMs = np.zeros([len(Es), 2, 2], dtype='complex')
        _TMs[:, 0, 0] = (Es-On[i])
        _TMs[:, 0, 1] = -1
        _TMs[:, 1, 0] = 1
        
        _v = np.matmul(_TMs, _v[..., None])[..., 0]
        _r = np.linalg.norm(_v, axis=1)
        _v /= _r[:, None]
        
        S = S + np.log(_r)
    gamma = S/len(On)
    return 1/gamma


def ScatQuan(Es, On, leadC=10):
    ks = np.arccos(-Es*0.5/leadC)
    On = np.append(0, np.append(On, 0))
    
    P0 = np.zeros([len(Es), 2, 2], dtype='complex')
    P0[:, 0, 0] = np.exp(1j*ks)
    P0[:, 0, 1] = np.exp(-1j*ks)
    P0[:, 1, 0] = 1
    P0[:, 1, 1] = 1
    
    PNinv = np.zeros([len(Es), 2, 2], dtype='complex')
    PNinv[:, 0, 0] = np.exp(-1j*ks*len(On))/(2*1j*np.sin(ks))
    PNinv[:, 0, 1] = -np.exp(-1j*ks*(len(On)+1))/(2*1j*np.sin(ks))
    PNinv[:, 1, 0] = -np.exp(1j*ks*len(On))/(2*1j*np.sin(ks))
    PNinv[:, 1, 1] = np.exp(1j*ks*(len(On)+1))/(2*1j*np.sin(ks))
    
    for i in range(len(On)):
        _TMs = np.zeros([len(Es), 2, 2], dtype='complex')
        if i == 0:
            _TMs[:, 0, 0] = -(Es-On[0])
            _TMs[:, 0, 1] = -leadC
        elif i == len(On)-1:
            _TMs[:, 0, 0] = -(Es-On[-1])/leadC
            _TMs[:, 0, 1] = -1/leadC
        else:
            _TMs[:, 0, 0] = -(Es-On[i])
            _TMs[:, 0, 1] = -1
        _TMs[:, 1, 0] = 1
        _TMs[:, 1, 1] = 0
        
        if i==0:
            TMs = _TMs
        else:
            TMs = np.matmul(_TMs, TMs)

    TMs = np.matmul(PNinv, np.matmul(TMs, P0))

    S = np.zeros([len(Es), 2, 2], dtype='complex')
    S[:, 0, 0] = -TMs[:, 1, 0]/TMs[:, 1, 1]
    S[:, 0, 1] = 1/TMs[:, 1, 1]
    S[:, 1, 0] = (TMs[:,0,0]*TMs[:,1,1] - TMs[:,0,1]*TMs[:,1,0])/TMs[:, 1, 1]
    S[:, 1, 1] = TMs[:, 0, 1]/TMs[:, 1, 1]
    return S


def TimeDelay(Es, On, DE, leadC=10):
    nEne = len(Es)
    EsP = Es + DE/2
    EsM = Es - DE/2
    Es = np.append(EsM, EsP)
    Es = np.sort(Es)
    
    S = ScatQuan(Es, On, leadC=leadC)
    detS = S[:,0,0]*S[:,1,1]-S[:,0,1]*S[:,1,0]
    t = S[:,0,1]
    r = S[:,0,0]
    
    tdS = np.zeros([nEne])
    tdt = np.zeros([nEne])
    tdr = np.zeros([nEne])
    for i in range(nEne):
        tdS[i] = np.angle(detS[2*i+1]*np.conjugate(detS[2*i]))/(2*DE)
        tdt[i] = np.angle(t[2*i+1]*np.conjugate(t[2*i]))/(2*DE)
        tdr[i] = np.angle(r[2*i+1]*np.conjugate(r[2*i]))/(2*DE)
    return tdS, tdt, tdr


def IPR(State, axis=0):
    return np.sum(np.abs(State)**4, axis=axis)/np.sum(np.abs(State)**2, axis=axis)**2


def BioIPR(StateL, StateR, axis=0):
    return np.sum(np.abs(StateL*StateR)**2, axis=axis)/(np.sum(np.abs(StateL)**2, axis=axis)*np.sum(np.abs(StateR)**2, axis=axis))


def ShannonEnt(State, axis=0):
    return -np.sum(np.abs(State)**2*np.log(np.abs(State)**2), axis=axis)


def BioShannonEnt(StateL, StateR, axis=0):
    return -np.sum(np.abs(StateL*StateR)*np.log(np.abs(StateL*StateR)), axis=axis)