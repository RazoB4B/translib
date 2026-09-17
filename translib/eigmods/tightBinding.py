#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 12:52:41 2026

@author: alberto-razo
"""

import numpy as np
from scipy.sparse.linalg import expm_multiply


def EigVec(H):
    val, vec = np.linalg.eig(H)
    _ind = np.argsort(np.real(val))
    val = val[_ind]
    vec = vec[:, _ind]
    
    return val, vec


def EigVecHerm(H):
    val, vec = np.linalg.eig(H)
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


def LocLengthPer(Es, H):
    S = np.zeros([len(Es)])
    _v = np.random.randn(2) + 1j*np.random.randn(2)
    _v /= np.linalg.norm(_v)
    for i in range(len(H)):
        _TMs = np.zeros([len(Es), 2, 2], dtype='complex')
        if i == 0:
            _TMs[:, 0, 0] = (Es-H[0, 0])/H[0, 1]
            _TMs[:, 0, 1] = -H[0, -1]/H[0, 1]
        elif i == len(H)-1:
            _TMs[:, 0, 0] = (Es-H[-1, -1])/H[-1, 0]
            _TMs[:, 0, 1] = -H[-1, -2]/H[-1, 0]
        else:
            _TMs[:, 0, 0] = (Es-H[i, i])/H[i, i+1]
            _TMs[:, 0, 1] = -H[i, i-1]/H[i, i+1]
        _TMs[:, 1, 0] = 1
        
        _v = np.matmul(_TMs, _v[..., None])[..., 0]
        _r = np.linalg.norm(_v, axis=1)
        _v /= _r[:, None]
        
        S = S + np.log(_r)
    gamma = S/len(H)
    return 1/gamma


def LocLengthOpen(Es, H, leadC=-1):
    S = np.zeros([len(Es)])
    _v = np.random.randn(2) + 1j*np.random.randn(2)
    _v /= np.linalg.norm(_v)
    _v = np.tile(_v, (len(Es), 1))
    for i in range(len(H)):
        _TMs = np.zeros([len(Es), 2, 2], dtype='complex')
        if i == 0:
            _TMs[:, 0, 0] = (Es-H[0, 0])/H[0, 1]
            _TMs[:, 0, 1] = -leadC/H[0, 1]
        elif i == len(H)-1:
            _TMs[:, 0, 0] = (Es-H[-1, -1])/leadC
            _TMs[:, 0, 1] = -H[-1, -2]/leadC
        else:
            _TMs[:, 0, 0] = (Es-H[i, i])/H[i, i+1]
            _TMs[:, 0, 1] = -H[i, i-1]/H[i, i+1]
        _TMs[:, 1, 0] = 1
        
        _v = np.matmul(_TMs, _v[..., None])[..., 0]
        _r = np.linalg.norm(_v, axis=1)
        _v /= _r[:, None]
        
        S = S + np.log(_r)
    gamma = S/len(H)
    return 1/gamma


def ScatMat_Trans(Es, H, leadC=-10, leadsysC=-1):
    ks = np.arccos(Es*0.5/leadC)
    
    TH = np.zeros([len(H) + 2, len(H) + 2], dtype=H.dtype)
    TH[1:len(H)+1, 1:len(H)+1] = H
    TH[0, 1] = leadsysC
    TH[1, 0] = leadsysC
    TH[-1, -2] = leadsysC
    TH[-2, -1] = leadsysC
    
    P0 = np.zeros([len(Es), 2, 2], dtype='complex')
    P0[:, 0, 0] = np.exp(1j*ks)
    P0[:, 0, 1] = np.exp(-1j*ks)
    P0[:, 1, 0] = 1
    P0[:, 1, 1] = 1
    
    PNinv = np.zeros([len(Es), 2, 2], dtype='complex')
    PNinv[:, 0, 0] = np.exp(-1j*ks*len(H))/(2*1j*np.sin(ks))
    PNinv[:, 0, 1] = -np.exp(-1j*ks*(len(H)+1))/(2*1j*np.sin(ks))
    PNinv[:, 1, 0] = -np.exp(1j*ks*len(H))/(2*1j*np.sin(ks))
    PNinv[:, 1, 1] = np.exp(1j*ks*(len(H)+1))/(2*1j*np.sin(ks))
    
    for i in range(len(TH)):
        _TMs = np.zeros([len(Es), 2, 2], dtype='complex')
        if i == 0:
            _TMs[:, 0, 0] = (Es-TH[0, 0])/TH[0, 1]
            _TMs[:, 0, 1] = -leadC/TH[0, 1]
        elif i == len(TH)-1:
            _TMs[:, 0, 0] = (Es-TH[-1, -1])/leadC
            _TMs[:, 0, 1] = -TH[-1, -2]/leadC
        else:
            _TMs[:, 0, 0] = (Es-TH[i, i])/TH[i, i+1]
            _TMs[:, 0, 1] = -TH[i, i-1]/TH[i, i+1]
        _TMs[:, 1, 0] = 1
        
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
   
    
    S[:, 0, 1] *= np.exp(1j*ks*(len(H)-1))
    S[:, 1, 0] *= np.exp(1j*ks*(len(H)-1))
#    S[:, 1, 1] *= np.cos(2*ks)*np.exp(2j*ks*(len(H)))
#    S[:, 0, 0] *= np.cos(2*ks)*np.exp(-2j*ks)
    # When comparing this results with the Green's function, a different pahse convention is
    # considered, resulting into different phase factors
    return S


def ScatMat_Green(Es, H, leadC=-10, leadsysC=-1):
    ks = np.arccos(Es*0.5/leadC)
    Gamma = -2*leadC*np.sin(ks)
    
    TH = np.zeros([len(H) + 2, len(H) + 2], dtype=H.dtype)
    TH[1:len(H)+1, 1:len(H)+1] = H
    TH[0, 1] = leadsysC
    TH[1, 0] = leadsysC
    TH[-1, -2] = leadsysC
    TH[-2, -1] = leadsysC
    
    Sigma = (Es-1j*np.sqrt(4*leadC**2 - Es**2))*0.5
    GNN = 1/(Es - TH[0, 0] - Sigma)
    
    G11 = GNN
    G1N = GNN
    GN1 = GNN
    
    for i in range(1,len(TH)-1):
        GNN_new = 1/(Es - TH[i, i] - TH[i, i-1]*TH[i-1, i]*GNN)
        G11 = G11 + TH[i, i-1]*TH[i-1, i]*GN1*G1N*GNN_new
        G1N = TH[i-1, i]*G1N*GNN_new
        GN1 = TH[i, i-1]*GN1*GNN_new
        GNN = GNN_new

    GNN = 1/(Es - TH[-1, -1] - Sigma - TH[-1, -2]*TH[-2, -1]*GNN)
    G11 = G11 + TH[-1, -2]*TH[-2, -1]*GN1*G1N*GNN
    G1N = TH[-2, -1]*G1N*GNN
    GN1 = TH[-1, -2]*GN1*GNN
    
    S = np.zeros([len(Es), 2, 2], dtype='complex')
    S[:, 0, 0] = 1-1j*Gamma*G11
    S[:, 0, 1] = 1j*Gamma*GN1
    S[:, 1, 0] = 1j*Gamma*G1N
    S[:, 1, 1] = 1-1j*Gamma*GNN
    return S


def Amplitude_Trans(Es, H, leadC=-10, leadsysC=-1):
    rs = ScatMat_Trans(Es, H, leadC)
    rs = rs[:, 0, 0]
    
    ks = np.arccos(Es*0.5/leadC)
    
    v = np.array([np.exp(1j*ks) + rs*np.exp(-1j*ks), 1 + rs]).T
    vs = np.zeros([len(Es), len(H)+3], dtype=complex)
    vs[:, 0] = v[:, 0]
    
    TH = np.zeros([len(H) + 2, len(H) + 2], dtype=H.dtype)
    TH[1:len(H)+1, 1:len(H)+1] = H
    TH[0, 1] = leadsysC
    TH[1, 0] = leadsysC
    TH[-1, -2] = leadsysC
    TH[-2, -1] = leadsysC
    
    for i in range(len(TH)):
        _TMs = np.zeros([len(Es), 2, 2], dtype='complex')
        if i == 0:
            _TMs[:, 0, 0] = (Es-TH[0, 0])/TH[0, 1]
            _TMs[:, 0, 1] = -leadC/TH[0, 1]
        elif i == len(TH)-1:
            _TMs[:, 0, 0] = (Es-TH[-1, -1])/leadC
            _TMs[:, 0, 1] = -TH[-1, -2]/leadC
        else:
            _TMs[:, 0, 0] = (Es-TH[i, i])/TH[i, i+1]
            _TMs[:, 0, 1] = -TH[i, i-1]/TH[i, i+1]
        _TMs[:, 1, 0] = 1
        
        v = np.matmul(_TMs, v[..., None])[..., 0]
        vs[:, i+1] = v[:, 0]
    return vs[:, :-1]


def TimeDelay(Es, H, DE, leadC=-10, leadsysC=-1):
    nEne = len(Es)
    EsP = Es + DE/2
    EsM = Es - DE/2
    Es = np.append(EsM, EsP)
    Es = np.sort(Es)
    
    S = ScatMat_Trans(Es, H, leadC, leadsysC)
    detS = S[:,0,0]*S[:,1,1]-S[:,0,1]*S[:,1,0]
    r = S[:,0,0]
    t = S[:,1,0]
    
    tdS = np.zeros([nEne])
    tdt = np.zeros([nEne])
    tdr = np.zeros([nEne])
    for i in range(nEne):
        tdS[i] = np.angle(detS[2*i+1]*np.conjugate(detS[2*i]))/(2*DE)
        tdt[i] = np.angle(t[2*i+1]*np.conjugate(t[2*i]))/(2*DE)
        tdr[i] = np.angle(r[2*i+1]*np.conjugate(r[2*i]))/(2*DE)
    return tdS, tdt, tdr


def TimeExitation(H, start, end, npoints, center, width):
    x = np.linspace(0, len(H)-1, len(H)) 
    IniState = np.exp(-(x-center)**2/(2*width**2))
    IniState /= np.linalg.norm(IniState)
    
    return expm_multiply(-1j*H, IniState, start, end, npoints)

#%%


def TimeExitationPh(H, tmax, Nsites):
    IniState = np.zeros([len(H)])
    for i in range(Nsites):
        IniState[2*i] = (-1)**i
    
    return expm_multiply(-1j*tmax*H, IniState)