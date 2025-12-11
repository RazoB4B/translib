#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 15:01:02 2025

@author: alberto-razo
"""
import numpy as np
import scipy.constants as sc
from collections.abc import Iterable

def RecWaveVector(ws, index=1, m=0, a=1, n=0, b=1, wnorm=True):
    """
    Takes the angular frequencies and computes the wave vector depending of the size of the
    waveguide, and the respective mode.

    ws: array of angular frequencies
    index: refractive index of the medium 
    m: index associated with the width a
    a: width of the waveguide
    n: index associated with the height b
    b: height of the waveguite 
    if wnorm true: the angular frequencies are weighted by the speed of light. The dimentions
                   a and b should correspond with the ones of c.
    if wnorm false: the wavevector is computed using the speed of light in m/s.

    Since we compute a complex wavevector, the system admits non propative solutions. A warning
    has been added to help.

    References: Classical Electrodynamics, J. Jackson, Chaps: 8
    """
    if wnorm:
        ks = ws*index
    else:
        ks = ws*index/sc.c
    
    # Complex vector that considers also the nonpropagative part
    mprop = m*np.pi/a
    nprop = n*np.pi/b
    if (not isinstance(ks, Iterable) and np.abs(ks)<(mprop+nprop)) or (isinstance(ks, Iterable) and (np.abs(ks)<(mprop+nprop)).any()):
        print(f'The array frequency array considers values out of the frequency cutoff of the mode {m,n}')

    return np.sqrt(ks**2 - mprop**2 - nprop**2 + 0j)
    

def TransMatOneSlabTE(ws, la, lb, tha, m=0, a=1, n=0, b=1, fromA=True, wnorm=True, epsa=None, epsb=None, mua=None, mub=None):
    """
    Computed the transfer matrices in the TE polarization corresponding to a dielectric material 
    contained in another dielectric following the Fresnel and Maxwell equations. If the system is
    quasi 1D, we assumed perfect reflection on the walls.

    ws: array of angular frequencies
    epsa: relative permettivity of the first material
    mua: relative permiability of the first material
    epsb:  relative permittivity of the second material
    mua: relative permiability of the second material
    la: lenght of the first medium in cm
    lb: lenght of the second medium in cm
    tha: incidence angle
    m: index associated with the width a
    a: width of the waveguide in cm
    n: index associated with the height b
    b: height of the waveguite in cm
    if fromA true: the wave starts propagate from the outside medium
    if fromA false: the wave starts propagate from the inside medium
    if wnorm true: the angular frequencies are weighted by the speed of light. The dimentions
                   a and b should correspond with the ones of c.
    if wnorm false: the wavevector is computed using the speed of light in m/s.

    References: Wave propagation, P. Markos and C. Soukoulis, Chaps: 1, 9, 10, 13
    """
    if epsa == None:
        epsa = 1
    if epsb == None:
        epsb = 2
    if mua == None:
        mua = 1
    if mub == None:
        mub = 1

    na = np.sqrt(epsa*mua)
    nb = np.sqrt(epsb*mub)
    
    if not wnorm:
        ws = ws/sc.c    
    thb = np.arcsin((na/nb)*np.sin(tha))

    kas = RecWaveVector(ws, na, m, a, n, b) * np.cos(tha)
    kbs = RecWaveVector(ws, nb, m, a, n, b) * np.cos(thb)

    zab = (mub*kas)/(mua*kbs)
    zba = 1/zab

    Mas = np.zeros([len(ws),2,2], dtype='complex')
    Mabs = np.zeros([len(ws),2,2], dtype='complex')
    Mbs = np.zeros([len(ws),2,2], dtype='complex')
    Mbas = np.zeros([len(ws),2,2], dtype='complex')

    Mas[:,0,0] = np.exp(1j*kas*la)
    Mas[:,1,1] = np.exp(-1j*kas*la)

    Mabs[:,0,0] = (1 + zab)*0.5
    Mabs[:,0,1] = (1 - zab)*0.5
    Mabs[:,1,0] = Mabs[:,0,1]
    Mabs[:,1,1] = Mabs[:,0,0]

    Mbas[:,0,0] = (1 + zba)*0.5
    Mbas[:,0,1] = (1 - zba)*0.5
    Mbas[:,1,0] = Mbas[:,0,1]
    Mbas[:,1,1] = Mbas[:,0,0]

    Mbs[:,0,0] = np.exp(1j*kbs*lb)
    Mbs[:,1,1] = np.exp(-1j*kbs*lb)

    if fromA:
        TMs = np.matmul(Mabs, Mas)
        TMs = np.matmul(Mbs, TMs)
    else:
        TMs = Mbs
    TMs = np.matmul(Mbas, TMs)

    return TMs


def TransMatOneSlabTM(ws, la, lb, tha, m=0, a=1, n=0, b=1, fromA=True, wnorm=True, epsa=None, epsb=None, mua=None, mub=None):
    """
    Computed the transfer matrices in the TM polarization corresponding to a dielectric material
    contained in another dielectric following the Fresnel and Maxwell equations. If the system is
    quasi 1D, we assumed perfect reflection on the walls.

    ws: array of angular frequencies
    epsa: relative permettivity of the first material
    mua: relative permiability of the first material
    epsb:  relative permittivity of the second material
    mua: relative permiability of the second material
    la: lenght of the first medium in cm
    lb: lenght of the second medium in cm
    tha: incidence angle
    m: index associated with the width a
    a: width of the waveguide in cm
    n: index associated with the height b
    b: height of the waveguite in cm
    if fromA true: the wave starts propagate from the outside medium
    if fromA false: the wave starts propagate from the inside medium
    if wnorm true: the angular frequencies are weighted by the speed of light. The dimentions
                   a and b should correspond with the ones of c.
    if wnorm false: the wavevector is computed using the speed of light in m/s.

    References: Wave propagation, P. Markos and C. Soukoulis, Chaps: 1, 9, 10, 13
    """
    if epsa == None:
        epsa = 1
    if epsb == None:
        epsb = 1
    if mua == None:
        mua = 1
    if mub == None:
        mub = 2

    na = np.sqrt(epsa*mua)
    nb = np.sqrt(epsb*mub)

    if not wnorm:
        ws = ws/sc.c
    thb = np.arcsin((na/nb)*np.sin(tha))

    kas = RecWaveVector(ws, na, m, a, n, b) * np.cos(tha)
    kbs = RecWaveVector(ws, nb, m, a, n, b) * np.cos(thb)

    zab = (epsb*kas)/(epsa*kbs)
    zba = 1/zab

    Mas = np.zeros([len(ws),2,2], dtype='complex')
    Mabs = np.zeros([len(ws),2,2], dtype='complex')
    Mbs = np.zeros([len(ws),2,2], dtype='complex')
    Mbas = np.zeros([len(ws),2,2], dtype='complex')

    Mas[:,0,0] = np.exp(1j*kas*la)
    Mas[:,1,1] = np.exp(-1j*kas*la)

    Mabs[:,0,0] = (1 + zab)*0.5
    Mabs[:,0,1] = (1 - zab)*0.5
    Mabs[:,1,0] = Mabs[:,0,1]
    Mabs[:,1,1] = Mabs[:,0,0]

    Mbas[:,0,0] = (1 + zba)*0.5
    Mbas[:,0,1] = (1 - zba)*0.5
    Mbas[:,1,0] = Mbas[:,0,1]
    Mbas[:,1,1] = Mbas[:,0,0]

    Mbs[:,0,0] = np.exp(1j*kbs*lb)
    Mbs[:,1,1] = np.exp(-1j*kbs*lb)

    if fromA:
        TMs = np.matmul(Mabs, Mas)
        TMs = np.matmul(Mbs, TMs)
    else:
        TMs = Mbs
    TMs = np.matmul(Mbas, TMs)

    return TMs
