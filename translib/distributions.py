#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 17:02:38 2025

@author: alberto-razo
"""
import numpy as np
import scipy.special as sc


def IntensChaos(_I, _phaseRigidity):
    '''
    Evaluates the statistical distribution of intensities predicted from RMT and
    considering the phase rigidity as a parameter. See Eq. 16 of PRE 93, 032108 (2016).
    
    I: The intensity domain
    phaseRigidity: The phase rigidity parameter
    '''
    _f = 1 - _phaseRigidity**2
    _P = (1/np.sqrt(_f))*np.exp(-_I/_f)*sc.i0(_phaseRigidity*_I/_f)
    return _P


def PhaseChaos(_phi, _complexness):
    '''
    Evaluates the statistical distribution of phases predicted from RMT and considering 
    the complexness as a parameter. See Eq. 4 of PRE 80, 035201(R) (2009).
    
    phi: The phase domain
    complexness: The complexness parameter
    '''
    return _complexness/(_complexness**2 * np.cos(_phi)**2 + np.sin(_phi)**2)/(2*np.pi)


def IntensOneFDSpeckle(_I, _Im):
    '''
    Evaluates the statistical distribution of intensities for a fully developed
    speckle pattern predicted considering random phasors with the average intensity 
    as parameter. See Eq. 3.15 of Goodman, Speckle Phenomena in Optics (2020)
    
    phi: The phase domain
    complexness: The complexness parameter
    '''
    return (1/_Im)*np.exp(-_I/_Im)


def Trans1DDisord(_G, _s):
    '''
    Evaluates the statistical distribution of the transmission through a 1D disordered
    system with the adimensional length as parameter. 
    See Eq. 2 of PRB 88, 205414 (2013)
    
    G: The transmission domain
    s: The adimensional length
    '''
    _G, _s = np.meshgrid(_G, _s, indexing="ij")
    _p = np.sqrt(np.arccosh(1/np.sqrt(_G)))*np.exp(-(1/_s)*(np.arccosh(1/np.sqrt(_G))**2))
    _p = _p/(np.sqrt(_G**3)*((1-_G)**(0.25)))
    _p = _p/np.trapezoid(_p, _G, axis=0)
    return _p


def Trans1DDisordSqrt(_G2, _s):
    '''
    Evaluates the statistical distribution of the transmission squared through 
    a 1D disordered system with the adimensional length as parameter. 
    See Eq. 2 of PRB 88, 205414 (2013)
    
    G: The transmission domain
    s: The adimensional length
    '''
    _G = np.sqrt(_G2)
    _p = Trans1DDisord(_G, _s)[:,0]/(2*_G)
    _p = _p/np.trapezoid(_p, _G2)
    return _p


def LogTrans1DDisord(_lnG, _s):
    '''
    Evaluates the statistical distribution of the logarithm of the transmission 
    through a 1D disordered system with the adimensional length as parameter. 
    See Eq. 2 of PRB 88, 205414 (2013)
    
    lnG: The logarithm of the transmission domain
    s: The adimensional length
    '''
    _G = np.exp(_lnG)
    _p = _G*Trans1DDisord(_G, _s)[:,0]
    _p = _p/np.trapezoid(_p, _lnG)
    return _p


def LogTrans1DDisordSqrt(_lnG, _s):
    '''
    Evaluates the statistical distribution of the logarithm of the transmission 
    squared through a 1D disordered system with the adimensional length as parameter. 
    See Eq. 2 of PRB 88, 205414 (2013)
    
    lnG: The logarithm of the transmission domain
    s: The adimensional length
    '''
    _G = np.exp(_lnG)
    _p = _G*Trans1DDisordSqrt(_G, _s)
    _p = _p/np.trapezoid(_p, _lnG)
    return _p


def Inten1DDisordNeupane(_x, _s, _etaL=1e1):
    '''
    Evaluates intensity as function of the position inside a 1D disordered system
    See Eq. 7 of PRB 92, 014207 (2015)
    
    x: The spatial domain
    s: The adimensional length
    etaL: Limits of the integral (Originally from -infinity to infinity)
    '''
    _eta = np.linspace(-_etaL, _etaL, 10001)
    X, Eta = np.meshgrid(_x, _eta, indexing="ij")
    _fun = np.exp(-(Eta-(X-0.5)*_s)**2/_s)*(np.tanh(Eta) + Eta/np.cosh(Eta)**2)
    return 1-np.sqrt(1/(_s*np.pi))*np.trapezoid(_fun, _eta)


def Inten1DDisordMello(_x, _s, _NG=1001):
    '''
    Evaluates intensity as function of the position inside a 1D disordered system
    See Eq. (11a) of Physica E 82, 261-265 (2016)
    
    x: The spatial domain
    s: The adimensional length
    etaL: Limits of the integral (Originally from -infinity to infinity)
    '''
    s1 = _s*_x
    s2 = _s*(1-_x)

    G = np.linspace(0, 1, _NG+2)
    G = G[1:-1]

    T1, T2 = np.meshgrid(G, G, indexing="ij")
    Dis = T1*(2-T2)/(T1+T2-T1*T2)
    Dis = np.dstack([Dis]*len(_x))
    
    P1 = Trans1DDisord(G, s1)
    P2 = Trans1DDisord(G, s2)
    P1 = np.dstack([P1]*len(G))
    P2 = np.dstack([P2]*len(G))

    P1 = np.transpose(P1, (0, 2, 1))
    P2 = np.transpose(P2, (2, 0, 1))
    return np.trapezoid(np.trapezoid(P1*P2*Dis, G, axis=0), G, axis=0)