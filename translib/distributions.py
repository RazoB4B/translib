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
    _p = np.sqrt(np.arccosh(1/np.sqrt(_G)))*np.exp(-(1/_s)*(np.arccosh(1/np.sqrt(_G))**2))
    _p = _p/(np.sqrt(_G**3)*((1-_G)**(0.25)))
    _p = _p/np.trapz(_p, _G)
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
    _p = _G*Trans1DDisord(_G, _s)
    _p = _p/np.trapz(_p, _lnG)
    return _p
