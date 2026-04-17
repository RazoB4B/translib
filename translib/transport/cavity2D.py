#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 12:46:48 2026

@author: alberto-razo
"""

import numpy as np
import mpmath as mp
import scipy.special as ss
import scipy.constants as sc
import translib.functions as fun


def cutoffFrq(n=0, h=1, index=1):
    """
    Computes the cutoff frequency in GHz of for a cavity with certain height 
    and certain mode

    n: index associated with the height h
    h: height of the cavity 
    index: refractive index of the medium 

    References: Classical Electrodynamics, J. Jackson, Chaps: 8
    """
    nuco = n*sc.c*1e-6/(2*h*index)
    
    return nuco


def WaveVector(ws, index=1, n=0, h=1, wnorm=True):
    """
    Takes the angular frequencies and computes the wave vector depending of the heigth
    of the cavity, and the respective mode.

    ws: array of angular frequencies
    index: refractive index of the medium 
    n: index associated with the height h
    b: height of the waveguite 
    if wnorm true: the angular frequencies are weighted by the speed of light. The dimentions
                   h should correspond with the ones of c.
    if wnorm false: the wavevector is computed using the speed of light in m/s.

    References: Classical Electrodynamics, J. Jackson, Chaps: 8
    """
    if wnorm:
        ks = ws*index
    else:
        ks = ws*index/sc.c
    
    # Complex vector that considers also the nonpropagative part
    nprop = n*np.pi/h
    return np.sqrt(ks**2 - nprop**2 + 0j)


def WaveVector_mp(ws, index=1, n=0, h=1, wnorm=True):
    """
    Takes the angular frequencies and computes the wave vector depending of the heigth
    of the cavity, and the respective mode.
    This function uses mpmath that which is a package of arbitrary precision. 

    ws: array of angular frequencies
    index: refractive index of the medium 
    n: index associated with the height h
    b: height of the waveguite 
    if wnorm true: the angular frequencies are weighted by the speed of light. The dimentions
                   h should correspond with the ones of c.
    if wnorm false: the wavevector is computed using the speed of light in m/s.

    References: Classical Electrodynamics, J. Jackson, Chaps: 8
    """
    if wnorm:
        ks = ws*index
    else:
        ks = ws*index/sc.c
        
    # Complex vector that considers also the nonpropagative part
    nprop = n*mp.pi/h
    return mp.sqrt(ks**2 - nprop**2 + 0j)


#%% ONE DIELECTRIC CYLINDER PHYSISCS
def QscaCylDiel(_m, _x):
    '''    
    Scattering efficiency of an individual Mie dielectric cylinder
    
    Parameters
    ----------
    m :Real number
        Ratio between the refractive index of the dielectric and the one of the medium 
    x : Real array
        "Frequency" axis - the array to consider is the wave number k, multiplied
        by the cylinder radius a
    '''
    _error = 1
    _result = abs(bnICylDiel(0, _m, _x))**2
    _n = 0
    while _error > 1E-3:
        _oldresult = _result
        for _i in range(10):
            _n += 1
            _result += 2*abs(bnICylDiel(_n, _m, _x))**2
        _error = max(abs(_oldresult - _result)/abs(_result))
    return _result*2/_x


def bnICylDiel(_n, _m, _x):
    '''
    The coefficiens b_n for the scattering efficiency of an individual Mie dielectric 
    cylinder
    
    Parameters
    ----------
    n : Integer
        Degree of the coefficient of the Scattering efficiency of an individual
        Mie dielectric cylinder
    m : Real number
        Ratio between the refractive index of the dielectric and the one of the medium 
    x : Real array
        "Frequency" axis - the array to consider is the wave number k, multiplied
        by the cylinder radius a
    '''
    _num = ss.jv(_n, _m*_x)*ss.jvp(_n, _x, 1) - _m*ss.jvp(_n, _m*_x, 1)*ss.jv(_n, _x)
    _den = ss.jv(_n, _m*_x) * ss.h1vp(_n, _x, 1) - _m*ss.jvp(_n, _m*_x)*ss.hankel1(_n, _x)
    return _num/_den


def QscaCylDiel_partial(_m, _x, _n):
    '''    
    Parameters
    ----------
    _m : Real number
         Ratio between the refractive index of the dielectric and the one of the medium 
    _x : Real array
         "Frequency" axis - the array to consider is the wave number k, multiplied
         by the cylinder radius a
    _n : Integer
         Degree of the coefficient of the Scattering efficiency of an individual
         Mie dielectric cylinder
    Returns
    -------
    Partial scattering efficiency of an individual Mie dielectric cylinder
    '''
    if _n == 0:
        _result = abs(bnICylDiel(0, _m, _x))**2
    else:
        _result = 2*abs(bnICylDiel(_n, _m, _x))**2
    return _result*2/_x


def FCylDiel(_ws, _epsilon, _h, _r, _l, _n):
    """
    Support function for CharEquationOneCyl.
    
    ws: array of angular frequencies
    epsilon: refractive index of the cylinder 
    h: height of the cylinder and the cavity
    r: radius of the cylinder
    l: index associated with the height h
    n: index associated with the radius r
                  
    References: Y. Kobayashi and S. Tanaka, IEEE 20, 10 (1980)
    """
    _u = _r*WaveVector(_ws, np.sqrt(_epsilon), _l, _h)
    
    return ss.jvp(_n, _u)/(_u*ss.jv(_n, _u))


def FCylDiel_mp(_ws, _epsilon, _h, _r, _l, _n):
    """
    Support function for CharEquationOneCyl_mp.
    
    ws: array of angular frequencies
    epsilon: refractive index of the cylinder 
    h: height of the cylinder and the cavity
    r: radius of the cylinder
    l: index associated with the height h
    n: index associated with the radius r
                  
    References: Y. Kobayashi and S. Tanaka, IEEE 20, 10 (1980)
    """
    _u = _r*WaveVector_mp(_ws, np.sqrt(_epsilon), _l, _h)
    
    return mp.besselj(_n, _u, 1)/(_u*mp.besselj(_n, _u))


def MCylDiel(_ws, _h, _r, _l, _n, _trap=False):
    """
    Support function for CharEquationOneCyl.
    
    ws: array of angular frequencies
    h: height of the cylinder and the cavity
    r: radius of the cylinder
    l: index associated with the height h
    n: index associated with the radius r
                  
    References: Y. Kobayashi and S. Tanaka, IEEE 20, 10 (1980)
    """
    _v = _r*WaveVector(_ws, 1, _l, _h)
    
    if _trap:
        _v = np.abs(_v)
        _fun = ss.kvp(_n, _v, 1)/(_v*ss.kn(_n, _v))
    else:
        _fun = -ss.h2vp(_n, _v)/(_v*ss.hankel2(_n, _v))
  
    return _fun


def MCylDiel_mp(_ws, _h, _r, _l, _n):
    """
    Support function for CharEquationOneCyl_mp.
    
    ws: array of angular frequencies
    epsilon: refractive index of the cylinder 
    h: height of the cylinder and the cavity
    r: radius of the cylinder
    l: index associated with the height h
    n: index associated with the radius r
                  
    References: Y. Kobayashi and S. Tanaka, IEEE 20, 10 (1980)
    """
    _v = _r*WaveVector_mp(_ws, 1, _l, _h)
    
    return -(mp.besselj(_n, _v, 1)-1j*mp.bessely(_n, _v, 1))/(_v*mp.hankel2(_n, _v))


def CharEquationCylDiel(_Frqs, _epsilon, _h, _r, _l, _n, _trap=False):
    """
    Characteristic equation of the resonant modes of a dielectric cylinder 
    sandwiched in a cavity.

    Frqs: array of frequencies
    epsilon: refractive index of the cylinder 
    h: height of the cylinder and the cavity
    r: radius of the cylinder
    l: index associated with the height h
    n: index associated with the radius r
    if trap true: considers that the modes cannot propagate outsite of the cylinder
                  therefore, they are evanecent
                  
    References: Y. Kobayashi and S. Tanaka, IEEE 20, 10 (1980)
    """
    _k = _Frqs*2*np.pi/sc.c
    _kz = _l*sc.pi/_h
    _u = _r*WaveVector(_k, np.sqrt(_epsilon), _l, _h)
    _v = _r*WaveVector(_k, 1, _l, _h)
    
    _F = FCylDiel(_k, _epsilon, _h, _r, _l, _n)
    
    if _trap:
        _v = -1j*np.abs(_v)
        _F = np.real(_F)
    
    _M = MCylDiel(_k, _h, _r, _l, _n, _trap=_trap)
    
    return (_F+_M)*(_k**2)*(_epsilon*_F+_M) - (_n*_kz*(1/_u**2 - 1/_v**2))**2


def CharEquationCylDiel_mp(_Frqs, _epsilon, _h, _r, _l, _n):
    """
    Characteristic equation of the resonant modes of a dielectric cylinder 
    sandwiched in a cavity.
    This function uses mpmath that which is a package of arbitrary precision.

    Frqs: array of frequencies
    epsilon: refractive index of the cylinder 
    h: height of the cylinder and the cavity
    r: radius of the cylinder
    l: index associated with the height h
    n: index associated with the radius r
    if trap true: considers that the modes cannot propagate outsite of the cylinder
                  therefore, they are evanecent
                  
    References: Y. Kobayashi and S. Tanaka, IEEE 20, 10 (1980)
    """
    _k = 2*mp.pi*_Frqs/sc.c
    _kz = _l*sc.pi/_h
    _u = _r*WaveVector_mp(_k, np.sqrt(_epsilon), _l, _h)
    _v = _r*WaveVector_mp(_k, 1, _l, _h)
    
    _F = FCylDiel_mp(_k, _epsilon, _h, _r, _l, _n)
    _M = MCylDiel_mp(_k, _h, _r, _l, _n)
    
    return (_F+_M)*(_k**2)*(_epsilon*_F+_M) - (_n*_kz*(1/_u**2 - 1/_v**2))**2


def PCylDiel(_Frq, _epsilon, _h, _r, _l, _n):
    """
    In the case of a dielectric cylinder sandwiched in a cavity, for a given 
    frequency, computes the proportion of electric and magnetic field;
    if |P|<1, electric field is predomintant, if |P|>1, magnetic field is predomintant
    
    Frq: the frequency
    epsilon: refractive index of the cylinder 
    h: height of the cylinder and the cavity
    r: radius of the cylinder
    l: index associated with the height h
    n: index associated with the radius r
                  
    References: Y. Kobayashi and S. Tanaka, IEEE 20, 10 (1980)
    """
    _k = 2*np.pi*_Frq/sc.c
    _u = _r*WaveVector(_k, np.sqrt(_epsilon), _l, _h)
    _v = np.abs(_r*WaveVector(_k, 1, _l, _h))
    
    _F = np.real(FCylDiel(_k, _epsilon, _h, _r, _l, _n))
    _M = MCylDiel(_k, _h, _r, _l, _n, True)
    return (_n*(1/_u**2 + 1/_v**2))/(_F+_M)


def ContConsZCylDiel(_ws, _epsilon, _h, _r, _l, _n, _trap=False):
    """
    In the case of a dielectric cylinder sandwiched in a cavity, it computes the
    proportional constant of the field at the boundary for E_z and B_z
    
    ws: the angular frequency
    epsilon: refractive index of the cylinder 
    h: height of the cylinder and the cavity
    r: radius of the cylinder
    l: index associated with the height h
    n: index associated with the radius r
    if trap true: considers that the modes cannot propagate outsite of the cylinder
                  therefore, they are evanecent
                  
    References: M. Reisner, Ph.D. thesis (2023)
    """
    _u = _r*WaveVector(_ws, np.sqrt(_epsilon), _l, _h)
    
    if _trap:
        _v = np.abs(_r*WaveVector(_ws, 1, _l, _h))
        _c = ss.jv(_n, _u)/ss.kv(_n, _v)
    else:
        _v = _r*WaveVector(_ws, 1, _l, _h)
        _c = ss.jv(_n, _u)/ss.hankel2(_n, _v)
    
    return _c


def EzCylDiel(_x, _y, _z, _Frq, _epsilon, _h, _r, _l, _n, _trap=False):
    """
    In the case of a dielectric cylinder sandwiched in a cavity, it computes the
    electric field E_z
    
    x: the x-coordinate
    y: the y-coordinate
    z: the z-coordinate
    Frq: the frequency
    epsilon: refractive index of the cylinder 
    h: height of the cylinder and the cavity
    r: radius of the cylinder
    l: index associated with the height h
    n: index associated with the radius r
    if trap true: considers that the modes cannot propagate outsite of the cylinder
                  therefore, they are evanecent
                  
    References: M. Reisner, Ph.D. thesis (2023)
    """
    _ws = 2*np.pi*_Frq/sc.c
    _rho, _phi = fun.polar(_x, _y)
    _u = _rho*WaveVector(_ws, np.sqrt(_epsilon), _l, _h)
    _v = _rho*WaveVector(_ws, 1, _l, _h)
    _c = ContConsZCylDiel(_ws, _epsilon, _h, _r, _l, _n, _trap)
    
    _Diel = np.exp(1j*_n*_phi) * np.cos(_l*np.pi*_z/_h) * ss.jv(_n, _u)
    
    if _trap:
        _v = np.abs(_v)
        _Air = _c*np.exp(1j*_n*_phi) * np.cos(_l*np.pi*_z/_h) * ss.kn(_n, _v)
    else:
        _Air = _c*np.exp(1j*_n*_phi) * np.cos(_l*np.pi*_z/_h) * ss.hankel2(_n, _v)
        
    return np.where(_rho<_r, _Diel, _Air)


def BzCylDiel(_x, _y, _z, _Frq, _epsilon, _h, _r, _l, _n, _trap=False):
    """
    In the case of a dielectric cylinder sandwiched in a cavity, it computes the
    magnetic field B_z
    
    x: the x-coordinate
    y: the y-coordinate
    z: the z-coordinate
    Frq: the frequency
    epsilon: refractive index of the cylinder 
    h: height of the cylinder and the cavity
    r: radius of the cylinder
    l: index associated with the height h
    n: index associated with the radius r
    if trap true: considers that the modes cannot propagate outsite of the cylinder
                  therefore, they are evanecent
                  
    References: M. Reisner, Ph.D. thesis (2023)
    """
    _ws = 2*np.pi*_Frq/sc.c
    _rho, _phi = fun.polar(_x, _y)
    _u = _rho*WaveVector(_ws, np.sqrt(_epsilon), _l, _h)
    _v = _rho*WaveVector(_ws, 1, _l, _h)
    _c = ContConsZCylDiel(_ws, _epsilon, _h, _r, _l, _n, _trap)
    
    _Diel = np.exp(1j*_n*_phi) * np.sin(_l*np.pi*_z/_h) * ss.jv(_n, _u)
    
    if _trap:
        _v = np.abs(_v)
        _Air = _c*np.exp(1j*_n*_phi) * np.sin(_l*np.pi*_z/_h) * ss.kn(_n, _v)
    else:
        _Air = _c*np.exp(1j*_n*_phi) * np.sin(_l*np.pi*_z/_h) * ss.hankel2(_n, _v)
        
    return np.where(_rho<_r, _Diel, _Air)


#%% ONE DIELECTRIC CYLINDER PHYSISCS

def PotFactCylMet(h1, h2, r):
    """
    Computes the constant of the potencial term of the differential equation of a
    metallic cylinder inside a cavity

    h1: space between the cylinder and the cavity
    h2: height of the cavity
    r: radius of the cylinder
    """
    return (h2 - h1)/(h1*h2*r)


def CharEquationCylMet(_Frqs, _h1, _h2, _r, _l, _n):
    """
    Characteristic equation of the resonant modes of a metallic cylinder in a cavity.

    Frqs: array of frequencies
    epsilon: refractive index of the cylinder 
    h1: space between the cylinder and the cavity
    h2: height of the cavity
    r: radius of the cylinder
    l: index associated with the height h
    n: index associated with the radius r
    """
    ws = _Frqs*2*np.pi/sc.c
    ks = WaveVector(ws, 1, _l, _h2)
    v = _r*ks

    J = ss.jv(_n, v)
    Jp = ss.jvp(_n, v)
    H = ss.hankel1(_n, v)
    Hp = ss.h1vp(_n, v)
    
    lmb = PotFactCylMet(_h1, _h2, _r)
    return (J*Hp/H - Jp)*ks - lmb*J


def CharEquationCylMet_mp(_Frqs, _h1, _h2, _r, _l, _n):
    """
    Characteristic equation of the resonant modes of a metallic cylinder in a cavity.
    This function uses mpmath that which is a package of arbitrary precision.

    Frqs: array of frequencies
    epsilon: refractive index of the cylinder 
    h1: space between the cylinder and the cavity
    h2: height of the cavity
    r: radius of the cylinder
    l: index associated with the height h
    n: index associated with the radius r
    """
    ws = 2*mp.pi*_Frqs/sc.c
    ks = WaveVector_mp(ws, 1, _l, _h2)
    v = _r*ks
        
    J = mp.besselj(_n, v)
    Jp = mp.besselj(_n, v, 1)
    H = mp.hankel1(_n, v)
    Hp = 0.5*(mp.hankel1(_n-1, v) - mp.hankel1(_n+1, v))
    
    lmb = PotFactCylMet(_h1, _h2, _r)
    return (J*Hp/H - Jp)*ks - lmb*J