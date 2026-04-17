#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 17:14:36 2026

@author: alberto-razo
"""

import numpy as np
from scipy.optimize import minimize


def FindPosibleResonances(_Frqs, _fun):
    """
    For a certain complex signal fun in a certain domain Frqs, it approximates
    the poles of the function by analyzing the sign changes.

    Frqs: the frequency domain
    fun: the function to analyze
    """
    _Res = []
    for _val in [_fun.real, _fun.imag]:
        _sign_changes = np.where(np.sign(_val[:-1]) * np.sign(_val[1:]) < 0)[0]
        for _i in _sign_changes:
            _fguess = (_Frqs[_i] + _Frqs[_i+1]) / 2
            _Res.append(_fguess)
    return sorted(set(_Res))


def FindComplexZeros(_Guess, _Fun, _args, tol=1e6):
    """
    For a certain complex function Fun in a certain range, it computes the roots
    of the function using initial guess.

    Guess: the initial guess
    Fun: the mpmath function
    args: the other arguments of the function
    tol: tolerance for repeated roots
    """
    results = []
    for g in _Guess:
        def objective(x):
            f_complex = x[0] + 1j*x[1]
            val = _Fun(f_complex, *_args)
            return abs(val)**2
        
        x0 = [np.real(g), np.imag(g)]
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead')
            f_sol = res.x[0] + 1j*res.x[1]
            
            if res.fun < 1e-6:  # tolerance
                results.append(f_sol)
        except:
            pass
        
    unique = []
    for res in results:
        if not any(abs(res - u) < tol for u in unique):
            unique.append(res)
    
    return np.array(unique)