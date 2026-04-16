#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 17:14:36 2026

@author: alberto-razo
"""

import numpy as np
import mpmath as mp


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


def FindComplexZeros(_Guess, _Fun, _args, _FrqI, _FrqF):
    """
    For a certain complex function Fun in a certain range, it computes the roots
    of the function using initial guess.

    Guess: the initial guess
    Fun: the mpmath function
    args: the other arguments of the function
    FrqI: the lowest frequency
    FrqF: the highest frequency
    """
    _Res = []
    for _guess in _Guess:
        _f = lambda _s: _Fun(_s, *_args)
        try:
            _r = mp.findroot(_f, _guess)
            _r = np.array(_r, dtype=complex)
            if (np.real(_r)>=0 and np.imag(_r)>=0) and np.real(_r)/np.imag(_r)>=1 and np.abs(_r)<_FrqF and np.abs(_r)>_FrqI:
                _Res.append(_r)
        except:
            pass
    _Res, _ = np.unique(_Res, return_counts=True)
    return _Res