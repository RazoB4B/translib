#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 12:47:29 2026

@author: alberto-razo
"""

import numpy as np

def GenerateSignal(Speckles, Pos):
    x, y = Pos
    Steps = len(x)

    signal = []
    for i in range(Steps):
        signal = np.append(signal, Speckles[:, x[i], y[i]])
    return signal


def CorrMap(Speckles, Signal):
    Speckles = Speckles - Speckles.mean(axis=0)
    Signal = Signal - Signal.mean()

    corrmap = np.sum(Speckles * Signal[:, None, None], axis=0) / (
        np.sqrt(np.sum(Speckles**2, axis=0)) * np.sqrt(np.sum(Signal**2)))
    return corrmap


def CheckLims(_grain, N, _x=None, _y=None):
    if _x is None:
        xmin = 0
        xmax = N
    else:
        xmin = _x - int(_grain)
        xmax = _x + int(_grain)
        
    if _y is None:
        ymin = 0
        ymax = N
    else:
        ymin = _y - int(_grain)
        ymax = _y + int(_grain)
    
    if xmin < 0:
        xmin = 0
    if ymin < 0:
        ymin = 0
        
    if xmax > N:
        xmax = N
    if ymax > N:
        ymax = N
    return xmin, xmax, ymin, ymax


def Tracking(Speckles, Signal, GrainSize):
    nPoints = len(Speckles)
    N = len(Speckles[0])
    cycles = len(Signal)//nPoints

    xmin, xmax, ymin, ymax = CheckLims(GrainSize, N)

    x = []
    y = []
    for i in range(cycles):
        _sig = Signal[nPoints*i:nPoints*(i+1)]
        _speckles = Speckles[:, xmin:xmax, ymin:ymax]

        _x, _y = CorrPos(_speckles, _sig)
        _x += xmin
        _y += ymin
        
        xmin, xmax, ymin, ymax = CheckLims(GrainSize, N, _x, _y)

        x.append(_x)
        y.append(_y)

    return x, y


def Corr(Guess, Signal):
    Guess = Guess - Guess.mean()
    Signal = Signal - Signal.mean()

    corrmap = np.sum(Guess * Signal)/(np.sqrt(np.sum(Guess**2)) * np.sqrt(np.sum(Signal**2)))
    return corrmap


def PhaseCorrPos(Speckles, Signal, axis=0):
    SpFreqC = np.fft.fft(Speckles, axis=axis)
    SFrq = np.fft.fft(Signal)

    R = SpFreqC * np.conj(SFrq[:, None, None])
    R /= np.maximum(np.abs(R), 1e-20)
    
    PhaseCorr = np.fft.ifft(R, axis=axis).real
    t, i, j = np.unravel_index(np.argmax(PhaseCorr), PhaseCorr.shape)
    return t, i, j


def NormCorrPos(Speckles, Signal, axis=0):
    SpFreqC = np.fft.fft(Speckles, axis=axis)
    SFrq = np.fft.fft(Signal)

    R = SpFreqC * np.conj(SFrq[:, None, None])
    PhaseCorr = np.fft.ifft(R, axis=axis).real
    Wei = np.sqrt(np.sum(Speckles**2, axis=axis) * np.sum(Signal**2))
    NPhaseCorr = PhaseCorr/Wei

    t, i, j = np.unravel_index(np.argmax(NPhaseCorr), NPhaseCorr.shape)
    return t, i, j


def CorrPos(Speckles, Signal, axis=0):
    tn, xn, yn = NormCorrPos(Speckles, Signal, axis=axis)
    ts, xs, ys = PhaseCorrPos(Speckles, Signal, axis=axis)

    foundn = np.append(Speckles[tn:, xn, yn], Speckles[:tn, xn, yn])
    founds = np.append(Speckles[ts:, xs, ys], Speckles[:ts, xs, ys])

    corrn = Corr(foundn, Signal)
    corrs = Corr(founds, Signal)

    if corrn>corrs:
        return xn, yn
    else:
        return xs, ys