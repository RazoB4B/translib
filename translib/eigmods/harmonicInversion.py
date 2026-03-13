#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 28 17:37:50 2026

@author original: kuhl
@author: larazolopez
"""

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from matplotlib import gridspec
from translib import plotOpts as po
po.Colorblind(True)


def rootPolish(a, uk, polish):
    '''
    A root polishing routine. It refines approximate roots of a polynomial using 
    Newton's method.
    
    a: Coefficients a_k obtained from the solution of c_j=Sum(a_k*c_{j+k}, k)
    uk: Approximated roots of the Hessenberg matrix
    polish: Number of iterations to polish
    
    Returns the list of polish eigenvalues uk
    '''
    jmax = len(a) - 2
    for _ in range(polish+1):
        for j in range(jmax+1):
            y1 = a[jmax+1]
            y  = a[jmax] + y1 * uk[j]
            for i in range(jmax-1, -1, -1):
                  y1 = y + y1 * uk[j]
                  y = a[i] + y * uk[j]
            uk[j] -= y/(1.0*y1)         # Newton's method to improve the estimation
    return uk


def HarmInvPade(FFTData, polish=10):
    '''
    Given the Fourier tranform of the spectrum, computes the complex resonant 
    frequencies, as well as the complex resonant amplitudes of the signal.
    
    FFTData: The data to fit (on the time domain)
    polish: Number of iterations to polish the result
    
    Returns the eigenfrequencies uk and the eigenamplitudes dk
    See Razo-Lopez, et. al., Optical Material Express 14, 3 (2024) - Eqs. (5) to (11)
    '''
    N = len(FFTData) - 1       # Number of point to take into account
    jmax = (N-2)//2
    
    # Generate the matrix in Eq. (8) - c_j = Sum(a_k * c_{j+k}, k)
    ac = np.zeros([jmax+1, jmax+1], dtype='complex')
    c = np.zeros([jmax+1], dtype='complex')
    a = np.zeros([jmax+2], dtype='complex')
    for j in range(jmax+1):
        ac[:, j] = FFTData[j+1 : j+1 + jmax+1]   # ac = a_k c_{j+k} 
    c[:jmax] = FFTData[:jmax]                    # c_{j}
    
    # Finding the coefficiens a_k 
    lu, piv = la.lu_factor(ac)
    a[1:] = la.lu_solve((lu, piv), c)
    a[0] = -1                                    # See equation (10)
    
    # Build Hessenberg-matrix in Eq. (11)
    A = np.diag(np.ones([jmax])+np.zeros([jmax])*1j, -1)
    A[0, :jmax+1] = -np.flip(a[:jmax+1])/a[jmax+1]
    
    # Calculate and polish the eigenvalues of the Hessenberg-matrix
    uk = la.eigvals(A)
    uk = rootPolish(a, uk, polish)
    
    # Generate the matrix in Eq. (6) - c_j = Sum(A_k * z_k^j, k)
    Az = np.zeros([jmax+1, jmax+1], dtype='complex')
    for j in range(jmax+1):
        Az[j, :jmax+1] = uk[j]**np.arange(jmax+1)
    # Az = A_k * z_k^j (This is the transverse of the Vandermonde matrix)
    d = FFTData[:jmax+1]         # c_{j}
    
    # Finding the coefficiens A_k
    lu, piv = la.lu_factor(Az.T)
    dk = la.lu_solve((lu, piv), d)
    return uk, dk


def HarInvFourier(FFTData, FrqI, FrqF, dFrq, polish=10, PtError=0.1, MinAmpl=1e-7):
    '''
    Computes the complex frequencies and the complex amplitudes of a given signal,
    only certain resonances in the interval and with larger amplitude than MinAmpl.
    
    FFTData: The data to fit (on the time domain)
    FrqI: the lowest frequency
    FrqF: the highest frequency
    dFrq: Frequency spacing 
    polish: Number of iterations to polish the result
    PtError: Maximum error allowed between the two extraction
    MinAmpl: Minimum amplitude allowed
    
    Returns the eigenfrequencies uk and the eigenamplitudes dk as
    [complex frequency, complex amplitude, absolute amplitude, err]
    '''
    N = len(FFTData)-1          # Number of Datapoints minus 2
    FrqC = (FrqI+FrqF)*0.5      # Central frequency
    tau = 2*np.pi/(FrqF-FrqI)   # Time-stepwidth in fft multiplied with 2 pi

    # HarmInvPade is calculated two times with data shifted by one point
    jmax = (N-3)//2  # jmax: (Number of data - 4)/2
    freq = np.zeros([2, jmax+1], dtype='complex')  
    ampl = np.zeros([2, jmax+1], dtype='complex')
    for i in range(2):
        uk, dk, = HarmInvPade(FFTData[i:N+i], polish)   # Croping the signal from 0 to N and form 1 to N+1, where N=len(Data)-1
        wk = (-1j/tau) * np.log(uk) + FrqC              # uk was exp(i*tau(Wk-FrqC)) ->wk is Wk-FrqC
        dk = dk * (-1j * dFrq / (2 * np.pi))            # The amplitudes are rescaled to come back to the continous Fourier transform
        dk = dk/uk**i                                   # Correction given by the shift stating form 0 and 1
        
        _ind = np.argsort(np.real(wk))  
        dk = dk[_ind]
        wk = wk[_ind]
        
        freq[i-1,:] = wk   
        ampl[i-1,:] = dk
        
    # The two results are compared to each other: 
    # Only resonances that appear in both reaults up to PtError are accepted
    result = []
    for i,_frq1 in enumerate(freq[0,:]):
        _err = np.inf
        for _frq2 in freq[1,:]:
            if np.abs(_frq1 - _frq2)<_err:
                _err = np.abs(_frq1 - _frq2) #The minimum distance between the resonances in both arrays is calculated
        # Resonances must fulfill
        # Error smaller than PtError and its real part between FrqI and FrqF
        # And amplitudes must be greater than MinAmpl
        if ((_err<PtError) and (np.real(_frq1)>=FrqI) and (np.real(_frq1)<FrqF) and (np.abs(ampl[0,i])>MinAmpl)):
            result.append([_frq1, ampl[0,i], _err])
    return np.array(result)


def DataReconstruction(Frqs, Res, Amps):
    '''
    Computes the sum of complex Lorentzians for the complex resonant frequencies 
    and amplitudes
    
    Frqs: The frequency domain
    Res: the complex resonant frequencies
    Amps: the complex resonant amplitudes
    '''
    result = np.zeros([len(Frqs)], dtype='complex')
    for i in range(len(Res)):
        result += Amps[i]/(Frqs - Res[i])
    return result


def HarInv(Frqs, Data, ResoMax=200, MinAmpl=1e-5, polish=10, PtError=0.1, Filter=1e-4, 
           Reflection=False, NoLine=True, Plot=False, FrqI=None, FrqF=None, FrqMin=None, 
           FrqMax=None, Phase=None):
    r'''
    Frqs: the frequency axis
    Data: the complex data to fit
    ResoMax: Maximun number of resonances to find
    polish: Number of iterations to polish the result
    PtError: Maximum error allowed between the two extraction
    Filter: Only resonances with sufficent resonance depth/height>Filter are taken 
            into account
    MinAmpl: Minimum amplitude allowed
    Reflection: if True adjusts the constant from s_ii=\delta_ii-\sum of Lorentzians
    NoLine: If True, no additional fit of a linear background is performed
    FrqI: the lowest frequency
    FrqF: the highest frequency
    FrqMin, FrqMax: defines the interval for which resonance frequencies are taken 
                    into account (remove border effects)
    Phase: if reflection an additional phase takes into account that the coupling 
           to the antenna is not constant and it is a function of the frequency. phi 
           is either a scalar or of length data 
    '''
    if FrqI is None:
        FrqI = Frqs[0]
    if FrqF is None:
        FrqF = Frqs[-1]  
    if FrqMin is None:
        FrqMin = FrqI
    if FrqMax is None:
        FrqMax = FrqF
          
    dFrq = np.ptp(Frqs)/Frqs.size       # Frequency spacing 
    FrqInds = np.where((Frqs >= FrqI) & (Frqs <=FrqF))[0]
    FrqsPart = Frqs[FrqInds]            # Cutting the frequency according to the imposed limits

    if Reflection: # in case of reflection the constant or the phase needs to be substracted
        if Phase is not None:
            DataPart = 1 - Data[FrqInds] * np.exp(1j*Phase[FrqInds])
        else:
            DataPart = 1 - Data[FrqInds]
    else:
        DataPart = Data[FrqInds]
        
    k_index = np.arange(0, len(DataPart))
    # ATTENTION: hi_fourier expects a data array which is the fourier transform of data(nu+(FrqI+FrqF)/2)
    # i.e the initial data on the intervall [FrqI,FrqF] but shifted by (FrqI+FrqF)/2. As the FFT asumes that
    # the data are on the frequency range [0,FrqF-FrqI] an aditional phase shift
    # has to be taken into account! This is done by (-1)^k_index
    FTData = np.fft.ifft(DataPart) * (-1)**k_index * len(DataPart)  # To check if this normalization is interesting or useful
    # multiplication of len() to have the same definition of FFT as in IDL
    FTDataPart = FTData[:ResoMax+1]  # cutoff of the Fourier Tranformed data
    HI = HarInvFourier(FTDataPart, FrqI, FrqF, dFrq, polish=polish, PtError=PtError, MinAmpl=MinAmpl)
    if len(HI) == 0:
        print('No resonances were found')
        return None
    
    # Croping the borders
    _ind = np.where((HI[:,0] >= FrqMin) & (HI[:,0] <= FrqMax))[0]
    HI =  HI[_ind,:]
    
    # Only resonances with sufficent resonance depth/height are taken into account
    _ind = np.where((np.abs(HI[:,1])/np.imag(HI[:,0])) >= Filter)[0]
    HI =  HI[_ind,:]
    
    # Deviation from the original data
    Recons = DataReconstruction(FrqsPart, HI[:, 0], HI[:, 1])
    Diff = DataPart - Recons
    
    # Fits a linear background
    if NoLine:
        slope_re, const_re, chinrRe = [0, 0, 0]
        slope_im, const_im, chinrIm = [0, 0, 0]
        Error = np.sum(np.abs(Diff)**2)
    else:
        _x = np.column_stack([FrqsPart, np.ones_like(FrqsPart)])
        (slope_re, const_re), res_re, *_ = np.linalg.lstsq(_x, np.real(Diff), rcond=None)
        (slope_im, const_im), res_im, *_ = np.linalg.lstsq(_x, np.imag(Diff), rcond=None)
        
        chinrRe = res_re[0] if res_re.size else 0
        chinrIm = res_im[0] if res_im.size else 0
    
        Error = chinrRe + chinrIm
        Recons += (slope_re+1j*slope_im)*FrqsPart + const_re+1j*const_im
        Diff = DataPart - Recons
    
    info = {'nRes': len(HI[:,0]), 'error':Error, 
            'FrqI':FrqI,'FrqF':FrqF, 'FrqMin':FrqMin,'FrqMax':FrqMax,
            'fit_slope':slope_re+1j*slope_im,'fit_const':const_re+1j*const_im}
    result = [info, HI]
    
    if Plot:
        BandLimited(FTData, ResoMax)
        Difference(FrqsPart, Diff)
        Comparison(FrqsPart, DataPart, Recons, np.real(HI[:,0]))
    return result


def BandLimited(FTData, Max):
    '''
    Plots the Band limited signal
    
    FTData: the band limited signal
    Max: the maximum number of points that are considered 
    '''
    Spec = gridspec.GridSpec(ncols=2, nrows=1, wspace=0.18)
    
    fig = plt.figure(figsize=(8, 3))
    ax = []
    for i in range(2):
        ax.append(fig.add_subplot(Spec[i]))
    
        ax[i].plot(np.abs(FTData)**2)
        ax[i].axvline(Max, ls='--')
        ax[i].set_xlabel(r'$j$', fontsize=15)
        ax[i].set_yscale('log')
    
    ax[0].set_xlim(0, len(FTData))
    ax[0].set_ylabel(r'$|c(j\tau)|^2$', fontsize=15)
    ax[1].set_xlim(0, Max+100)
    ax[1].set_title('Band Limited signal', x=-0.1, fontsize=15)
    plt.show()
    
    
def Difference(Frqs, Data):
    '''
    Plots the differences between the original data and the reconstruction
    
    Frqs: the frequency domain
    Data: the difference between the original and the reconstruction datas
    '''
    Spec = gridspec.GridSpec(ncols=2, nrows=2, wspace=0.3, hspace=0.12)
    
    fig = plt.figure(figsize=(8, 6))
    ax = []
    for i in range(4):
        ax.append(fig.add_subplot(Spec[i]))
        ax[i].set_xlim(Frqs[0], Frqs[-1])
        
    ax[0].plot(Frqs, np.abs(Data))
    ax[1].plot(Frqs, np.angle(Data))
    ax[2].plot(Frqs, np.real(Data))
    ax[3].plot(Frqs, np.imag(Data))
    
    ax[0].set_ylabel(r'Amplitude', fontsize=15)
    ax[1].set_ylabel(r'Phase', fontsize=15)
    ax[2].set_ylabel(r'Real part', fontsize=15)
    ax[3].set_ylabel(r'Imaginary part', fontsize=15)
    
    ax[2].set_xlabel(r'Frequency', fontsize=15)
    ax[3].set_xlabel(r'Frequency', fontsize=15)
    
    ax[1].set_title('Differences', x=-0.15, fontsize=15)
    plt.show()
    
    
def Comparison(Frqs, DataOr, DataRec, ResFrq):
    '''
    Plots a comparison between the original data and the reconstruction
    
    Frqs: the frequency domain
    DataOr: the original data
    DataRec: the reconstructed data
    ResFrq: the resonance frequencies
    '''
    Spec = gridspec.GridSpec(ncols=2, nrows=2, wspace=0.3, hspace=0.12)
    
    fig = plt.figure(figsize=(8, 6))
    ax = []
    for i in range(4):
        ax.append(fig.add_subplot(Spec[i]))
        ax[i].set_xlim(Frqs[0], Frqs[-1])
        
    ax[0].plot(Frqs, np.abs(DataOr))
    ax[1].plot(Frqs, np.angle(DataOr))
    ax[2].plot(Frqs, np.real(DataOr))
    ax[3].plot(Frqs, np.imag(DataOr), label='Original data')
    
    ax[0].plot(Frqs, np.abs(DataRec))
    ax[1].plot(Frqs, np.angle(DataRec))
    ax[2].plot(Frqs, np.real(DataRec))
    ax[3].plot(Frqs, np.imag(DataRec), label='Reconstruction')
    
    for _res in ResFrq:
        ax[0].axvline(_res, ls=':', color='C05')
    
    ax[0].set_ylabel(r'Amplitude', fontsize=15)
    ax[1].set_ylabel(r'Phase', fontsize=15)
    ax[2].set_ylabel(r'Real part', fontsize=15)
    ax[3].set_ylabel(r'Imaginary part', fontsize=15)
    
    ax[2].set_xlabel(r'Frequency', fontsize=15)
    ax[3].set_xlabel(r'Frequency', fontsize=15)
    ax[3].legend(fontsize=12)
    
    ax[1].set_title('Comparison', x=-0.15, fontsize=15)
    plt.show()
    