#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 28 17:37:50 2026

@author original: kuhl
@author: larazolopez
"""

import numpy as np
import scipy.linalg as la


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
    Az = np.vander(uk, jmax+1, increasing=True).T
    # Az = A_k * z_k^j (This is the transverse of the Vandermonde matrix)
    d = FFTData[:jmax+1]         # c_{j}
    
    # Finding the coefficiens A_k
    lu, piv = la.lu_factor(Az)
    dk = la.lu_solve((lu, piv), d)
    return uk, dk


def hi_fourier(cd, w0, wl, wr, tau, polish=10, Stepfrq=None, PtError=0.1, MinAmpl=1e-7):
    r'''
    returns an complex np.array 
    each element contains [complex frequency, complex amplitude, absolute amplitude, err]
    
    cd: FFTData which is assumed to be equal to \sum_k (-i*Ak)*e^{i(wk-w0)t} if cd is obtained by Fourier Transformation of Frequency data, data has to be hifted by wo before transforming
    wl: left border of frequency range, 
    wr: right border of frequency range, 
    w0: middle of frequency range, 
    tau: 2*pi*deltaT where deltaT =1/(w2-w1)
    is taken from FFT,
    Stepfrq: distance between two measured frequencies
    polish: defining how often rootpolish is used in HarmInvPade
    
    hi_fourier returns the complex positions wk of the resonances and the amplitudes Ak which contribute to the initial signal cd
    Only resonances between wl and wr are accepted. 
    Furthermore only resonaces which shift less then PtError when the initial data cd is shifted by one point are accepted 
      and the abs-value of the amplitude has to exeed MinAmpl.
    ''' 
    #Set Constants
    nd = len(cd)-2   # nd: Number of Datapoints minus 2
    result=[]        # Complex array to return result

    jmax = (nd-2)//2  # jmax: (Number of data- 4)/2
    freq=np.zeros(shape=(2,jmax+1), dtype=complex) #here the frequencies of the resonances will which are obtained by the shifted and unshifted data
    ampl=np.zeros(shape=(2,jmax+1), dtype=complex) #here the amplitudes if the resonances will be stored which are obtained by the shifted and unshifted data

    # HarmInv-Pade is calculated two times with data shifted by one Datapoint
    for p in np.arange(0,2):
        uk,dk=HarmInvPade(cd[p:nd+p+1], nd, polish=polish)
        wk=np.zeros(shape=(len(dk)), dtype=np.complex128) # here the extracted Resonances Wk will be stored
        wk = (-1j / tau) * np.log(uk)                     # uk was exp(i*tau(Wk-w0)) ->wk is Wk-wo
        dk = dk * (-1j * Stepfrq / (2 * np.pi))           # ??? Wieso werden die dks noch einmal reskaliert
        dk = dk/uk**p  # When data has been shifted the Amplitude has to be corrected

        tmp=np.argsort(np.real(wk))  # the resonances are sorted by incrasing size
        dk= dk[tmp]          # Also the amplitudes are sorted the same way
        wk= wk[tmp]

        wk = w0 + wk #now wk=Wk
        index = 1-p
        # frequencies and amplitudes are stored in the corresponding arrays
        freq[index,:] = wk   
        ampl[index,:] = dk

    # The two results are compared to each other: 
    # Only resonances that appear in both reaults up to en Error of PtError are accepted
    err = np.zeros(shape=(jmax+1), dtype=float)
    err[:] = 99999999999.0e0
    for l in np.arange(0, jmax):
        for j in np.arange(0, jmax): 
            #The minimum distance between a resonance in Result 0 to an arbitrary other resonance in Result 1 is calculated
            if (abs(freq[0,l] - freq[1,j]) < err[l]) :
                err[l] = abs(freq[0,l] - freq[1,j])
        # Filtering of resonances:Resonances must fulfill
        # Error smaller then PtError and its np.real must be between wl and wr
        # and amplitude greater then MinAmpl
        if ( (err[l] < PtError) 
            and (np.real(freq[0,l]) >= wl) and (np.real(freq[0,l]) < wr) 
            and (np.abs(ampl[0,l]) > MinAmpl) ) :
                result.append([freq[0,l], ampl[0,l], abs(ampl[0,l]), err[l]])
    return np.array(result)


def hi_base(Frqs, Data, ResoMax=200, Reflection=False, FrqI=None, FrqF=None, FrqMin=None, 
            FrqMax=None, Phase=None,
            plot_flag=False, noline=True, 
            polish=10, MinAmpl=1e-5, PtError=0.1, hi_filter=1e-4, oldflag=False):
    '''
    Frqs: the frequency axis
    Data: the conmplex data to fit
    ResoMax: Maximun number of resonances to find
    Reflection: if True adjusts the constant from s_ii=\delta_ii-\sum of Lorentzians
    FrqI: the lowest frequency
    FrqF: the highest frequency
    FrqMin, FrqMax: defines the interval for which resonance frequencies are taken into account (remove border effects)
    Phase: if reflection an additional phase takes into account that the coupling to the antenna is not constant and it is a function of the frequency
         phi is either a scalar or of length data 
    ################
    ------------------
    returns (info, ListOFResonances)
    info is a dictionary that contains:
        'nRes': number of resonances
        'FrqI','FrqF' : maximal and minimal frequency
        'valmin','valmax' : 
        'fit_slope',fit_const' : slope and constant of the linear fit performed 
            after hi to calculate the deviations
    if oldflag=True the return gives back erg structure originating from idl
    '''
    if FrqI in None:
        FrqI = Frqs[0]
    if FrqF is None:
        FrqF = Frqs[-1]  
    if FrqMin is None:
        FrqMin = FrqI
    if FrqMax is None:
        FrqMax = FrqF   
          
    dFrq = np.ptp(Frqs)/Frqs.size       # Frequency spacing 
    FrqC = (FrqI+FrqF)*0.5              # Central frequency
    FrqInds = np.where((Frqs >= FrqI) & (Frqs <=FrqF))[0]
    FrqsPart = Frqs[FrqInds]            # Cutting the frequency according to the imposed limits
    tau = 2*np.pi/(FrqF-FrqI)           # Time-stepwidth in fft multiplied with 2 pi

    if Reflection: # in case of reflection the constant or the phase needs to be substracted
        if Phase is not None:
            DataPart = 1 - Data[FrqInds] * np.exp(1j*Phase[FrqInds])
        else:
            DataPart = 1 - Data[FrqInds]
    else:
        DataPart = Data[FrqInds]
        
    k_index = np.arange(0, len(DataPart))
    # ATTENTION: hi_fourier expects a data array which is the fourier transform of data(nu+FrqC)
    # i.e the initial data on the intervall [FrqI,FrqF] but shifted by FrqC. As the FFT asumes that
    # the data are on the frequency range [0,FrqF-FrqI] an aditional phase shift
    # has to be taken into account! This is done by (-1)^k_index
    FTData = np.fft.ifft(DataPart) * (-1)**k_index * len(DataPart)  # To check if this normalization is interesting or useful
    # multiplication of len() to have the same definition of FFT as in IDL
    FTDataPart = FTData[:ResoMax+1]  # cutoff of the Fourier Tranformed data
    ###################
   
    hi_erg = hi_fourier(FTDataPart, FrqC, FrqI, FrqF, tau, Stepfrq=dFrq, polish=polish, PtError=PtError, MinAmpl=MinAmpl)  #call hi fourier
    if len(hi_erg) == 0:
        """No resonances returned by hi_fourier"""
        return None
    # Filter noise resonances: 
    # Only resonances with sufficent resonance depth/height are taken into account
    val = np.where((np.abs(hi_erg[:,1])/np.imag(hi_erg[:,0])) >= hi_filter)[0]
    n_res= len(val) # number of valid resonances
    hi_erg =  hi_erg[val,:]

    #Start reconstruction
    Ausschnitt=np.where((FrqsPart >= FrqMin) & (FrqsPart <= FrqMax))[0]

    Rekonstr = resToData(FrqsPart[Ausschnitt], hi_erg)
    ErrorArr = Rekonstr - DataPart[Ausschnitt] #deviation of hi reconstruction and original data
    erg=np.zeros(9+2*n_res,dtype=complex)
    if noline:  #if keyword /noline is set, then no additional fit of a linear backgrund is performed
        slope_re,const_re,chinrRe=(0,0,0)
        slope_im,const_im,chinrIm=(0,0,0)
        Error=np.sum(np.abs(ErrorArr)**2) # Error is square sum of ErrorArr
    else :  #else the linear line is fitted
        #fit linear slope to ErrorArr
        (slope_re,const_re),chinrRe=np.linalg.lstsq(np.vstack([FrqsPart[Ausschnitt],np.ones(len(FrqsPart[Ausschnitt]))]).T,np.real(ErrorArr), rcond=None)[0:2]
        (slope_im,const_im),chinrIm=np.linalg.lstsq(np.vstack([FrqsPart[Ausschnitt],np.ones(len(FrqsPart[Ausschnitt]))]).T,np.imag(ErrorArr), rcond=None)[0:2]
        Error=chinrRe+chinrIm  #error is the non reduced chisquare of the two linear fits
    #Store Data in erg-Array
    if oldflag:
        erg[0]=n_res
        erg[1]=Error  #error is the non reduced chisquare of the two linear fits
        erg[2]=FrqI
        erg[3]=FrqF
        erg[4]=FrqMin
        erg[5]=FrqMax
        erg[6]=slope_re+1j*slope_im # complex slope for the linear fit
        erg[7]=const_re+1j*const_im # complex constant for the linear fit
        erg[8:8+n_res]=hi_erg[:,0]
        erg[8+n_res:8+2*n_res]=hi_erg[:,1]
        result=erg
    else:
        info={'nRes': n_res, 'error':Error, 
              'FrqI':FrqI,'FrqF':FrqF, 'FrqMin':FrqMin,'FrqMax':FrqMax,
              'fit_slope':slope_re+1j*slope_im, # complex slope for the linear fit              
              'fit_const':const_re+1j*const_im # complex constant for the linear fit
              }
        result=(info,hi_erg[:,:])
    if plot_flag:
        plt.figure('Hi_FFT');plt.clf()
        plt.semilogy(np.abs(FTData)**2,label='FFT_data')
        plt.ylabel('|FFT|$^2$');plt.xlabel('datapoint n')
        plt.axvline(ResoMax, ls=':', label=str(ResoMax))
        if not noline:
            Rekonstr+=(slope_re+1j*slope_im)*FrqsPart[Ausschnitt] + const_re+1j*const_im
        if Reflection:
            ylabel_str='Reflection {0}S_{{ii}}{1}' # {{ ans }} used due to ''.format later on
            DataPart+=0
            Rekonstr+=1#1-.5
            #Rekonstr+=-( (slope_re+1j*slope_im)*frq_part[Ausschnitt]+(const_re+1j*const_im))
        else:
            ylabel_str='Transmission {0}S_{{ij}}{1}'
        res=np.array([ np.real(i[0]) for i in hi_erg])
        plot4x4((FrqsPart,FrqsPart[Ausschnitt]),(DataPart,Rekonstr),labels=['Original data','Reconstruction'],title='Hi_ControlAll', res=res)
        diff=DataPart[Ausschnitt]-Rekonstr
        frq_diff=FrqsPart[Ausschnitt]
        plot4x4((frq_diff,),(diff,),labels=('data-reconstr',),title='Hi_ControlAll Diff')
    
    return result#erg

# -*- coding: utf-8 -*-
"""
Originally from IDL implementation
call either: 
    hi_base to perform a single evaluation of with harmonic inversion 
    example: 
      hi.hi_base(frq,data_part,frq[0],frq[-1],reflection_flag=True, phi=phase_corr, plot_flag=True, noline=False)
      where phase_corr is an array of len(frq) doing the phase correction for the reflection
or 
    hi


Created on Thu Mar 31 18:33:24 2016

@author: kuhl
"""
import matplotlib.pyplot as plt
from mesopylib.utilities.plot.plot_exp_spectra import plot4x4
    
def resToData(frq, Res, Ampl=None):
    """
    calculates a sum of complex Lorentzians, 
    where the Res=[complex frequency, complex amplitude]
    or Res=complex frequency and Ampl=complex amplitude
    """
    result=np.zeros(len(frq), dtype=complex)
    if Ampl is None:
        for i in np.arange(0, len(Res[:,0])) :
            result += Res[i,1]/(frq - Res[i,0])
    else:
        for i in np.arange(0, len(Res)) :
            result += Ampl[i]/(frq - Res[i])
    return result