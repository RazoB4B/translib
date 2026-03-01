#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 28 17:37:50 2026

@author original: kuhl
@author: larazolopez
"""

import numpy as np

def hi_base(Frqs, Data, Reflection=False, FrqI=None, FrqF=None, FrqMin=None, 
            FrqMax=None, Phase=None,
            var_trunc=200, 
            plot_flag=False, noline=True, 
            polish=10, MinAmpl=1e-5, PtError=0.1, hi_filter=1e-4, oldflag=False):
    '''
    Frqs: the frequency axis
    Data: the conmplex data to fit
    Reflection: if True adjusts the constant from s_ii=\delta_ii-\sum of Lorentzians
    FrqI: the lowest frequency
    FrqF: the highest frequency
    FrqMin, FrqMax: defines the interval for which resonance frequencies are taken into account (remove border effects)
    Phase: if reflection an additional phase takes into account that the coupling to the antenna is not constant and it is a function of the frequency
         phi is either a scalar or of length data 
    ################
    var_trunc: number of points taken from the time signal obbtained by FFT\n
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
    ###################
   
    data_fft_trunc = FTData[0:var_trunc+1]  #cutoff of fft-Data after trunc datapoints
    
    hi_erg = hi_fourier(data_fft_trunc, FrqC, FrqI, FrqF, tau, Stepfrq=dFrq, polish=polish, PtError=PtError, MinAmpl=MinAmpl)  #call hi fourier
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
        plt.axvline(var_trunc, ls=':', label=str(var_trunc))
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
import scipy.linalg as la
import matplotlib.pyplot as plt
from mesopylib.utilities.plot.plot_exp_spectra import plot4x4

def extBest(erg31, deltaval=0, RekDet=None, windows=None):
    """
    extract the best resonances from different Hi's
    """
    #;@RekDet funktioniert noch nicht
    if windows is None:
        #if keyword window is not set, take all windows of erg31
        #extract from erg31
        windows = np.arange(0,len(erg31[:,0,0])) 
    numwin = len(windows)
    nvar= len(erg31[0,:,0]) #extracts the number of variations per window
    #initialize GRes and Ampl
    GesRes=[]
    Ampl = []

    valmin = np.float(np.real(erg31[windows[0],0,4]))/1e9
    #valmin is set on valmin value of first window
    for k in np.arange(0, numwin):
        i=windows[k]
        BestError=10000000
        valmax = np.float(np.real(erg31[i,0,5]))
        #;@kann man viel einfacher mit min statt schleife machen
        BRes=None
        BAmpl=None
        for j in np.arange(0, nvar):
            if (np.real(erg31[i,j,1]) < BestError) :
                BestError = np.real(erg31[i,j,1])
                n_res=np.int(np.real(erg31[i,j,0]))
                BRes = erg31[i,j,8:8+n_res]
                BAmpl =  erg31[i,j,8+n_res: 8+2*n_res]
        # Find appropriate valmax such that none of the best resonances is near the valmax-border by increasing valmax while it is not the case
        if BRes is not None:
            dummy=np.where((np.real(BRes) >= (valmax-deltaval)) & (np.real(BRes) <= (valmax+deltaval)))[0]
            count = len(dummy)
            while (count != 0):
                valmax += deltaval/4.
                dummy=np.where((np.real(BRes) >= valmax-deltaval) & (np.real(BRes) <= valmax+deltaval))[0]
                count = len(dummy)
            #only resonances with real part between valmin and valmax are accepted
            pos  = np.where((np.real(BRes) >= valmin) & (np.real(BRes) <= valmax))[0]
            count = len(pos)
            #if valid resonances are found, attach them
            if (count > 0):
                GesRes=np.append(GesRes,BRes[pos])
                Ampl=np.append(Ampl,BAmpl[pos])
            #set next valmin at current valmax position
        valmin=valmax
    return GesRes,Ampl

def get_HiBorders(frq, numWin=1, deltaFft=None, FirstBorder=True, LastBorder=True):
    """
    determines borders for the harmonic inversion
    input:
        frq in GHz
        numWin : number of windows to create
        deltaFft : overlap of the FFT Windows
        FirstBorder : first Window starts at first element (default=True)
        LastBorder : last Window ends at last element (default=True)
    returns:
        [fftRange, hi_range]
        fftRange are the frequency ranges used for the FFT for the individual windows
        hi_range are the frequency ranges returning the resonance frequencies for the individual windows
        typically fftRange is larger then hi_range to reduce border effects 
        (in the first and the last window the begining/end of the windows are the same respectively)
    """
    frqRange = frq.ptp() # frequency width of frq array
    nFrqRange = len(frq)  # number of datapoints
    nuToN = np.float(nFrqRange)/frqRange  # Factor which transfers frequency values into the corresponding number of datapoints
    deltaVal= (frqRange - 2.* deltaFft)/np.float(numWin) # calculate width of Val-Window
    nDeltaVal= np.floor(deltaVal*nuToN) # corresponding widths in datapoints
    nDeltaFft= np.floor(deltaFft*nuToN)

    #initialize empty arrays
    valRange = np.zeros(shape=(numWin,2),dtype=int)#float)
    fftRange = np.zeros(shape=(numWin,2),dtype=int)#float)
    # fill them
    valRange[:,0] = np.rint(nDeltaFft + nDeltaVal*np.arange(0,numWin))
    valRange[:,1] = np.rint(nDeltaFft + nDeltaVal*(np.arange(0,numWin)+1))
    fftRange[:,0] = valRange[:,0] - nDeltaFft
    fftRange[:,1] = valRange[:,1] + nDeltaFft    
    
    # set evaluation for the first and last windows to the starting end endpoint
    if FirstBorder:
        valRange[0,0] = 0
        fftRange[0,0] = 0
    if LastBorder:
        valRange[-1,1] = nFrqRange-1
        fftRange[-1,1] = nFrqRange-1
    return fftRange, valRange

def harmonic_inversion(data, frq, numWin=1, trunc=100, reflexion=True, NoFFT=False, All=True):
    """
    """
    k=len(data[0, :, 0, 0])
    l=len(data[0, 0, :, 0])
    m=len(data[0, 0, 0, :])
    

def root_polish(cof, uk, jmax, polish):
# in polish is stored the number of cycles
    for iter in np.arange(0,np.floor(polish)+1):
        for j in np.arange(0,jmax+1):
            y1 = cof[jmax+1]
            y  = cof[jmax] + y1 * uk[j]
            for i in np.arange(jmax-1,-1,-1):
                  y1 = y + y1 * uk[j]
                  y = cof[i] + y * uk[j]
            uk[j] -= y/(1.0*y1)
    return uk

def HarmInv_Pade( cd, nd, polish=10):
                #, status_ludc1=None, status_hqr=None, status_ludc2=None):
    """
        eq{11} and 
        c_(n) = Sum(a_k * c_(n+k), k)\n
        c_(n) = Sum(a_k * c_(n+k), k) for a_k -> cof\n
        c_n = Sum(d_k * z_k^n, k) to calculate dk	{see eq. 7}
        --------------------\n
        Input:
          cd : data (on time axis)
          nd : number of points taken into account
          polish = 10 : int (number of polishing the result)
        Example: 
        HarmInv_Pade( data[], 120, polish=10):
    """
    #Variables
    jmax = (nd-2)//2
    a = np.zeros(shape=(jmax+1,jmax+1), dtype=np.complex128)
    af = np.zeros(shape=(jmax+1,jmax+1), dtype=np.complex128)
    b = np.zeros(shape=(jmax+1), dtype=np.complex128)
    cof = np.zeros(shape=(jmax+2), dtype=np.complex128)
    dk = np.zeros(shape=(jmax+1), dtype=np.complex128)
    dcomplex1 = np.complex128(1) # 1+0j

    #Generate eqnarr c_(n) = Sum(a_k * c_(n+k), k)     {see eq. 11}
    for j in np.arange(0, jmax+1): 
        a[:,j]=cd[j+1 : j+1+jmax+1]
    af = a[:,:]
    b[0:jmax] = cd[0:jmax]
    cof[1:] = b

    #Solve c_(n) = Sum(a_k * c_(n+k), k) for a_k -> cof 	 {see eq. 11}
    af,ipiv,status_ludc1 = la.lapack.zgetrf(a)
    #ipiv is the permutation in the LU decomp. A = ipiv * L * U
    cof[1:],status_ludsol = la.lapack.zgetrs(af,ipiv,b)
#    lu_piv = la.lu_factor(a)       # use highlvel functions
#    cof[1:]=la.lu_solve(lu_piv,b)  # "
    cof[0] = -1.0e0	#{see eq. 10} a_0 = -1 if we sum k from 0 instead

    #Build Hessenberg-matrix		{see eq. 14}
    a *= 0
    a[0, 0:jmax+1] = -cof[jmax-np.arange(jmax+1)]/cof[jmax+1]
    for j in np.arange(1, jmax+1): 
        a[j,j-1] = dcomplex1
    #Calculate eigenvalues of the Hessenberg-matrix
    uk=la.eigvals(a)
    status_hqr=0
    if (status_hqr != 0): 
        print(('1: la_hqr (<-f08psf): status={0}'.format(status_hqr)))
    if polish is not None :
        uk=root_polish(cof, uk, jmax, polish)
    #Generate eqnarr c_n = Sum(d_k * z_k^n, k) to calculate dk	{see eq. 7}
    for j in np.arange(0, jmax+1): 
        a[j,0:jmax+1] = (uk[j])**np.arange(jmax+1) * dcomplex1 #Vandermondsche-matrix
    af = a[:,:]
    dk[0:jmax+1] = cd[0:jmax+1]    
    b = dk

    #Solve eqnarr c_n = Sum(d_k * z_k^n, k) for dk	{see eq. 7}
    af,ipiv,status_ludc2 = la.lapack.zgetrf(a.T)
    if (status_ludc2 != 0): 
        print(('2: la_ludc (<-f07arf): status=', status_ludc2))
    dk,status_ludsol = la.lapack.zgetrs(af,ipiv,b)
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
    ii=np.complex128(1j)
    nd = len(cd)-2   # nd: Number of Datapoints minus 2
    result=[]        # Complex array to return result

    jmax = (nd-2)//2  # jmax: (Number of data- 4)/2
    freq=np.zeros(shape=(2,jmax+1), dtype=complex) #here the frequencies of the resonances will which are obtained by the shifted and unshifted data
    ampl=np.zeros(shape=(2,jmax+1), dtype=complex) #here the amplitudes if the resonances will be stored which are obtained by the shifted and unshifted data

    # HarmInv-Pade is calculated two times with data shifted by one Datapoint
    for p in np.arange(0,2):
        uk,dk=HarmInv_Pade(cd[p:nd+p+1], nd, polish=polish)
        wk=np.zeros(shape=(len(dk)), dtype=np.complex128) # here the extracted Resonances Wk will be stored
        wk = (-ii / tau) * np.log(uk)                     # uk was exp(i*tau(Wk-w0)) ->wk is Wk-wo
        dk = dk * (-ii * Stepfrq / (2 * np.pi))           # ??? Wieso werden die dks noch einmal reskaliert
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
    
def hi_border( data, frq,  fftmin=0, fftmax=0, valmin=None, valmax=None, 
    hi_filter=0, var_fftmin=None, var_fftmax=None, var_trunc=None, PtError=0.01, 
    MinAmpl=1e-7, polish=10, noline=None, plot_flag=False):
    """
    returns all information from all reconstructions
    calls hi_base(...) with different values for:
        fftrange given by (var_fftmin,var_fftmax)
        truncation given by var_trunc
    """
    #initialize unset parameters
    if var_trunc is None:
        var_trunc = np.arange(160,241, dtype=int)

    #determine size of erg array
    mtrunc = max(var_trunc)*2+8 # eigenfrequency amplitude + info
    nvar= len(var_fftmin)*len(var_fftmax)*len(var_trunc)
    erg = np.zeros(shape=(nvar, mtrunc), dtype=complex)

    #extract frequency information
    stepfrq=frq.ptp()/frq.size
    #Start to vary fftborderies and trunc
    varcount=0 #Counter that determines the psoition in the erg array
    #import pdb;pdb.set_trace()
    for i in np.arange(0, len(var_fftmin)):
        #print(('Outer loop i: {}'.format(i)))
        for j in np.arange(0, len(var_fftmax)):
            #print('  Inner Loop j: {}'.format(j))
            for k in np.arange(0, len(var_trunc)):
                w1 = fftmin+var_fftmin[i]  #w1/w2/w0 are the left/right border 
                w2 = fftmax+var_fftmax[j]  #respectively middle of the cut out frequency array
                hi_erg_tmp=hi_base(frq, data, w1, w2, var_trunc[k]+1, valmin, valmax, stepfrq=stepfrq, PtError=PtError, 
                        MinAmpl=MinAmpl, polish=polish, noline=noline, plot_flag=plot_flag, oldflag=True)
                if hi_erg_tmp is None or len(hi_erg_tmp)> mtrunc:
                    print('hi_base return wrong array or None: {}'.format(hi_erg_tmp)) 
                    #import pdb;pdb.set_trace()
                else:
                    erg[varcount,0:len(hi_erg_tmp)]=hi_erg_tmp
                
                varcount+=1
    return erg 
    #erg is returned

def hi(data, frq, numWin=1, deltaFft=None, var_fftmin=None, var_fftmax=None, 
       var_trunc=None, hi_filter=None,
       PtError=0.01, MinAmpl=1e-7, polish=10, noline=None, plot_flag=False):
    """
    input:
      data : complex spectra
      frq : corresponding frequency axis
      numWin : in how many windows the data array should be seperated
      var_fftmin:
      var_fftmax:
      var_trunc: truncation array in number of points usd from the FFT data
      hi_filter:
      PtError: (default=0.01)
      MinAmpl: (default=1e-7)
      polish:  (default==10)
      noline: 
      
    returns:
      an np.array that includes the returned 
      [complex resonance frq, complex amplitude, errorinfos]
    """
    erg = []
    
    [fftRange,valRange]=get_HiBorders(frq, numWin, deltaFft)
    #the borders of the frq windows are saved in fftRange/valRange
    valmin=frq[valRange[0,0]]

    #split Data into windows indicated by valRange and fftRange
    for q in np.arange(0, numWin):
        valmax=frq[valRange[q,1]]
        #call hi_border
        hi_erg=hi_border(data, frq, frq[fftRange[q,0]], frq[fftRange[q,1]], valmin=valmin, valmax=valmax, PtError=PtError, MinAmpl=MinAmpl,
	            hi_filter=hi_filter, var_fftmin=var_fftmin, var_fftmax=var_fftmax, var_trunc=var_trunc, polish=polish, noline=noline, plot_flag=plot_flag)

        valmin=valmax #set next valmin at the adapted valmax value
        erg.append(hi_erg)
        #erg=[erg, hi_erg] #attatch hi_erg to erg
    return np.array(erg)