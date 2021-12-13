# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 21:53:03 2021

@author: Ольга
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy import abs as np_abs
from numpy.fft import rfft, rfftfreq,irfft
from scipy.signal import butter, lfilter, lfilter_zi
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
import math

def snr (spec):
    '''
    
    Соотношение сигнал шум

    Parameters
    ----------
    spec : ndarray
        Спектр сигнала.

    Returns
    -------
    float
        Значение SNR в децибелах.

    '''
    peaks,_=find_peaks(spec)
    n=len(peaks)
    s=0
    for i in range(n):
        s+=spec[peaks[i]]
    m=np.max(spec)
    return 10*math.log10(m/(s-m))

def r (s1,s2):
    '''
    коэффицент корреляции Пирсона,
    который применяется для исследования взаимосвязи двух сигнгалов переменных,
    измеренных в метрических шкалах на одной и той же выборке.

    Parameters
    ----------
    s1 : ndarray
        сигнал 1.
    s2 : ndarray
        Сигнал 2.

    Returns
    -------
    float
        Значение коэффицент корреляции Пирсона.

    '''
    n=len(s1)
    X=np.sum(s1)/n
    Y=np.sum(s2)/n
    a=0
    b=0
    c=0
    for i in range(n):
        a+=(s1[i]-X)*(s2[i]-Y)
        b+=(s2[i]-Y)**2
        c+=(s1[i]-X)**2
    return a/((b*c)**0.5)

def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    '''
    
    Фильр Баттерворта
    
    Parameters
    ----------
    data : ndarray
        сигнал
    lowcut : float
        нижняя частота
    highcut : float
        верхняя частота
    fs : int
        частота дискретизации
    order : int, optional
        order. The default is 5.

    Returns
    -------
    y : ndarray
        отфильтрованный сигнал.

    '''
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    zi = lfilter_zi(b, a)
    y, z = lfilter(b, a, data, zi=zi*data[0])
    return y
    
def spectrum (sig,fs):
    '''
    
    Спектр сигнала

    Parameters
    ----------
    sig : ndarray
        Сигнал
    fs : int
        Частота дискретизации

    Returns
    -------
    xf : ndarray
        Частоты спектра
    yf : ndarray
        Амплитуды спектра

    '''
    yf = np_abs(rfft(sig))
    xf = rfftfreq(len(sig), 1/fs)
    return xf,yf

def ppg (f,spec):
    '''
    Расчёт ЧСС

    Parameters
    ----------
    f : ndarray
        Частоты спектра.
    spec : ndarray
        Амплитуды спектра.

    Returns
    -------
    j : float
        ЧСС.
    m : float
        Амплитуда соответствующая ЧСС.

    '''
    n=len(spec)
    m=-1
    j=-1
    for i in range(n):
        if (spec[i]>m):
            m=spec[i]
            j=f[i]
    return j,m