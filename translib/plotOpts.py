#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 12:00:26 2025

@author: alberto-razo
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from colorsys import hls_to_rgb
from matplotlib.colors import LinearSegmentedColormap


def ColorMaps(_CLabel=None):
    """
    Defines a colormap that goes from a black background to a bright color

    CLabel: Limite colors of the array
    """
    if _CLabel == None:
        _CLabel = 'BlackGreen'
        
    if _CLabel == 'BlackRed':
        _colors = [(0, 0, 0), (1, 0, 0)]
        _cmap = LinearSegmentedColormap.from_list('Custom', _colors, N=1000)
    elif _CLabel == 'BlackGreen':
        _colors = [(0, 0, 0), (0, 1, 0)]
        _cmap = LinearSegmentedColormap.from_list('Custom', _colors, N=1000)
    elif _CLabel == 'BlackBlue':
        _colors = [(0, 0, 0), (0, 0, 1)]
        _cmap = LinearSegmentedColormap.from_list('Custom', _colors, N=1000)
    elif _CLabel == 'RedBlue':
        _colors = [(1, 0, 0), (0, 0, 0), (0, 0, 1)]
        _cmap = LinearSegmentedColormap.from_list('Custom', _colors, N=1000)
    elif _CLabel == 'BlackPink':
        _colors = [(0, 0, 0), (159/255, 43/255, 104/255)]
        _cmap = LinearSegmentedColormap.from_list('Custom', _colors, N=1000)
    elif _CLabel == 'BlackBrown':
        _colors = [(0, 0, 0), (193/255, 154/255, 107/255)]
        _cmap = LinearSegmentedColormap.from_list('Custom', _colors, N=1000)
    else:
        print('ColorMap not defined. Existing options: Black[Red, Green, Blue, Pink, Brown]')
    return _cmap


def useTex(_useTex):
    """
    If True activates the latex font in the plots
    """
    if _useTex:
        plt.rc('text', usetex=True)
        plt.rc('font',**{'family':'serif','serif':['Helvetica']})
    else:
        plt.rc('text', usetex=False)
        plt.rc('font', family='sans-serif')
        

def Colorblind(blind):
    """
    If True activates the colorblind palette
    """
    if blind:
        sns.set_palette('colorblind')
        
        
def Colorize(array, theme="dark", saturation=1.0, beta=1.4, transparent=False,
             alpha=1.0, max_threshold=1):
    """
    Returns a vectorial array with the intensity going from black to a transparence
    that shows the phase of the array
    
    array: The complex array
    theme: if dark the background if black, if white background if white
    saturation: defines the saturation of the image
    beta: 
    transparent: if True, the plot blackground is transparent
    alpha: if transparent, defines the degree of transparency
    max_threshold: representes of the maxima of the intensity in the vectorize function 
                   which is normalized to its maxima
    """
    r = np.abs(array)
    r /= max_threshold * np.max(np.abs(r))
    arg = np.angle(array)

    h = (arg + np.pi) / (2 * np.pi) + 0.5
    l = 1.0 / (1.0 + r**beta) if theme == "white" else 1.0 - 1.0 / (1.0 + r**beta)
    s = saturation

    c = np.vectorize(hls_to_rgb)(h, l, s)  # --> tuple
    c = np.array(c)  # -->  array of (3,n,m) shape, but need (n,m,3)
    c = c.swapaxes(0, 2)
    if transparent:
        a = 1.0 - np.sum(c**2, axis=-1) / 3
        alpha_channel = a[..., None] ** alpha
        return np.concatenate([c, alpha_channel], axis=-1)
    else:
        return c