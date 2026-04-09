#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 12:52:41 2026

@author: alberto-razo
"""

import numpy as np


def EigVec(H):
    val, vec = np.linalg.eig(H)
    _ind = np.argsort(np.real(val))
    val = val[_ind]
    vec = vec[:, _ind]
    
    return val, vec