#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 13:20:16 2016

@author: matt
"""

#Import all of the necessary packages
from __future__ import division
import pyfits
import os
import glob
import math
import time
import numpy as np
from numpy import pi, sin, cos, random, zeros, ones, ediff1d
#from numpy import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
#import scipy
import scipy.stats as stats
import scipy.special as sp
#import scipy.interpolate as si
import scipy.optimize as opt
#import scipy.sparse as sparse
import scipy.signal as signal
#import scipy.linalg as linalg
#import solar
import special as sf
#import argparse
#import lmfit
#import emcee
#import mcint
#import psf_utils
import source_utils as src
#import mcmc_tools
#import multiprocessing as mp
#from multiprocessing import Pool, Process

def sum_mags(marray):
    farr = 10**(-0.4*marray)
    fsum = np.sum(farr)
    msum = -2.5*np.log10(fsum)
    return msum
    
msource = np.zeros(17)
magn = [14, 26, 15, 9, 16, 14, 6, 7, 18, 8, 17, 4, 8, 12, 13, 6, 23]
mags_dict = {}
mags_dict[0] = np.array(([25.3, 25.6, 26.1, 27.3]))
mags_dict[1] = np.array(([27.1, 27.1]))
mags_dict[2] = np.array(([24.8, 24.5, 24.8]))
mags_dict[3] = np.array(([27.4]))
mags_dict[4] = np.array(([27.1, 27.3, 25.6]))
mags_dict[5] = np.array(([24.5, 25.5, 26.1]))
mags_dict[6] = np.array(([25.5]))
mags_dict[7] = np.array(([25.6, 24.5]))
mags_dict[8] = np.array(([24.9, 27.8, 24.7]))
mags_dict[9] = np.array(([26.2]))
mags_dict[10] = np.array(([27.0, 26.6, 27.6, 25.2]))
mags_dict[11] = np.array(([25.1, 25.5]))
mags_dict[12] = np.array(([24.9, 27.8]))
mags_dict[13] = np.array(([24.2, 26.6, 26.5, 29.6]))
mags_dict[14] = np.array(([24.7, 26.9, 27.8, 21.2]))
mags_dict[15] = np.array(([25.5, 25.5]))
mags_dict[16] = np.array(([26.4, 28.5, 28.5]))

for i in range(17):
    msource[i] = sum_mags(mags_dict[i])
    
msource_mag = np.zeros(17)
for j in range(17):
    msource_mag[j] = msource[j] - 2.5*np.log10(magn[j])
    
print msource_mag
exit(0)
mu1, mu2, mu3 = 5, 10, 20
msrc = np.arange(21,29)
mint1 = msrc - 2.5*np.log10(mu1)
mint2 = msrc - 2.5*np.log10(mu2)
mint3 = msrc - 2.5*np.log10(mu3)
plt.plot(msource,msource_mag, 'k.', markersize = 15)
plt.plot(msrc, mint1, 'k-.', linewidth = 2)
plt.plot(msrc, mint2, 'k-', linewidth = 2)
plt.plot(msrc, mint3, 'k--', linewidth = 2)
plt.xlabel("magnitude (intrinsic)", fontsize = 20)
plt.ylabel("magnitude (lensed)", fontsize = 20)
ax = plt.gca()
plt.text(22.5, 18.3, "$\mu = 20$", rotation = 30, fontsize = 18)
plt.text(23, 19.6, "$\mu = 10$", rotation = 30, fontsize = 18)
plt.text(23.5, 21, "$\mu = 5$", rotation = 30, fontsize = 18)
ax.invert_yaxis()
ax.invert_xaxis()
#ax.set_ylim([22.5, 25.0])
#ax.set_xlim([21, 28])
#plt.show()

zvals = np.loadtxt('/home/matt/software/matttest/docs/Galaxy_redshifts.csv',delimiter=',',skiprows=1)
z_src = zvals[:,1]
z_lens = zvals[:,2]
plt.plot(z_lens, z_src, 'kd', markersize = 12)
plt.xlabel('$z_{lens}$', fontsize = 22)
plt.ylabel('$z_{source}$', fontsize = 22)
#plt.show()

cnts = 0.1*10**4
mpas = src.find_abmag(cnts)
print mpas