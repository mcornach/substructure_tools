#!/usr/bin/env python 2.7
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 12:10:27 2016

@author: matt

Test - delete this code when done (or turn into a package if its useful)
"""

#Import all of the necessary packages
from __future__ import division
import pyfits
import os
import math
import time
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from matplotlib import cm
plt.rcParams['image.cmap'] = 'gray'
from matplotlib.colors import LinearSegmentedColormap
#import scipy
#import scipy.stats as stats
import scipy.special as sp
import scipy.interpolate as si
import scipy.optimize as opt
import scipy.integrate as integrate
import lmfit
import scipy.sparse as sparse
import multiprocessing as mp
from multiprocessing import Pool, Process
import thread
import argparse
import mcmc_tools as mcmc
import special as sf
import source_utils as src


#from multiprocessing import Process
#import os

#def info(title):
#    print(title)
#    print('module name:', __name__)
#    print('parent process:', os.getppid())
#    print('process id:', os.getpid())
#
#def f(name):
#    info('function f')
#    print('hello', name)
#
#if __name__ == '__main__':
#    info('main line')
#    p = Process(target=f, args=('bob',))
#    p.start()
#    p.join()
#
#exit(0)

myargs = {'interpolation':'none'}

### Take input arguments
parser = argparse.ArgumentParser()
parser.add_argument("-n","--sim_num",help="Number (0-16) of simulated image file", default=1, type=int)
parser.add_argument("-p","--parallel",help="Run MCMC in parallel mode", action='store_true')
parser.add_argument("-t","--plot",help="Plots the output MCMC chains for all variables", action='store_true')
parser.add_argument("-m","--form",help="Noise form option (gaussian or poisson)", default='gaussian')
args_in = parser.parse_args()
parallel = args_in.parallel
plot = args_in.plot
form = args_in.form


### Import simulated galaxy, with true parameters saved (output from 
### master.py).  Will use this as the MCMC test to fit for clump 
### parameters.

res_dir = '/home/matt/software/matttest/results'
num=args_in.sim_num
fname = os.path.join(res_dir,'sim_gxy_{:03d}_eff.fits'.format(num))
sim_fits = pyfits.open(fname)
img = sim_fits[0].data
noise = sim_fits[1].data
img_clean = img-noise
params = sim_fits[2].data
nclumps = params.shape[1]
print "{} Clumps".format(nclumps)
### Now make a scaled up image with Poisson noise + poisson background
scale = 10000
img_scale = img_clean*scale
params[2] *= scale #increase intensity by scale factor
bg_mean = 5
bg_arr = bg_mean*np.ones(img.shape)
bg_arr = np.random.poisson(bg_arr)
img_sn = np.random.poisson(img_scale)
img_sn += bg_arr

#x1, y1 = 31, 38
#x2, y2 = 34, 29

def randomize_guess(params):
    rpars = np.zeros(params.shape)
    for i in range(len(rpars)):
        rpars[i] = np.random.randn()*params[i]/10
    return params+rpars

### Now set up emcee to run
paramsr = np.zeros((6*nclumps))
paramsrmn = np.zeros((6*nclumps))
paramsrmx = np.zeros((6*nclumps))
inds = np.array(([0, 1, 2, 6, 4, 5]), dtype=int)
for i in range(nclumps):
    paramsr[6*i:6*i+6] = randomize_guess(params[:,i][inds])
    paramsrmn[6*i:6*i+6] = (params[:,i][inds])/2
    paramsrmx[6*i:6*i+6] = (params[:,i][inds])*2

xvals, yvals = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
fit_params = mcmc.run_emcee(paramsr, paramsrmn, paramsrmx, xvals, yvals, img_sn, nwalkers=100, ntrials=4000, burn_in=2000, percents = [16,50,84], form=form, plot_samples=plot, parallel=parallel)

### build model from fitted parameters
def trim_fit_errs(fit_params):
    new_pars = np.zeros(fit_params.shape)
    for i in range(new_pars):
        new_pars[i] = fit_params[i][0]
    return new_pars
    
model = np.zeros(img_sn.shape)
new_params = trim_fit_errs(fit_params)
x, y = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
for i in range(nclumps):
    paramsi = lmfit.Parameters()
    paramsi = sf.array_to_Parameters(paramsi, new_params[6*i:6*i+6], arraynames=['xcb0', 'ycb0', 'Ieb0', 'reb0', 'PAb0', 'qb0'])
    model += src.galaxy_profile(paramsi, x, y, img_sn, 1/(abs(img_sn)+5**2), return_residuals=False,blob_type='eff')

plt.imshow(np.hstack(img_sn, model, img_sn-model), **myargs)
plt.ion()
plt.show()















#############################################################################
#############################################################################
###
###        MULTIPROCESSING ATTEMPTS - REVISIT SOON 
###
#############################################################################
#############################################################################



#'''
### From the python 17.2 multiprocessing introduction
#def f(x):
#    return x*x
#    
#def info(title):
#    print(title)
#    print('module name:', __name__)
#    print('parent process:', os.getppid())
#    print('process id:', os.getpid())
#    
#    
#def name(name):
#    info('function name')
#    print('Hello {}'.format(name))
#    
#def foo(q):
#    q.put('hello')
#    
#if __name__ == '__main__':
#    info('main line')
##    mp.set_start_method('fork')
#    q = mp.Queue()
##    p = Process(target=name, args=('Jimbo',))
#    p = Process(target=foo, args=(q,))
#    p.start()
#    print(q.get())
#    p.join()
#'''

'''
def wait(a, i):
    name = mp.current_process().name
#    time.sleep(a)
    value = np.dot(np.ones(100000),np.arange(100000))
    print name
    print 'done', value, i
    return
    
iters = 1
t0 = time.time()
#    if __name__ == '__main__':
#        p = Process(target=wait, args=(2,))
#        p.start()
#        p.join()
if __name__ == '__main__':
    jobs = []
    for i in range(iters):
        p = Process(name='waiting', target=wait, args=(2, i))
        jobs.append(p)
        p.start()
    p.join()
#    tid = thread.start_new_thread(wait, (2,))

tf = time.time()
print("Total time is {}s".format(tf-t0))
#'''