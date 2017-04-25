#!/usr/bin/env python 2.7
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 11:35:28 2016

@author: matt

Functions for use with MCMC fitting.
Designed to be used with emcee v2.2.1
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
import scipy
#import scipy.stats as stats
import scipy.special as sp
#import scipy.interpolate as si
#import scipy.optimize as opt
#import scipy.integrate as integrate
#import lmfit
#import scipy.sparse as sparse
#import scipy.signal as sig
#import scipy.linalg as linalg
#import astropy.stats as stats
import special as sf
import source_utils as src
import emcee

def run_emcee(params, param_mins, param_maxes, xvals, nwalkers=100, ntrials=10000, burn_in=500, percents = [16,50,84], form='gaussian', plot_samples=False):
    ''' Run an instance of emcee with given inputs.
        Returns parameter estimates and error estimates (based upon percents)
    '''
    ndim = len(params)
    pos = [params + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
    t0 = time.time()
    kwargs = dict()
    kwargs['form'] = form
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(param_mins,param_maxes,xvals), kwargs=kwargs)
    sampler.run_mcmc(pos, ntrials)
    
#    ###Attempts to integrate MP    
#    def samp(pos, ntrials):
#        sampler.run_mcmc(pos, ntrials)
#        return
#    
#    print __name__
#    if __name__ == '__main__':
#        for i in range(nwalkers):
#            p = Process(name='MCMC', target=samp, args=(pos[i], ntrials))
#            p.start()
#        p.join()
#    ### End MP Attempt


    t1 = time.time()
    print("emcee time = {}s".format(t1-t0))

    if plot_samples:
        for i in range(sampler.chain.shape[2]):
            for j in range(100):
                plt.plot(sampler.chain[j,:,i])
            plt.show()
            plt.close()


    ### Remove burn in and resize
    samples = sampler.chain[:, burn_in:, :].reshape((-1, ndim))
    ### Map percentiles to values
    new_params = np.zeros((ndim))
    new_params = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, percents, axis=0)))
    return new_params

#def lnlike(params,x,y,z,inv_z,psf,mode='sersic'):
#    """ log likelihood function for emcee
#    """
##    model = galaxy_profile(params,x,y,z,return_residuals=False)
#    if mode == 'sersic':
#        xc, yc, q, PA, Ie, re, n = params ### Unpack parameters
#        model = sersic2D(x,y,xc,yc,Ie,re,n,q=q,PA=PA)
##        model = signal.convolve2d(model,psf,mode='same') ### Convolve with psf
#    elif mode == 'gaussian':
#        q, PA, xc, yc, sigx, sigy, hght = params
#        model = hght*sf.gauss2d(x,y,sigx,sigy,xcenter=xc,ycenter=yc,q=q,PA=PA)
#    else:
#        print("Invalid mode in lnlike")
#        exit(0)
#    log_likelihood = -0.5*(np.sum((z-model)**2*inv_z - np.log(inv_z)))
#    return log_likelihood
    
def lnlike_pdf(params,x,form='gaussian'):
    """ log likelihood function for gaussian pdf
        params is a list object, names must match those below...
    """
    if type(form) == list:
        prob = sf.gen_central2d(params, x, dists=form)
        kwargs = dict()
        kwargs['dists'] = form
        norm = src.get_norm(x,params,sf.gen_central2d, kwargs=kwargs)
        log_likelihood = np.sum(np.log(prob)-np.log(norm))
        return log_likelihood
    if form=='gaussian':
        xc = params[0]
        sig = params[1]
        log_likelihood = np.sum(-0.5*((x-xc)**2/(sig**2) + 2*np.log(sig)))
    elif form=='trunc_gaussian':
        xc = params[0]
        sig = params[1]
        xl = 0
        err = 0.5*(1-sp.erf((xl-xc)/(np.sqrt(2)*sig)))
        log_likelihood = np.sum(-0.5*(x-xc)**2/(sig**2) - 0.5*np.log(2*np.pi) - np.log(sig) - np.log(err))
    elif form=='trunc_cauchy':
        xc = params[0]
        gam = params[1]
        xl = 0
        err = 0.5-1/np.pi*np.arctan((xl-xc)/gam)
        log_likelihood = np.sum(-np.log(np.pi) - np.log(gam) - np.log(1+((x-xc)/gam)**2) - np.log(err))
    elif form=='exponential':
        tau = params[0]
        log_likelihood = np.sum(-np.log(tau)-x/tau)
    elif form=='trunc_cauchy_2d':
        xc = params[0]
        xsig = params[1]
        yc = params[2]
        ysig = params[3]
        gam = params[4]
        ### approximate, but cheap, integral
        fc = 2
        ln = 200
        xarr = np.linspace(0,fc*np.max(x[0]),ln)
        yarr = np.linspace(0,fc*np.max(x[1]),ln)
        X, Y = np.meshgrid(xarr, yarr)
        norm = np.sum(sf.cauchy_2d(X,Y,xc,yc,xsig,ysig,gam))
        ### Now get log likelihood
        val = sf.cauchy_2d(x[0],x[1],xc,yc,xsig,ysig,gam)
        log_likelihood = np.sum(np.log(val)-np.log(norm))
    else:
        print("Invalid form")
        exit(0)
    return log_likelihood    

def lnprior(params,param_mins,param_maxes,mode='uniform',form='gaussian'):
    """ Uniform only for now
    """
    if mode == 'uniform':
        ok = True
        for i in range(len(params)):
            if param_mins[i] > params[i] or param_maxes[i] < params[i]:
                ok = False
        if ok:
            return 0.0
        else:
            return -np.inf
#    if form == 'gaussian' or form == 'trunc_gaussian':
#        if param_mins[0] < params[0] < param_maxes[0] and param_mins[1] < params[1] < param_maxes[1]:
#            return 0.0
#        else:
#            return -np.inf
#    elif form == 'trunc_cauchy':
#        if param_mins[0] < params[0] < param_maxes[0] and param_mins[1] < params[1] < param_maxes[1] and param_mins[2] < params[2] < param_maxes[2]:
#            return 0.0
#        else:
#            return -np.inf
    else:
        print("Invalid mode")
        exit(0)
        
def lnprob(params,param_mins,param_maxes,x,form='gaussian'):
    """ log probability for emcee
    """
    lp = lnprior(params,param_mins,param_maxes,form=form)
    if not np.isfinite(lp):
        return -np.inf
    ll = lnlike_pdf(params,x,form=form)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll
    
#def run_emcee2d(params, param_mins, param_maxes, X, nwalkers=100, ntrials=10000, burn_in=500, percents = [16,50,84], form=['gaussian','gaussian'], plot_samples=False):
#    ''' Run an instance of emcee with given inputs.
#        Returns parameter estimates and error estimates (based upon percents)
#        Only works for uniform priors right now...
#    '''
#    ndim = len(params)
#    pos = [params + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
#    t0 = time.time()
#    kwargs = dict()
#    kwargs['form'] = form
##    kwargs['form1'] = form[1]
#    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob2d, args=(param_mins,param_maxes,X), kwargs=kwargs)
#    sampler.run_mcmc(pos, ntrials)
#
#    t1 = time.time()
#    print("emcee time = {}s".format(t1-t0))
#
#    if plot_samples:
#        for i in range(sampler.chain.shape[2]):
#            for j in range(100):
#                plt.plot(sampler.chain[j,:,i])
#            plt.show()
#            plt.close()
#
#
#    ### Remove burn in and resize
#    samples = sampler.chain[:, burn_in:, :].reshape((-1, ndim))
#    ### Map percentiles to values
#    new_params = np.zeros((ndim))
#    new_params = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, percents, axis=0)))
#    return new_params
#    
#def lnprob2(params,param_mins,param_maxes,X,form=['gaussian','gaussian']):
#    """ log probability for emcee, assuming 2d input
#    """
#    lp = lnprior(params,param_mins,param_maxes,form=form)
#    if not np.isfinite(lp):
#        return -np.inf
#    ll = lnlike_pdf2(params,x,form=form)
#    if not np.isfinite(ll):
#        return -np.inf
#    return lp + ll