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
import argparse
import lmfit
import emcee
import mcint
import psf_utils
import source_utils as src
import mcmc_tools as mcmc
import multiprocessing as mp
#from multiprocessing import Pool, Process
from matplotlib_scalebar.scalebar import ScaleBar

try:
    hst_dir = os.environ['HST_DIR']
except:
    print("set environvental varable HST_DIR")
    exit(0)
    
cycle = '23'
program = '14189'
root = os.path.join(hst_dir,cycle,program)
### make list of filenames
filenames = [f for f in glob.glob(os.path.join(root,'*','*SIE*pix.fits')) if not 'sersic' in f and not 'SLACS' in f]
if False:
    filenames = [f for f in glob.glob(os.path.join(root,'*','*sersic*.fits')) if not 'pix' in f and not 'SLACS' in f]
f1 = '/home/matt/data/hst_lens/23/14189/08/SDSSJ085621.59+201040.5_sersic_SIE_pix.fits'
f1 = '/home/matt/data/hst_lens/23/14189/08/SDSSJ085621.59+201040.5_source_pix_reg46.fits'
fits1 = pyfits.open(f1)

#img = fits1[4].data
#plt.imshow(img)
#scalebar = ScaleBar(1, units='m', label='1 kpc',frameon=False,color='w', height_fraction = 0.002, length_fraction = 0.08)
#plt.gca().add_artist(scalebar)
#plt.show()
#plt.close()
#exit(0)

vlims = [0, 0.2] ### set common limits on intensity in images
#vlims = None
#ylims = [0, 1.5]
#n = 10000
#ys = sf.gauss_draw(n=n, params = np.array(([0.2, 0.5])), lims=ylims)
#plt.hist(ys, normed=True, bins = 100)
#plt.show()

### Test correlations
#def bi_modal_gauss(n=1,frac=0.5,mu1=0.5,mu2=0.2,sig1=0.07,sig2=0.07):
#    out = np.zeros((n))
#    for i in range(n):
#        a = np.random.rand()
#        if a > frac:
#            out[i] = sig1*np.random.randn() + mu1
#        else:
#            out[i] = sig2*np.random.randn() + mu2
#    if len(out) == 1:
#        out = out[0]
#    return out
#trials = 1000
#ccs = np.zeros((trials))
#for i in range(trials):
#    n = 100
#    frac = 0.2
#    big = bi_modal_gauss(n=n, frac=frac)
#    ex = np.random.exponential(size=n)
#    big2 = bi_modal_gauss(n=n, frac=frac)
#    big2 = np.random.standard_cauchy(size=n)
#    ccs[i] = sf.correlation_coeff(big, big2)
#    
#plt.hist(ccs, 50)
#plt.show()
#plt.hist(big, 50)
#plt.plot(big, ex, 'k.')
#plt.show()


### Find equivalent widths...
#files = np.loadtxt('/home/matt/software/matttest/data/pix_source_models.txt', dtype=np.str)
#inds = np.array(([657, 666],[867,877],[1093,1108],[453,461],[602,612],[898,907],[380,390],[566,576],[584,592],[587,596],[740,750],[652,662],[1080,1086],[271,285],[1037,1045],[1106,1116],[427,436]))
#EWs = np.zeros(len(files))
#for i in range(len(files)):
#    EWs[i] = src.get_ew(files[i], inds[i])
#
#exit(0)

### Test magnitude conversions
#iy = 14.96533
#ny = 0.3257795
#ry = 0.0061528
#ry /= 0.01
#PAy = 3.54162
#qy = 0.078677
#xc = 3
#yc = 6
#xd = 0.03
#yd = 0.03
#xarr = np.arange(0,int(2*xc),xd)
#yarr = np.arange(0,int(2*yc),yd)
#sers = sf.sersic2d(xarr, yarr, xc, yc, iy, ry, ny, PA=PAy, q=qy)
#plt.imshow(sers)
#plt.show()
#print np.sum(sers)
#intcnt = np.sum(sers)*yd*xd
#mag = src.find_abmag(intcnt,scl=4**2)
#print mag
#exit(0)

### 0
### Set profile
param_kw = '_eff'
#param_kw='_sersic'
blob_type = 'eff'
#blob_type = 'sersic'
if blob_type == 'eff':
    gam_fixed = True
    if not gam_fixed:
        param_kw += '_gam_free'

#########################################################################
### 1
### Load Fitted values (found using LAE.py)
sim = False
if sim:
    filenames = np.loadtxt('/home/matt/software/matttest/results/sim_gxys/sim_gxys.txt', dtype=str)
else:
    filenames = np.loadtxt('/home/matt/software/matttest/data/pix_source_models.txt', dtype=str)
    
#src.source_mosaic(filenames, vlims=vlims)
#exit(0)    
  
#if sim:
#    src.fitted_mosaic(data='/home/matt/software/matttest/results/sim_gxys/sim_gxys.txt',param_kw=param_kw, blob_type=blob_type, vlims=vlims)
#else:
#    src.fitted_mosaic(param_kw=param_kw, blob_type=blob_type, vlims=vlims)
#exit(0)
  
logscale = False
overwrite1 = False
#############################################################################
'''
lum = True

if lum:
    centroids, qgals, PAgals = src.get_centroids(filenames, use_model=True, return_axis_ratio=True, param_kw=param_kw)
    lum = src.import_fitted_blob_values(plot_dists=False,centroids=centroids, qgal=qgals, PAgal=PAgals, logscale=logscale, bins=14, param_kw=param_kw,sim=sim,return_lum=lum)
    
lfracs = dict()
lfrac_arr = []
Ltots = np.zeros(len(lum))
magn = [14, 26, 15, 9, 16, 14, 6, 7, 18, 8, 17, 4, 8, 12, 13, 6, 23]
magn = np.array(magn)
for i in range(len(lum)):
    Ltot = np.sum(lum[i])
    Ltots[i] = Ltot
    lfracs[i] = lum[i]/Ltot
    lfrac_arr += list(lum[i]/Ltot)

lfrac_arr = np.array(lfrac_arr) 

plt.hist(Ltots)
plt.show()
plt.hist(Ltots*magn)
plt.show()
plt.close()

exit(0)
#'''
#####################################

if os.path.isfile('/home/matt/software/matttest/results/fitted_clump_values{}_{}.npy'.format(param_kw,"log" if logscale else "lin")) and not overwrite1 and not sim:
    fname = '/home/matt/software/matttest/results/fitted_clump_values{}_{}.npy'.format(param_kw,"log" if logscale else "lin")
    [nb, re, ie, rsep, nsers, qb, PAb, Qgal, PAgal] = np.load(fname)
elif os.path.isfile('/home/matt/software/matttest/results/sim_clump_values{}_{}.npy'.format(param_kw,"log" if logscale else "lin")) and not overwrite1 and sim:
    fname = '/home/matt/software/matttest/results/sim_clump_values{}_{}.npy'.format(param_kw,"log" if logscale else "lin")
    [nb, re, ie, rsep, nsers, qb, PAb, Qgal, PAgal] = np.load(fname)
else:
    centroids, qgals, PAgals = src.get_centroids(filenames, use_model=True, return_axis_ratio=True, param_kw=param_kw)
    nb, re, ie, rsep, nsers, qb, PAb, Qgal, PAgal = src.import_fitted_blob_values(plot_dists=True,centroids=centroids, qgal=qgals, PAgal=PAgals, logscale=logscale, bins=14, param_kw=param_kw,sim=sim)
    if sim:
        np.save('/home/matt/software/matttest/results/sim_clump_values{}_{}.npy'.format(param_kw,"log" if logscale else "lin"), [nb, re, ie, rsep, nsers, qb, PAb, Qgal, PAgal])
    else:
        np.save('/home/matt/software/matttest/results/fitted_clump_values{}_{}.npy'.format(param_kw,"log" if logscale else "lin"), [nb, re, ie, rsep, nsers, qb, PAb, Qgal, PAgal])

if overwrite1 == False:
    plt.close()
ie = ie.astype(float)

#exit(0)
if logscale:
#    logre = re
    logie = ie
#thta = np.pi/15
#thta = src.min_rotation(ie, nsers, thta0=0)

use_qb_mask = False
if use_qb_mask:
    qbm = qb > 0.101
    ie = ie[qbm]
    re = re[qbm]
    nsers = nsers[qbm]
    qb = qb[qbm]
    PAb = PAb[qbm]

#############################################################################
######## Space for testing various other functions ##########################
#############################################################################
'''
### Luminosity distributions (need to actuall use correct z)
### May also be good to do by galaxy to compare to literature
    
def io_r_to_lum(io,ro,ang,z,photflam=1.1845489*10**-19):
    ro = src.px_to_pc(ro*1000, 0.01, z, inv=True)
    dL = sf.luminosity_dist(z)
    dL *= 3.086e18 ### conver pc to cm
    io *= photflam*ang
    F = 2*np.pi*io*ro**2
    L = F*4*np.pi*dL**2
    return L
    
def mAB(fnu):
    return -2.5*np.log10(fnu) - 48.6

def get_fnu(io,ro,z,lam=1700,photflam=1.1845489*10**-19):
    ro = src.px_to_pc(ro*1000, 0.01, z, inv=True)
    io *= photflam
    flam = 2*np.pi*io*ro**2
    fnu = lam**2*flam/3E8*10**8 ### convert to fnu in erg/cm2/s/Hz
    return fnu

fnu = get_fnu(ie,re,2.5)
mags = mAB(fnu)
plt.hist(mags)
plt.show()
exit(0)

photflam = 1.1845489*10**-19
ang = 2340
ang = 1 ## to compare
Lum = io_r_to_lum(ie,re,ang,2.5)
#plt.hist(Lum, bins=14)
#plt.show()
plt.plot(np.log10(np.sort(Lum)))
plt.show()
exit(0)
#'''

#############################################################################
### TRYING 2D FITTING METHOD...
#'''

#def func(params, x):#, dist='gaussian'):
#    dist = 'gaussian'
#    xc = params[0]
#    sc = params[1]
#    if dist == 'gaussian':
#        print 'gauss'
#        return np.exp(-(x-xc)**2/(2*sc**2))
#    elif dist == 'cauchy':
#        print 'cauchy'
#        return 1/(1+(x-xc)**2/sc**2)
#        
#def find_norm(params, x, kwargs=dict()):
#    spacing = x[1]-x[0] ### assume even...
#    val = func(params,x,**kwargs)
#    return np.sum(val)*spacing
#    
#x = np.linspace(-1000,1000,10000)
#params = np.array(([0.2, 0.8]))    
#kwargs = dict()
##kwargs['dist'] = 'cauchy'
#norm = find_norm(params, x, kwargs)
#print "Gaussian if 1:", norm/np.sqrt(2*np.pi*params[1]**2)
#print "Cauchy if 1:", norm/(np.pi*params[1])
#exit(0)
    
### Test on q, re
#params = np.array(([0.253, 0.48, 0.1, 0.15, -np.pi/6]))
params = np.array(([0.26, 0.482, 0.232, 0.2135, 0])) #re, qb #-0.8
pmin = np.array(([0.1, 0.4, 0.05, 0.05, -np.pi/4]))
pmax = np.array(([0.5, 0.8, 1, 1, np.pi/4]))
bounds = np.vstack((pmin, pmax)).T
#params = np.array(([0.26, 0, 0.232, 0.1135, 0])) #re, ie
#pmin = np.array(([0.1, -0.6, 0.05, 0.005, -np.pi/4]))
#pmax = np.array(([0.8, 0.2, 1, 0.3, np.pi/4]))
#bounds = np.vstack((pmin, pmax)).T
#params = np.array(([-0.2, 0.5, 0.1, 0.2135, 0])) #ie, qb
#pmin = np.array(([-0.6, 0.01, 0.005, 0.05, -1000]))
#pmax = np.array(([0.2, 0.7, 0.4, 0.4, 1000]))
#bounds = np.vstack((pmin, pmax)).T

dists=['gaussian','cauchy']
#X = np.array(([re, qb]))
#qb = ie
X = np.vstack((re, qb))
x, y = np.meshgrid(np.linspace(np.min(re),np.max(re),100), np.linspace(np.min(qb),np.max(qb),100))
Xim = np.array(([x, y]))
#lims = [np.min(x), np.max(x), np.min(y), np.max(y)]
#imgr = sf.gen_gauss2d(params, X, idx=0, lims=None, rots=True)
#norm = src.get_norm(X, params, sf.gen_gauss2d)
img = sf.gen_central2d(params, Xim, idx=0, lims=None, dists=['gaussian','cauchy'])
#print np.min(img)
#exit(0)
plt.imshow(img, extent=[np.min(re), np.max(re), np.max(qb), np.min(qb)])
plt.plot(re, qb, 'b.')
ax = plt.gca()
ax.invert_yaxis
plt.show()
plt.close()
#rproj = np.sum(img,axis=0)
#rarr = np.linspace(np.min(re),np.max(re),len(rproj))
#rproj /= np.sum(rproj)*(rarr[1]-rarr[0])
#qproj = np.sum(img,axis=1)
#qarr = np.linspace(np.min(qb),np.max(qb),len(qproj))
#qproj /= np.sum(qproj)*(qarr[1]-qarr[0])
#plt.hist(re,bins=14,normed=True)
#plt.plot(np.linspace(np.min(re),np.max(re),len(rproj)),rproj)
#plt.show()
#plt.close()
#plt.hist(qb,bins=14,normed=True)
#plt.plot(np.linspace(np.min(qb),np.max(qb),len(qproj)),qproj)
#plt.show()
#plt.close()


Test = src.Pdf_info('fit_test')
Test.add_data(X, sf.gen_central2d, normalized=False)#, dists=['cauchy','cauchy'])
Test.add_paramarray(params, bounds)
Test.add_params(params)

fit_params = src.fit_nd_dist(Test, index_cnt=0, method='opt_minimize')
fit_params2 = mcmc.run_emcee(params, pmin, pmax, X, form=dists, plot_samples=True)
print fit_params
print fit_params2
new_params = np.array(([fit_params2[i][0] for i in range(len(fit_params2))]))
    #    quant.params = fit_params
#        if plot:
#            src.plot_1d_projections(quant, index_cnt=index_cnt, plot_2d=True, use_array=True)

img = sf.gen_central2d(new_params, Xim, idx=0, lims=None, dists=['gaussian','cauchy'])
plt.imshow(img, extent=[np.min(re), np.max(re), np.max(qb), np.min(qb)])
ax = plt.gca()
ax.invert_yaxis
plt.plot(re, qb, 'b.')
plt.show()
plt.close()

rproj = np.sum(img,axis=0)
rarr = np.linspace(np.min(re),np.max(re),len(rproj))
rproj /= np.sum(rproj)*(rarr[1]-rarr[0])
qproj = np.sum(img,axis=1)
qarr = np.linspace(np.min(qb),np.max(qb),len(qproj))
qproj /= np.sum(qproj)*(qarr[1]-qarr[0])
plt.hist(re,bins=14,normed=True)
plt.plot(np.linspace(np.min(re),np.max(re),len(rproj)),rproj)
plt.show()
plt.close()
plt.hist(qb,bins=14,normed=True)
plt.plot(np.linspace(np.min(qb),np.max(qb),len(qproj)),qproj)
plt.show()
plt.close()
exit(0)
#'''
#########################################################################
### 2
### Turn into Pdf_info objects (single or joint as needed) and fit
### for parameters.  Specify initial parameters on source_utils.pdf_param_guesses

### Start w/1D
#dists = dict()
#dists['nb'] = [nb, sf.poisson]
#dists['qgal'] = [qgal,'sine']
#dists['qblob'] = [qb,'sine']
#dists['PAgal'] = [PAgal, 'uniform']
#dists['PAblob'] = [PAblob, 'gaussian']
#dists['rsep'] = [rsep, sf.cauchy_lmfit]
#dists['logie'] = [logie, sf.gaussian_lmfit]
#dists['logre'] = [logre, sf.gaussian_lmfit]
#dists['nsers'] = [nsers, sf.cauchy_lmfit]

if blob_type == 'sersic':
    NBS = src.Pdf_info('nb')
    NBS.add_data(nb,np.random.uniform)
    QGS = src.Pdf_info('qgal')
    QGS.add_data(Qgal,sf.gaussian_lmfit_trunc, normalized=True)
    QBS = src.Pdf_info('qb')
    QBS.add_data(qb, sf.cauchy_lmfit, normalized=False)
    RCS = src.Pdf_info('rsep')
    RCS.add_data(rsep, sf.weibull_lmfit, normalized=True)
#    RE = src.Pdf_info('re')
#    RE.add_data(re, sf.weibull_lmfit)
#    RE.add_dependence(RC)
#    NS = src.Pdf_info('nsers_re') #P(nsers|re)
#    NS.add_data(nsers, sf.weibull_lmfit)
#    NS.add_dependence(RB)
#    IE = src.Pdf_info('ie_nsers') #P(ie|nsers)
#    IE.add_data(ie, sf.exponential_lmfit)
#    IE.add_dependence(NS)
    Sersic = src.Pdf_info('ie_re_nsers_comb')
    ### Right now I have to make special functions for each shape, can't think of a 
    ### more general way to cover all cases :(
    rmx = 2
    recut = re[re<rmx]
    iecut = ie[re<rmx]
    nserscut = nsers[re<rmx]
#    Sersic.add_data(np.vstack((recut,iecut,nserscut)),[sf.gaussian_re_lmfit, sf.exponential_ie_lmfit, sf.gaussian_nsers_lmfit])
#    Sersic.add_data(np.vstack((recut,iecut,nserscut)),[sf.cauchy_re_lmfit, sf.exponential_ie_lmfit, sf.weibull_nsers_lmfit])
    Sersic.add_data(np.vstack((recut,iecut,nserscut)),sf.gauss3_irn, normalized=False)
#    dists = [RC, Sersic]
    dists = [QGS, QBS, RCS, Sersic]
elif blob_type == 'eff':
    NBE = src.Pdf_info('nbe')
    NBE.add_data(nb,np.random.uniform)
    QGE = src.Pdf_info('qgale')
    QGE.add_data(Qgal,np.random.uniform)
#    QBE = src.Pdf_info('qb')
#    QBE.add_data(qb, sf.cauchy_lmfit_trunc)
    RCE = src.Pdf_info('rsepe')
    RCE.add_data(rsep, sf.weibull_lmfit, normalized=True)
#    IOE = src.Pdf_info('io')
#    IOE.add_data(ie, sf.exponential_lmfit)
#    RSE = src.Pdf_info('a')
#    RSE.add_data(re, sf.gaussian_lmfit)
    EFF = src.Pdf_info('eff_joint')
    EFF.add_data(np.vstack((re, ie, qb)), sf.cauchy3_irq, normalized=False)
    if not gam_fixed:
        GAM = src.Pdf_info('gamma')
        GAM.add_data(nsers, sf.gaussian_lmfit)
    dists = [RCE, EFF]

#dists = [RC, LogIE, NS, LogRE]
#dists = [RC, Sersic]

def find_dist_params(dists, overwrite=False, bootstrap=False, plot=False, sim=False):
    idx = 0
    if type(dists) != list:
        dists = [dists]
    for quant in dists:
        if not bootstrap:
            if os.path.isfile('/home/matt/software/matttest/results/pdf_params_{}.npy'.format(quant.name)) and not overwrite:
                if sim:
                    qq = np.load('/home/matt/software/matttest/results/sim_params_{}.npy'.format(quant.name))[()]    
                else:
                    qq = np.load('/home/matt/software/matttest/results/pdf_params_{}.npy'.format(quant.name))[()]
                dists[idx] = qq
        #        src.plot_1d_projections(qq, index_cnt=0, plot_2d=True, use_array=True)
                idx += 1
                continue
            params_0 = src.pdf_param_guesses(quant.name)
            quant.add_params(params_0)
            pa, bnds = src.pdf_param_guesses(quant.name, return_array=True)
            quant.add_paramarray(pa, bnds)
            index_cnt = 0
        #    src.plot_1d_projections(quant, index_cnt=index_cnt, plot_2d=True)
            if sim:
                fit_params = src.fit_nd_dist(quant, save_name = 'sim'+quant.name, index_cnt=index_cnt, method='opt_minimize')
            else:
                fit_params = src.fit_nd_dist(quant, save_name = quant.name, index_cnt=index_cnt, method='opt_minimize')
        else:
#            params_0 = src.pdf_param_guesses(quant.name)
#            quant.add_params(params_0)
#            pa, bnds = src.pdf_param_guesses(quant.name, return_array=True)
#            quant.add_paramarray(pa, bnds)
            index_cnt = 0
            fit_params = src.fit_nd_dist(quant, index_cnt=index_cnt, method='opt_minimize')
    #    quant.params = fit_params
        if plot:
            src.plot_1d_projections(quant, index_cnt=index_cnt, plot_2d=True, use_array=True)
    #    np.save('/home/matt/software/matttest/results/pdf_params_{}.npy'.format(quant.name), quant)
        idx += 1
    if bootstrap:
        return fit_params

print "Running initial parameter fit"
### Normalized behavior for cauchy3_irq needs to be specially set within the function (for now)
find_dist_params(dists, overwrite = True, plot=True, sim=sim)

if blob_type == 'eff':
    RCE = np.load('/home/matt/software/matttest/results/pdf_params_{}.npy'.format(RCE.name))[()]
    EFF = np.load('/home/matt/software/matttest/results/pdf_params_{}.npy'.format(EFF.name))[()]
    EFF.normalized = True ## For now, set to true for faster processing

### Error analysis via Bootstrapping
print "Running bootstrap (to estimate errors)"
tb0 = time.time()
overwriteb = False # overwrite bootstrap param files?
if blob_type == 'eff':
    if sim:
        frce = '/home/matt/software/matttest/results/sim_params_boot_{}.npy'.format(RCE.name)
        feff = '/home/matt/software/matttest/results/sim_params_boot_{}.npy'.format(EFF.name)
    else:
        frce = '/home/matt/software/matttest/results/pdf_params_boot_{}.npy'.format(RCE.name)
        feff = '/home/matt/software/matttest/results/pdf_params_boot_{}.npy'.format(EFF.name)
    if os.path.isfile(frce) and os.path.isfile(feff) and not overwriteb:
        rce_params = np.load(frce)
        eff_params = np.load(feff)
    else:
        nboot = 1000 #Set to 1000? - might need to parallelize...
        rce_params = np.zeros((2, nboot)) ### Weibull params
        eff_params = np.zeros((9, nboot)) ### Joint pdf params
        for bt in range(nboot):
            if bt%50 == 0:
                print("On bootstrap iteration {}/{}".format(bt,nboot))
            ### Resample all clump, galaxy parameters
            nbbs, rebs, iebs, rsepbs, nsersbs, qbbs, PAbb, Qgalbs, PAgalbs = src.resample(nb, re, ie, rsep, nsers, qb, PAb, Qgal, PAgal)
            RCE.data = rsepbs.reshape(1,len(rsepbs))
            EFF.data = np.vstack((rebs, iebs, qbbs))
            rce_params[:, bt] = find_dist_params(RCE, bootstrap=True)
            eff_params[:, bt] = find_dist_params(EFF, bootstrap=True)
    
tbf = time.time()
if blob_type == 'eff':
    if sim:
        np.save('/home/matt/software/matttest/results/sim_params_boot_{}.npy'.format(RCE.name),rce_params)
        np.save('/home/matt/software/matttest/results/sim_params_boot_{}.npy'.format(EFF.name),eff_params)
    else:
        np.save('/home/matt/software/matttest/results/pdf_params_boot_{}.npy'.format(RCE.name),rce_params)
        np.save('/home/matt/software/matttest/results/pdf_params_boot_{}.npy'.format(EFF.name),eff_params)
print "Bootstrap time =", tbf-tb0
#for k in range(2):
#    plt.plot(rce_params[k,:],'k.')
#    plt.title("RCE param {}".format(k))
#    plt.show()
#    plt.close()
#for j in range(9):
#    plt.plot(eff_params[j,:],'k.')
#    plt.title("EFF param {}".format(j))
#    plt.show()
#    plt.close()


### Calculate error bars for each parameter, save final results
### col 1, -err, col 2 + err, row1 = param1, row2 = param2, etc...
def bootstrap_errs(bootstrap_params,low=16,high=84, true_params=None):
    """ gets errors from array of bootstrap parameters.
        Can adjust limits (default is 16th and 84th percentiles)
        Returns median, -err, +err
    """
    if len(bootstrap_params.shape) == 1:
        iters = 1
        bootstrap_params = np.reshape(bootstrap_params, (1, len(bootstrap_params)))
    else:
        iters = bootstrap_params.shape[0]
    errs = np.zeros((iters, 3))
    for i in range(iters):
        if true_params is None:
            errs[i, 0] = np.median(bootstrap_params[i])
        else:
            errs[i,0] = true_params[i]
        errs[i, 1] = np.percentile(bootstrap_params[i], low) - np.median(bootstrap_params[i])
        errs[i, 2] = np.percentile(bootstrap_params[i], high) - np.median(bootstrap_params[i])
    return errs
        
if blob_type == 'eff':
    rce_params_errs = bootstrap_errs(rce_params, true_params=RCE.paramarray)
    eff_params_errs = bootstrap_errs(eff_params, true_params=EFF.paramarray)
    if sim:
        np.save('/home/matt/software/matttest/results/sim_params_final_{}.npy'.format(RCE.name), rce_params_errs)
        np.save('/home/matt/software/matttest/results/sim_params_final_{}.npy'.format(EFF.name), eff_params_errs)
    else:
        np.save('/home/matt/software/matttest/results/pdf_params_final_{}.npy'.format(RCE.name), rce_params_errs)
        np.save('/home/matt/software/matttest/results/pdf_params_final_{}.npy'.format(EFF.name), eff_params_errs)

#print rce_params_errs
#print eff_params_errs
#exit(0)

### Draw from the distributions above to create simulated galaxies
if sim:
    exit(0) ## - don't re-make simulated galaxies
dists = [RCE, EFF]
if blob_type == 'eff':
    for Pdf in dists:
        if Pdf.name == 'rsepe':
            Pdf.add_draw(np.random.weibull)
        elif Pdf.name == 'eff_joint':
            Pdf.add_draw(np.array((3*[sf.cauchy_draw])))


##############################################################################
'''
### run empirical corrections for rc as a fct of nclumps
n_gxys = 20
nclumps_mx = 8/0.66
profile = blob_type
dim=76
xc, yc = dim/2, dim/2
delta_cents = []# np.zeros((n_gxys))
qs = []
ratio = np.zeros((n_gxys))
for i in range(n_gxys):
    nclumps = int(np.random.rand()*nclumps_mx + 1)
    fake_gal, qtmp, xytmp = src.make_sim_gal(dists, dim=dim, profile=profile, return_noise=False, nclumps=nclumps)
    qs += list(qtmp)
    xs = xytmp[0,:]
    ys = xytmp[1,:]
    center = sf.centroid(fake_gal)
    rcenters = np.sqrt((xs-center[0])**2 + (ys-center[0])**2)
    ratio[i] = np.mean(rcenters)/np.mean(qtmp)
#    print center
#    print rcenters
#    print qtmp
#    print ratio[i]
#    plt.imshow(fake_gal,interpolation='none')
#    plt.show()
#    plt.close()
#    delta_cents[i] = np.sqrt((center[0]-xc)**2 + (center[1]-yc)**2)
    

print "nclumps = ", nclumps
print np.nanmean(ratio)
print np.nanstd(ratio)
plt.hist(ratio,100)
plt.show()
#print np.nanmean(delta_cents)
#print np.nanstd(delta_cents)
#print np.mean(qs)
#print np.mean(qs)/np.sqrt((np.mean(qs)**2+np.nanmean(delta_cents)**2))
exit(0)
#'''
##############################################################################
            
n_gxys = 17
#fig, ax = plt.subplots(5,4,sharex=True,sharey=True)
#fig.subplots_adjust(wspace=0,hspace=0)
#profile = 'sersic'
profile = blob_type
dim=76
#plt.ion()
fig = plt.figure(figsize=(10,7.75))
#fig.suptitle("Simulated Source Images, {} profile, {}x{}px".format(profile,dim,dim),fontsize=18)
fig.subplots_adjust(wspace=0,hspace=0)
#fig.subplots_adjust(vspace=0)
qs = []
save= True
for i in range(n_gxys):
    i1 = int(np.mod(i,5))
    i2 = int(np.floor(i/5))
    dim1 = 100
    fake_gal, qtmp, noise = src.make_sim_gal(dists, dim=dim, profile=profile, return_noise=True, save=save, save_idx=i)
    ### save for later analysis - overwrite every time right now...
    if not save:
        hdu0 = pyfits.PrimaryHDU(fake_gal)
        hdu1 = pyfits.PrimaryHDU(noise)
        ### Additional new header values
        hdu0.header.append(('UNITS','Counts','Relative photon counts (no flat fielding)'))
        hdu1.header.append(('UNITS','Noise/Error','standard deviation'))
        hdulist = pyfits.HDUList([hdu0])
        hdulist.append(hdu1)
        hdulist.writeto('/home/matt/software/matttest/results/sim_gxy_{:03d}.fits'.format(i),clobber=True)
    qs += list(qtmp)
#    dims = [100, 100]
#    fake_gal = fake_gal[dim]
#    fake_gal, junk = src.crop_images(fake_gal, np.ones(fake_gal.shape), bgcut = 0.1*np.max(fake_gal-np.mean(fake_gal))+np.mean(fake_gal), dims=dims)
    plt.subplot(4,5,i+1)
    ax = plt.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
#    vlims = None
    if vlims is None:
        fake_gal[70:71,5:15] = np.max(fake_gal)
        plt.imshow(fake_gal,interpolation='none',cmap=cm.hot)
    else:
        fake_gal[70:71,5:15] = vlims[1]
        plt.imshow(fake_gal,interpolation='none',cmap=cm.hot, vmin=vlims[0], vmax=vlims[1])
    ax.text(4, 68, '0.1"', color='w', fontsize=6)
#    ratio = (ax.get_xlim()[0]-ax.get_xlim()[1])/(ax.get_ylim()[1]-ax.get_ylim()[0])
#    ax.set_aspect(1*ratio, adjustable='box')
#    plt.show()
#    plt.close()
plt.savefig('/home/matt/software/matttest/results/figs/sim_mosaic.pdf')
plt.show()
plt.close()
### can check on various quantities to make sure draw is working right
#plt.hist(qs, bins=14)
#plt.show()

'''
#Primitive framework for evaluating magnitudes of sources/lensed images
photflam = 1.1845489*10**-19
photplam = 5887.4326
abmag_zpt = -2.5*np.log10(photflam) - 21.10 - 5*np.log10(photplam) + 18.6921
idx = 3
img = pyfits.open(filenames[idx])
print filenames[idx]
### lens plane source only
src_mag = img[7].data
src_unmag = img[8].data#[382:,:]
#plt.ion()
xct, yct = sf.centroid(src_unmag)
print xct, yct
plt.imshow(src_unmag,interpolation='none')
plt.show()
abs_mag = -2.5*np.log10(np.sum(src_mag)) + abmag_zpt
abs_unmag = -2.5*np.log10(np.sum(src_unmag)*(4)**-2) + abmag_zpt
print "Lensed Magnitude =", abs_mag
print "Unlensed Magnitude =", abs_unmag
print "Magnification =", 10**((abs_unmag-abs_mag)/2.5)

### Try K-S tests
xcw = 54.4285 # 31.4487 # 
ycw = 44.8143 # 36.1865 #
xinds = np.arange(-50,50)+src_unmag.shape[1]/2
yinds = np.arange(-50,50)+src_unmag.shape[0]/2
src_unmag = src_unmag[yinds[0]:yinds[-1],xinds[0]:xinds[-1]]
rarr = sf.make_rarr(np.arange(src_unmag.shape[1]),np.arange(src_unmag.shape[0]),xcw,ycw)
rinds = np.argsort(np.ravel(rarr))
xarr = np.ravel(rarr)[rinds]
yarr = np.ravel(src_unmag)[rinds]
yarr[yarr<0] = 0 # set all positive
yarr /= np.sum(yarr) # Normalize
ycum = np.cumsum(yarr)
plt.plot(xarr,ycum)
plt.show()
exit(0)
#'''


''' Initial simulated source draws
xarr = np.linspace(0,1,50)
#xarr = 0.5
xblob = np.linspace(0,10,100)
n = 2
sig = 0.1
#ldist = src.trial_lfrac(xarr,n,sig)
#nblobs = src.trial_nblobs(xblob)
#plt.ion()
#plt.plot(xarr,ldist,'k.')
#plt.plot(xblob,nblobs*300)
#print np.sum(nblobs)*(xblob[1]-xblob[0])

#blobs = sf.pdf_draw(src.trial_nblobs,n=10000,args=None,int_lims=[0,10],res=1000)
#blobs = np.ceil(blobs)
#plt.hist(blobs)

#lfrac = src.lfrac_nblobs()
#print lfrac

#nblobs, fwhm, sbmx, rsep, Qrat = src.import_visual_blob_values(plot_dists=True)
nblobs, fwhm, sbmx, rsep, Qrat = src.import_visual_blob_values(plot_dists=True)
exit(0)

#Qxx = np.mean(Qij,axis=0)[0]
#Qyy = np.mean(Qij,axis=0)[1]
#Qxy = np.mean(Qij,axis=0)[2]
#Qxx = Qij[1,0]
#Qyy = Qij[1,1]
#Qxy = Qij[1,2]
#plt.plot(xv,yv,'bo')
#plt.show()
#plt.hist(xv)
#plt.figure()
#plt.hist(yv)
#plt.ion()
#plt.show()
#exit(0)

#rdist = src.trial_rrel(np.arange(55))
#plt.plot(np.arange(55),rdist)

### Blob parameters
params = dict()
params['p_lam'] = 5.5
params['g_mu'] = 5.5/2.4
params['g_sig'] = 2.2/4
params['w_k'] = 1.4
params['w_lam'] = 10
params['e_tau'] = 0.2

nf = 17
#fig, ax = plt.subplots(5,4,sharex=True,sharey=True)
#fig.subplots_adjust(wspace=0,hspace=0)
profile = 'gaussian'
fig = plt.figure()
fig.suptitle("Simulated Source Images, {} profile, 100x100px".format(profile),fontsize=18)
fig.subplots_adjust(wspace=0,hspace=0)
#fig.subplots_adjust(vspace=0)
for i in range(nf):
    i1 = int(np.mod(i,5))
    i2 = int(np.floor(i/5))
    fake_gal = src.make_blob_gal(params,profile=profile)
#    plt.ion()
    plt.subplot(3,6,i+1)
    ax = plt.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    plt.imshow(fake_gal,interpolation='none')
plt.show()
plt.close()
#'''