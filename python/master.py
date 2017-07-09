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
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-e","--evalfit",help="Extracts fitted blob values", action='store_true')
parser.add_argument("-o","--overwrite",help="Overwrites existing fits for parameters", action='store_true')
parser.add_argument("-p","--plotfit",help="Plots fitted values", action='store_true')
parser.add_argument("-b","--bootstrap",help="Overwrites saved bootstrap", action='store_true')
parser.add_argument("-s","--sim",help="Take inputs from simulated galaxies", action='store_true')
parser.add_argument("-t","--blob_type",help="SB profile model for clumps/blobs", default='eff')
parser.add_argument("-k","--kw_extra",help="append param_kw for special cases", default='')
parser.add_argument("-x","--exact",help="use true/exact parameters from simulations", action='store_true')
parser.add_argument("-c","--compare",help="look at sim bias-true", action='store_true')
args_in = parser.parse_args()
blob_type = args_in.blob_type
kw_extra = args_in.kw_extra
exact = args_in.exact
compare = args_in.compare
if kw_extra is not '':
    kw_extra = '_' + kw_extra

try:
    hst_dir = os.environ['HST_DIR']
except:
    print("set environvental varable HST_DIR")
    exit(0)
    
cycle = '23'
program = '14189'
root = os.path.join(hst_dir,cycle,program)
### make list of filenames
if kw_extra == '':
    filenames = [f for f in glob.glob(os.path.join(root,'*','*SIE*pix.fits')) if not 'sersic' in f and not 'SLACS' in f and not 'div' in f and not 'tim' in f]
elif kw_extra == 'div10':
    filenames = [f for f in glob.glob(os.path.join(root,'*','*SIE*pix_div10.fits')) if not 'sersic' in f and not 'SLACS' in f]
elif kw_extra == 'tim10':
    filenames = [f for f in glob.glob(os.path.join(root,'*','*SIE*pix_tim10.fits')) if not 'sersic' in f and not 'SLACS' in f]
if False:
    filenames = [f for f in glob.glob(os.path.join(root,'*','*sersic*.fits')) if not 'pix' in f and not 'SLACS' in f]

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

### Test bias on sigma from gaussian
#trials = 1000
#sig_ns = np.zeros((trials))
#for i in range(trials):
#    nt = 10000
#    xar = np.linspace(-3,3,nt)
#    ys = sf.gaussian(xar, 1)
#    yar = np.random.randn(nt)
#    ns = 0.6*np.random.randn(nt)
#    dat = yar + ns
#    mnd = np.sum(dat)/nt
#    sig_emp = (1/(nt-1))*np.sum((dat-mnd)**2)
#    sig_ns[i] = np.sqrt(abs(sig_emp-nt/(nt-1)))
#print np.mean(sig_ns)
##plt.ion()
##plt.hist(dat, min(nt/10,100), normed = True)
##plt.plot(xar, ys, 'k')
##plt.show()
#exit(0)

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
param_kw = '_' + blob_type + kw_extra
#param_kw='_sersic'
#blob_type = 'eff'
#blob_type = 'sersic'
if blob_type == 'eff':
    gam_fixed = True
    if not gam_fixed:
        param_kw += '_gam_free'

#########################################################################
#########################################################################
###
###  11
### 111
###  11
###  11
###  11
### 1111
### 
#########################################################################
#########################################################################
### Load Fitted values (found using LAE.py)
sim = args_in.sim
if sim:
    ### Specially saved sim galaxies
#    filenames = np.loadtxt('/home/matt/software/matttest/results/sim_gxys/sim_gxys.txt', dtype=str)
    ### Readily updated sim galaxies
    if exact:
        filenames = np.loadtxt('/home/matt/software/matttest/results/sim_gxys.txt', dtype=str)
    else:
        filenames = np.loadtxt('/home/matt/software/matttest/results/sim_gxys_fit.txt', dtype=str)
elif kw_extra == '':
    filenames = np.loadtxt('/home/matt/software/matttest/data/pix_source_models.txt', dtype=str)
elif kw_extra == '_div10':
    filenames = np.loadtxt('/home/matt/software/matttest/data/pix_source_models_div10.txt', dtype=str)
elif kw_extra == '_tim10':
    filenames = np.loadtxt('/home/matt/software/matttest/data/pix_source_models_tim10.txt', dtype=str)
    
#src.source_mosaic(filenames, vlims=vlims)
#exit(0)    
  
#if sim:
#    src.fitted_mosaic(data='/home/matt/software/matttest/results/sim_gxys/sim_gxys.txt',param_kw=param_kw, blob_type=blob_type, vlims=vlims)
#else:
#    src.fitted_mosaic(param_kw=param_kw, blob_type=blob_type, vlims=vlims)
#exit(0)
  
logscale = False
overwrite1 = args_in.evalfit
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
    [nb, re, ie, rsep, nsers, qb, PAb, Qgal, PAgal, rpx] = np.load(fname)
elif os.path.isfile('/home/matt/software/matttest/results/sim_clump_values{}_{}.npy'.format(param_kw,"log" if logscale else "lin")) and not overwrite1 and sim:
    fname = '/home/matt/software/matttest/results/sim_clump_values{}_{}.npy'.format(param_kw,"log" if logscale else "lin")
    [nb, re, ie, rsep, nsers, qb, PAb, Qgal, PAgal, rpx] = np.load(fname)
else:
    use_model = False#not sim
    if sim:
        filenamesc = np.loadtxt('/home/matt/software/matttest/results/sim_gxys.txt', dtype=str)
        centroids, qgals, PAgals = src.get_centroids(filenamesc, use_model=use_model, return_axis_ratio=True, param_kw=param_kw)
    else:
        return_rhalf = False
        if return_rhalf:
            rhalf = src.get_centroids(filenames, use_model=True, return_axis_ratio=False, param_kw=param_kw, return_rhalf=return_rhalf)
            print rhalf
            print np.median(rhalf)
            plt.hist(rhalf)
            plt.show()
            exit(0)
        else:
            centroids, qgals, PAgals = src.get_centroids(filenames, use_model=use_model, return_axis_ratio=True, param_kw=param_kw)
    ### Check nblobs with Icut
#    nblobs = src.import_fitted_blob_values(filenames, plot_dists=True, centroids=centroids, qgal=qgals, PAgal=PAgals, logscale=logscale, bins=14, param_kw=param_kw,sim=sim, exact=exact, Icut=True)
#    print nblobs
#    print np.sum(nblobs)
#    plt.hist(nblobs)
#    plt.show()
#    plt.close()
#    exit(0)
    print "Importing fitted blob values"
    nb, re, ie, rsep, nsers, qb, PAb, Qgal, PAgal, rpx = src.import_fitted_blob_values(filenames, plot_dists=True, centroids=centroids, qgal=qgals, PAgal=PAgals, logscale=logscale, bins=14, param_kw=param_kw,sim=sim, exact=exact)
#    plt.hist(Qgal,bins=9)
#    plt.show()
#    plt.close()
    if sim:
        np.save('/home/matt/software/matttest/results/sim_clump_values{}_{}.npy'.format(param_kw,"log" if logscale else "lin"), [nb, re, ie, rsep, nsers, qb, PAb, Qgal, PAgal, rpx])
    else:
        np.save('/home/matt/software/matttest/results/fitted_clump_values{}_{}.npy'.format(param_kw,"log" if logscale else "lin"), [nb, re, ie, rsep, nsers, qb, PAb, Qgal, PAgal, rpx])

if overwrite1 == True:
    print "total number of clumps =", re.size
    exit(0)
    plt.close()
    plt.close()
    plt.close()
    plt.ioff()
ie = ie.astype(float)


### Load "true" sim values for comparison
#exact = True # use exact values from input simulations, rather than fitted
#if sim and exact:
#    filenames = np.loadtxt('/home/matt/software/matttest/results/sim_gxys.txt', dtype=str)
#    re, rpx, rsep, ie, qb, PAb = src.import_sim_blob_values(filenames)
#    nsers = 1.5*np.ones((len(re)))

if logscale:
#    logre = re
    logie = ie
#thta = np.pi/15
#thta = src.min_rotation(ie, nsers, thta0=0)

### check significance of correlation coefficients
ccs = src.corr_coeff_bootstrap(nb, re, ie, rsep, nsers, qb, PAb, Qgal, PAgal, sig=2)
#print ccs
#exit(0)

use_qb_mask = False
if use_qb_mask:
    qbm = qb > 0.101
    ie = ie[qbm]
    re = re[qbm]
    nsers = nsers[qbm]
    qb = qb[qbm]
    PAb = PAb[qbm]
    
use_re_mask = False
if use_re_mask:
    rem = re < 1
    ie = ie[rem]
    re = re[rem]
    rpx = rpx[rem]
    rsep = rsep[rem]
    nsers = nsers[rem]
    qb = qb[rem]
    PAb = PAb[rem]

#############################################################################
######## Space for testing various other functions ##########################
#############################################################################
#'''
### Selection function
trials = 100000
bb_array = src.calc_bb_array(ie, rpx)  
sf_array = src.calc_selection_fct(bb_array, trials=trials)
#try:
#    pcnts_array = (sf_array[:,:,0]/sf_array[:,:,1])
#except:
#    print "need more trials"
pcnts_total = (np.sum(sf_array,axis=1)[:,0]/np.sum(sf_array,axis=1)[:,1])
#bmsh = (bb_array >0.00063) * (bb_array < 0.00065)
#bmsh = (bb_array >0.00395) * (bb_array < 0.00397)
#sf1 = sf_array[np.argmax(bmsh)]
#print sf1
#print sf1[:,0]/sf1[:,1]
#plt.plot(sf1[:,0]/sf1[:,1])
#plt.show()
#plt.plot(bb_array, pcnts_total, 'k.')
#plt.show()
#exit(0)

bad_find_msk = pcnts_total < 0.01 ## cut out anything below 1% as unlikely to have been found originally...
weights = 1/pcnts_total
weights[weights > 100] = 100
weights[weights == np.inf] = 100
weights[weights == 0] = 1
if kw_extra != '':
    weights = np.ones(weights.shape)#*0.01
#weights[1] = 1000
#print weights
#exit(0)
copies = weights.astype(int)
rextra = np.array(([]))#np.zeros((np.sum(copies)-len(copies)))
iextra = np.array(([]))#np.zeros((np.sum(copies)-len(copies)))
dextra = np.array(([]))
for i in range(len(copies)):
    if copies[i] > 1:
        rextra = np.append(rextra,np.ones((copies[i]-1))*re[i])
        iextra = np.append(iextra,np.ones((copies[i]-1))*ie[i])
        dextra = np.append(dextra,np.ones((copies[i]-1))*rsep[i])
rtot = np.append(re,rextra)
itot = np.append(ie,iextra)
dtot = np.append(rsep,dextra)
#plt.hist(rtot,20)
#plt.show()
#plt.hist(itot,20)
#plt.show()
#plt.hist(dtot,20)
#plt.show()
#exit(0)
#'''

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

#########################################################################
#########################################################################
###
###  222
### 22 22
###    22
###   22
###  22
### 222222
###
#########################################################################
#########################################################################
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
    Sersic = src.Pdf_info('ie_re_nsers_comb')
    ### Right now I have to make special functions for each shape, can't think of a 
    ### more general way to cover all cases :(
    rmx = 2
    recut = re[re<rmx]
    iecut = ie[re<rmx]
    nserscut = nsers[re<rmx]
    Sersic.add_data(np.vstack((recut,iecut,nserscut)),sf.gauss3_irn, normalized=False)
#    dists = [RC, Sersic]
    dists = [QGS, QBS, RCS, Sersic]
elif blob_type == 'eff':
    NBE = src.Pdf_info('nbe')
    NBE.add_data(nb,np.random.uniform)
    QGE = src.Pdf_info('qgale')
    QGE.add_data(Qgal,sf.cauchy_lmfit_trunc)
#    QGE.add_invar(np.ones(Qgal.shape))
    QBE = src.Pdf_info('qb')
#    if kw_extra == '':
    QBE.add_data(qb, sf.cauchy_lmfit_trunc)#, normalized=False)
#    el:
#        QBE.add_data(qb, sf.gaussian_lmfit_trunc) #for div10
#    if sim:
    QBE.add_invar(np.ones(qb.shape))
#    else:
#        QBE.add_invar(weights)
    DCE = src.Pdf_info('rsepe')
    DCE.add_data(rsep, sf.weibull_lmfit)#_trunc)#, normalized=True)
#    if sim:
    DCE.add_invar(np.ones(rsep.shape))
#    else:
#        RCE.add_invar(weights)
    ICE = src.Pdf_info('ic')
#    ICE.add_data(ie, sf.cauchy_lmfit_trunc)
    ICE.add_data(ie, sf.weibull_lmfit)
    ICE.add_invar(weights)
    RCE = src.Pdf_info('rc')
    RCE.add_data(re, sf.cauchy_lmfit_trunc)
    RCE.add_invar(weights)
#    EFF = src.Pdf_info('eff_joint')
##    EFF.add_data(np.vstack((re, ie, qb)), sf.cauchy3_irq, normalized=False)
#    EFF.add_data(np.vstack((re, ie, qb)), sf.gen_central2d, normalized=False)
#    EFF.add_kwargs(['cauchy','cauchy','cauchy']) ### distribution for each...
    '''
    #Various efforts at joint distributions if they need to be revived later
    IRD = src.Pdf_info('ird')
    IRD.add_data(np.vstack((re, ie, rsep)), sf.gen_central2d, normalized=False)
#    IRD.add_data(np.vstack((rtot, itot, dtot)), sf.gen_central2d, normalized=False)
    if sim:
        IRD.add_invar(np.vstack((np.ones(re.shape), np.ones(ie.shape), np.ones(rsep.shape))))
    else:
        IRD.add_invar(np.vstack((weights,weights,np.ones(rsep.shape))))
    IRD.add_kwargs(['cauchy','cauchy','gaussian'])
    IRJ = src.Pdf_info('irj')
    IRJ.add_data(np.vstack((re, ie)), sf.gen_central2d, normalized=False)
#    IRD.add_data(np.vstack((rtot, itot, dtot)), sf.gen_central2d, normalized=False)
    if sim:
        IRJ.add_invar(np.vstack((np.ones(re.shape), np.ones(ie.shape))))
    else:
        IRJ.add_invar(np.vstack((weights,weights)))
#    IRJ.add_kwargs(['cauchy','cauchy'])
    IRJ.add_kwargs(['lorentz'])
    #'''
    if not gam_fixed:
        GAM = src.Pdf_info('gamma')
        GAM.add_data(nsers, sf.gaussian_lmfit)
#    dists = [RCE, EFF]
    dists = [QGE, QBE, DCE, ICE, RCE] #IRJ]# IRD] 
#    dists = [ICE, RCE]

#dists = [RC, LogIE, NS, LogRE]
#dists = [RC, Sersic]

def find_dist_params(dists, overwrite=False, bootstrap=False, plot=False, sim=False):
    idx = 0
    if type(dists) != list:
        dists = [dists]
    for quant in dists:
        if not bootstrap:
            if os.path.isfile('/home/matt/software/matttest/results/pdf_params_{}{}.npy'.format(quant.name, kw_extra)) and not overwrite:
                if sim:
                    if exact:
                        extra = '_exact'
                    else:
                        extra = ""
                    qq = np.load('/home/matt/software/matttest/results/pdf_params_sim_{}{}.npy'.format(quant.name,extra))[()]    
                else:
                    qq = np.load('/home/matt/software/matttest/results/pdf_params_{}{}.npy'.format(quant.name, kw_extra))[()]
                dists[idx] = qq
        #        src.plot_1d_projections(qq, index_cnt=0, plot_2d=True, use_array=True)
                idx += 1
                continue
            params_0 = src.pdf_param_guesses(quant.name)
            quant.add_params(params_0)
            pa, bnds = src.pdf_param_guesses(quant.name, return_array=True)
            quant.add_paramarray(pa, bnds)
            index_cnt = 0
#            src.plot_1d_projections(quant, index_cnt=index_cnt, plot_2d=True)#, use_array=True)
            if sim:
                if exact:
                    extra = '_exact'
                else:
                    extra = ""
                fit_params = src.fit_nd_dist(quant, save_name = 'sim_'+quant.name+extra, index_cnt=index_cnt, method='opt_minimize')
            else:
                print "fitting:", quant.name
                time.sleep(2)
                fit_params = src.fit_nd_dist(quant, save_name = quant.name+kw_extra, index_cnt=index_cnt, method='opt_minimize')
                print quant.paramarray
        else:
#            params_0 = src.pdf_param_guesses(quant.name)
#            quant.add_params(params_0)
#            pa, bnds = src.pdf_param_guesses(quant.name, return_array=True)
#            quant.add_paramarray(pa, bnds)
            index_cnt = 0
            fit_params = src.fit_nd_dist(quant, index_cnt=index_cnt, method='opt_minimize')
#            print quant.paramarray
    #    quant.params = fit_params
        if plot:
            src.plot_1d_projections(quant, index_cnt=index_cnt, plot_2d=True, use_array=True)
    #    np.save('/home/matt/software/matttest/results/pdf_params_{}.npy'.format(quant.name), quant)
        idx += 1
    if bootstrap:
        return fit_params

print "Running initial parameter fit"
### Normalized behavior for cauchy3_irq needs to be specially set within the function (for now)
find_dist_params(dists, overwrite = args_in.overwrite, plot=args_in.plotfit, sim=sim)

if blob_type == 'eff':
#    RCE = np.load('/home/matt/software/matttest/results/pdf_params_{}.npy'.format(RCE.name))[()]
#    EFF = np.load('/home/matt/software/matttest/results/pdf_params_{}.npy'.format(EFF.name))[()]
#    EFF.normalized = True ## For now, set to true for faster processing
    if sim:
        for d in dists:
            if compare:
                d1 = np.load('/home/matt/software/matttest/results/pdf_params_sim_{}{}.npy'.format(d.name,'_exact'))[()]
                d2 = np.load('/home/matt/software/matttest/results/pdf_params_sim_{}{}.npy'.format(d.name,""))[()]
                print d1.paramarray-d2.paramarray
            else: 
                if exact:
                    extra = '_exact'
                else:
                    extra = ""
                d = np.load('/home/matt/software/matttest/results/pdf_params_sim_{}{}.npy'.format(d.name,extra))[()]
                print d.paramarray
        if compare:
            exit(0)
#        QGE = np.load('/home/matt/software/matttest/results/pdf_params_sim_{}.npy'.format(QGE.name))[()]
#        QBE = np.load('/home/matt/software/matttest/results/pdf_params_sim_{}.npy'.format(QBE.name))[()]
#        IRD = np.load('/home/matt/software/matttest/results/pdf_params_sim_{}.npy'.format(IRD.name))[()]
    else:
        for d in dists:
            d = np.load('/home/matt/software/matttest/results/pdf_params_{}.npy'.format(d.name))[()]
#        QGE = np.load('/home/matt/software/matttest/results/pdf_params_{}{}.npy'.format(QGE.name,kw_extra))[()]
#        QBE = np.load('/home/matt/software/matttest/results/pdf_params_{}{}.npy'.format(QBE.name,kw_extra))[()]
#        IRD = np.load('/home/matt/software/matttest/results/pdf_params_{}{}.npy'.format(IRD.name,kw_extra))[()]
    

#########################################################################
#########################################################################
###
###  3333
### 33  33
###    33
###  333 
###    33
### 33  33
###  3333
###
#########################################################################
#########################################################################

### Error analysis via Bootstrapping
print "Running bootstrap (to estimate errors)"
tb0 = time.time()
overwriteb = args_in.bootstrap # overwrite bootstrap param files?
if blob_type == 'eff':
    fnames = [""]*len(dists)#np.zeros((len(dists)))#['fqge', fqbe, fdce, fice, frce]
    pnames = [""]*len(dists)#np.zeros((len(dists)))#[qge_params, qbe_params, dce_params, ice_params, rce_params]
    if sim:
        for idx in range(len(fnames)):
            fnames[idx] = '/home/matt/software/matttest/results/sim_params_boot_{}.npy'.format(dists[idx].name)
#        frce = '/home/matt/software/matttest/results/sim_params_boot_{}.npy'.format(RCE.name)
#        feff = '/home/matt/software/matttest/results/sim_params_boot_{}.npy'.format(EFF.name
#        fqge = '/home/matt/software/matttest/results/sim_params_boot_{}.npy'.format(QGE.name)
#        fqbe = '/home/matt/software/matttest/results/sim_params_boot_{}.npy'.format(QBE.name)
#        frce = '/home/matt/software/matttest/results/sim_params_boot_{}.npy'.format(RCE.name)
#        fird = '/home/matt/software/matttest/results/sim_params_boot_{}.npy'.format(IRD.name)
#        firj = '/home/matt/software/matttest/results/sim_params_boot_{}.npy'.format(IRJ.name)
    else:
        for idx in range(len(fnames)):
            fnames[idx] = '/home/matt/software/matttest/results/pdf_params_boot_{}{}.npy'.format(dists[idx].name,kw_extra)
#        frce = '/home/matt/software/matttest/results/pdf_params_boot_{}.npy'.format(RCE.name)
#        feff = '/home/matt/software/matttest/results/pdf_params_boot_{}.npy'.format(EFF.name)
#        fqge = '/home/matt/software/matttest/results/pdf_params_boot_{}{}.npy'.format(QGE.name,kw_extra)
#        fqbe = '/home/matt/software/matttest/results/pdf_params_boot_{}{}.npy'.format(QBE.name,kw_extra)
#        frce = '/home/matt/software/matttest/results/pdf_params_boot_{}{}.npy'.format(RCE.name,kw_extra)
#        fird = '/home/matt/software/matttest/results/pdf_params_boot_{}{}.npy'.format(IRD.name,kw_extra)
#        firj = '/home/matt/software/matttest/results/pdf_params_boot_{}{}.npy'.format(IRJ.name,kw_extra)
#    if os.path.isfile(frce) and os.path.isfile(feff) and not overwriteb:
#        rce_params = np.load(frce)
#        eff_params = np.load(feff)
    pths = [os.path.isfile(f) for f in fnames]
    if np.prod(pths) == 1 and not overwriteb:
        for i in range(len(pnames)):
            pnames[i] = np.load(fnames[i])
#        qge_params = np.load(fqge)
#        qbe_params = np.load(fqbe)
#        ird_params = np.load(fird)
#    else:
#        nboot = 1000 #Set to 1000? - might need to parallelize...
#        rce_params = np.zeros((2, nboot)) ### Weibull params
#        eff_params = np.zeros((9, nboot)) ### Joint pdf params
#        for bt in range(nboot):
#            if bt%50 == 0:
#                print("On bootstrap iteration {}/{}".format(bt,nboot))
#            ### Resample all clump, galaxy parameters
#            nbbs, rebs, iebs, rsepbs, nsersbs, qbbs, PAbb, Qgalbs, PAgalbs = src.resample(nb, re, ie, rsep, nsers, qb, PAb, Qgal, PAgal)
#            RCE.data = rsepbs.reshape(1,len(rsepbs))
#            EFF.data = np.vstack((rebs, iebs, qbbs))
#            rce_params[:, bt] = find_dist_params(RCE, bootstrap=True)
#            eff_params[:, bt] = find_dist_params(EFF, bootstrap=True)
    else:
        nboot = 1000 #Set to 1000? - might need to parallelize...
        ### have to manually adjust this if distributions change...
        for i in range(len(pnames)):
            pnames[i] = np.zeros((2,nboot))
#        qge_params = np.zeros((2, nboot))
#        qbe_params = np.zeros((2, nboot))
#        ird_params = np.zeros((9, nboot)) ### Joint pdf params
        for bt in range(nboot):
            if bt%50 == 0:
                print("On bootstrap iteration {}/{}".format(bt,nboot))
            ### Resample all clump, galaxy parameters
            nbbs, rebs, iebs, rsepbs, nsersbs, qbbs, PAbb, Qgalbs, PAgalbs = src.resample(nb, re, ie, rsep, nsers, qb, PAb, Qgal, PAgal)
            ### also need to manually adjust this if distributions change
            for idx in range(len(pnames)):
                if idx == 0:
                    quant = 1.0*Qgalbs
                elif idx == 1:
                    quant = 1.0*qbbs
                elif idx == 2:
                    quant = 1.0*rsepbs
                elif idx == 3:
                    quant = 1.0*iebs
                elif idx == 4:
                    quant = 1.0*rebs
                dists[idx].data = quant.reshape(1,len(quant))
#            QGE.data = Qgalbs.reshape(1,len(Qgalbs))
#            QBE.data = qbbs.reshape(1,len(qbbs))
#            IRD.data = np.vstack((rebs, iebs, rsepbs))
                pnames[idx][:, bt] = find_dist_params(dists[idx], bootstrap=True)
#            qge_params[:, bt] = find_dist_params(QGE, bootstrap=True)
#            qbe_params[:, bt] = find_dist_params(QBE, bootstrap=True)
#            ird_params[:, bt] = find_dist_params(IRD, bootstrap=True)
    
tbf = time.time()
if blob_type == 'eff':
#    if sim:
#        np.save('/home/matt/software/matttest/results/sim_params_boot_{}.npy'.format(RCE.name),rce_params)
#        np.save('/home/matt/software/matttest/results/sim_params_boot_{}.npy'.format(EFF.name),eff_params)
#    else:
#        np.save('/home/matt/software/matttest/results/pdf_params_boot_{}.npy'.format(RCE.name),rce_params)
#        np.save('/home/matt/software/matttest/results/pdf_params_boot_{}.npy'.format(EFF.name),eff_params)
    if sim:
        for idx in range(len(dists)):
            print idx, pnames[idx], dists[idx]
            np.save('/home/matt/software/matttest/results/sim_params_boot_{}.npy'.format(dists[idx].name),pnames[idx])
#        np.save('/home/matt/software/matttest/results/sim_params_boot_{}.npy'.format(QGE.name),qbe_params)
#        np.save('/home/matt/software/matttest/results/sim_params_boot_{}.npy'.format(QBE.name),qbe_params)
#        np.save('/home/matt/software/matttest/results/sim_params_boot_{}.npy'.format(IRD.name),ird_params)
    else:
        for idx in range(len(dists)):
            np.save('/home/matt/software/matttest/results/pdf_params_boot_{}{}.npy'.format(dists[idx].name,kw_extra),pnames[idx])
#        np.save('/home/matt/software/matttest/results/pdf_params_boot_{}{}.npy'.format(QGE.name,kw_extra),qge_params)
#        np.save('/home/matt/software/matttest/results/pdf_params_boot_{}{}.npy'.format(QBE.name,kw_extra),qbe_params)
#        np.save('/home/matt/software/matttest/results/pdf_params_boot_{}{}.npy'.format(IRD.name,kw_extra),ird_params)
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
#    rce_params_errs = bootstrap_errs(rce_params, true_params=RCE.paramarray)
#    eff_params_errs = bootstrap_errs(eff_params, true_params=EFF.paramarray)
#    if sim:
#        np.save('/home/matt/software/matttest/results/sim_params_final_{}.npy'.format(RCE.name), rce_params_errs)
#        np.save('/home/matt/software/matttest/results/sim_params_final_{}.npy'.format(EFF.name), eff_params_errs)
#    else:
#        np.save('/home/matt/software/matttest/results/pdf_params_final_{}.npy'.format(RCE.name), rce_params_errs)
#        np.save('/home/matt/software/matttest/results/pdf_params_final_{}.npy'.format(EFF.name), eff_params_errs)
    earrs = [""]*len(dists)
    for idx in range(len(earrs)):
        earrs[idx] = bootstrap_errs(pnames[idx], true_params=dists[idx].paramarray)
#        print earrs[idx]
#    qge_params_errs = bootstrap_errs(qge_params, true_params=QGE.paramarray)    
#    qbe_params_errs = bootstrap_errs(qbe_params, true_params=QBE.paramarray)
#    ird_params_errs = bootstrap_errs(ird_params, true_params=IRD.paramarray)
    if sim:
        for idx in range(len(dists)):
            np.save('/home/matt/software/matttest/results/sim_params_final_{}.npy'.format(dists[idx].name), earrs[idx])
#        np.save('/home/matt/software/matttest/results/sim_params_final_{}.npy'.format(QGE.name), qge_params_errs)
#        np.save('/home/matt/software/matttest/results/sim_params_final_{}.npy'.format(QBE.name), qbe_params_errs)
#        np.save('/home/matt/software/matttest/results/sim_params_final_{}.npy'.format(IRD.name), ird_params_errs)
    else:
        for idx in range(len(dists)):
            np.save('/home/matt/software/matttest/results/pdf_params_final_{}{}.npy'.format(dists[idx].name,kw_extra), earrs[idx])
#        np.save('/home/matt/software/matttest/results/pdf_params_final_{}{}.npy'.format(QGE.name,kw_extra), qge_params_errs)
#        np.save('/home/matt/software/matttest/results/pdf_params_final_{}{}.npy'.format(QBE.name,kw_extra), qbe_params_errs)
#        np.save('/home/matt/software/matttest/results/pdf_params_final_{}{}.npy'.format(IRD.name,kw_extra), ird_params_errs)

#print rce_params_errs
#print eff_params_errs
#exit(0)

#########################################################################
#########################################################################
###
###  44 44
###  44 44
###  44444
###     44
###     44
###
#########################################################################
#########################################################################

#print "Temporarily froze simulations - finish end to end analysis"
#exit(0)
### Draw from the distributions above to create simulated galaxies
if sim:
    exit(0) ## - don't re-make simulated galaxies
### For bias corrections from sim
#qgf = np.load('/home/matt/software/matttest/results/sim_gxys/sim_params_final_qg_fit.npy')
#qgt = np.load('/home/matt/software/matttest/results/sim_gxys/sim_params_final_qg_true.npy')
#qbf = np.load('/home/matt/software/matttest/results/sim_gxys/sim_params_final_qb_fit.npy')
#qbt = np.load('/home/matt/software/matttest/results/sim_gxys/sim_params_final_qb_true.npy')
#irdf = np.load('/home/matt/software/matttest/results/sim_gxys/sim_params_final_ird_fit.npy') 
#irdt = np.load('/home/matt/software/matttest/results/sim_gxys/sim_params_final_ird_true.npy')
##qgc = qgt[:,0] - qgf[:,0]
#qbc = qbt[:,0] - qbf[:,0]
#irdc = irdt[:,0] - irdf[:,0]
#dists = [RCE, EFF]
#dists = [QGE, QBE, IRD]
if blob_type == 'eff':
    for Pdf in dists:
        if Pdf.name == 'rsepe':
            Pdf.add_draw(sf.weibull_draw)
        elif Pdf.name == 'eff_joint':
            Pdf.add_draw(np.array((3*[sf.cauchy_draw])))
        elif Pdf.name == 'qb':
#            Pdf.add_draw(sf.gauss_draw)
            Pdf.add_draw(sf.cauchy_draw)
        elif Pdf.name == 'ird':
#            Pdf.add_draw(np.array(([sf.gauss_draw, sf.cauchy_draw, sf.gauss_draw])))
            Pdf.add_draw(np.array(([sf.cauchy_draw, sf.cauchy_draw, sf.gauss_draw])))
        elif Pdf.name == 'qgale':
            Pdf.add_draw(sf.cauchy_draw)
        elif Pdf.name == 'rc':
            Pdf.add_draw(sf.cauchy_draw)
        elif Pdf.name == 'ic':
            Pdf.add_draw(sf.weibull_draw)
#            Pdf.add_draw(sf.cauchy_draw)
### Add bias correction estimates
#QGE.paramarray = QGE.paramarray + qgc
#QBE.paramarray = QBE.paramarray + qbc
#IRD.paramarray = IRD.paramarray + irdc

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
            
n_gxys = 15
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
    print "Making sim galaxy {}".format(i)
    i1 = int(np.mod(i,5))
    i2 = int(np.floor(i/5))
    dim1 = 120
    fake_gal, qtmp, noise = src.make_sim_gal(dists, dim=dim, profile=profile, return_noise=True, save=save, save_idx=i, plot_results=False)
    ### save for later analysis - overwrite every time right now...
    if not save:
        hdu0 = pyfits.PrimaryHDU(fake_gal)
        hdu1 = pyfits.PrimaryHDU(noise)
        ### Additional new header values
        hdu0.header.append(('UNITS','Counts','Relative photon counts (no flat fielding)'))
        hdu0.header.append(('MODEL',profile,'clump model used to generate galaxy'))
        hdu1.header.append(('UNITS','Noise/Error','standard deviation'))
        hdulist = pyfits.HDUList([hdu0])
        hdulist.append(hdu1)
        hdulist.writeto('/home/matt/software/matttest/results/sim_gxy_{:03d}{}.fits'.format(i, param_kw),clobber=True)
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
#        fake_gal[70:71,5:15] = np.max(fake_gal)
        plt.imshow(fake_gal,interpolation='none',cmap=cm.hot)
    else:
#        fake_gal[70:71,5:15] = vlims[1]
        plt.imshow(fake_gal,interpolation='none',cmap=cm.hot, vmin=vlims[0], vmax=vlims[1])
    lf = src.px_to_pc(1000,0.01,2.5, inv=True)
#    ax.text(4, 68, '0.1"', color='w', fontsize=6)
    scalebar = ScaleBar(1, units='m', label='1 kpc',frameon=False,color='w', height_fraction = 0.002, length_fraction = 1/lf, location='lower left',font_properties=dict(size=8))
    ax.add_artist(scalebar)
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
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