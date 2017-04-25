#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 12:26:10 2016

@author: matt

try to make schechter functions from info in CANDELS paper (Guo 2015)
This got subsumed into other work.  Everything here is now scrap.  Delete this
code once the final version of others is up and running.
"""

#Import all of the necessary packages
from __future__ import division
import pyfits
import os
import math
import time
import numpy as np
#from numpy import pi, sin, cos, random, zeros, ones, ediff1d
#from numpy import *
import matplotlib.pyplot as plt
#from matplotlib import cm
#import scipy
#import scipy.stats as stats
#import scipy.special as sp
#import scipy.interpolate as si
#import scipy.optimize as opt
#import scipy.sparse as sparse
import scipy.signal as signal
#import scipy.linalg as linalg
import scipy.integrate as integrate
#import solar
import special as sf
import psf_utils as psf
#import argparse
#import lmfit

def schechter_fct(L,L_star,Phi_star,alpha):
    """ To match eqn. 1
    """
    N_L = Phi_star*(L/L_star)**alpha*np.exp(-L/L_star)
    return N_L
    
L = np.logspace(-2,0,num=100)
### Params estimated by eye from info provided 2<z<3, middle stellar mass bin
L_star = 10**(-1.15)
Phi_star = 0.9
alpha = 1.5
##############################################################################
N_L = schechter_fct(L,L_star,Phi_star,alpha)
#plt.plot(np.log10(L),np.log10(N_L))
#plt.plot(L,N_L)
#plt.show()

N_gal = 20 ### No idea how to actually find this, but start here for estimation

### Now try to build galaxies from random draws

#for i in range(200):
#    ub = -10+i/10.0
#result = integrate.quad(sf.gaussian,-np.inf,10,args=(1,))
#print result


#args = (L_star,Phi_star,alpha)
#int_lims = [0, 1]
#n = 50000
#pdf_vals = sf.pdf_draw(sf.schechter_fct,n=n,res=5e3,args=args,int_lims=int_lims)
#plt.hist(pdf_vals,51)
#plt.plot(L,N_L*2.2*n/10,linewidth=2)
##plt.hist(np.random.randn(10000),51)
##print np.mean(pdf_vals), np.std(pdf_vals)
#plt.show()

def sersic1D(rarr,Ie,re,n):
    bn = 0.868*n-0.142
    I_r = Ie*10**(-bn*((rarr/re)**(1/n)-1))
    return I_r

def sersic2D(x,y,xc,yc,Ie,re,n,q=1,PA=0):
    """ makes a 2D image (dimx x dimy) of a Sersic profile centered at [xc,yc]
        and parameters Ie, re, and n.
        Optionally can add in ellipticity with axis ratio (q) and position
        angle (PA).
    """
    rarr = sf.make_rarr(x,y,xc,yc,q=q,PA=PA)
    image = sersic1D(rarr,Ie,re,n)
    return image

Ie = 1e3 ### what are the units???
n = 1
Re = 12
dimx = 40
dimy = 40
xc = dimx/2
yc = dimy/2
q = 0.8
PA = np.pi/4
image = sersic2D(np.arange(dimx),np.arange(dimy),xc,yc,Ie,Re,n,q=q,PA=PA)

kernel = np.array(([3, 2, 1],[2, 1, 2],[1, 2, 3]))

conv_image = psf.convolve_image(image,kernel)
conv_image_sci = signal.convolve2d(image,kernel,mode='same')
plt.imshow(conv_image_sci-conv_image,interpolation='none')
plt.show()
plt.imshow(conv_image_sci,interpolation='none')
plt.show()

###########################################################################
########### START OF LAE SCRAP ############################################
###########################################################################

'''
### Test sersic
Ie = 1e2 ### what are the units???
n = 1
Re = 12
dimx = 40
dimy = 40
xc = dimx/2+0.2
yc = dimy/2-0.11
q = 0.9
PA = np.pi/3
image = sersic2D(np.arange(dimx),np.arange(dimy),xc,yc,Ie,Re,n,q=q,PA=PA)
L_tot = np.sum(image)
#plt.imshow(np.log(image),interpolation='none')
#plt.show()
#rarr = np.arange(0,10,0.1)
#sers = sersic1D(rarr,Ie,Re,n)
#plt.ion()
#plt.plot(rarr,np.log(sers))

### Add in clumps
pc = 0.2 ## Percent of total luminosity from clumps
num_c = 1
nc = 1
Ic = Ie
sigc = Re/10
rc = Re/3
### Set (arbitrary) min/mix radii
rmax = Re
rmin = 0.1*Re
for i in range(num_c):
#    xcc = xc + Re*np.random.randn()
#    ycc = yc + Re*np.random.randn()
    xcc = 26.34
    ycc = 15.90
    rcc = np.sum(np.sqrt((xcc-xc)**2+(ycc-yc)**2))
#    while rcc > rmax or rcc < rmin:
#        xcc = xc + Re*np.random.randn()
#        ycc = yc + Re*np.random.randn()
#        rcc = np.sum(np.sqrt((xcc-xc)**2+(ycc-yc)**2))
    print("clump at ({},{})".format(xcc, ycc))
    image_clump = sersic2D(np.arange(dimx),np.arange(dimy),xcc,ycc,Ic,rc,nc)
    L_clump = pc/(1-pc)*(L_tot/num_c)
#    image_clump = sf.gauss2d(np.arange(dimx),np.arange(dimy),sigc,sigc,xcenter=xcc,ycenter=ycc,q=q,PA=PA)
    image_clump *= L_clump/np.sum(image_clump)
#    print np.sum(image_clump)
#    print np.sum(image)
#    print np.max(image_clump)
#    print np.max(image)
#    plt.imshow(image_clump,interpolation='none')
#    plt.show()
#    plt.close()
    image += image_clump
#    
#### add noise
image = np.random.poisson(image)
invar = 1/(image+0.0000001)
invar[image==0] = 0
#plt.imshow(image,interpolation='none')
#plt.show()
#'''
    


#### Fitting stuff from CANDELS

L = np.logspace(-2,0,num=100)
### Params estimated by eye from info provided 2<z<3, middle stellar mass bin
L_star = 10**(-1.15)
Phi_star = 0.9
alpha = 1.5
##############################################################################
N_L = schechter_fct(L,L_star,Phi_star,alpha)
#plt.plot(np.log10(L),np.log10(N_L))
#plt.show()

N_gal = 20 ### No idea how to actually find this, but start here for estimation

### Now try to build galaxies from random draws
#L_draw = np.random.rand() ### Uniform dist
#N_draw = schechter_fct(L_draw,L_star,Phi_star,alpha)
#N_draw = int(np.round(N_draw))
#print N_draw
#N_draw = 4 ## pick number of blob for now
#int_lims = [0.08, 1]
#args = (L_star,Phi_star,alpha)
#for i in range(N_draw):
#    ### Need to look up formalism to draw from an arbitrary PDF...
#    L_i = sf.pdf_draw(sf.schechter_fct,n=1,res=5e3,args=args,int_lims=int_lims)
#    Iblob = L_i * Ie
#    xrand = np.random.randn()
#    yrand = np.random.randn()
#    xc = 0.4*Re*xrand + dimx/2 + 0.1*Re*np.sign(xrand)
#    yc = 0.4*Re*yrand + dimy/2 + 0.1*Re*np.sign(yrand)
#    print yc, xc
#    blob_image = sersic2D(np.arange(dimx),np.arange(dimy),xc,yc,Ie,Re/10,n=1,q=q,PA=PA)
#    image = image + blob_image
#    plt.imshow(blob_image,interpolation='none')
#    plt.show()
 
tt = pyfits.open('tim_tim_sample_psf.fits')
psf = tt[0].data

def find_blob_centers(pix_image,pix_invar,smooth_sizes=[2,3],smooth_type='gaussian'):
    """ Takes pix_image, convolves with several smoothing filters, finds peaks,
        finds centers, takes some sort of weighted average over filters.
        Returns array of x, y centers of peaks (col0 = xc, col1 = yc).
    """
    if type(smooth_sizes) == int:
        iters = 1
    else:
        iters = len(smooth_sizes)
    xycntrs_dict = dict()
    for i in range(iters):
        if iters == 1:
            smooth_size = smooth_sizes
        else:
            smooth_size = smooth_sizes[i]
        gauss_sig = smooth_size/(2*np.sqrt(2*np.log(2)))
        if smooth_type == 'gaussian':
            smooth_kernel = sf.gauss2d(np.arange(-5,6),np.arange(-5,6),gauss_sig,gauss_sig)
        elif smooth_type == 'tophat' or smooth_type == 'boxcar':
            smooth_kernel = np.ones((smooth_size,smooth_size))
        else:
            print("invalid smooth_type")
            print("Choose gaussian or tophat")
            exit(0)
        smooth_kernel /= np.sum(smooth_kernel)
        conv_image = signal.convolve2d(pix_image,smooth_kernel,mode='same')
        conv_var = signal.convolve2d(1/pix_invar,smooth_kernel,mode='same')
        conv_invar = 1/(conv_var)
#        print smooth_size
#        plt.imshow(conv_image,interpolation='none')
#        plt.show()
#        plt.close()
        peaks = sf.find_peaks2D(conv_image, bg_thresh=0,min_amp=np.sum(conv_image)*0.01)
        peak_pos = np.transpose(np.nonzero(peaks))
        peak_pos = peak_pos[::-1]
        min_space = 4
        peak_pos = sf.merge_close_peaks(peak_pos,min_space)
        xycntrs = np.zeros((peak_pos.shape[0],2))
        pp = 6
        for j in range(peak_pos.shape[0]):
            sub_image = conv_image[peak_pos[j,0]-pp:peak_pos[j,0]+pp+1,peak_pos[j,1]-pp:peak_pos[j,1]+pp+1]
            sub_invar = conv_invar[peak_pos[j,0]-pp:peak_pos[j,0]+pp+1,peak_pos[j,1]-pp:peak_pos[j,1]+pp+1]
            sub_cntrs = sf.find2dcenter(sub_image,sub_invar,[pp,pp],method='gaussian')
            xycntrs[j] = peak_pos[j] + sub_cntrs - pp
        xycntrs_dict[i] = xycntrs
    num_blobs = np.zeros((iters))
    for k in range(iters):
        num_blobs[k] = len(xycntrs_dict[k])
    if len(np.unique(num_blobs)) == 1:
        xycntrs = np.zeros((peak_pos.shape[0],2))
        for l in range(iters):
            xycntrs += xycntrs_dict[l]
        xycntrs /= iters
        return xycntrs
#    else:
#        print("Different convolutions return different blob counts")
#        print("Figure out how to handle it!")
#        exit(0)


#blob_cntrs = find_blob_centers(pix_image,pix_invar)
#print blob_cntrs
plt.imshow(pix_image,interpolation='none')
#plt.plot(blob_cntrs[1,0],blob_cntrs[0,0],'ko')
#plt.plot(blob_cntrs[1,1],blob_cntrs[0,1],'co')
#plt.show()
#exit(0)

def smooth_image(image,pix=10,mode='boxcar'):
    """ Smooths an image using the boxcar method with a square box, side 
        length pix.
    """
    if mode=='boxcar':
        filt = np.ones((pix,pix))/pix**2
    else:
        print("Not a valid mode.  Select 'boxcar'")
        exit(0)
    return signal.convolve2d(image,filt,mode='same')
    
#conv_image = np.copy(pix_image)
im_smooth = smooth_image(pix_image)
#print np.max(im_smooth)/Ie
#print np.max(conv_image)/Ie
#print np.sum(im_smooth)
#print np.sqrt(np.sum(im_smooth))
#print np.sum(im_smooth)/im_smooth.size

#def find_center_radius(image):
#    
#plt.imshow(im_smooth,interpolation='none')
#plt.show()
#plt.imshow(conv_image-im_smooth,interpolation='none')
#plt.show()
trast_im = pix_image-im_smooth

def zero_background(image,sigma=3):
    """ Using sigma clipping, estimate, then remove the background.
    """
    im_mask = sf.sigma_clip(image,sigma=sigma,max_iters=np.size(image))
    im_mask = np.reshape(im_mask,image.shape)
    bg = np.mean(image[im_mask])
    bg_std = np.std(image[im_mask])
    bg_mask = image < (bg+2*sigma/3*bg_std)
    image[bg_mask] = 0
    return image
    
zbg_im = trast_im
#zbg_im = zero_background(trast_im)
#plt.figure()
#plt.imshow(zbg_im,interpolation='none')
#plt.show()


### reference special functions to find 2D peaks in (high SNR) image
#peaks = sf.find_peaks2D(zbg_im,bg_thresh=0,min_amp=np.sum(conv_image)*0.01)#,offset=[1,1])
#peaks = sf.find_peaks2D(conv_image, bg_thresh=0,min_amp=np.sum(conv_image)*0.01)

def min_radius(image,center,min_radius):
    """ Zeros any elements inside min_radius from center.
    """
    rarr = sf.make_rarr(np.arange(image.shape[1]),np.arange(image.shape[0]),center[0],center[1])
    rmask = rarr < min_radius
#    plt.imshow(rmask)
    image[rmask] = 0
    return image

### Remove center bulge of galaxy
#peaks = min_radius(peaks,[dimx/2,dimy/2],0.001)
#peak_pos = np.transpose(np.nonzero(peaks))
#peak_mask = np.ones((peak_pos.shape[0]),dtype=bool)
#for i in range(peak_pos.shape[0]):
#    if peak_pos[i,0] > dimy/2-2 and peak_pos[i,0] < dimy/2+2:
#        if peak_pos[i,1] > dimx/2-2 and peak_pos[i,1] < dimx/2+2:
#            peak_mask[1] = False
#peak_pos = peak_pos[peak_mask]
#print peak_pos

x0, x1, y0, y1 = 350, 430, 360, 430
#x0, x1, y0, y1 = 381, 401, 383, 395 #blob1
#x0, x1, y0, y1 = 373, 393, 406, 418 #blob2
#dimx = x1-x0
#dimy = y1-y0
#blobs = trast_im[y0:y1,x0:x1]
#pix_image = pix_image[y0:y1,x0:x1]
#pix_invar = pix_invar[y0:y1,x0:x1]

peaks = sf.find_peaks2D(pix_image, bg_thresh=0,min_amp=np.sum(pix_image)*0.01)
peak_pos = np.transpose(np.nonzero(peaks))
peak_pos = peak_pos[::-1]
print peak_pos
#plt.imshow(conv_image,interpolation='none')
#conv_invar = abs(conv_invar[y0:y1,x0:x1])
#conv_invar = np.ones(conv_image.shape)

if args.observe:
    plt.imshow(conv_image,interpolation='none')
    plt.show()
#    fig = plt.figure()
#    ax = fig.add_subplot(111, projection='3d')
#    x_matrix = np.tile(np.arange(dimx),(dimy,1))
#    y_matrix = np.tile(np.arange(dimy),(dimx,1)).T
#    ax.plot_surface(x_matrix,y_matrix,conv_image,rstride=1,cstride=1)
#    plt.show()
#    plt.close()
    exit(0)

#plt.figure()
#plt.imshow(image+np.max(image)*peaks,interpolation='none')
#plt.show()

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#x_matrix = np.tile(np.arange(dimx),(dimy,1))
#y_matrix = np.tile(np.arange(dimy),(dimx,1)).T
#ax.plot_surface(x_matrix,y_matrix,zbg_im,rstride=1,cstride=1)
#plt.show()


#pspec = np.fft.fft(conv_image)
#plt.imshow(abs(pspec),interpolation='none')
#plt.plot(abs(pspec[0:int(len(pspec)/2)]))
#plt.show()

### Make later export file to use with SExtractor program as per Guo paper
#hdu1 = pyfits.PrimaryHDU(zbg_im)
#hdu1.header.append(('UNITS','Flux','Relative photon counts (no flat fielding)'))
#hdulist = pyfits.HDUList([hdu1])
#hdulist.writeto('proc_im_test.fits',clobber=True)

### Supporting parameter estimation functions
### (mostly) following conventions in emcee documentation
def lnlike(params,x,y,z,inv_z,psf,mode='sersic'):
    """ log likelihood function for emcee
    """
#    model = galaxy_profile(params,x,y,z,return_residuals=False)
    if mode == 'sersic':
        xc, yc, q, PA, Ie, re, n = params ### Unpack parameters
        model = sersic2D(x,y,xc,yc,Ie,re,n,q=q,PA=PA)
#        model = signal.convolve2d(model,psf,mode='same') ### Convolve with psf
    elif mode == 'gaussian':
        q, PA, xc, yc, sigx, sigy, hght = params
        model = hght*sf.gauss2d(x,y,sigx,sigy,xcenter=xc,ycenter=yc,q=q,PA=PA)
    else:
        print("Invalid mode in lnlike")
        exit(0)
    log_likelihood = -0.5*(np.sum((z-model)**2*inv_z - np.log(inv_z)))
    return log_likelihood
    
def lnlike_lmfit(params,x,y,z,inv_z):
    """ log likelihood function for lmfit
    """
    model = galaxy_profile(params,x,y,z,psf,return_residuals=False)    
#    xc, yc, q, PA, Ie, re, n = params ### Unpack parameters
#    model = sersic2D(x,y,xc,yc,Ie,re,n,q=q,PA=PA)
#    model = signal.convolve2d(model,psf,mode='same') ### Convolve with psf
    log_likelihood = -0.5*((z-model)**2*inv_z - np.log(inv_z))
    return np.ravel(log_likelihood)
    
def lnlike_poisson(params,x,y,z,psf):    
    """ log likelihood function using poisson instead of Gaussian dist.
        Set up for use with lmfit.
    """
    
    model = galaxy_profile(params,x,y,z,psf,return_residuals=False)    
    
#    num_iters = int(len(params)/7)
#    model = np.zeros((z.shape))
#    for i in range(num_iters):
#        xc, yc, q, PA, Ie, re, n = params[7*i:7*(i+1)]
#        xc, yc, q, PA, Ie, re, n = [params[k].value for k in \
#        ['xc{}'.format(i),'yc{}'.format(i),'q{}'.format(i),'PA{}'.format(i),\
#        'Ie{}'.format(i),'re{}'.format(i),'n{}'.format(i)]] ### Unpack parameters
#        tmp_model = sersic2D(x,y,xc,yc,Ie,re,n,q=q,PA=PA)
#        model += signal.convolve2d(tmp_model,psf,mode='same') ### Convolve with psf
#    log_likelihood = np.sum(z*np.log(model) - model - sp.gammaln(z+1))
    log_likelihood = z*np.log(model) - model - sp.gammaln(z+1)
    ### minus sign to minimize
    return -np.ravel(log_likelihood)
    
def lnlike_poisson_opt(params,x,y,z,psf,zmask):    
    """ log likelihood function using poisson instead of Gaussian dist.
        Set up for use with scipy.optimize.minimize
    """
    num_iters = int(len(params)/7)
    model = np.zeros((z.shape))
    for i in range(num_iters):
        xc, yc, q, PA, Ie, re, n = params[7*i:7*(i+1)]
        tmp_model = sersic2D(x,y,xc,yc,Ie,re,n,q=q,PA=PA)
        model += signal.convolve2d(tmp_model,psf,mode='same') ### Convolve with psf
    log_likelihood = np.sum(z[zmask]*np.log(model[zmask]) - model[zmask] - sp.gammaln(z[zmask]+1))
    ### minus sign to minimize
    return -log_likelihood
   
def lnprior(params,param_mins,param_maxes,mode='sersic'):
    """ Uniform only for now
    """
    if mode == 'sersic':
        xc, yc, q, PA, Ie, re, n = params ### Unpack parameters
        xc_mn, yc_mn, q_mn, PA_mn, Ie_mn, re_mn, n_mn = param_mins ### Unpack lower limits
        xc_mx, yc_mx, q_mx, PA_mx, Ie_mx, re_mx, n_mx = param_maxes ### Unpack upper limits
        if xc_mn < xc < xc_mx and yc_mn < yc < yc_mx and q_mn < q < q_mx and PA_mn < PA < PA_mx and Ie_mn < Ie < Ie_mx and re_mn < re < re_mx and n_mn < n < n_mx:
            return 0.0
        else:
            return -np.inf
    if mode == 'gaussian':
        xc, yc, sigx, sigy, hght = params ### Unpack parameters
        xc_mn, yc_mn, sigx_mn, sigy_mn, hght_mn = param_mins ### Unpack lower limits
        xc_mx, yc_mx, sigx_mx, sigy_mx, hght_mx = param_maxes ### Unpack upper limits
        if xc_mn < xc < xc_mx and yc_mn < yc < yc_mx and sigx_mn < sigx < sigx_mx and sigy_mn < sigy < sigy_mx and hght_mn < hght < hght_mx:
            return 0.0
        else:
            return -np.inf
        
def lnprob(params,param_mins,param_maxes,x,y,z,inv_z,psf):
    """ log probability for emcee
    """
    lp = 0
    num_iters = int((len(params)-7)/5)+1
    for i in range(num_iters):
        if i == 0:
            lp += lnprior(params[i:i+7],param_mins[i:i+7],param_maxes[i:i+7],mode='sersic')
        else:
            lp += lnprior(params[7+5*(i-1):7+5*i],param_mins[7+5*(i-1):7+5*i],param_maxes[7+5*(i-1):7+5*i],mode='gaussian')
#    for i in range(int(len(params)/7)):
#        lp += lnprior(params[i:i+7],param_mins[i:i+7],param_maxes[i:i+7])
    if not np.isfinite(lp):
        return -np.inf
    ll = 0
    for j in range(num_iters):
        if j == 0:
            ll += lnlike(params[j:j+7],x,y,z,inv_z,psf)
        else:
            ll += lnlike(np.append(params[2:4],params[7+5*(i-1):7+5*i]),x,y,z,inv_z,psf,mode='gaussian')
#    for j in range(int(len(params)/7)):
#        ll += lnlike(params[j:j+7],x,y,z,inv_z,psf)
    return lp + ll

#    if gal_model is 'sersic':
#        nblobs = int((len(params)-7)/5)
#        galaxy_model = sersic2D_lmfit(xarr,yarr,params)
#        if nblobs > 0:
#            for i in range(nblobs):
#                xc = params['xcb{}'.format(i)].value
#                yc = params['ycb{}'.format(i)].value
#                sigx = params['sigx{}'.format(i)].value
#                sigy = params['sigy{}'.format(i)].value
#                hght = params['hght{}'.format(i)].value
#                ### Assume q, PA of host galaxy apply to all blobs...
#                q = params['q'].value
#                PA = params['PA'].value
#                blob_model = hght*sf.gauss2d(xarr,yarr,sigx,sigy,xcenter=xc,
#                                             ycenter=yc,q=q,PA=PA)
#                galaxy_model += blob_model
#    elif gal_model is 'gaussian':
#        nblobs = int((len(params)-2)/5)
#        galaxy_model = np.zeros(z.shape)
#        for i in range(nblobs):
#            xc = params['xcb{}'.format(i)].value
#            yc = params['ycb{}'.format(i)].value
#            sigx = params['sigx{}'.format(i)].value
#            sigy = params['sigy{}'.format(i)].value
#            hght = params['hght{}'.format(i)].value
#            ### Assume q, PA of host galaxy apply to all blobs...
#            q = params['q'].value
#            PA = params['PA'].value
#            galaxy_model += hght*sf.gauss2d(xarr,yarr,sigx,sigy,xcenter=xc,
#                                            ycenter=yc,q=q,PA=PA)
#    else:
#        print("Choose either 'sersic' or 'gaussian' for keyword gal_model")
#        exit(0)
#    if return_residuals:
#        galaxy_residuals = np.ravel((z-galaxy_model)**2*inv_z)
#        return galaxy_residuals
#    else:
#        return galaxy_model

#### MCMC fit (below) is super slow.  Try using lmfit to minimize loglikelihood
nparams = 7
#npeaks = 2
npeaks = peak_pos.shape[0]+1
zmask = np.ones(pix_image.shape,dtype=bool)
pd = 3
for i in range(zmask.shape[0]):
    for j in range(zmask.shape[1]):
        if peaks[i,j]==1:
            zmask[i-pd:i+pd+1,j-pd:j+pd+1] = 0
            
#plt.imshow(zmask,interpolation='none')
#plt.show()
#initial_params = [dimx/2,dimy/2,0.7,np.pi/4,np.max(conv_image)/3,10,1]
#initial_params = [dimx/2,dimy/2,q,PA,Ie,Re,n]
#for i in range(npeaks):
#    if i == 0: ### the galaxy in which these are embedded...
#        initial_params += [dimx/2,dimy/2,0.7,np.pi/4,np.max(conv_image)/3,10,1]
#    else:
#        initial_params += [peak_pos[i-1,1],peak_pos[i-1,0],0.7,np.pi/4,trast_im[peak_pos[i-1,0],peak_pos[i-1,1]]/3,1,1]
    
#params = lmfit.Parameters()
#for i in range(npeaks):
#    if i == 0: ### the galaxy in which these are embedded...
#        params.add('xc{}'.format(i), value = dimx/2)
#        params.add('yc{}'.format(i), value = dimy/2)
#        params.add('q{}'.format(i), value = 0.7)
#        params.add('PA{}'.format(i), value = np.pi/4)
#        params.add('Ie{}'.format(i), value = np.max(conv_image)/3)
#        params.add('re{}'.format(i), value = 10)
#        params.add('n{}'.format(i), value = 1)
#    else:
#        params.add('xc{}'.format(i), value = peak_pos[i-1,1])
#        params.add('yc{}'.format(i), value = peak_pos[i-1,0])
#        params.add('q{}'.format(i), value = 0.7)
#        params.add('PA{}'.format(i), value = np.pi/4)
#        params.add('Ie{}'.format(i), value = trast_im[peak_pos[i-1,0],peak_pos[i-1,1]]/3)
#        params.add('re{}'.format(i), value = 1)
#        params.add('n{}'.format(i), value = 1)

#### Params for galaxy_profile with sersic modeling
#params = lmfit.Parameters()
#for i in range(npeaks):
#    if i == 0: ### the galaxy in which these are embedded...
#        params.add('xcg', value = dimx/2, min = dimx/2-2, max = dimx/2+2)
#        params.add('ycg', value = dimy/2, min = dimy/2-2, max = dimy/2+2)
#        params.add('q', value = 0.7, min = 0, max = 1)
#        params.add('PA', value = np.pi/4, min = 0, max = np.pi)
#        params.add('Ie', value = np.max(conv_image)/3)
#        params.add('re', value = 10)
#        params.add('n', value = 1, min = 0.5, max = 4)
#    else:
#        xcc = peak_pos[i-1,1]
#        ycc = peak_pos[i-1,0]
#        params.add('xcb{}'.format(i-1), value = xcc, min = xcc-2, max = xcc+2)#peak_pos[i-1,1])
#        params.add('ycb{}'.format(i-1), value = ycc, min = ycc-2, max = ycc+2)#peak_pos[i-1,0])
#        params.add('sigx{}'.format(i-1), value = 1, min = 0.5, max = 3)
#        params.add('sigy{}'.format(i-1), value = 1, min = 0.5, max = 3)
#        params.add('hght{}'.format(i-1), value = trast_im[peak_pos[i-1,0],peak_pos[i-1,1]])

#### Params for galaxy_profile with sersic modeling for emcee
#params = np.arange(7 + npeaks*5)
#for i in range(npeaks+1):
#    if i == 0: ### the galaxy in which these are embedded...
#        params[0] = dimx/2
#        params[1] = dimy/2
#        params[2] = 0.7
#        params[3] = np.pi/4
#        params[4] = np.max(conv_image)/3
#        params[5] = 10
#        params[6] = 1
#    else:
#        params[7+5*(i-1)] = peak_pos[i-1,1]
#        params[7+5*(i-1)+1] = peak_pos[i-1,0]
#        params[7+5*(i-1)+2] = 1
#        params[7+5*(i-1)+3] = 1
#        params[7+5*(i-1)+4] = trast_im[peak_pos[i-1,0],peak_pos[i-1,1]]

#### Params for galaxy_profile with gaussian modeling
#params = lmfit.Parameters()
#for i in range(npeaks):
#    if i == 0: ### the galaxy in which these are embedded...
#        params.add('xcb0', value = dimx/2, min = dimx/2-3, max = dimx/2+3)
#        params.add('ycb0', value = dimy/2, min = dimy/2-3, max = dimy/2+3)
#        params.add('sigx0', value = dimx/4, min = 3, max = dimx/2)
#        params.add('sigy0', value = dimy/4, min = 3, max = dimy/2)
#        params.add('hght0', value = np.max(conv_image[int(dimy/2)-3:int(dimy/2)+3,int(dimx/2)-3:int(dimx/2)+3]))#trast_im[peak_pos[i-1,0],peak_pos[i-1,1]])        
#        params.add('q', value = 0.7, min = 0, max = 1)
#        params.add('PA', value = np.pi/4, min = 0, max = np.pi)
#    else:
#        xcc = peak_pos[i-1,1]
#        ycc = peak_pos[i-1,0]
#        params.add('xcb{}'.format(i), value = xcc, min = xcc-2, max = xcc+2)#peak_pos[i-1,1])
#        params.add('ycb{}'.format(i), value = ycc, min = ycc-2, max = ycc+2)#peak_pos[i-1,0])
#        params.add('sigx{}'.format(i), value = 1, min = 0.5, max = 3)
#        params.add('sigy{}'.format(i), value = 1, min = 0.5, max = 3)
#        params.add('hght{}'.format(i), value = trast_im[peak_pos[i-1,0],peak_pos[i-1,1]])

##############################################################################
############ First fit convolved image to get initial guesses ################
##############################################################################
gauss_sig = 4
smooth_kernel = sf.gauss2d(np.arange(-5,6),np.arange(-5,6),gauss_sig,gauss_sig)
pix_smooth = signal.convolve2d(pix_image,smooth_kernel,mode='same')
pix_var_smooth = signal.convolve2d(1/pix_invar,smooth_kernel,mode='same')
pix_invar_smooth = 1/(pix_var_smooth)

#best_model = sersic2D(np.arange(dimx),np.arange(dimy),results["x"][0],results["x"][1],
#                      results["x"][4],results["x"][5],results["x"][6],
#                      q=results["x"][2],PA=results["x"][3])
#best_model = sersic2D(np.arange(dimx),np.arange(dimy),xc,yc,Ie,Re,n,q=q,PA=PA)
#conv_model = signal.convolve2d(best_model,psf,mode='same')
#blobs = conv_image-conv_model
#plt.imshow(blobs,interpolation='none')
#plt.show()

'''
### Now estimate parameters of galaxies with MCMC
    
nparams = 7
npeaks = peak_pos.shape[0]
ndim, nwalkers = nparams + 5*npeaks, 100
#initial_params = [48.1,52.6,0.7,0.1,1500,20.8,3]
#initial_params = [49.5,50.5,0.78,np.pi/4.1,1003,29.8,1.02]
#param_mins = [48,48,0.5,0,500,20,0.5]
#param_maxes = [53,53,1,np.pi/2,2000,40,4]
initial_params = []
param_mins = []
param_maxes = []
for i in range(npeaks+1):
    if i == 0: ### the galaxy in which these are embedded...
        initial_params += [dimx/2,dimy/2,0.7,np.pi/4,np.max(conv_image)/3,10,1]
        param_mins += [dimx/2-2,dimy/2-2,0.4,0,100,3,0.1]
        param_maxes += [dimx/2+2,dimy/2+2,1,np.pi,np.max(conv_image),20,4]
#        initial_params += [dimx/2,dimy/2,0.7,np.pi/4,np.max(conv_image)/3,10,1]
#        param_mins += [dimx/2-2,dimy/2-2,0.4,0,100,3,0.1]
#        param_maxes += [dimx/2+2,dimy/2+2,1,np.pi,np.max(conv_image),20,4]
    else:
        initial_params += [peak_pos[i-1,1],peak_pos[i-1,0],1,1,trast_im[peak_pos[i-1,0],peak_pos[i-1,1]]]
        param_mins += [peak_pos[i-1,1]-2,peak_pos[i-1,0]-2,0.5,0.5,10]
        param_maxes += [peak_pos[i-1,1]+2,peak_pos[i-1,0]+2,3,3,trast_im[peak_pos[i-1,0],peak_pos[i-1,1]]*3]
        #        params[7+5*(i-1)] = peak_pos[i-1,1]
#        params[7+5*(i-1)+1] = peak_pos[i-1,0]
#        params[7+5*(i-1)+2] = 1
#        params[7+5*(i-1)+3] = 1
#        params[7+5*(i-1)+4] = trast_im[peak_pos[i-1,0],peak_pos[i-1,1]]
#        initial_params += [peak_pos[i-1,1],peak_pos[i-1,0],0.7,np.pi/4,trast_im[peak_pos[i-1,0],peak_pos[i-1,1]]/3,1,1]
#        param_mins += [peak_pos[i-1,1]-2,peak_pos[i-1,0]-2,0.4,0,10,0.1,0.1]
#        param_maxes += [peak_pos[i-1,1]+2,peak_pos[i-1,0]+2,1,np.pi,trast_im[peak_pos[i-1,0],peak_pos[i-1,1]]/3,3,4]
pos = [initial_params + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
xvals = np.arange(dimx)
yvals = np.arange(dimy)
t0 = time.time()
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(param_mins,param_maxes,xvals,yvals,trast_im,invar,psf))
sampler.run_mcmc(pos, 10000)
t1 = time.time()
print("emcee time = {}s".format(t1-t0))

for i in range(sampler.chain.shape[2]):
    for j in range(100):
        plt.plot(sampler.chain[j,:,i])
    plt.show()
    plt.close()
#'''

###########################################################################
############ END OF LAE SCRAP #############################################
###########################################################################