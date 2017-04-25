#!/usr/bin/env python

#Code to fit LAE galaxies with clumps

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
plt.rcParams['image.cmap'] = 'gray'
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
#import scipy
#import scipy.stats as stats
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
import psf_utils
import source_utils as src
from gal_fit_input import *

#########################################
data_dir = os.path.join(os.environ['HST_DIR'],'23','14189')
parser = argparse.ArgumentParser()
parser.add_argument("-f","--filename",help="Name of image file (.fits) to evaluate",
                    default=os.path.join(data_dir,'10','SDSSJ091859.21+510452.5_SIE+shear_pix.fits'))#'SDSSJ002927.38+254401.7_SIE_pix_04.fits'))
parser.add_argument("-a","--param_file",help="Use fitted parameters", action='store_true')
parser.add_argument("-o","--observe",help="View 3D Contrast image", action='store_true')
parser.add_argument("-c","--crop",help="Make sure crop didn't remove a portion that needs fitting", action='store_true')
parser.add_argument("-p","--peaks",help="Show 3D and 2D plots of image to help find peaks", action='store_true')
parser.add_argument("-b","--brightness",help="Print array of surface brightnesses", action='store_true')
parser.add_argument("-g","--guess",help="Look at 3D plot of residuals (data-model) for guess", action='store_true')
parser.add_argument("-P","--plot_results",help="Look at 2D plot of data/model/residuals for fitted values", action='store_true')
parser.add_argument("-n","--no_save",help="Run code without saving fitted parameters", action='store_true')
parser.add_argument("-s","--sim",help="Take inputs from simulated galaxies", action='store_true')
parser.add_argument("-C","--clump_profile",help="Name of profile to use for clump fitting", default='sersic')
args_in = parser.parse_args()

########## Booleans for testing control #########################
eval_crop = args_in.crop # Make sure crop didn't remove a portion that needs fitting
look_for_peaks = args_in.peaks # Show 3D and 2D plots of image to help find peaks
check_sbb = args_in.brightness # Print array of surface brightnesses
eval_guess = args_in.guess # Look at 3D and 2D plots of residuals (data-model) for guess
plot_results = args_in.plot_results
####################################################################

def schechter_fct(L,L_star,Phi_star,alpha):
    """ To match eqn. 1
    """
    N_L = Phi_star*(L/L_star)**alpha*np.exp(-L/L_star)
    return N_L

def sersic2D(x,y,xc,yc,Ie,re,n,q=1,PA=0):
    """ makes a 2D image (dimx x dimy) of a Sersic profile centered at [xc,yc]
        and parameters Ie, re, and n.
        Optionally can add in ellipticity with axis ratio (q) and position
        angle (PA).
    """
    rarr = sf.make_rarr(x,y,xc,yc,q=q,PA=PA)
    image = sersic1D(rarr,Ie,re,n)
    return image
    
def sersic2D_lmfit(params,x,y):
    """ makes a 2D image (dimx x dimy) of a Sersic profile with parameters
        in lmfit object params (xc, yc, Ie, re, n, q, PA)
    """
    xc = params['xcg'].value
    yc = params['ycg'].value
    q = params['q'].value
    PA = params['PA'].value
    Ie = params['Ie'].value
    re = params['re'].value
    n = params['n'].value
    rarr = sf.make_rarr(x,y,xc,yc,q=q,PA=PA)
    image = sersic1D(rarr,Ie,re,n)
    return image
    
def sersic1D(rarr,Ie,re,n):
    bn = 0.868*n-0.142
    I_r = Ie*10**(-bn*((rarr/re)**(1/n)-1))
    return I_r

def sersic_mag(rarr,ue,re,n):
    bn = 0.868*n-0.142
    ur = ue + 2.5*bn*((rarr/re)**(1/n)-1)
    return ur
    
def galaxy_profile(params, xarr, yarr, z, inv_z, return_residuals=True, fdg=False, dgt='sersic', fb=True, blob_type='sersic', nblobs=1):
    """ Simple galaxy profile - sersic + 2D gaussian blobs.
        params is an lmfit object, dimensionality is 7 + 5*nblobs
        Assume no psf convolution.
        Returns residuals from profile for lmfit fitting.
        Galaxy model can be 'sersic' or 'gaussian'
    """
    if fdg:
        if dgt is 'sersic':
            galaxy_model = sersic2D_lmfit(params,xarr,yarr)
        else:
            print("Invalid diffuse galaxy type")
            exit(0)
    else:
        galaxy_model = np.zeros(z.shape)
    if fb:
        for i in range(nblobs):
            if blob_type is 'gaussian':
                galaxy_model += sf.gauss2d_lmfit(params,xarr,yarr,i)
            elif blob_type is 'sersic':                
                galaxy_model += sf.sersic2d_lmfit(params,xarr,yarr,i)
            else:
                print("Invalid blob type")
                exit(0)
    if return_residuals:
        galaxy_residuals = np.ravel((z-galaxy_model)**2*inv_z)
        return galaxy_residuals
    else:
        return galaxy_model  

pix_fits = pyfits.open(args_in.filename)
if 'sim_gxy' in args_in.filename:
    pix_image = pix_fits[0].data
    pix_image -= np.median(pix_image)
    pix_err = pix_fits[1].data
    pix_err[pix_err<0.01*np.max(pix_image)] = 0.01*np.max(pix_image)
#    pix_err = np.sqrt(abs(pix_image)) + 0.1*np.max(pix_image)
#    bgcut = 0.1*np.max(pix_image-np.mean(pix_image))+np.mean(pix_image)
    bgcut = None
else:
    pix_image = pix_fits[8].data
    pix_err = pix_fits[9].data
    bgcut = None
### Find inverse variance
pix_invar = 1/(pix_err)**2
### Remove infinities and nans
pix_invar[pix_invar==np.inf] = 0
pix_invar[np.isnan(pix_invar)] = 0

### Full image with nonzero portions cropped - used for cropping evaluation only
pix_full = np.copy(pix_image)
xinds = np.nonzero(np.sum(pix_full,axis=0))[0]
yinds = np.nonzero(np.sum(pix_full,axis=1))[0]
pix_full = pix_full[yinds[0]:yinds[-1],xinds[0]:xinds[-1]]
pix_invar_full = pix_invar[yinds[0]:yinds[-1],xinds[0]:xinds[-1]]


#print np.max(pix_invar)
#print np.min(abs(pix_err))
#print np.median(pix_invar)
#plt.imshow(pix_invar)
#plt.show()
#plt.close()
#plt.imshow(pix_image*pix_invar)
#plt.show()
#plt.close()

### Crop zero portions to make fitting go faster
if args_in.sim:
#    pix_image, pix_invar, bgcut, pd = src.crop_images(pix_image,pix_invar,bgcut=bgcut,dim=None,return_crop_info=True)
    bgcut, pd = 1, 1
else:
    pix_image, pix_invar, bgcut, pd = src.crop_images(pix_image,pix_invar,bgcut=bgcut,return_crop_info=True)
if pix_full.size < pix_image.size:
    pix_image = pix_full
    pix_invar = pix_invar_full
### Manual second crop, automate later
#pix_image = pix_image[30:90,40:108]
#pix_invar = pix_invar[30:90,40:108]
#print pix_invar.shape
if eval_crop:
    plt.figure('Nonzero image')
    plt.imshow(pix_full,interpolation='none')
    plt.figure('Cropped image')
    plt.imshow(pix_image,interpolation='none')
    plt.show()
    plt.close()
#pix_pix = pix_fits[8].data
#pix_pix = pix_pix[400-65:400+66,400-65:400+66]
dimx = pix_image.shape[1]
dimy = pix_image.shape[0]
### Need inverse variance estimates from Yiping's algorithm, if possible
#pix_invar = 1/(pix_image+0.0000001)
#pix_invar[(pix_image==0)] = 0
#plt.imshow(pix_pix[::-1],vmin=0.0, vmax = 5*np.std(pix_pix), interpolation='none')
#plt.show()
#plt.close()    

if look_for_peaks:
    plt.figure('Galaxy in 2D')
    sigma = 80/80.0 ### Guassian width
    gx = np.arange(-10,11)
    gy = np.arange(-10,11)
    gauss2d = sf.gauss2d(gx,gy,sigma,sigma)
    gauss2d /= np.sum(gauss2d)
#    plt.imshow(gauss2d,interpolation='none')
#    plt.show()
#    plt.close()
    pix_conv = signal.convolve2d(pix_image,gauss2d,mode='same')
#    plt.imshow(pix_conv,interpolation='none')
#    plt.show()
#    plt.close()
    pix_down = sf.downsample(pix_image,4)
#    plt.imshow(pix_down,interpolation='none')
#    plt.show()
#    plt.close()
    tt = pyfits.open('tiny_tim_sample_psf.fits')[0].data
#    plt.imshow(tt,interpolation='none')
#    plt.show()
#    plt.close()
    pix_tt = signal.convolve2d(pix_image,tt,mode='same')
#    plt.imshow(pix_tt,interpolation='none')
#    plt.show()
#    plt.close()
#    plt.imshow(pix_image,interpolation='none')
#    sf.plot_3D(pix_image)
#    plt.show()    
#    plt.close()
#    plt.figure()
#    plt.imshow(np.hstack((pix_image,pix_conv,pix_tt)),interpolation='none')
#    plt.show()
    tt_down = sf.downsample(pix_tt,4)
    pix_down_tt = signal.convolve2d(pix_down,tt,mode='same')
    pix_down_tt *= np.sum(pix_down)/np.sum(pix_down_tt)
    plt.imshow(pix_down_tt,interpolation='none')
    plt.show()
#    sf.plot_3D(pix_tt)
#    plt.show()
    
### Manual initial guesses ###########################

profile_type = 'eff'
#profile_type = 'sersic'
gam_fixed = True

if not args_in.param_file and not args_in.sim:
    img_num = os.path.split(os.path.split(args_in.filename)[0])[1]
    
    qgen = 0.8
    PAgen = np.pi/4
    ## 01:
    if img_num == '01':
        xb = np.array(([22, 26, 30, 14, 17.5, 10, 14, 15]))
        yb = np.array(([14.5, 15, 15, 15, 15, 40, 39, 43]))
        sbb = np.array(([0.025, 0.0225, 0.011, 0.0034, 0.0034, 0.0022, 0.0016, 0.0016]))*1.5
        sigb = np.array(([2.6, 1.5, 0.9, 2.5, 2.5, 3.5, 3.5, 2.5]))*2
        nsb = np.array(([2.2, 1.6, 1.6, 1.5, 1.5, 1.5, 1.5, 1.5]))
        qgal = np.array(([0.7, 0.8, 0.8, 0.7, 0.5, 0.6, 0.8, 0.8]))
        PAgal = np.array(([130, 130, 180, 135, 45, 90, 90, 0]))*np.pi/180
    #    qgal = np.ones((len(xb)))*qgen
    #    PAgal = np.ones((len(xb)))*PAgen
        
    ## 03:
    elif img_num == '03':
        xb = np.array(([27, 26, 35, 44, 52, 52, 57, 42]))
        yb = np.array(([22, 12, 24, 28.5, 26, 29, 24, 54]))
        sbb = np.array(([0.004, 0.003, 0.0011, 0.003, 0.0025, 0.001, 0.00092, 0.0025]))
        sigb = np.array(([4, 8, 5, 3, 4, 2, 3, 8]))*2
        nsb = np.array(([1.4, 1, 1.5, 1.5, 1.5, 1.5, 1.5, 1]))
        qgal = np.array(([0.33, 0.7, 0.7, 0.7, 0.8, 0.9, 0.2, 0.8]))
        PAgal = np.array(([150, 120, 90, 140, 90, 90, 60, 70]))*np.pi/180
        
    ## 04:
    elif img_num == '04':        
        xb = np.array(([29.5, 26, 27, 23, 14]))
        yb = np.array(([31, 46, 33, 30, 26]))
        sbb = np.array(([0.02, 0.006, 0.005, 0.001, 0.006]))*2.3
        sigb = np.array(([6, 6, 4, 3.5, 10]))*2
        nsb = np.array(([1.2, 1.2, 1.6, 1.5, 1]))
        qgal = np.array(([0.3, 0.8, 0.3, 0.8, 0.9]))
        PAgal = np.array(([80, 30, 40, 45, 0]))*np.pi/180+np.pi/2
    
    ## 05:
    elif img_num == '05':
        xb = np.array(([18, 15, 5, 29]))
        yb = np.array(([13, 12, 4, 16]))
        sbb = np.array(([0.0052, 0.0037, 0.0006, 0.0004]))*1.3
        sigb = np.array(([4.5, 2.5, 5.5, 2.5]))*2
        nsb = np.array(([1.7, 1.5, 1.5, 1.5]))
        qgal = np.array(([0.8, 0.9, 0.8, 0.8]))
        PAgal = np.array(([45, 45, 45, 45]))*np.pi/180
    
    ## 06:
    elif img_num == '06':
        xb = np.array(([15, 15, 19.5, 24]))
        yb = np.array(([24, 20, 21, 18]))
        sbb = np.array(([0.0071, 0.0047, 0.016, 0.0072]))*1.2
        sigb = np.array(([1, 3, 5, 6]))*2
        nsb = np.array(([1.5, 1.5, 1.5, 1.5]))*1.1
        qgal = np.array(([0.8, 0.7, 0.6, 0.9]))
        PAgal = np.array(([10, 110, 130, 45]))*np.pi/180
    
    ## 07:
    elif img_num == '07':
        xb = np.array(([26, 20]))
        yb = np.array(([31, 38]))
        sbb = np.array(([0.0064, 0.0114]))*4.5
        sigb = np.array(([12, 8]))
        nsb = np.array(([1.13, 1.16]))
        qgal = np.array(([0.7, 0.8]))
        PAgal = np.array(([135, 170]))*np.pi/180
    
    ## 08:
    elif img_num == '08':
        xb = np.array(([18,]))
        yb = np.array(([18,]))
        sbb = np.array(([0.0112]))*5.5
        sigb = np.array(([4,]))*2
        nsb = np.array(([1.0,]))*3
        qgal = np.array(([0.6,]))
        PAgal = np.array(([145,]))*np.pi/180
    
    ## 09:
    elif img_num == '09':
        xb = np.array(([15,]))
        yb = np.array(([18,]))
        sbb = np.array(([0.06,]))
        sigb = np.array(([4,]))*2
        nsb = np.array(([1.2,]))
        qgal = np.array(([0.5,]))
        PAgal = np.array(([145]))*np.pi/180
    
    ## 10:    
    elif img_num == '10':
        xb = np.array(([24.5, 21, 22, 28, 28.5, 32.5, 35, 43, 43]))-8    
        yb = np.array(([28.5, 28, 23, 27, 30.5, 31.5, 36, 34, 37.5]))-8
        sbb = np.array(([0.037,0.015,0.035,0.005,0.007,0.006,0.008,0.006,0.021]))
        sigb = np.array(([3, 1.75, 4, 1.25, 2.25, 3, 4.5, 2.5, 4]))*1.7
        nsb = np.array(([2.5, 1.5, 1.2, 2, 2.5, 2, 1, 1.5, 1]))
        qgal = np.ones((len(xb)))*qgen
        PAgal = np.ones((len(xb)))*PAgen
#        xb = np.array(([24.5, 21, 22, 28, 28.5, 35, 43, 43]))-8    
#        yb = np.array(([28.5, 28, 23, 27, 30.5, 36, 34, 37.5]))-8
#        sbb = np.array(([0.037,0.015,0.035,0.005,0.007,0.008,0.006,0.021]))
#        sigb = np.array(([3, 1.75, 4, 1.25, 2.25, 4.5, 2.5, 4]))*1.7
#        nsb = np.array(([2.5, 1.5, 1.2, 2, 2.5, 1, 1.5, 1]))
#        qgal = np.ones((len(xb)))*qgen
#        PAgal = np.ones((len(xb)))*PAgen
    
    ## 11:
    elif img_num == '11':
        xb = np.array(([14, 23, 27]))
        yb = np.array(([15, 23, 24]))
        sbb = np.array(([0.015, 0.042, 0.006]))
        sigb = np.array(([3.5, 5, 6.3]))*1.0
        nsb = np.array(([1.0, 1.0, 1.3]))
        qgal = np.array(([0.7, .5, 0.7]))
        PAgal = np.array(([45, 45, 160]))*np.pi/180
    
    ## 12:
    elif img_num == '12':
        xb = np.array(([15.5, 15.5, 17, 19]))
        yb = np.array(([18.5, 27, 34.5, 41]))
        sbb = np.array(([0.025, 0.016, 0.06, 0.024]))
        sigb = np.array(([7, 2.5, 3.5, 5]))
        nsb = np.array(([1.0, 0.6, 1.0, 1.0]))
        qgal = np.array(([0.7, 0.70101, 0.70013, 0.7]))
        PAgal = np.array(([160, 70, 100, 90]))*np.pi/180
    
    ## 13:
    elif img_num == '13':
        xb = np.array(([26, 25, 24.5, 23, 10.5, 13, 19]))
        yb = np.array(([22, 27, 33.5, 37, 47.5, 28, 46]))
        sbb = np.array(([0.035, 0.029, 0.024, 0.022, 0.01, 0.009, 0.004]))
        sigb = np.array(([4.5, 3.5, 3.5, 3.5, 2.5, 7, 1.5]))*2
        nsb = np.array(([1.0, 1.0, 1.0, 1.0, 0.8, 0.6, 1.0]))
        qgal = np.array(([0.5, 0.5, 0.6, 0.5, 0.8, 0.6, 0.8]))*0.8
        PAgal = np.array(([175, 175, 165, 175, 175, 10, 10]))*np.pi/180
    
    ## 14:
    elif img_num == '14':
        xb = np.array(([15, 19, 20.5]))
        yb = np.array(([16, 16, 25.5]))
        sbb = np.array(([0.21, 0.12, 0.024]))
        sigb = np.array(([3.3, 2.2, 5.0]))
        nsb = np.array(([0.6, 0.55, 1.0]))
        qgal = np.array(([0.7, 0.7, 0.8]))
        PAgal = np.array(([75, 75, 75]))*np.pi/180
    
    ## 15:
    elif img_num == '15':
        xb = np.array(([11, 18.5, 24, 36, 47, 48.5, 57]))
        yb = np.array(([29, 45, 46, 29, 27, 24, 18]))
        sbb = np.array(([0.003, 0.005, 0.004, 0.003, 0.018, 0.009, 0.018]))*1.3
        sigb = np.array(([2.5, 4.5, 6, 3.5, 6.5, 4, 7]))*1.4
        nsb = np.array(([1.0, 1.0, 1.0, 1.0, 1.2, 1.5, 1.0]))*1.1
        qgal = np.array(([0.8, 0.7, 0.4, 0.8, 0.9, 0.3, 0.5]))
        PAgal = np.array(([90, 135, 20, 10, 135, 60, 90]))*np.pi/180
    
    ## 16:
    elif img_num == '16':
        xb = np.array(([15, 20, 25, 24, 23, 32, 51.5, 55.5]))
        yb = np.array(([36, 28.5, 23, 18, 14, 21, 33.5, 37.5]))
        sbb = np.array(([0.0033, 0.0038, 0.015, 0.0025, 0.003, 0.0046, 0.02, 0.0019]))*2
        sigb = np.array(([4, 2.5, 2.5, 2, 1.5, 1, 2.0, 1.5]))*1.5
        nsb = np.array(([1.0, 1.0, 1.0, 1.5, 1.0, 1.0, 1.2, 1.0]))*0.9
        qgal = np.array(([0.7, 0.7, 0.6, 0.8, 0.8, 0.8, 0.8, 0.8]))
        PAgal = np.array(([130, 130, 140, 80, 130, 45, 135, 135]))*np.pi/180
    
    ## 19:
    elif img_num == '19':
        xb = np.array(([36, 28.5, 31.5]))
        yb = np.array(([24, 23.5, 35.5]))
        sbb = np.array(([0.0058, 0.004, 0.06]))*1.5
        sigb = np.array(([3, 1.5, 2.5]))*1.5
        nsb = np.array(([1.0, 1.0, 1.0]))
        qgal = np.array(([0.5, 0.8, 0.6]))
        PAgal = np.array(([120, 90, 130]))*np.pi/180
    
    ## 21:
    elif img_num == '21':
        xb = np.array(([13.5, 20.5, 24, 26, 17, 34, 29, 15, 35]))
        yb = np.array(([19, 32, 36, 41, 38, 34, 46.5, 56, 53]))
        sbb = np.array(([0.004, 0.0074, 0.014, 0.008, 0.003, 0.0017, 0.0021, 0.00122, 0.0009]))*1.1
        sigb = np.array(([4.5, 3, 2.5, 1.5, 2, 1, 1.5, 5, 1]))*1.5
        nsb = np.array(([1.0, 1.5, 1.1, 1.0, 1.0, 1.0, 1.0, 0.6, 1.0]))
        qgal = np.array(([.7, .6, .6, .8, .8, .8, .8, .9, .8]))
        PAgal = np.array(([40, 10, 110, 45, 135, 90, 10, 10, 45]))*np.pi/180
    
    else:
        print("Add initial guess for image {}".format(img_num))
        
    if profile_type == 'sersic':
        for j in range(len(xb)):
            sbb[j], sigb[j], nsb[j] = src.convert_sersic(sbb[j], sigb[j], nsb[j], direction='m2y')
    elif profile_type == 'eff':
        sbb *= 4 ## approximate rescaling
        nsb = 3*np.ones((len(xb)))

###############################################################################

if args_in.sim:
#    img_num = int(os.path.split(args_in.filename)[1][8:11])
#    xb, yb, sbb, sigb, nsb, qgal, PAgal = src.import_guesses(img_num)
    spar = pyfits.open(args_in.filename)[2].data
    img = pyfits.open(args_in.filename)[1].data
    msk = spar[2] > 0.1*np.max(spar[2])
#    print msk
    xb, yb, sbb, sigb, qgal, PAgal = spar[0][msk], spar[1][msk], spar[2][msk], spar[3][msk], spar[4][msk], spar[5][msk]
#    if len(xb) > 4:
#        exit(0)
    ### can add read noise, but don't think it'll really matter that much...
#    xb -= 5
#    yb -= 3
    nsb = 3*np.ones((len(xb))) ## assuming eff...
    

if check_sbb:
    sbb = np.zeros((len(xb)))
    for i in range(len(xb)):
        sbb[i] = pix_image[int(yb[i]),int(xb[i])]/10
    print "Scaled Surface Brightnesses from pix_image:"
    print sbb
    exit(0)

########## Option to use fitted values as initial guesses #####################
if args_in.param_file:
    root, filename = os.path.split(args_in.filename)
    if gam_fixed and profile_type == 'eff':
        filename = "{}_{}.fits".format(filename[:11], profile_type)
    elif profile_type == 'sersic':
        filename = "{}_{}.fits".format(filename[:11], profile_type)
    else:
        filename = "{}_{}_gam_free.fits".format(filename[:11], profile_type)
    savedir = '/home/matt/software/matttest/results'
    param_file = pyfits.open(os.path.join(savedir,filename))
    root, targ_num = os.path.split(root)
    print("Target Number is {}".format(targ_num))
    best_params = param_file[0].data
    if param_file[0].header['MODEL'] == 'sersic' or param_file[0].header['MODEL'] == 'eff':
        div = 7
    else:
        print("Cannot accommodate model type {}".format(param_file[0].header['MODEL']))
        exit(0)
    nblobs = int(len(best_params)/div)
    xb = np.zeros((nblobs))
    yb = np.zeros((nblobs))
    sbb = np.zeros((nblobs))
    sigb = np.zeros((nblobs))
    nsb = np.zeros((nblobs))
    qgal = np.zeros((nblobs))
    PAgal = np.zeros((nblobs))    
    for i in range(nblobs):
        xb[i] = best_params[div*i]# + 0.01*np.random.randn()
        yb[i] = best_params[div*i+1]# + 0.01*np.random.randn()
        sbb[i] = best_params[div*i+2]# + 0.0001*np.random.randn()
        sigb[i] = best_params[div*i+3]# + 0.01*np.random.randn()
        nsb[i] = best_params[div*i+4]# + 0.01*np.random.randn()
        qgal[i] = best_params[div*i+5]# + 0.01*np.random.randn()
        PAgal[i] = best_params[div*i+6]# + np.random.randn()
        ### Manual Overrides
#        #For 01:
#        xb[3] = 15
#        qgal[3] = 0.7

###############################################################################
### Try fitting in a, b space, not r, q space
### by my defn's a = r/sqrt(q), b = r*sqrt(q)

use_ab = True
if use_ab:
    ab = sigb/np.sqrt(qgal)
    bb = sigb*np.sqrt(qgal)
#    ab = 1.0*sigb
#    bb = 1.0*sigb

###############################################################################

kws = dict()
kws['nblobs'] = len(xb)
kws['blob_type'] = profile_type
kws['ab'] = use_ab

### For Gaussian Model
#gauss_names = []
#gauss_estimates = []
#gauss_mins = []
#gauss_maxes = []
#for i in range(kws['nblobs']):
#    gauss_names += ['xcb{}'.format(i),'ycb{}'.format(i),'sigb{}'.format(i),'hght{}'.format(i),'qb{}'.format(i),'PAb{}'.format(i)]
##    xc = blob_cntrs[i,1]
##    yc = blob_cntrs[i,0]
##    hght = pix_smooth.shape[0]*pix_smooth[int(yc),int(xc)]
#    gauss_estimates += [xb[i], yb[i], sigb[i], sbb[i], qgal, PAgal]
#    gauss_mins += [xb[i]-0.25, yb[i]-0.25, sigb[i]/1.2, sbb[i]/3, qgal-0.2, 0]
#    gauss_maxes += [xb[i]+0.25, yb[i]+0.25, sigb[i]*1.2, sbb[i]*3, 1, np.pi]
#
#gauss_fixed = np.ones(len(gauss_names))

### For Sersic Model
### Switch to Yiping's Sersic definition - remove this part once fully switched over

nparams = 7
sersic_names = []
sersic_names_0 = []
sersic_estimates = []
sersic_mins = []
sersic_maxes = []
sersic_fixed = np.ones((nparams*kws['nblobs']))
xb = xb + 0.001*np.random.randn(len(xb))
yb = yb + 0.001*np.random.randn(len(yb))
for i in range(kws['nblobs']):
    ### r, q space ...
    if not use_ab:
        sersic_names += ['xcb{}'.format(i),'ycb{}'.format(i),'Ieb{}'.format(i),'reb{}'.format(i),'nb{}'.format(i),'qb{}'.format(i),'PAb{}'.format(i)]
        sersic_estimates += [xb[i], yb[i], sbb[i], sigb[i], nsb[i], qgal[i], PAgal[i]]
        sersic_mins += [xb[i]-0.5, yb[i]-0.5, sbb[i]/10, sigb[i]/3, 0.01, 0.1, -1000*np.pi]
        sersic_maxes += [xb[i]+0.5, yb[i]+0.5, sbb[i]*10, sigb[i]*3, 20, 10, 1000*np.pi]
    ### a, b space ...
    else:
        sersic_names += ['xcb{}'.format(i),'ycb{}'.format(i),'Ieb{}'.format(i),'ab{}'.format(i),'nb{}'.format(i),'bb{}'.format(i),'PAb{}'.format(i)]
        sersic_names_0 += ['xcb{}'.format(i),'ycb{}'.format(i),'Ieb{}'.format(i),'reb{}'.format(i),'nb{}'.format(i),'qb{}'.format(i),'PAb{}'.format(i)]
        sersic_estimates += [xb[i], yb[i], sbb[i], ab[i], nsb[i], bb[i], PAgal[i]]
        if args_in.param_file:
            sersic_mins += [xb[i]-0.5, yb[i]-0.5, sbb[i]/10, ab[i]/1.5, 0.01, bb[i]/1.5, -1000*np.pi]
            sersic_maxes += [xb[i]+0.5, yb[i]+0.5, sbb[i]*10, ab[i]*1.5, 20, bb[i]*1.5, 1000*np.pi]
        else:
            sersic_mins += [xb[i]-0.5, yb[i]-0.5, sbb[i]/10, ab[i]/3, 0.01, bb[i]/3, -1000*np.pi]
            sersic_maxes += [xb[i]+0.5, yb[i]+0.5, sbb[i]*10, ab[i]*3, 20, bb[i]*3, 1000*np.pi]
    ## For EFF, fix gamma parameters (nb)
    if kws['blob_type'] == 'eff' and gam_fixed:
        sersic_fixed[7*i+4] = 0

#sersic_fixed = np.ones((len(sersic_names)))

#params = lmfit.Parameters()
#params = sf.array_to_Parameters(params,gauss_estimates,arraynames=gauss_names,minarray=gauss_mins,maxarray=gauss_maxes,fixed=gauss_fixed)
params_sers = lmfit.Parameters()
params_sers = sf.array_to_Parameters(params_sers,sersic_estimates,arraynames=sersic_names,minarray=sersic_mins,maxarray=sersic_maxes,fixed=sersic_fixed)
dimx = pix_image.shape[1]
dimy = pix_image.shape[0]

t0 = time.time()
args = (np.arange(dimx),np.arange(dimy),pix_image,pix_invar)
guess_model = src.galaxy_profile(params_sers,np.arange(dimx),np.arange(dimy),pix_image,pix_invar,return_residuals=False,blob_type=kws['blob_type'],nblobs=kws['nblobs'], ab=kws['ab'])
chi2_guess = np.sum((pix_image-guess_model)**2*pix_invar)/(np.size(pix_image)-np.size(sersic_estimates))
print("Guess chi^2 red = {}".format(chi2_guess))

if eval_guess:
    plt.imshow(np.hstack((pix_image,guess_model,pix_image-guess_model)))
#    sf.plot_3D((pix_image-guess_model))
    plt.show()
    plt.close()

### Actual fitting:
#leastsq_kws = {'maxfev':16000} #Option to adjust maximum function evaluations
results = lmfit.minimize(src.galaxy_profile,params_sers,args=args,kws=kws,method='leastsq')#,**leastsq_kws)
### Enforce 0 < q < 1, 0 < PA < pi
res_params = results.params
if use_ab:
    ### add in keywords for qb, re
    for key in res_params.keys():
        if 'ab' in key:
            cnt = key[-1]
            try:
                if res_params['qb{}'.format(cnt)].value is not None:
                    cnt = str(10+int(cnt))
            except:
                cnt = cnt
            ab = res_params[key].value
            bb = res_params['bb{}'.format(cnt)].value
            qtmp = bb/ab
            retmp = np.sqrt(ab*bb)
            res_params.add('qb{}'.format(cnt), value = qtmp)
            res_params.add('reb{}'.format(cnt), value = retmp)
    sersic_names = sersic_names_0 ### Change this for proper saving
            
for key in res_params.keys():
    if 'qb' in key:
        if res_params[key].value > 1:
            res_params[key].value = 1/res_params[key].value
            PAkey = 'PAb' + key[-1]
            res_params[PAkey].value += np.pi/2
    elif 'PAb' in key:
        res_params[key].value = np.mod(res_params[key].value,2*np.pi)
    
print args_in.filename + ':'    
for i in range(kws['nblobs']):
    if res_params['qb{}'.format(i)].value < 0.11:
        print("erratic axis ratio fit on blob {}".format(i))        
  
      
best_model = src.galaxy_profile(res_params,np.arange(dimx),np.arange(dimy),pix_image,pix_invar,return_residuals=False,blob_type=kws['blob_type'],nblobs=kws['nblobs'])
chi2r = np.sum((pix_image-best_model)**2*pix_invar)/(np.size(pix_image)-np.size(sersic_estimates))
t1 = time.time()
print("lmfit time is {}s".format(t1-t0))
print("Reduced chi^2 = {}".format(chi2r))
if plot_results:
    plt.imshow(np.hstack((pix_image,best_model,(pix_image-
    best_model))),interpolation='none')
    plt.title("Data/Model/Residuals with raw image, chi2red = {:6.3f}".format(chi2r))


if args_in.param_file:
    try:
        chiold = param_file[0].header['CHISQR']
    except:
        chiold = chi2_guess
    if chi2r > chiold or np.isnan(chi2r):
        print("Repeat fit worse than previous fit")
        if np.isnan(chi2r):
            print("Reduced Chi^2 returned nan")
        print("Updates will not be saved")
        if plot_results:
            plt.show()
            plt.close()
        exit(0)

if plot_results:
    plt.show()
    plt.close()

if args_in.no_save:
    exit(0)

##### Put results into proper format for saving
best_params = sf.Parameters_to_array(res_params,arraynames=sersic_names)
best_covar = results.covar
good_covar = True
if best_covar is None:
    good_covar = False
    best_covar = np.ones((best_params.size,best_params.size))
sersic_names = np.array(sersic_names,ndmin=2)
sersic_estimates = np.array(sersic_estimates,ndmin=2)
sersic_mins = np.array(sersic_mins,ndmin=2)
sersic_maxes = np.array(sersic_maxes,ndmin=2)
sersic_fixed = sersic_fixed.reshape(1,len(sersic_fixed))
init_guesses = np.vstack((sersic_estimates, sersic_mins, sersic_maxes, sersic_fixed))

##### Save results
savedir = '/home/matt/software/matttest/results'
junk, filename = os.path.split(args_in.filename)
if gam_fixed or kws['blob_type'] != 'eff':
    filename = "{}_{}.fits".format(filename[:11],kws['blob_type'])
else:
    filename = "{}_{}_gam_free.fits".format(filename[:11],kws['blob_type'])
savefile = os.path.join(savedir,filename)

hdu1 = pyfits.PrimaryHDU(best_params) #parameters from lmfit
hdu2 = pyfits.PrimaryHDU(best_covar) #covariance matrix from lmfit
hdu3 = pyfits.PrimaryHDU(init_guesses) #guesses, lower bound, upper bound, fixed array

hdu1.header.comments['NAXIS1'] = 'Best fit parameters'
hdu2.header.comments['NAXIS1'] = 'Covariance matrix, by parameters'
hdu2.header.comments['NAXIS2'] = 'Covariance matrix, by parameters'
hdu3.header.comments['NAXIS1'] = 'Initial parameter guesses'
hdu3.header.comments['NAXIS2'] = 'Guesses, mins, maxes, fixed, q, PA'
### Additional new header values
hdu1.header.append(('MODEL',kws['blob_type'],'Model profile used for clump fitting'))
if kws['blob_type'] == 'sersic':
    hdu1.header.append(('PARAMORD','','Parameter order: [xc,yc,Ie,Re,n,q,PA]'))
elif kws['blob_type'] == 'eff':
    hdu1.header.append(('PARAMORD','','Parameter order: [xc,yc,Io,a,gamma,q,PA]'))
hdu1.header.append(('BGCUT',bgcut,'Crop regions below BGCUT value'))
hdu1.header.append(('CUTPAD',pd,'Padding applied to cropped image'))
hdu1.header.append(('CHISQR',chi2r,'Reduced chi2 of best_guess'))
hdu2.header.append(('OKCOVAR',good_covar,'Boolean, whether covar is valid'))
hdulist = pyfits.HDUList([hdu1])
hdulist.append(hdu2)
hdulist.append(hdu3)
if not os.path.isdir(savedir):
    os.makedirs(savedir)
#    print os.path.join(redux_dir,savedate,savefile+'.proc.fits')
hdulist.writeto(savefile,clobber=True)