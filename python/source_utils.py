#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 12:49:40 2016

@author: matt
"""

#Import all of the necessary packages
from __future__ import division
import pyfits
import os
import glob
#import math
import time
import numpy as np
#from numpy import pi, sin, cos, random, zeros, ones, ediff1d
#from numpy import *
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from matplotlib import mlab, cm
from matplotlib.font_manager import FontProperties
#import scipy
from scipy.misc import comb
#import scipy.stats as stats
#import scipy.special as sp
#import scipy.interpolate as si
import scipy.optimize as opt
#import scipy.sparse as sparse
import scipy.signal as signal
#import scipy.linalg as linalg
#import solar
import special as sf
#import argparse
import lmfit
#import emcee
#import psf_utils
from matplotlib_scalebar.scalebar import ScaleBar

class Pdf_info:
    
    def __init__(self,name):
        self.name = name
        self.dependence = []
    
    def add_data(self, data, distribution, normalized=False):
        if len(data.shape) == 1:
            data = data.reshape((1,len(data)))
        self.data = data
        ### Distribution should be a function
        self.dist = distribution
        self.normalized = normalized

    def add_invar(self, invar):
        if len(invar.shape) == 1:
            invar = invar.reshape((1,len(invar)))
        self.invar = invar

    def add_kwargs(self, kwargs):
        self.kwargs = kwargs

    def add_params(self, params):
        ### params should be an lmfit.Parameters() object
        self.params = params
        
    def add_paramarray(self, paramarray, bounds):
        ### param guesses as an array (for minimize function)
#        if len(paramarray.shape) == 1:
#            paramarray = paramarray.reshape(1,len(paramarray))
        if type(paramarray) == np.ndarray:
            self.paramarray = paramarray
        else:
            self.paramarray = np.array((paramarray))
#        if len(bounds.shape) == 1:
#            bounds = bounds.reshape(1,len(bounds))
        if type(bounds) == np.ndarray:
            self.bounds = bounds
        else:
            self.bounds = np.array((bounds))
            
    def add_dependence(self, Pdf):
        ### Pdf should be another instance of Pdf_info
        self.dependence += Pdf
        
    def add_draw(self, draw):
        ### Function to use when drawing from this distribution
        ### Should be related to self.dist
        self.draw = draw

def eff_flux(r, io, a, eta=3/2):
    flux = 2*np.pi*io*(-(a**2+r**2)*(1+(r/a)**2)**-eta/(2*(eta-1)) + a**2/(2*(eta-1)))
    return flux

def get_axis_ratio(image, centroid, return_rhalf=False):
    """ Trying to get PA, q for galaxies as a whole...
    """
    thetas = np.arange(0,np.pi,np.pi/200)
    def get_Q_el(el,xgrid,ygrid,Igrid):
        if el[0] == el[1] and el[0] == 0:
            Q_el = np.sum(2*xgrid**2*Igrid)
        elif el[0] == el[1] and el[0] == 1:
            Q_el = np.sum(2*ygrid**2*Igrid)
        elif el[0] != el[1]:
            Q_el = np.sum(3*xgrid*ygrid*Igrid)
        return Q_el
    def Q_mat(xgrid, ygrid, Igrid, mask=None, return_xy=False, return_mat=False, plot_grids=False):
        Q = np.zeros((2,2))
        if mask is None:
            mask = np.ones((Igrid.shape), dtype=bool)
        xgrid[mask==0] = 0
        ygrid[mask==0] = 0
        Igrid[mask==0] = 0
        Qxx = get_Q_el((0,0),xgrid, ygrid, Igrid)
        Qyy = get_Q_el((1,1),xgrid, ygrid, Igrid)
        Qxy = get_Q_el((0,1),xgrid, ygrid, Igrid)
        Q[0,0] = Qxx
        Q[0,1] = Qxy
        Q[1,0] = Qxy
        Q[1,1] = Qyy
#        Q_det = np.linalg.det(Q)
#        if plot_grids:
#            plt.imshow(np.hstack((xgrid, ygrid)), interpolation='none', origin='lower')
##            print ygrid[ygrid==0]
##            plt.plot(xgrid.shape[1]+xgrid[0], ygrid[ygrid==0])
##            plt.plot(xgrid[xgrid==0], ygrid[0])
#            plt.show()
#            plt.close()
        if return_xy:
            return Qxy
        elif return_mat:
            return Q
        else:
            return Qxx/Qyy
    def get_moment_ratio(image,theta, centroid):
        xarr = np.arange(image.shape[1]) - centroid[0]
        yarr = np.arange(image.shape[0]) - centroid[1]
        Xg, Yg = np.meshgrid(xarr, yarr)
        Xm = np.cos(theta)*Xg - np.sin(theta)*Yg
        Ym = np.sin(theta)*Xg + np.cos(theta)*Yg
        rprime = sf.make_rarr(xarr, yarr, 0, 0)
#        mask1 = (Xm > 0) * (Ym > 0) * (rprime < min(centroid[0], centroid[1]))
#        mask2 = (Xm < 0) * (Ym > 0) * (rprime < min(centroid[0], centroid[1]))
        mask1 = (rprime < min(centroid[0], centroid[1]))
        mask2 = (rprime < min(centroid[0], centroid[1]))
#        M1 = np.sum(rprime[mask1] * image[mask1])
#        M2 = np.sum(rprime[mask2] * image[mask2])
        M1 = np.sum(abs(Xm[mask1]) * image[mask1])
        M2 = np.sum(abs(Ym[mask2]) * image[mask2])
        rat = M1/M2
        Q = Q_mat(Xm, Ym, image, return_xy=True)
#        print Q
#        rat = Q
        M1 = Q
        if rat > 1:
            rat = 1/rat
#        return rat
#        if M1 < 1:
#            M1 = 1/M1
        return M1
    def get_q(image, theta, centroid):
        xarr = np.arange(image.shape[1]) - centroid[0]
        yarr = np.arange(image.shape[0]) - centroid[1]
        Xg, Yg = np.meshgrid(xarr, yarr)
#        theta = np.pi/4
#        print theta
        theta= - theta
        Xm = np.cos(theta)*Xg - np.sin(theta)*Yg
        Ym = np.sin(theta)*Xg + np.cos(theta)*Yg
        rprime = sf.make_rarr(xarr, yarr, 0, 0)
        ### There are a number of ways to calculate this moment
        ### I use the X and Y moments for positive and negative, averaged together
        ### I also apply a mask for a circular radius centered on the centroid
        ### to mitigate artifacts introduced by the rectangular image space
        maskX = (Xm > 0)# * (rprime < min(centroid[0], centroid[1]))
        maskY = (Ym > 0)# * (rprime < min(centroid[0], centroid[1]))
        maskX2 = (Xm < 0)# * (rprime < min(centroid[0], centroid[1]))
        maskY2 = (Ym < 0)# * (rprime < min(centroid[0], centroid[1]))
        Mx = np.sum(image[maskX]*Xm[maskX])
        My = np.sum(image[maskY]*Ym[maskY])
        Mx2 = -1*np.sum(image[maskX2]*Xm[maskX2])
        My2 = -1*np.sum(image[maskY2]*Ym[maskY2])
        rat = (Mx+Mx2)/(My+My2)
        Q = Q_mat(Xm, Ym, image)
#        rat = Q
        Qxy = Q_mat(Xm, Ym, image, return_xy=True)
#        rat = Q[0,0]/Q[1,1]
        theta_extra = 0
        if rat > 1:
            rat = 1/rat
        else:
            theta_extra = np.pi/2
#        rat = np.sqrt(abs(rat))
        if np.isnan(rat):
            print np.max(image)
            plt.imshow(image, interpolation='none')
            plt.plot(centroid[0], centroid[1], 'bo')
            plt.show()
            plt.close()
        return rat, theta_extra
#    def get_q(image, theta, centroid):
#        xarr = np.arange(image.shape[1]) - centroid[0]
#        yarr = np.arange(image.shape[0]) - centroid[1]
#        Xg, Yg = np.meshgrid(xarr, yarr)
#        Xm = np.cos(theta)*Xg - np.sin(theta)*Yg
#        Ym = np.sin(theta)*Xg + np.cos(theta)*Yg
#        rprime = sf.make_rarr(xarr, yarr, 0, 0)
##        maskX = (Xm > 0) * (rprime < min(centroid[0], centroid[1]))
##        maskY = (Ym > 0) * (rprime < min(centroid[0], centroid[1]))
#        maskr = (rprime < min(centroid[0], centroid[1]))
##        image[image<0.05] = 0
#        Mx = np.sum(image*abs(Xm)*maskr)
#        My = np.sum(image*abs(Ym)*maskr)
##        print Mx, My
##        plt.imshow(np.hstack((image*20, image*abs(Xm)*maskr, image*abs(Ym*maskr))),interpolation='none')
##        plt.plot(centroid[0],centroid[1],'bo')
##        plt.show()
##        plt.close()
#        rat = Mx/My
#        theta_extra = 0
#        if rat > 1:
#            rat = 1/rat
#            theta_extra = np.pi/2
#        return abs(rat), theta_extra
    moments = np.zeros((len(thetas)))
    for i in range(len(thetas)):
        MR = get_moment_ratio(image, thetas[i], centroid)
        moments[i] = MR
    min_arg = np.argmin(moments)
    theta_min = thetas[min_arg]
    if min_arg == 0:
        ml = get_moment_ratio(image, -np.pi/20, centroid)
        mh = moments[min_arg+1]
        if mh < ml:
            theta_2min = thetas[min_arg+1]
        else:
            theta_2min = -np.pi/20
    elif min_arg == len(thetas)-1:
        ml = moments[min_arg-1]
        mh = get_moment_ratio(image, np.pi, centroid)
        if mh < ml:
            theta_2min = np.pi
        else:
            theta_2min = thetas[min_arg-1]
    else:
        ml = moments[min_arg-1]
        mh = moments[min_arg+1]
        if mh < ml:
            theta_2min = thetas[min_arg+1]
        else:
            theta_2min = thetas[min_arg-1]
    mm = moments[min_arg]
    thetalow_old = 10
    thetalow = 1.0*theta_min
    thetadiff = 10
    cnt = 0
    while (thetadiff > 0.001) and (cnt < 10):
        thetalow_old = thetalow*1.0
        mh = get_moment_ratio(image, theta_2min, centroid)
        ml = get_moment_ratio(image, theta_min, centroid)
#        [m, b] = np.polyfit([theta_2max, theta_max], [ml, mh], 1)
        thetamean = (theta_2min + theta_min)/2
        mn = get_moment_ratio(image, thetamean, centroid)
#        print "iteration", cnt
#        print theta_min, ml
#        print theta_2min, mh
#        print thetamean, mn
#        print ""
        if mh > ml:
            mh = mn
            theta_2min = thetamean
        else:
            ml = mn
            theta_min = thetamean
#        if mh > ml:
#            thetalow = theta_min*1.0
#        else:
#            thetalow = theta_2min*1.0
        thetalow = (theta_2min + theta_min)/2
        thetadiff = abs(thetalow-thetalow_old)
        cnt += 1
    def get_half_light(image,centroid,q,PA):
        ltot = np.sum(image)
        rhalf = 3
        lhalf = 0
        while lhalf < ltot/2:
            rhalf += 0.1
            xarr = np.arange(image.shape[1]) - centroid[0]
            yarr = np.arange(image.shape[0]) - centroid[1]
            rprime = sf.make_rarr(xarr, yarr, 0, 0)
            lhalf = np.sum(image*(rprime<rhalf))
        return rhalf
    '''
    max_moment = np.argmax(moments)
    theta_max = thetas[max_moment]
    if max_moment == 0:
        ml = get_moment_ratio(image, -np.pi/20, centroid)
        mh = moments[max_moment+1]
        if mh > ml:
            theta_2max = thetas[max_moment+1]
        else:
            theta_2max = -np.pi/20
    elif max_moment == len(thetas)-1:
        ml = moments[max_moment-1]
        mh = get_moment_ratio(image, np.pi/2, centroid)
        if mh > ml:
            theta_2max = np.pi/2
        else:
            theta_2max = thetas[max_moment-1]
    else:
        ml = moments[max_moment-1]
        mh = moments[max_moment+1]
        if mh > ml:
            theta_2max = thetas[max_moment+1]
        else:
            theta_2max = thetas[max_moment-1]
    mm = moments[max_moment]
    thetamean = thetas[max_moment]
#    print theta_max, theta_2max, thetamean
    cnt = 0
    while (abs(1-mm) > 0.01) and (cnt < 10):
        ml = get_moment_ratio(image, theta_2max, centroid)
        mh = get_moment_ratio(image, theta_max, centroid)
#        [m, b] = np.polyfit([theta_2max, theta_max], [ml, mh], 1)
        thetamean = (theta_2max + theta_max)/2
        mn = get_moment_ratio(image, thetamean, centroid)
        if abs(mn) > abs(ml):
            ml = mn
            theta_2max = thetamean
        else:
            mh = mn
            theta_max = thetamean
        mm = max(ml, mh)
        cnt += 1
    '''
    PA = thetamean
    PA += np.pi/4
#    plt.plot(thetas, moments)
#    plt.show()
#    plt.close()
    q, theta_extra = get_q(image, PA, centroid)
    if return_rhalf:
        return get_half_light(image, centroid, q, PA)
#    print q, np.mod(PA+theta_extra, np.pi)
#    image = image[centroid[1]-20:centroid[1]+21,centroid[0]-20:centroid[0]+21]
#    plt.imshow(image[::-1,:],interpolation='none',extent=[centroid[0]-20,centroid[0]+21,centroid[1]-20,centroid[1]+21,])
#    print q, np.mod(PA+theta_extra,np.pi)
#    plt.imshow(image,interpolation='none',origin='lower')
#    plt.imshow(image>0,interpolation='none',origin='lower')
#    plt.plot(centroid[0],centroid[1],'bo')
#    sf.plot_circle((centroid[0],centroid[1]),10,q=q,PA=np.mod(PA+theta_extra,np.pi))
#    plt.show()
#    plt.close()
    return abs(q), np.mod(PA+theta_extra,np.pi)

def get_centroids(filenames,use_model=False,ext=8, dims=None, return_axis_ratio=False, param_kw='sersic', return_rhalf=False):
    """ Grabs centroids from .fits files with data in given extension
    """
    if type(filenames) == str:
        print("Error, input 'filenames' must be a list")
        exit(0)
    centroids = np.zeros((len(filenames),2))
    if return_axis_ratio:
        q = np.zeros((len(filenames)))
        PA = np.zeros((len(filenames)))
    if return_rhalf:
        rhalf = np.zeros((len(filenames)))
    idx = 0
    for f in filenames:
        fits = pyfits.open(f)
#        centroid_known = False
        arr_pad = 0
        if 'sim' in f:
            image = fits[0].data#-fits[1].data
            junk_invar = np.ones(image.shape)
        else:
            image = fits[ext].data
            image, junk_invar = crop_images(image, np.ones((image.shape)))
        if use_model:
            paramarray = load_sersic_params(f, param_kw=param_kw)
            params = convert_sersic_params(paramarray)
            blob_type = param_kw[1:]
            if '10' in blob_type:
                blob_type = blob_type[0:-6]
            arr_pad = 20
            step = 1
            xarr = np.arange(-arr_pad,image.shape[1]+arr_pad,step)
            yarr = np.arange(-arr_pad,image.shape[0]+arr_pad,step)
            image = galaxy_profile(params,xarr,yarr,image,junk_invar,return_residuals=False,blob_type=blob_type,nblobs=paramarray.shape[1])
#            if paramarray.shape[1] == 1:
#                centroid_known = True
#            fake_image = np.zeros(image.shape)
#            fake_image[20,30] = 1
#            fake_image[40,30] = 1
#            fake_image[30,25] = 1
#            fake_image[30,35] = 1
        if return_rhalf:
            image_orig = 1.0*image
        noise = 0.005
        if 'sim' in f:
            noise = np.median(image)+np.std(image)
        noise = np.median(image)+np.std(image)
        image[image<noise] = 0
#        if centroid_known:
#            centroids[idx] = [paramarray[0]+arr_pad, paramarray[1]+arr_pad]
#        else:
        centroids[idx] = sf.centroid(image, view_plot=False)
        if return_axis_ratio:
            q[idx], PA[idx] = get_axis_ratio(image, centroids[idx])
        if return_rhalf:
            redshifts = np.loadtxt('/home/matt/software/matttest/docs/Galaxy_redshifts.csv', delimiter=',', dtype=str)
            redshifts = redshifts[1:,:]
            for j in range(redshifts.shape[0]):
                if redshifts[j,0][1:11] in f:
                    zgal = float(redshifts[j,1])
            rhalf_px = get_axis_ratio(image_orig, centroids[idx], return_rhalf=return_rhalf)
            rhalf[idx] = px_to_pc(rhalf_px, 0.01, zgal)/1000
        centroids[idx,0] -= arr_pad
        centroids[idx,1] -= arr_pad
        if dims is not None:
            centroids[idx,0] = dims[0]/2 + (image.shape[1]/2-centroids[idx,0])
            centroids[idx,1] = dims[1]/2 + (image.shape[0]/2-centroids[idx,1])
        idx += 1
    if return_axis_ratio:
        return centroids, q, PA
    if return_rhalf:
        return rhalf
    else:
        return centroids
    
def crop_images(image,invar,bgcut=None,dim=76,return_crop_info=False):
    if bgcut is None:
        bgcut = 0.1*np.max(image) ## Aggressive cutting
    pd = 10 ## extra padding to go with aggressive cutting
    pix_tmp = np.copy(image)
    pix_tmp[pix_tmp<bgcut] = 0
    xinds = np.nonzero(np.sum(pix_tmp,axis=0))[0]
    yinds = np.nonzero(np.sum(pix_tmp,axis=1))[0]
    yl = max(yinds[0]-pd,0)
    yr = min(yinds[-1]+pd,image.shape[0])
    xl = max(xinds[0]-pd,0)
    xr = min(xinds[-1]+pd,image.shape[1])
    if dim is not None:
        dlty = yr-yl
        dll = int(np.ceil((dim-dlty)/2))
        dlr = int(np.floor((dim-dlty)/2))
        yl -= dll
        yr += dlr
        dltx = xr-xl
        dll = int(np.ceil((dim-dltx)/2))
        dlr = int(np.floor((dim-dltx)/2))
        xl -= dll
        xr += dlr
    pix_image = image[yl:yr,xl:xr]
    pix_invar = invar[yl:yr,xl:xr]
    if return_crop_info:
        return pix_image, pix_invar, bgcut, pd
    else:
        return pix_image, pix_invar

def source_mosaic(filenames,ext=8, vlims=[0,0.6]):
    """ List of filenames (must be .fits) to import and tile.  ext is the
        fits extension to use for the mosaic (8=pixelized source)
    """
    if type(filenames) == str:
        print("Error, input 'filenames' must be a list")
        exit(0)
    data='/home/matt/software/matttest/data/pix_source_models.txt'
    filenames = np.loadtxt(data,np.str)
    fig = plt.figure(figsize=(10,7.75))
    fig.subplots_adjust(wspace=0,hspace=0)
#    fig.subplots_adjust(wspace=0.2,hspace=0.3)
#    fig.suptitle("Pixelized Source Reconstructions",fontsize=18)    
    idx = 0 ### index for subplots
    filenames = filenames[:-2]
    for filename in filenames:
        pix_image, pix_invar = open_image(filename)
        pix_val = pix_image[pix_image != 0]
#        ps = pix_val[pix_val<1.5*np.mean(pix_val)]
#        bg = np.median(ps)
#        rms = np.sqrt(bg**2)
#        std = np.std(ps)
#        mx = np.max(pix_image)
#        print filename
#        print "rms:", rms, "std:", std
#        print "rms/mx:", rms/mx, "std/mx:", std/mx
#        plt.close()
#        plt.hist(ps,20)
#        plt.show()
#        plt.close()
#        plt.hist(pix_val,200)
#        plt.show()
        plt.subplot(4,5,idx+1)
        ax = plt.gca()
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ### set white strip on image, 10px across - 0.1"
#        vlims = None
        if vlims is None:
            pix_image[70:71,5:15] = np.max(pix_image)
            plt.imshow(pix_image,interpolation='none', cmap=cm.hot)
        else:
            pix_image[70:71,5:15] = vlims[1]
            plt.imshow(pix_image,interpolation='none', cmap=cm.hot, vmin=vlims[0], vmax=vlims[1])
#        plt.imshow(np.hstack((pix_image,best_model,(pix_image-
#    best_model))),interpolation='none')
#        plt.title(os.path.split(filename)[1][0:11],fontsize=12)
        ax.text(4, 68, '0.1"', color='w', fontsize=6)
#        ratio = (ax.get_xlim()[0]-ax.get_xlim()[1])/(ax.get_ylim()[1]-ax.get_ylim()[0])
#        ax.set_aspect(1*ratio, adjustable='box')
        idx += 1
    plt.savefig('/home/matt/software/matttest/results/figs/raw_mosaic.pdf')
    plt.show()
    plt.close()    

  
def fitted_mosaic(data='/home/matt/software/matttest/data/pix_source_models.txt', param_kw=None, overwrite=True, blob_type='sersic', vlims=[0,0.6]):
    def save_or_show(fig,param_kw,cnt=''):
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.01,hspace=0.01)
        if 'sim' in data:
            if param_kw is None:
                plt.savefig('/home/matt/software/matttest/results/figs/sim/mosaic{}.pdf'.format(cnt))
            else:
                plt.savefig('/home/matt/software/matttest/results/figs/sim/mosaic{}{}.pdf'.format(param_kw,cnt))
            plt.show()
            plt.close()
        else:
            if param_kw is None:
                plt.savefig('/home/matt/software/matttest/results/figs/mosaic{}.pdf'.format(cnt), bbox_inches='tight', pad_inches=0)
            else:
                plt.savefig('/home/matt/software/matttest/results/figs/mosaic{}{}.pdf'.format(param_kw,cnt), bbox_inches='tight', pad_inches=0)
            plt.show()
            plt.close()
    filenames = np.loadtxt(data,np.str)
    redshifts = np.loadtxt('/home/matt/software/matttest/docs/Galaxy_redshifts.csv', delimiter=',', dtype=str)
#    redshift_hdrs = redshifts[0,:]
    redshifts = redshifts[1:,:]    
    num_plots = len(filenames)
    num_cols = 2
    max_rows = 7 ### split into two figs...
    num_rows = int(np.ceil(num_plots/num_cols))
    if num_rows > max_rows:
        cnt = 0
    else:
        cnt = ''
    fig = plt.figure(dpi=300,figsize=(10,15))
#    fig.suptitle("Data/Model/Residuals for all LAE Galaxies",fontsize=18)
    idx = 0 ### index for subplots
    idxo = 0
    if 'sim' in data:
        yshft = np.zeros((len(filenames)))
        xshft = np.zeros((len(filenames)))
    else:
        yshft = [14, 6, 7, 27, 20, 10, 21, 21, 16, 18, 11, 8, 19, 7, 13, 13, 4]
        xshft = [18, -7, 13, 19, 17, 11, 20, 20, 13, 18, 20, 18, 22, -2, 4, 12, 16]
    font = FontProperties()
    font.set_family('serif')
    font.set_size(10)
    for filename in filenames:
        ### Get Redshifts
        for j in range(redshifts.shape[0]):
                if redshifts[j,0][1:11] in filename:
                    zgal = float(redshifts[j,1])
        ### Open file and get fit parameters
        pix_image, pix_invar = open_image(filename)
        if 'sim' in data:
            pix_image = pyfits.open(filename)[0].data
            pix_err = pyfits.open(filename)[1].data
            pix_image -= np.median(pix_image)
            pix_err[pix_err<0.01*np.max(pix_image)] = 0.01*np.max(pix_image)
            pix_invar = 1/(pix_err)**2
            ### Remove infinities and nans
            pix_invar[pix_invar==np.inf] = 0
            pix_invar[np.isnan(pix_invar)] = 0
            zgal = 2.5
        paramarray = load_sersic_params(filename, param_kw=param_kw)
        paramarray[0] += xshft[idxo+idx]
        paramarray[1] += yshft[idxo+idx]
        if idx == 1:
            print paramarray[1]
        params = convert_sersic_params(paramarray)
        best_model = galaxy_profile(params,np.arange(pix_image.shape[1]),np.arange(pix_image.shape[0]),pix_image,pix_invar,return_residuals=False,blob_type=blob_type,nblobs=paramarray.shape[1])
        chi2r = np.sum((pix_image-best_model)**2*pix_invar)/(np.size(pix_image)-paramarray.size)
        if idx == 1:
            print best_model.shape
        print filename
        print "  Reduced chi^2 = ", chi2r
#        hidx = np.mod(idx,num_cols)
#        vidx = int(np.floor(idx/num_rows))
        plt.subplot(num_rows,num_cols,idx+1)
        ax = plt.gca()
        for bl in range(paramarray.shape[1]):
#            rad = px_to_pc(paramarray[2,bl]*1000,0.01,zgal,inv=True)
#            sf.plot_circle((paramarray[0,bl], paramarray[1,bl]), 2*rad, q=paramarray[5,bl], PA=paramarray[6,bl],lw=0.5,color='w')
            shft = 0
            if '011300' in filename:
                shft = 2
            sf.plot_circle((paramarray[0,bl]+shft, paramarray[1,bl]), 2, lw=0.5, color='w')
#            ax.plot(paramarray[0,bl],paramarray[1,bl], 'wx')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        res = pix_image-best_model
        stck = np.hstack((pix_image,best_model,res))
        vlims = None
        if vlims is None:
            pix_image[:,-1] = np.max(stck)
            best_model[:,-1] = np.max(stck)
#            pix_image[70:71,5:15] = np.max(pix_image)
            plt.imshow(np.hstack((pix_image,best_model,res)),interpolation='none', cmap=cm.hot)
        else:
            pix_image[:,-1] = vlims[1]
            best_model[:,-1] = vlims[1]
#            pix_image[70:71,5:15] = vlims[1]
            plt.imshow(np.hstack((pix_image,best_model,res)),interpolation='none', cmap=cm.hot, vmin=vlims[0], vmax=vlims[1])
#        plt.title(os.path.split(filename)[1][0:11],fontsize=10)
        ax.text(89,10,os.path.split(filename)[1][0:11], color='w', fontproperties=font) #transform=ax1a.transAxes)
        ntext = "N$_{\mathrm{c}}$=" + str(paramarray.shape[1])
        ax.text(5,10, ntext, color='w', fontproperties=font)
        ### Get pixel scale
        lf = px_to_pc(1000,0.01,zgal, inv=True)
#        print 1/lf
        scalebar = ScaleBar(1, units='m', label='1 kpc',frameon=False, color='w', height_fraction = 0.002, length_fraction = 1/lf, location='lower left',font_properties=dict(family='serif',size=8))
        ax.add_artist(scalebar)
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')
#        xdlt = np.array(([pix_image.shape[1],2*pix_image.shape[1]]))
#        ydlt = np.ones((2))*pix_image.shape[0]
#        sf.plt_deltas(xdlt, ydlt, color='w')
#        ax.text(4, 68, '0.1"', color='w', fontsize=6)
#        ratio = (ax.get_xlim()[0]-ax.get_xlim()[1])/(ax.get_ylim()[1]-ax.get_ylim()[0])
#        ax.set_aspect(0.3*ratio, adjustable='box')
        idx += 1
        if idx >= max_rows*num_cols:
            save_or_show(fig,param_kw,cnt)
            fig = plt.figure(figsize=(10,15),dpi=300)
#            fig.subplots_adjust(wspace=0,hspace=0.1)
            cnt += 1
            idxo = idx
            idx = 0
    save_or_show(fig,param_kw,cnt)
    
  
def open_image(filename):
    ''' For opening pixelized source images, then cropping
    '''
    if 'sim' in filename and '_fit' in filename:
        filename = filename[0:-9]+'.fits'
    pix_fits = pyfits.open(filename)
    if 'sim' in filename:
        pix_image = pix_fits[0].data
        pix_err = pix_fits[1].data
        dim=pix_image.shape[0]
        bgcut = 0
#        bgcut = 0.1*np.max(pix_image-np.mean(pix_image))+np.mean(pix_image)
    else:
        pix_image = pix_fits[8].data
        pix_err = pix_fits[9].data #Not sure of the units..., assume standard dev
        bgcut=None
        dim=76
    ### S/N approach to pix_invar...not necessarily valid
    #pix_invar = 1/(abs(pix_image)+np.mean(pix_err)**2)
    #pix_invar = (pix_err/pix_image)**2
    pix_invar = 1/(pix_err)**2
    pix_invar[pix_invar==np.inf] = 0
    pix_invar[np.isnan(pix_invar)] = 0    
    pix_full = np.copy(pix_image)
    xinds = np.nonzero(np.sum(pix_full,axis=0))[0]
    yinds = np.nonzero(np.sum(pix_full,axis=1))[0]
    pix_full = pix_full[yinds[0]:yinds[-1],xinds[0]:xinds[-1]]
    pix_invar_full = pix_invar[yinds[0]:yinds[-1],xinds[0]:xinds[-1]]
    ### Crop zero portions to make fitting go faster
    pix_image, pix_invar, bgcut, pd = crop_images(pix_image, pix_invar,bgcut=bgcut, return_crop_info=True,dim=dim)
#    if pix_full.size < pix_image.size:
#        pix_image = pix_full
#        pix_invar = pix_invar_full
    return pix_image, pix_invar
    
def load_sersic_params(filename, savedir = '/home/matt/software/matttest/results', param_kw='sersic', verbose=False):
    ''' Loads saved values of parameters (right now only does sersic, eff)
        Returns array of size (nparams x nblobs)
    '''
    root, filename = os.path.split(filename)
#    if 'sim' not in filename:
    if param_kw is None:
        filename = "{}.fits".format(filename[:11])
    else:
        filename = "{}{}.fits".format(filename[:11],param_kw)
    param_file = pyfits.open(os.path.join(savedir,filename))
    root, targ_num = os.path.split(root)
    if verbose:
        print("Target Number is {}".format(targ_num))
    if 'sim' in filename and '_fit' in filename:
        best_params = param_file[0].data
        best_params = np.ravel(best_params.T)
    elif 'sim' in filename:
        best_params = param_file[2].data
        best_params = np.ravel(best_params.T)
    else:
        best_params = param_file[0].data
    if param_file[0].header['MODEL'] == 'sersic':
        div = 7
    elif param_file[0].header['MODEL'] == 'eff':
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
    return np.vstack((xb, yb, sbb, sigb, nsb, qgal, PAgal))
    
def convert_sersic_params(paramarray, randomize_centers=False):
    ''' Returns lmfit.Parameters() object with sersic params and my naming
        convention...
    '''
    xb, yb, sbb, sigb, nsb, qgal, PAgal = paramarray[0], paramarray[1], paramarray[2], paramarray[3], paramarray[4], paramarray[5], paramarray[6]
    nblobs = paramarray.shape[1]
    sersic_names = []
    sersic_estimates = []
#    sersic_mins = []
#    sersic_maxes = []
    if randomize_centers:
        xb = xb + 0.001*np.random.randn(len(xb))
        yb = yb + 0.001*np.random.randn(len(yb))
    for i in range(nblobs):
        sersic_names += ['xcb{}'.format(i),'ycb{}'.format(i),'Ieb{}'.format(i),'reb{}'.format(i),'nb{}'.format(i),'qb{}'.format(i),'PAb{}'.format(i)]
    #    xc = blob_cntrs[i,1]
    #    yc = blob_cntrs[i,0]
    #    hght = pix_smooth.shape[0]*pix_smooth[int(yc),int(xc)]
        sersic_estimates += [xb[i], yb[i], sbb[i], sigb[i], nsb[i], qgal[i], PAgal[i]]
#        qmin = 0.05
#        sersic_mins += [xb[i]-0.5, yb[i]-0.5, sbb[i]/3, sigb[i]/3, 0.01, max(qgal[i]-qmin,0.1), 0]
#        sersic_maxes += [xb[i]+0.5, yb[i]+0.5, sbb[i]*3, sigb[i]*3, 8, 100, 2*np.pi]
    #sersic_fixed = np.ones((len(sersic_names)))
    params_sers = lmfit.Parameters()
    params_sers = sf.array_to_Parameters(params_sers, sersic_estimates, arraynames=sersic_names)#, minarray=sersic_mins, maxarray=sersic_maxes, fixed=sersic_fixed)
    return params_sers

def io_r_to_lum(io,ro,z,ang=2340,photflam=1.1845489*10**-19):
    ro = px_to_pc(ro*1000, 0.01, z, inv=True)
    dL = sf.luminosity_dist(z)
    dL *= 3.086e18 ### conver pc to cm
    io *= photflam*ang
    F = 2*np.pi*io*ro**2
    L = F*4*np.pi*dL**2
    return L
    
def galaxy_profile(params, xarr, yarr, z, inv_z, return_residuals=True, fdg=False, dgt='sersic', fb=True, blob_type='sersic', nblobs=1, ab=False):
    """ Simple galaxy profile - sersic + 2D gaussian blobs.
        params is an lmfit object, dimensionality is 7 + 5*nblobs
        Assume no psf convolution.
        Returns residuals from profile for lmfit fitting.
        Galaxy model can be 'sersic' or 'gaussian'
    """
    if fdg:
        if dgt is 'sersic':
            galaxy_model = sf.sersic2d_lmfit(params,xarr,yarr)
        else:
            print("Invalid diffuse galaxy type")
            exit(0)
    else:
        galaxy_model = np.zeros((len(yarr),len(xarr)))
    if fb:
        for i in range(nblobs):
            if blob_type == 'gaussian':
                galaxy_model += sf.gauss2d_lmfit(params,xarr,yarr,i,ab=ab)
            elif blob_type == 'sersic':     
                galaxy_model += sf.sersic2d_lmfit(params,xarr,yarr,i,ab=ab)
            elif blob_type == 'eff':
                galaxy_model += sf.eff2d_lmfit(params,xarr,yarr,i,ab=ab)
            else:
                print("Invalid blob type: {}".format(blob_type))
                exit(0)
    if return_residuals:
        if len(xarr) == z.shape[1] and len(yarr) == z.shape[0]:
            galaxy_residuals = np.ravel((z-galaxy_model)**2*inv_z)
        else:
            print "Cannot return galaxy_profile residuals if input array sizes don't match image size."
            exit(0)
        return galaxy_residuals
    else:
        return galaxy_model
        
def import_visual_blob_values(plot_dists=False):
    """ Imports the values I found by eye and plots their distributions.
        Only useful for creating initial guesses of PDF form
    """
    bv = np.loadtxt('/home/matt/software/matttest/docs/Visual Blob Distribution.csv',str,delimiter=',')
    bv_hdr = bv[0]
    bv = bv[1:]
    rad = bv[:,2].astype(float)
    lfrac = bv[:,3].astype(float)
    xc = bv[:,4].astype(float)
    yc = bv[:,5].astype(float)
    gals = np.nonzero(bv[:,6]!='')[0]
    q = bv[:,6][gals].astype(float)
    sbmx = bv[:,8].astype(float)
    fwhm = bv[:,9].astype(float)
    nblobs = np.zeros((len(gals)),dtype=int)
    for i in range(len(nblobs)-1):
        nblobs[i] = int(bv[:,1][gals[i+1]-1])
    nblobs[-1] = int(bv[-1,1])
    ############ spacing between blob points ##########################
#    rrel = np.zeros((10000))
#    ridx = 0
#    for j in range(len(gals)):
#        if nblobs[j] == 1:
#            continue
#        rmat = np.zeros((nblobs[j],nblobs[j]))
#        xcs = xc[gals[j]:gals[j]+nblobs[j]]
#        ycs = yc[gals[j]:gals[j]+nblobs[j]]
#        for k in range(nblobs[j]):
#            for l in range(k,nblobs[j]):
#                if k == l:
#                    continue
#                rmat[k,l] = np.sqrt((xcs[k]-xcs[l])**2 + (ycs[k]-ycs[l])**2)
#        rarr = rmat[rmat!=0]
#        rrel[ridx:ridx+len(rarr)] = rarr
#        ridx += len(rarr)
#    rrel = rrel[0:ridx]
    ############ spacing from central point ##########################
    rsep = np.zeros((len(xc)))
    ridx = 0
#    Qij = np.zeros((len(gals),3))
    Qrat = np.zeros((len(gals)))
    for j in range(len(gals)):
        if nblobs[j] == 1:
            ridx += 1
            continue
        xcs = xc[gals[j]:gals[j]+nblobs[j]]
        ycs = yc[gals[j]:gals[j]+nblobs[j]]
        sbs = sbmx[gals[j]:gals[j]+nblobs[j]]*fwhm[gals[j]:gals[j]+nblobs[j]]**2
        xcw = np.sum(xcs*sbs)/np.sum(sbs)
        ycw = np.sum(ycs*sbs)/np.sum(sbs)
        rsep[ridx:ridx+nblobs[j]] = np.sqrt((xcs-xcw)**2+(ycs-ycw)**2)
        ### Qij, col0 = Qxx, col2 = Qyy, col3 = Qxy
        Qxx = np.sum((3*(xcs-xcw)**2-rsep[ridx:ridx+nblobs[j]]**2)*sbs)
        Qyy = np.sum((3*(ycs-ycw)**2-rsep[ridx:ridx+nblobs[j]]**2)*sbs)
        Qxy = np.sum((3*(xcs-xcw)*(ycs-ycw))*sbs)
        Qmat = np.array(([Qxx, Qxy],[Qxy, Qyy]))
        print "Iteration {}".format(j)
        print Qmat
        evl, evc = np.linalg.eig(Qmat)
        Qrat[j] = np.sqrt(abs((2+evl[0]/evl[1])/(2*evl[0]/evl[1]+1)))
        if Qrat[j] > 1:
            Qrat[j] = 1/Qrat[j]
        print Qrat[j]
#        print xcw, ycw
#        print rsep[ridx:ridx+nblobs[j]]
        ridx += nblobs[j]
    if plot_dists:
        plt.figure('Number of Blobs')
        plt.hist(nblobs,9)
        plt.figure('Blob FWHM Distribution')
        plt.hist(fwhm)
        plt.figure('Blob Maximum Surface Brightness')
        plt.hist(sbmx)
        plt.figure('Blob Separation from Center')
        plt.hist(rsep,10)
        plt.figure('Axis Ratio for Quadrupole Moment')
        plt.hist(Qrat,8)
        plt.ion()
        plt.show()     
    return nblobs, fwhm, sbmx, rsep, Qrat

def import_guesses(img_num,fl='/home/matt/software/matttest/docs/Sim Blob Distribution.csv'):
    """ Imports guesses for xb, yb, sbb, sigb, nsb, PAb, qb for file 'fl'
    """
    bv = np.loadtxt(fl,str,delimiter=',')
    bv_hdr = bv[1]
    bv = bv[2:]
    xc = bv[:,1].astype(float)
    yc = bv[:,2].astype(float)
    try:
        sbmx = bv[:,3].astype(float)
    except:
        sbmx = np.zeros(xc.shape)
    rad = bv[:,4].astype(float)
    qb = bv[:,5].astype(float)
    PA = bv[:,6].astype(float)
    nsb = 3*np.ones(xc.shape)
    ind1s = np.nonzero(bv[:,0]!='')[0]
    nblobs = np.ediff1d(ind1s)
    nblobs = np.append(nblobs,len(xc)-ind1s[-1])
    sect = range(ind1s[img_num], ind1s[img_num]+nblobs[img_num])
    return xc[sect], yc[sect], sbmx[sect], rad[sect], nsb[sect], qb[sect], PA[sect]
    
def import_sim_blob_values(filenames):
    re = np.zeros((1000))
    rpx = np.zeros((1000))
    dc = np.zeros((1000))
    Ie = np.zeros((1000))
    qb = np.zeros((1000))
    PA = np.zeros((1000))
    gidx = 0
    for f in filenames:
        spar = pyfits.open(f)[2].data
        noise = pyfits.open(f)[1].data
        img = pyfits.open(f)[0].data
        ### make a cut on the peak flux in ~1pixel from each clump - depends on i and r
        fluxes = eff_flux(np.sqrt(1/np.pi),spar[2], spar[3])
#        print np.std(noise)
        msk = fluxes > 1*np.std(noise)
    #    print msk
        xb, yb, sbb, sigb, qgal, PAgal = spar[0][msk], spar[1][msk], spar[2][msk], spar[3][msk], spar[5][msk], spar[4][msk]
        xc, yc = sf.centroid(img)
        bidx = 0
        nb = len(xb)
        for i in range(nb):
            re[gidx+bidx] = px_to_pc(sigb[bidx], 0.01, 2.5)/1000
            rpx[gidx+bidx] = sigb[bidx]
            dc[gidx+bidx] = px_to_pc(np.sqrt((xb[bidx]-xc)**2 + (yb[bidx]-yc)**2), 0.01, 2.5)/1000
            Ie[gidx+bidx] = sbb[bidx]
            qb[gidx+bidx] = qgal[bidx]
            PA[gidx+bidx] = PAgal[bidx]
            bidx += 1
        gidx += nb
    re = re[re!=0]
    rpx = rpx[rpx!=0]
    dc = dc[dc!=0]
    Ie = Ie[Ie!=0]
    qb = qb[qb!=0]
    PA = PA[PA!=0]
    return re, rpx, dc, Ie, qb, PA
    
def import_fitted_blob_values(filenames, plot_dists=False, centroids=None, qgal=None, PAgal=None, logscale=False, bins=14, Icut = False, param_kw=None, sim=False, return_lum=False, exact=False, view_PA_corr=False):
    """ Imports the values I found by fitting sersic models.  This isn't
        very flexible right now...
    """
    if not sim:
        param_files = glob.glob('/home/matt/software/matttest/results/*{}.fits'.format(param_kw))
        param_files = np.array(param_files)
        npf = len(param_files)
        pf_msk = np.ones(len(param_files), dtype=bool)
        for i in range(npf):
            if 'sim' in param_files[i]:
                pf_msk[i] = 0
        param_files = param_files[pf_msk]
    else:
        param_files = filenames
    param_files = np.sort(param_files)
    redshifts = np.loadtxt('/home/matt/software/matttest/docs/Galaxy_redshifts.csv', delimiter=',', dtype=str)
#    redshift_hdrs = redshifts[0,:]
    redshifts = redshifts[1:,:]
    re = np.zeros((1000))
    rpx = np.zeros((1000))
    Ie = np.zeros((1000))
    Sbe = np.zeros((1000))
    nsers = np.zeros((1000))
    qb = np.zeros((1000))
    PAb = np.ones((1000))*100
    QG = np.zeros((len(filenames)))
    PAG = np.zeros((len(filenames)))
    PAcorr = np.zeros((1000))
    lums = dict()
    rshft = np.zeros((len(filenames)))
    if centroids is not None:
        rsep = np.zeros((1000))
        if len(centroids) != len(filenames):
            print len(centroids)
            print len(filenames)
            print("Number of centroids doesn't match number of fitted parameter files")
            exit(0)
    nblobs = np.empty((len(filenames)),dtype=int)
    bidx = 0
    gidx = 0
    zref = 2.6
    for f in param_files:
        param_file = pyfits.open(f)
        if sim:
            zgal=zref
        else:
            for j in range(redshifts.shape[0]):
                if redshifts[j,0][1:11] in f:
                    zgal = float(redshifts[j,1])
                    gidx = j
        rshft[gidx] = zgal
        k_corr = k_correction(zgal)
        if sim and '_fit' not in f:
            best_params = param_file[2].data
            best_params = np.ravel(best_params.T)
        else:
            best_params = param_file[0].data
#        exptime = param_file[0].header['EXPTIME']
#        photflam = param_file[0].header['PHOTFLAM']
#        exptime = 626 #varies - need to key on this
        photflam = 1.1845489#*10**-19 #constant
        if param_file[0].header['MODEL'] == 'sersic':
            div = 7
        elif param_file[0].header['MODEL'] == 'eff':
            div = 7
        else:
            print("Invalid MODEL: {}".format(param_file[0].header['MODEL']))
            exit(0)
        pix_image, pix_invar = open_image(filenames[gidx])
        paramarray = load_sersic_params(filenames[gidx], param_kw=param_kw)
        params = convert_sersic_params(paramarray)
        if qgal is not None and PAgal is not None:
            QG[gidx] = qgal[gidx]
            PAG[gidx] = PAgal[gidx]
        else:
            best_model = galaxy_profile(params,np.arange(pix_image.shape[1]),np.arange(pix_image.shape[0]),pix_image,pix_invar,return_residuals=False,blob_type=param_file[0].header['MODEL'],nblobs=paramarray.shape[1])
            if sim:
                QG[gidx], PAG[gidx] = get_axis_ratio(pix_image, centroids[gidx])#best_model
            else:
                noise = 0.005
                best_model[best_model < noise] = 0
                QG[gidx], PAG[gidx] = get_axis_ratio(best_model, centroids[gidx])
        blob_cnt = int(len(best_params)/div)
        lum_tmp = np.zeros((blob_cnt))
        # Apply a dynamic intensity cut to not count low value blobs
        if Icut:
            Ies = best_params[3::div]
            res = px_to_pc(best_params[4::div],0.01,zgal)
            lum_clump = io_r_to_lum(Ies,res/1000,zgal)
            ltot = np.sum(lum_clump)
            lmask = lum_clump > 0.08*ltot
#            nblobs_bright = len(Ies[Ies>0.2*np.max(Ies)]) #Can vary cutoff
            nblobs[gidx] = np.sum(lmask)
        else:
            nblobs[gidx] = blob_cnt
        PAdelt = np.zeros((blob_cnt))  ### For evaluating PA correlations
        check_vals = False
        if check_vals:
            print f
        for i in range(blob_cnt):
            if centroids is not None:
                xb = best_params[div*i]
                yb = best_params[div*i+1]
                if sim and exact:
                    ### apply deviant centroid cut
                    if xb < 0 or xb > pix_image.shape[1] or yb <0 or yb > pix_image.shape[1]:
                        continue
                rsep[bidx] = px_to_pc(np.sqrt((xb-centroids[gidx,0])**2 + (yb-centroids[gidx,1])**2),0.01,zgal)
            Ie[bidx] = best_params[div*i+2]*k_corr*(1+zgal)**5/(1+zref)**5
            Sbe[bidx] = best_params[div*i+2]*photflam*k_corr
            if sim and '_fit' not in f:
                re[bidx] = best_params[div*i+3]*1000
            else:
                re[bidx] = px_to_pc(best_params[div*i+3],0.01,zgal)
            rpx[bidx] = best_params[div*i+3]
            nsers[bidx] = best_params[div*i+4]
            qb[bidx] = best_params[div*i+5]
            PAb[bidx] = best_params[div*i+6]# - PAgal[gidx]
            lum_tmp[i] = io_r_to_lum(Ie[bidx],re[bidx]/1000,zgal)
            if check_vals:
                print " Clump", i
                print "  Pixel radius =", rpx[bidx]
                print "  Parsec radius =", re[bidx]
                print "  Axis ratio =", qb[bidx]
                print "  Luminosity =", lum_tmp[i]
            if sim and exact:
                ### apply flux cut
                eff_flx = eff_flux(np.sqrt(1/np.pi),Ie[bidx],best_params[div*i+3])
                noise = pyfits.open(filenames[gidx])[1].data
                keep = eff_flx > 1*np.std(noise)
                if not keep:
                    Ie[bidx] = 0
                    Sbe[bidx] = 0
                    re[bidx] = 0
                    rpx[bidx] = 0
                    nsers[bidx] = 0
                    qb[bidx] = 0
                    PAb[bidx] = 0
                    lum_tmp[i] = 0
                    continue
            PAdelt[i] = np.mod(PAG[gidx] - PAb[bidx], np.pi) #Linear difference
            PAcorr[bidx] = np.mod(PAG[gidx] - PAb[bidx], np.pi) #Linear difference
#            print PAcorr[bidx]
            PAcorr[bidx] = (np.pi - PAcorr[bidx]) if PAcorr[bidx] > np.pi/2 else PAcorr[bidx]
#            print PAcorr[bidx]
#            time.sleep(1)
#            print ""
#            PAdelt[i] = np.mod(np.arctan(yb/xb)-PAb[bidx],np.pi/2) #radial difference
#            PAcorr[bidx] = np.mod(np.arctan(yb/xb)-PAb[bidx],np.pi/2)
            bidx += 1
        lums[gidx] = lum_tmp
        if sim:
            gidx += 1
    if Icut:
        return nblobs
    if view_PA_corr:
        PAcorr = PAcorr[PAcorr != 0]
        plt.hist(PAcorr, 14)
        plt.show()
        plt.close()
    param_kw = param_kw[1:]
    re = re[re!=0]
    rpx = rpx[rpx!=0]
    rscl = 1
#    if 'eff' in param_kw:
#        rscl = (np.sqrt((0.5)**(1/(1-3/2))-1)) ##convert a to r1/2
    rsc = 5
    re /= 1000
    rsep /= 1000
    Ie = Ie[Ie!=0]
    Sbe = Sbe[Sbe!=0]
#    exptime = 626 #varies - need to key on this
    photflam = 1.1845489#*10**-19 #constant
    arcsec2 = (0.01)**2
    iscl = photflam/arcsec2/10**4#10**5#10 #*10**22 to get on scale [0, 1]
    ### convert to magnitudes
#    Ie = find_abmag(Ie)
    ### or convert to flux
#    Ie = Sbe*10**22
#    Ie.astype(float)
    nsers = nsers[nsers!=0]
    qb = qb[qb!=0]
    PAb = PAb[PAb!=100]
    rlim = (0, 1.2)
    ilim = (0, 0.8)
    dlim = (0, 3.5)
    qlim = (0, 1)
    if centroids is not None:
        rsep = rsep[rsep!=0]
    if logscale:
        Ie = np.log10(Ie)
#        re = np.log10(re)
    if plot_dists:
        font = FontProperties()
        font.set_family('serif')
        font.set_size(14)
        tfont = font.copy()
        tfont.set_size(12)
        savedir = '/home/matt/software/matttest/results/figs'
        if sim:
            savedir = '/home/matt/software/matttest/results/figs/sim'
        ### Galaxy Histograms
        ######################################################
        hcolor = [0.3, 0.3, 0.3]
#        fig0, ((ax1, ax2)) = plt.subplots(1,2,figsize=(10,3.5))
#        fig.suptitle('Histograms of Clump Parameters',fontsize=24)
#        fig0.subplots_adjust(top = 0.9, bottom = 0.2, hspace = 0.4)
#        ax1.set_title('Number of Clumps')
        fig = plt.figure()
        ax1 = plt.gca()
        ax1.hist(nblobs,bins=(np.max(nblobs)-np.min(nblobs)+1),color=hcolor, normed=True)
        xblb = np.arange(1,10)
        yblb = (1/9.0)*np.ones(len(xblb))
        ax1.plot(xblb, yblb, 'k', linewidth = 2)
        xlabel = ax1.set_xlabel("N$_{\mathrm{c}}$ (per galaxy)", fontproperties=font)
        ax1.set_ylabel("N$_{\mathrm{c}}$/N$_{\mathrm{tot}}$ (Probability)", fontproperties=font)
        ax1.set_xticklabels(ax1.get_xticks(), tfont)
        ax1.set_yticklabels(ax1.get_yticks(), tfont)
#        print centroids
#        print QG
        if param_kw is None:
            plt.savefig(os.path.join(savedir,'Ngal.pdf'), bbox_extra_artists=[xlabel], bbox_inches='tight')
        else:
            plt.savefig(os.path.join(savedir,'Ngal_{}.pdf'.format(param_kw)), bbox_extra_artists=[xlabel], bbox_inches='tight')
        fig = plt.figure()
        ax2 = plt.gca()
        ax2.hist(QG,bins=int(len(QG)/2),color=hcolor, normed=True)
        xqgl = np.linspace(0.1,np.max(QG),300)
        cpars = np.array([0.582, 0.142])
        yqgl = sf.cauchy_lmfit_trunc(cpars, xqgl)
        ax2.plot(xqgl, yqgl, 'k', linewidth = 2)
        ax2.set_xlabel("Q$_{\mathrm{c}}$ (galaxy axis ratio)", fontproperties=font)
        ax2.set_ylabel("Probability", fontproperties=font)
        ax2.set_xticklabels(ax2.get_xticks(), tfont)
        ax2.set_yticklabels(ax2.get_yticks(), tfont)
        if param_kw is None:
            plt.savefig(os.path.join(savedir,'Qgal.pdf'), bbox_extra_artists=[xlabel], bbox_inches='tight')
        else:
            plt.savefig(os.path.join(savedir,'Qgal_{}.pdf'.format(param_kw)), bbox_extra_artists=[xlabel], bbox_inches='tight')
#        fig0, ((ax1, ax2)) = plt.subplots(1,2,figsize=(10,3.5))
##        fig.suptitle('Histograms of Clump Parameters',fontsize=24)
#        fig0.subplots_adjust(top = 0.9, bottom = 0.2, hspace = 0.4)
##        ax1.set_title('Number of Clumps')
#        ax1.hist(nblobs,bins=(np.max(nblobs)-np.min(nblobs)+1),color=hcolor, normed=True)
#        xblb = np.arange(1,10)
#        yblb = (1/9.0)*np.ones(len(xblb))
#        ax1.plot(xblb, yblb, 'k', linewidth = 2)
#        xlabel = ax1.set_xlabel("N_${\mathrm{c}}$ (per galaxy)", fontproperties=font)
#        ax1.set_ylabel("N$_{\mathrm{c}}$/N$_{\mathrm{tot}}$ (Probability)", fontproperties=font)
##        print centroids
##        print QG
#        ax2.hist(QG,bins=int(len(QG)/2),color=hcolor, normed=True)
#        xqgl = np.linspace(0.1,np.max(QG),300)
#        cpars = np.array([0.582, 0.142])
#        yqgl = sf.cauchy_lmfit_trunc(cpars, xqgl)
#        ax2.plot(xqgl, yqgl, 'k', linewidth = 2)
#        ax2.set_xlabel("Q$_{\mathrm{c}}$ (galaxy axis ratio)", fontproperties=font)
#        ax2.set_ylabel("Probability", fontproperties=font)
#        if param_kw is None:
#            plt.savefig(os.path.join(savedir,'Gxy_histograms.pdf'), bbox_extra_artists=[xlabel], bbox_inches='tight')
#        else:
#            plt.savefig(os.path.join(savedir,'Gxy_histograms_{}.pdf'.format(param_kw)), bbox_extra_artists=[xlabel], bbox_inches='tight')
        ### Clump Histograms
        ######################################################
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3,2,figsize=(10,10))
#        fig.suptitle('Histograms of Clump Parameters',fontsize=24)
        fig.subplots_adjust(top = 0.9, hspace = 0.4)
#        ax1.set_title('Radius of Clumps')
#        ax1.hist(re,bins,color=hcolor)
        if logscale:
            ax1.hist(re,bins,color=hcolor)
        else:
            ax1.hist(re[re<rsc*np.median(re)]*rscl,bins,color=hcolor, range=rlim)
        ax1.set_xlabel("r$_{\mathrm{c}}$ (kpc)", fontproperties=font)
        ax1.set_xlim(rlim)
#        ax2.set_title('Intensity of Clumps')
        ax2.hist(Ie*iscl,bins,color=hcolor, range=ilim)
        ax2.set_xlim(ilim)
        ax2.set_xlabel(r"i$_{\mathrm{c}}$ (ergs cm$^{-2}$ s$^{-1}$ $\AA^{-1}$ arcsec$^{-2}$$\times$10$^{-19})$", fontproperties=font)
#        ax2.set_xlabel("$Intensity\ (counts)$")
#        ax3.set_title('Clump Separation from Centroid')
        ax3.hist(rsep,bins,color=hcolor, range=dlim)
        ax3.set_xlabel("d$_{\mathrm{c}}$ (kpc)", fontproperties=font)
        ax3.set_xlim(dlim)
#        ax4.set_title('Sersic Index')
        if 'eff' in param_kw:
            ax4.hist(qb,bins,color=hcolor, range=qlim)
            ax4.set_xlim(qlim)
            ax4.set_xlabel("q$_{\mathrm{c}}$ (axis ratio)", fontproperties=font)
            fig.delaxes(ax5)
            fig.delaxes(ax6)
        else:
            ax4.hist(nsers,bins,color=hcolor)
            ax4.set_xlabel("$Sersic\ Index$", fontproperties=font)
    #        ax5.set_title('Clump Ellipticity')
            ax5.hist(qb,bins,color=hcolor)
            ax5.set_xlabel("$Axis\ Ratio$", fontproperties=font)
            fig.delaxes(ax6)
        plt.draw()
        if logscale:
            if param_kw is None:
                plt.savefig(os.path.join(savedir,'Clump_histograms_log.pdf'), bbox_inches='tight')
            else:
                plt.savefig(os.path.join(savedir,'Clump_histograms_log{}.pdf'.format(param_kw)), bbox_inches='tight')
        else:
            if param_kw is None:
                plt.savefig(os.path.join(savedir,'Clump_histograms.pdf'), bbox_inches='tight')
            else:
                plt.savefig(os.path.join(savedir,'Clump_histograms_{}.pdf'.format(param_kw)), bbox_inches='tight')
        ### Scatter plots
        ######################################################
#        spfnt = 14
        tfont = FontProperties()
        tfont.set_family('serif')
        tfont.set_size(12)
        ifont = font.copy()
        ifont.set_size(12)
        rcut = np.arange(0.25,1.4,0.1)
        icut = 1-rcut/1.25
        ic2 = np.arange(0,0.9,0.1)
        dc2 = 3.1*(1-ic2/0.85)
#        (3.0*(1-ied/0.7))
        if 'eff' in param_kw:
            fig2, ((ax1a, ax2a, ax3a), (ax4a, ax5a, ax6a)) = plt.subplots(2,3,figsize=(14,8))
        else:
            fig2, ((ax1a, ax2a), (ax3a, ax4a), (ax5a, ax6a), (ax7a, ax8a), (ax9a, ax10a)) = plt.subplots(5,2,figsize=(10,15))
#        fig2.suptitle('Scatterplots of Clump Parameters',fontsize=24)
        fig2.subplots_adjust(top = 0.9, wspace = 0.3, hspace = 0.3)
#        ax1a.set_title('Radius vs Intensity')
#        ax1a.plot(re,Ie,'k.')
        if logscale:
            ax1a.plot(re,Ie,'k.')
        else:
            ax1a.plot(re[re<rsc*np.median(re)]*rscl,Ie[re<rsc*np.median(re)]*iscl,'k.')
            ax1a.set_xlim(rlim)
            ax1a.set_ylim(ilim)
        ax1a.plot(rcut,icut, 'k--', linewidth = 1)
        ax1a.set_xlabel("r$_{\mathrm{c}}$ (kpc)", fontproperties=font)
        ax1a.set_ylabel(r"i$_{\mathrm{c}}$ (ergs cm$^{-2}$ s$^{-1}$ $\AA^{-1}$ arcsec$^{-2}$$\times$10$^{-19})$", fontproperties=ifont)
#        ax1a.set_ylabel("$Intensity\ (counts)$")
        corr1 = sf.correlation_coeff(re,Ie)
        ax1a.text(0.68,0.88,'r = {:.3f}'.format(corr1),transform=ax1a.transAxes, fontproperties=tfont)
        ax1a.set_xticklabels(ax1a.get_xticks(), tfont)
        ax1a.set_yticklabels(ax1a.get_yticks(), tfont)
        if 'eff' in param_kw:
    #        ax4a.set_title('r_c vs Ellpticity')
    #        ax4a.plot(re,qb,'k.')        
            if logscale:
                ax2a.plot(re,qb,'k.')
            else:
                ax2a.plot(re[re<rsc*np.median(re)]*rscl,qb[re<rsc*np.median(re)],'k.')
                ax2a.set_xlim(rlim)
                ax2a.set_ylim(qlim)
            ax2a.set_xlabel("r$_{\mathrm{c}}$ (kpc)", fontproperties=font)
            ax2a.set_ylabel("q$_{\mathrm{c}}$ (axis ratio)", fontproperties=font)
            corr2 = sf.correlation_coeff(re,qb)
            ax2a.text(0.68,0.88,'r = {:.3f}'.format(corr2),transform=ax2a.transAxes, fontproperties=tfont)
            ax2a.set_xticklabels(ax2a.get_xticks(), tfont)
            ax2a.set_yticklabels(ax2a.get_yticks(), tfont)
    #        ax3a.set_title('Intensity vs Ellipticity')
            ax3a.plot(Ie*iscl,qb,'k.')
            ax3a.set_xlim(ilim)
            ax3a.set_ylim(qlim)
#            ax3a.set_xlabel("$Intensity\ (counts)$")
            ax3a.set_xlabel(r"i$_{\mathrm{c}}$ (ergs cm$^{-2}$ s$^{-1}$ $\AA^{-1}$ arcsec$^{-2}$$\times$10$^{-19})$", fontproperties=ifont)
            ax3a.set_ylabel("q$_{\mathrm{c}}$ (axis ratio)", fontproperties=font)
            corr3 = sf.correlation_coeff(Ie,qb)
            ax3a.text(0.68,0.88,'r = {:.3f}'.format(corr3),transform=ax3a.transAxes, fontproperties=tfont)
            ax3a.set_xticklabels(ax3a.get_xticks(), tfont)
            ax3a.set_yticklabels(ax3a.get_yticks(), tfont)
    #        ax7a.set_title('r_c vs Distance from Centroid')
            ax4a.plot(re[re<rsc*np.median(re)],rsep[re<rsc*np.median(re)],'k.')
    #        ax7a.plot(re,rsep,'k.')        
            if logscale:
                ax4a.plot(re,rsep,'k.')
            else:
                ax4a.plot(re[re<rsc*np.median(re)]*rscl,rsep[re<rsc*np.median(re)],'k.')
                ax4a.set_xlim(rlim)
                ax4a.set_ylim(dlim)
            ax4a.set_xlabel("r$_{\mathrm{c}}$ (kpc)", fontproperties=font)
            ax4a.set_ylabel("d$_{\mathrm{c}}$ (kpc)", fontproperties=font)
            corr4 = sf.correlation_coeff(re,rsep)
            ax4a.text(0.68,0.88,'r = {:.3f}'.format(corr4),transform=ax4a.transAxes, fontproperties=tfont)
            ax4a.set_xticklabels(ax4a.get_xticks(), tfont)
            ax4a.set_yticklabels(ax4a.get_yticks(), tfont)
    #        ax8a.set_title('Intensity vs Distance from Centroid')
            ax5a.plot(Ie*iscl,rsep,'k.')
            ax5a.plot(ic2,dc2, 'k--', linewidth = 1)
            ax5a.set_xlim(ilim)
            ax5a.set_ylim(dlim)
#            ax5a.set_xlabel("$Intensity\ (counts)$")
            ax5a.set_xlabel(r"i$_{\mathrm{c}}$ (ergs cm$^{-2}$ s$^{-1}$ $\AA^{-1}$ arcsec$^{-2}$$\times$10$^{-19})$", fontproperties=ifont)
            ax5a.set_ylabel("d$_{\mathrm{c}}$ (kpc)", fontproperties=font)
            corr5 = sf.correlation_coeff(Ie,rsep)
            ax5a.text(0.68,0.88,'r = {:.3f}'.format(corr5),transform=ax5a.transAxes, fontproperties=tfont)
            ax5a.set_xticklabels(ax5a.get_xticks(), tfont)
            ax5a.set_yticklabels(ax5a.get_yticks(), tfont)
    #        ax10a.set_title('Ellipticity vs Distance from Centroid')
            ax6a.plot(qb,rsep,'k.')
            ax6a.set_xlim(qlim)
            ax6a.set_ylim(dlim)
            corr6 = sf.correlation_coeff(qb,rsep)
            ax6a.text(0.68,0.88,'r = {:.3f}'.format(corr6),transform=ax6a.transAxes, fontproperties=tfont)
            ax6a.set_xticklabels(ax6a.get_xticks(), tfont)
            ax6a.set_yticklabels(ax6a.get_yticks(), tfont)
            ax6a.set_xlabel("q$_{\mathrm{c}}$ (axis ratio)", fontproperties=font)
            ax6a.set_ylabel("d$_{\mathrm{c}}$ (kpc)", fontproperties=font)        
        else:
    #        ax2a.set_title('r_c vs Sersic Index')
    #        ax2a.plot(re,nsers,'k.')
            if logscale:        
                ax2a.plot(re,nsers,'k.')
            else:
                ax2a.plot(re[re<rsc*np.median(re)],nsers[re<rsc*np.median(re)],'k.')
            ax2a.set_xlabel("r$_{\mathrm{c}}$ (kpc)")
            ax2a.set_ylabel("$Sersic\ Index$")
            corr2 = sf.correlation_coeff(re,nsers)
            ax2a.text(0.68,0.88,'r = {:.3f}'.format(corr2),transform=ax2a.transAxes, fontproperties=tfont)
    #        ax3a.set_title('Intensity vs Sersic Index')
            ax3a.plot(Ie,nsers,'k.')
#            ax3a.set_xlabel("$Intensity\ (counts)$")
            ax3a.set_xlabel("$i_c\ (ergs/cm^{2}/s/\AA/arcsec^{2}\ *\ 10^{-19})$")
            ax3a.set_ylabel("$Sersic\ Index$")
            corr3 = sf.correlation_coeff(Ie,nsers)
            ax3a.text(0.68,0.88,'r = {:.3f}'.format(corr3),transform=ax3a.transAxes, fontproperties=tfont)
    #        ax4a.set_title('r_c vs Ellpticity')
    #        ax4a.plot(re,qb,'k.')        
            if logscale:
                ax4a.plot(re,qb,'k.')
            else:
                ax4a.plot(re[re<rsc*np.median(re)]*rscl,qb[re<rsc*np.median(re)],'k.')
            ax4a.set_xlabel("r$_{\mathrm{c}}$ (kpc)")
            ax4a.set_ylabel("q$_{\mathrm{c}}$ (axis ratio)")
            corr4 = sf.correlation_coeff(re,qb)
            ax4a.text(0.68,0.88,'r = {:.3f}'.format(corr4),transform=ax4a.transAxes, fontproperties=tfont)
    #        ax5a.set_title('Intensity vs Ellipticity')
            ax5a.plot(Ie,qb,'k.')
            ax5a.set_xlabel("$i_c\ (ergs/cm^{2}/s/\AA/arcsec^{2}\ *\ 10^{-19})$")
#            ax5a.set_xlabel("$Intensity\ (counts)$")
            ax5a.set_ylabel("q$_{\mathrm{c}}$ (axis ratio)")
            corr5 = sf.correlation_coeff(Ie,qb)
            ax5a.text(0.68,0.88,'r = {:.3f}'.format(corr5),transform=ax5a.transAxes, fontproperties=tfont)
    #        ax6a.set_title('Sersic Index vs Ellipticity')
            ax6a.plot(nsers,qb,'k.')
            corr6 = sf.correlation_coeff(nsers,qb)
            ax6a.text(0.68,0.88,'r = {:.3f}'.format(corr6),transform=ax6a.transAxes, fontproperties=tfont)
            ax6a.set_xlabel("$Sersic\ Index$")
            ax6a.set_ylabel("q$_{\mathrm{c}}$ (axis ratio)")
    #        ax7a.set_title('r_c vs Distance from Centroid')
            ax7a.plot(re[re<rsc*np.median(re)],rsep[re<rsc*np.median(re)],'k.')
    #        ax7a.plot(re,rsep,'k.')        
            if logscale:
                ax7a.plot(re,rsep,'k.')
            else:
                ax7a.plot(re[re<rsc*np.median(re)],rsep[re<rsc*np.median(re)],'k.')           
            ax7a.set_xlabel("r$_{\mathrm{c}}$ (kpc)")
            ax7a.set_ylabel("$Distance\ (kpc)$")
            corr7 = sf.correlation_coeff(re,rsep)
            ax7a.text(0.68,0.88,'r = {:.3f}'.format(corr7),transform=ax7a.transAxes, fontproperties=tfont)
    #        ax8a.set_title('Intensity vs Distance from Centroid')
            ax8a.plot(Ie,rsep,'k.')
#            ax8a.set_xlabel("$Intensity\ (counts)$")
            ax8a.set_xlabel("$i_c\ (ergs/cm^{2}/s/\AA/arcsec^{2}\ *\ 10^{-19})$")
            ax8a.set_ylabel("d$_{\mathrm{c}}$ (kpc)")
            corr8 = sf.correlation_coeff(Ie,rsep)
            ax8a.text(0.68,0.88,'r = {:.3f}'.format(corr8),transform=ax8a.transAxes, fontproperties=tfont)
    #        ax9a.set_title('Sersic Index vs Distance from Centroid')
            ax9a.plot(nsers,rsep,'k.')
            corr9 = sf.correlation_coeff(nsers,rsep)
            ax9a.text(0.68,0.88,'r = {:.3f}'.format(corr9),transform=ax9a.transAxes, fontproperties=tfont)
            ax9a.set_xlabel("$Sersic\ Index$")
            ax9a.set_ylabel("d$_{\mathrm{c}}$ (kpc)")
    #        ax10a.set_title('Ellipticity vs Distance from Centroid')
            ax10a.plot(qb,rsep,'k.')
            corr10 = sf.correlation_coeff(qb,rsep)
            ax10a.text(0.68,0.88,'r = {:.3f}'.format(corr10),transform=ax10a.transAxes, fontproperties=tfont)
            ax10a.set_xlabel("q$_{\mathrm{c}}$ (axis ratio)")
            ax10a.set_ylabel("d$_{\mathrm{c}}$ (kpc)")
        if logscale:
            if param_kw is None:
                plt.savefig(os.path.join(savedir,'Clump_scatterplots_log.pdf'), bbox_inches='tight')
            else:
                plt.savefig(os.path.join(savedir,'Clump_scatterplots_log{}.pdf'.format(param_kw)), bbox_inches='tight')
        else:
            if param_kw is None:
                plt.savefig(os.path.join(savedir,'Clump_scatterplots.pdf'), bbox_inches='tight')
            else:
                plt.savefig(os.path.join(savedir,'Clump_scatterplots_{}.pdf'.format(param_kw)), bbox_inches='tight')
        plt.ion()
        plt.show() 
    if return_lum:
        return lums
    else:
        return nblobs, re, Ie, rsep, nsers, qb, PAb, QG, PAG, rpx
    
def trial_lfrac(xarr,n,sig):
    """ Test empirical function for luminosity fraction.
        This is pretty arbitrary at this point
    """
    try:
        junk = len(xarr)
    except:
        xarr = np.array(([xarr]))
    if n == 1:
        return xarr*(xarr==1)
    else:
        ### start with normalized gaussian
        ret = sf.gaussian(xarr,sig*n,center=1/n,height=1/(np.sqrt(2*np.pi)*n*sig))
        ### cut out values outside valid interval [0,1]        
        xtrue = (xarr>=0)*(xarr<=1)
        ret *= xtrue
#        ret /= np.sum(ret)
        ### Apply exponential decay envelope
        ret *= np.exp(-xarr*(n**1.5))
        return ret
        
def trial_nblobs(xarr):
    try:
        junk = len(xarr)
    except:
        xarr = np.array(([xarr]))
    ret = -(1.0/(100**2))*xarr**5 + xarr
    ### cut out values outside valid interval [0,1]        
    xtrue = (xarr>=0)*(xarr<=10)
    ret *= xtrue
#    ret /= np.sum(ret)
    return ret
        
def trial_rad(xarr):
    try:
        junk = len(xarr)
    except:
        xarr = np.array(([xarr]))
    c4 = [0.01, 0, -1, 3, 5.5]
    ret = np.poly1d(c4)(xarr)
    ### cut out values outside valid interval [0,1]        
    xtrue = (xarr>=0)*(xarr<=10)
    ret *= xtrue
#    ret /= np.sum(ret)
    return ret   
      
def trial_rrel(xarr):
    """ All in pixel space for now - translate to kpc later?
    """
    try:
        junk = len(xarr)
    except:
        xarr = np.array(([xarr]))
    c1 = [-56/52, 56]
    ret = np.poly1d(c1)(xarr)
    xtrue = (xarr>=0)*(xarr<=52)
    ret *= xtrue
    return ret
      
def lfrac_nblobs():
    """ 
    """
    blobs = sf.pdf_draw(trial_nblobs,n=1,args=None,int_lims=[0,10],res=1000)
    blobs = int(np.ceil(blobs))
    lfrac = np.zeros((blobs,2))
    ltot = 1
    for i in range(blobs-1):
        li = sf.pdf_draw(trial_lfrac,n=1,args=(blobs-i,0.1),int_lims=[0,1],res=1e3)
        li = li*ltot
        ltot -= li
        lfrac[i,0] = li
    lfrac[-1,0] = 1-np.sum(lfrac)
    for j in range(blobs):
        lfrac[j,1] = sf.pdf_draw(trial_rad,int_lims=[0,6])
    return lfrac

def get_rots(x, y, z, th1, th2, th3, inv=False):
    if not inv:
        xr = x*np.cos(th2)*np.cos(th3) + y*(np.cos(th3)*np.sin(th1)*np.sin(th2) - np.cos(th1)*np.sin(th3)) + z*(np.cos(th3)*np.cos(th1)*np.sin(th2) + np.sin(th3)*np.sin(th1))
        yr = x*np.sin(th3)*np.cos(th2) + y*(np.cos(th1)*np.cos(th3) + np.sin(th1)*np.sin(th2)*np.sin(th3)) + z*(np.cos(th1)*np.sin(th2)*np.sin(th3)-np.cos(th3)*np.sin(th1))
        zr = -x*np.sin(th2) + y*np.cos(th2)*np.sin(th1) + z*np.cos(th1)*np.cos(th2)
    else:
        Rxi = np.array(([1, 0 , 0], [0, np.cos(th1), np.sin(th1)], [0, -np.sin(th1), np.cos(th1)]))
        Ryi = np.array(([np.cos(th2), 0, -np.sin(th2)], [0, 1, 0], [np.sin(th2), 0, np.cos(th2)]))
        Rzi = np.array(([np.cos(th3), np.sin(th3), 0], [-np.sin(th3), np.cos(th3), 0], [0, 0, 1]))
        Ri = np.dot(Rxi,np.dot(Ryi,Rzi))
        xr = x*Ri[0,0] + y*Ri[0,1] + z*Ri[0,2]
        yr = x*Ri[1,0] + y*Ri[1,1] + z*Ri[1,2]
        zr = x*Ri[2,0] + y*Ri[2,1] + z*Ri[2,2]
#    print th1, th2, th3
#    print x, xr
#    print y, yr
#    print z, zr
    return xr, yr, zr
    
def make_sim_gal(clump_object_list, gal_object_list=None, dim=100, profile='sersic', return_noise=False, nclumps=None, save=False, save_idx=0, plot_results=False):
    """ Generates one sample galaxy with PDF form and params given
        in a saved value (load from *_object_list)
        Some alternatives are hardcoded for now (mostly q, PA, nblobs)
    """
#    bg = 20 ### Add arbitrary background if desired
#    gal_img = np.ones((dim,dim))*bg
    gal_img = np.zeros((dim,dim))
#    snr = 100
#    ltot = snr*bg*dim**2
    xc = int(dim/2)
    yc = int(dim/2)
    ### first pick nblobs from poisson distributions
#    lam = 5.0588#params['p_lam']
#    nclumps = np.random.poisson(lam)
        ### make sure there is at least one clump
#    while nclumps == 0:
#        nclumps = np.random.poisson(lam)
    ### change - pick nclumps from uniform dist on [1, 9]
    if nclumps is None:
        nclumps = int((10*np.random.random()))+1
#    nclumps = 30
#    plot_results=True
    rsarr = np.zeros(nclumps)
    rbarr = np.zeros(nclumps)
    rpxarr = np.zeros(nclumps)
    qbarr = np.zeros(nclumps)
    iearr = np.zeros(nclumps)
    ycarr = np.zeros(nclumps)
    xcarr = np.zeros(nclumps)
    paarr = np.zeros(nclumps)
#    print nclumps, " Clumps"
    zsim = 2.6
    arcsec_per_px = 0.01
#    qgal = sf.sine_draw()
    ### for now, qgal looks uniform for eff...
    QGE = clump_object_list[0]
    if profile == 'eff':
#        qgal = 0.8*np.random.rand()+0.1
#        qgal = 0.15*np.random.randn() + 0.5
#        while qgal <= 0 or qgal > 1:
#            qgal = 0.15*np.random.randn() + 0.5
#        qgal = 1
        qg_pars = QGE.paramarray
#        qg_pars[1] /= 2
        qgal = QGE.draw(params=qg_pars, lims=[0.1, np.max(QGE.data)])
        thgal = 2*np.pi*np.random.rand()
    ### Then assign each clump other qualities and add to galaxy
    if profile == 'sersic':
        rsep = clump_object_list[0] ### Need to find dynamically later
        lirn = clump_object_list[1] #logie, logre, nsers
        lp = lirn.params
#        nd_args = (lp['c0'], lp['s0'], lp['c1'], lp['s1'], lp['c2'], lp['s2'], lp['gam0'])
        lv = lirn.data
        lims = np.array(([np.min(lv[0]), np.max(lv[0])],[np.min(lv[1]), np.max(lv[1])],[np.min(lv[2]), np.max(lv[2])]))
    #    print lims
        grid, spacing = sf.pdf_nd_grid(sf.cauchy_nd_lmfit, lp, lims, mx_pnts = 10**6, bpd=1, save=False)
        for i in range(nclumps):
#        q = 0.2*np.random.randn()+0.7
            qc = sf.sine_draw()
            ### For now, enforce minimum ellipticity
            while qc < 0.1:
                qc = sf.sine_draw()
            PA = np.pi*np.random.rand()
            rc = np.random.randn()*rsep.params['s0'] + rsep.params['c0']
            while rc < 0 or rc > np.max(rsep.data):
                rc = np.random.randn()*rsep.params['s0'] + rsep.params['c0']
            zsim = 2.5
            
            lirn_vals = sf.pdf_draw_nd(grid, lims, spacing, 3)
            logie, logre, nsers = lirn_vals[0], lirn_vals[1], lirn_vals[2]
            ie = 10**logie
            ie = np.random.exponential(0.2)
            re = 10**logre
            rc /= 0.08
            re /= 0.08
            thpr = 2*np.pi*np.random.rand()
            xcb = rc/np.sqrt(qgal)*np.cos(thpr)
            ycb = rc*np.sqrt(qgal)*np.sin(thpr)
    elif profile == 'eff':
        QBE = clump_object_list[1]
        indep = (len(clump_object_list) == 5)
        if indep:
            DCE = clump_object_list[2]
            ICE = clump_object_list[3]
            RCE = clump_object_list[4]
        elif len(clump_object_list) ==4:
            RCE = clump_object_list[2]
            IDJ = clump_object_list[3]
        else:
            IRD = clump_object_list[2]
        rcc = np.zeros(nclumps)
        xycc = np.zeros((2,nclumps))
        for i in range(nclumps):
            ### draw parameters
            mf = 1
            qbd = QBE.draw(params=QBE.paramarray, lims=[np.min(QBE.data), np.max(QBE.data)])#lims = [0, 1])
            if indep:
    #            PA = np.pi*np.random.rand()
    #            if dcd > 3.8:
    #                dcd = DCE.draw(params=DCE.paramarray, lims=[np.min(DCE.data), np.max(DCE.data)])
                icd_ok = False
                while not icd_ok:
                    icd = ICE.draw(params=ICE.paramarray, lims=[2*np.min(ICE.data), np.max(ICE.data)])
                    rcd = RCE.draw(params=RCE.paramarray, lims=[np.min(RCE.data), np.max(RCE.data)*mf])
                    dcd = DCE.draw(params=DCE.paramarray, lims=[np.min(DCE.data), np.max(DCE.data)])
    #                icd_ok = True
                    ### Manually exclude high I and R combo...
                    icdr_ok = (icd < (1-rcd/1.25))
                    ### Manually exclude high I and D combo...
#                    icdd_ok = (dcd < (1/(icd+0.5)**2-0.6))
                    icdd_ok = (dcd < (3.1*(1-icd/0.7)))
                    icd_ok = icdr_ok*icdd_ok
    #                icd = ICE.draw(params=ICE.paramarray, lims=[np.min(ICE.data), np.max(ICE.data)])
    #                rcd = RCE.draw(params=RCE.paramarray, lims=[np.min(RCE.data), np.max(RCE.data)*mf])
                ### Convert units and apply corrections if needed
                rc = px_to_pc(dcd*1000, arcsec_per_px, zsim, inv=True)
                rc /= 1.2 ### Empirical correction
                ref = px_to_pc(rcd*1000, arcsec_per_px, zsim, inv=True)
                ied = icd
            elif len(clump_object_list) ==4:
                mf = 0.75
                rcd = RCE.draw(params=RCE.paramarray, lims=[np.min(RCE.data), np.max(RCE.data)*mf])
                def draw_idj(IDJ):
                    ie = IDJ.data[0]
                    dc = IDJ.data[1]
                    mni, mnd, sci, scd, th = IDJ.paramarray
                    irot = (ie-mni)*np.cos(th) - (dc-mnd)*np.sin(th)
                    drot = (dc-mnd)*np.cos(th) + (ie-mni)*np.sin(th)
                    ird = IDJ.draw[0](params=np.array(([0,sci])),lims=[np.min(irot),np.max(irot)])
                    drd = IDJ.draw[1](params=np.array(([0,scd])),lims=[np.min(drot),np.max(drot)])
                    ied, dcd, jnk = get_rots(ird, drd, ird, th, 0, 0, inv=True)
                    ied += mni
                    dcd += mnd
                    dcf = px_to_pc(dcd*1000, arcsec_per_px, zsim, inv=True)
#                    icdr_ok = (ied < (1-red/1.25))
#                    icdd_ok = (dcd < (1/(icd+0.5)**2-0.6))
                    icdd_ok = (dcd < (3.0*(1-ied/0.7)))
#                    icd_ok = icdr_ok*icdd_ok
#                    if not icdd_ok:
#                        ied, dcf = draw_idj(IDJ)
                    return ied, dcf
                ied, dcf = draw_idj(IDJ)
                while ied < 0 or dcf < 0:
                    ied, dcf = draw_idj(IDJ)
                ref = px_to_pc(rcd*1000, arcsec_per_px, zsim, inv=True)
                rc = dcf
            else:
                def draw_ird(IRD):
                    re = IRD.data[0]
                    ie = IRD.data[1]
                    dc = IRD.data[2]
                    th1, th2, th3 = IRD.paramarray[6:]
                    rpars = np.array(([IRD.paramarray[0],IRD.paramarray[3]]))
                    ipars = np.array(([IRD.paramarray[1],IRD.paramarray[4]]))
                    dpars = np.array(([IRD.paramarray[2],IRD.paramarray[5]]))
                    rrpars = np.array(([0,rpars[1]]))
                    ripars = np.array(([0,ipars[1]]))
                    rdpars = np.array(([0,dpars[1]]))
                    ### Draw from "independent" PDFs in rotated frame, then rotate back
                    rrot, irot, drot = get_rots(re-rpars[0],ie-ipars[0],dc-dpars[0],th1,th2,th3)
                    mf = 0.75 #scales up/down the max allowable radius
                    rrd = IRD.draw[0](params=rrpars,lims=[np.min(rrot), np.max(rrot)*mf])
                    ird = IRD.draw[1](params=ripars,lims=[np.min(irot), np.max(irot)])
                    drd = IRD.draw[2](params=rdpars,lims=[np.min(drot), np.max(drot)])
                    red, ied, dcd = get_rots(rrd, ird, drd, th1, th2, th3, inv=True)
                    red += rpars[0]
                    ied += ipars[0]
                    dcd += dpars[0]
    #                red *= (4.0/5) ### Empirical correction, not sure why this happens...
                    ref = px_to_pc(red*1000, arcsec_per_px, zsim, inv=True)
                    dcf = px_to_pc(dcd*1000, arcsec_per_px, zsim, inv=True)
    #                ref = red
                    ### Enforce a minimum on axis ratio...
    #                if qbd < 0.3:
    #                    draw_irq(IRQ)
                    icdr_ok = (ied < (1-red/1.25))
#                    icdd_ok = (dcd < (1/(icd+0.5)**2-0.6))
                    icdd_ok = (dcd < (3.0*(1-ied/0.7)))
                    icd_ok = icdr_ok*icdd_ok
                    if not icd_ok:
                        ref, ied, dcf = draw_ird(IRD)
                    return ref, ied, dcf
                ref, ied, dcf = draw_ird(IRD)
                while ied < 0 or ref < 0 or dcf < 0:
                    print "re-drawing"
                    ref, ied, dcf = draw_ird(IRD)
                rc = dcf #/1.2
            thpr = np.pi*np.random.rand()
            xcb = rc/np.sqrt(qgal)*np.cos(thpr)
            ycb = rc*np.sqrt(qgal)*np.sin(thpr)
            ### Rotate to galaxy position angle thgal
            xcf = xcb*np.cos(thgal) - ycb*np.sin(thgal) + xc
            ycf = ycb*np.cos(thgal) + xcb*np.sin(thgal) + yc
            rcc[i] = rc
            xycc[:,i] = [xcf, ycf]
            clump_im = sf.eff2d(np.arange(dim), np.arange(dim), xcf, ycf, ied, ref, q=qbd, PA=thpr)
            gal_img += clump_im
#            rsarr[i] = rcpc
            rsarr[i] = px_to_pc(rc, arcsec_per_px, zsim)/1000
            rbarr[i] = px_to_pc(ref, arcsec_per_px, zsim)/1000#ref
            rpxarr[i] = ref
            qbarr[i] = qbd
            iearr[i] = ied
            ycarr[i] = ycf
            xcarr[i] = xcf
            paarr[i] = thpr
        ### Manually check various distributions
        if plot_results:
#            print "q clump"
#            drng = np.linspace(QBE.data.min(),QBE.data.max(),100)
#            dpdf = QBE.dist(QBE.paramarray, drng)
#            plt.plot(drng,dpdf,'k',linewidth=2)
#            plt.hist(qbarr, bins=14, normed=True)
#            plt.show()
#            plt.close()
            print "ie vs dc"
            drng = np.linspace(0,0.8,100)
            dy = 1/(drng+0.5)**2-0.6
            plt.plot(drng,dy,'k',linewidth=2)
#            plt.hist(qbarr, bins=14, normed=True)
            plt.plot(iearr,rsarr,'b.')
            plt.show()
            plt.close()
#            print "I, r, d"
#            plot_1d_projections(IRD,alt_data=np.vstack((rbarr, iearr, rsarr)), use_array=True)
#            plt.imshow(gal_img, interpolation='none')
#            plt.show()
#            plt.close()
    ### Option to apply poisson noise on user set scale
#    gal_cpy = 1.0*gal_img
#    gal_img = np.random.poisson(gal_img*1000)/1000
#    noise_p = gal_cpy-gal_img
#    gal_img += np.random.poisson(bg*np.ones((gal_img.shape)))
#    gal_img /= snr
    ### These error parameters are estimated
    mu_err = 0.16*np.max(gal_img)
    ### incompleteness estimate
#    bg_cut = 0.1*np.max(gal_img)
    bg_cut = np.random.randn()*0.002 + 0.006
    while bg_cut < 0.002 or bg_cut > 0.01:
        bg_cut = np.random.randn()*0.002 + 0.006
    print bg_cut
#    print "% complete =", len(iearr[iearr>bg_cut])/len(iearr)
    farr = eff_flux(np.sqrt(1/np.pi),iearr, rbarr)
    clumps_detected = len(farr[farr>1.5*bg_cut])
    print "clumps detected =", clumps_detected, "/", nclumps
    mu_err = 0.05*np.max(gal_img)
    mu_err = 0.1*np.max(gal_img)
#    print mu_err/np.max(gal_img)
    std_err = mu_err/8
#    print std_err
    std_err = np.random.randn()*bg_cut
    noise = std_err*np.random.randn(gal_img.size)# + mu_err/3
    noise = np.reshape(noise,gal_img.shape)
#    print np.max(noise)/np.max(gal_img)#, np.min(noise), np.max(gal_img), np.mean(gal_img)
    ### Recreate entire galaxy if it is too dim to actually be detected (SNR <= 2)
    if clumps_detected < 1:
        print "re-drawing"
        gal_img, rcc, noise = make_sim_gal(clump_object_list, gal_object_list=gal_object_list, dim=dim, profile=profile, return_noise=return_noise, nclumps=nclumps, save=False, save_idx=save_idx, plot_results=plot_results)
    else:
        gal_img += noise
    if save:    
        hdu0 = pyfits.PrimaryHDU(gal_img)
        hdu1 = pyfits.PrimaryHDU(noise)
        hdu2 = pyfits.PrimaryHDU(np.vstack((xcarr,ycarr,iearr,rbarr,paarr,qbarr,rpxarr)))
        ### Additional new header values
        hdu0.header.append(('UNITS','Counts','Relative photon counts (no flat fielding)'))
        hdu0.header.append(('MODEL',profile,'clump model used to generate galaxy'))
        hdu1.header.append(('UNITS','Noise/Error','standard deviation'))
        hdulist = pyfits.HDUList([hdu0])
        hdulist.append(hdu1)
        hdulist.append(hdu2)
        if profile is not None:
            param_kw = '_' + profile
        hdulist.writeto('/home/matt/software/matttest/results/sim_gxy_{:03d}{}.fits'.format(save_idx,param_kw),clobber=True)    
    
    if return_noise:
        return gal_img, rcc, noise#+noise_p
    else:
        return gal_img, rcc, xycc
    
'''
def make_blob_gal(params,dim=100,profile='gaussian'):
    """ Generates one sample galaxy with PDF params given in parmams
        Params can be a simple dict, lmfit object, or whatever is convenient
        To start use the following distributions
          nblobs - poisson
          fwhm - gaussian
          rsep - weibull
          sbmx - exponential
          Q tensor - gaussian
          q - gaussian
    """
    bg = 3 ### Add arbitrary background if desired
    gal_img = np.ones((dim,dim))*bg
    snr = 500
#    ltot = snr*bg*dim**2
    xc = int(dim/2)
    yc = int(dim/2)
    ### first pick nblobs from poisson distributions
    lam = params['p_lam']
    nblobs = np.random.poisson(lam)
    ### make sure there is at least one blob
    while nblobs == 0:
        nblobs = np.random.poisson(lam)
    ### Then assign each blob other qualities and add to galaxy
#    Qxx = np.random.randn()*130
#    Qyy = np.random.randn()*180
#    Qxy = np.random.randn()*130
    Q = 0.15*np.random.randn()+0.3
    for i in range(nblobs):
        q = 0.2*np.random.randn()+0.7
        if q <= 0:
            q = 0.1
        if q > 1:
            q = 1
        PA = np.pi*np.random.rand()
        sbmxb = snr*np.random.exponential(params['e_tau'])
        fwhmb = abs(params['g_sig']*np.random.randn()+params['g_mu'])
        rposb = params['w_lam']*np.random.weibull(params['w_k'])
#        args = (rposb,Qxx,Qyy,Qxy)
#        xv = sf.pdf_draw(sf.random_quadrupole_2d,n=1,args=args,int_lims=[-rposb,rposb],res=1e3)
        xv1 = rposb*np.random.randn()/3
        if abs(xv1) > rposb:
            xv1 /= rposb/xv1
        xv = xv1
        yv = np.sqrt(rposb**2-xv**2)*2.0*((np.random.rand()>0.5)-0.5)
        xcb = xv+xc
        ycb = yv+yc
        sig = fwhmb/2.35
#        print xcb, ycb
#        print sig
#        print sbmxb
        if profile == 'gaussian':
            blob_im = sbmxb*sf.gauss2d(np.arange(dim),np.arange(dim),sig,sig,xcenter=xcb,ycenter=ycb,q=q,PA=PA,unity_height=True)
        elif profile == 'sersic':
            blob_im = sf.sersic2d(np.arange(dim),np.arange(dim),xcb,ycb,sbmxb,fwhmb,1,q=q,PA=PA)
        else:
            print("Invalid blob profile. Choose one of the following:")
            print("  gaussian")
            print("  sersic")
#        plt.imshow(blob_im)
#        plt.show()
        gal_img += blob_im
    gal_img = np.random.poisson(gal_img)
#    gal_img /= snr
    return gal_img
#'''
           
def test_sim_dists(params1,params2,dim=100,ndist=200,stat='KS'):
    """ Compares two distributions with params1 and params2.  Aggregates ndist
        number of galaxies.  Final statistic returned is a keyword input,
        'KS' is default
    """
    def params_to_gals(params,ndist,dim):
        r_arr = np.zeros((dim,dim*ndist))
        img = np.zeros((dim,dim*ndist))
        for i in range(ndist):
            gal = make_blob_gal(params1)
            xcg, ycg = sf.centroid(gal)
            rs = sf.make_rarr(np.arange(dim),np.arange(dim),xcg,ycg)
#            print r_arr.shape
#            print dim*i, dim*(i+1)
            r_arr[:,dim*i:(dim*(i+1))] = rs
            img[:,dim*i:(dim*(i+1))] = gal
        rinds = np.argsort(np.ravel(r_arr))
        r = np.ravel(r_arr)[rinds]
        img = np.ravel(img)[rinds]
        return r, img
    r1, img1 = params_to_gals(params1,ndist,dim)
    r2, img2 = params_to_gals(params2,ndist,dim)
    if stat == 'KS':
        im1 = np.cumsum(img1)/np.sum(img1)
        im2 = np.cumsum(img2)/np.sum(img2)
        plt.plot(r1,im1,r2,im2)
        plt.show()
        im1b, im1e = np.histogram(img1,bins=10000)
        im2b, im2e = np.histogram(img2,bins=10000)
        im1b = np.cumsum(im1b)/np.sum(im1b)
        im2b = np.cumsum(im2b)/np.sum(im2b)
#        plt.close()
#        plt.plot(im1b)
#        plt.plot(im2b)
#        plt.show()
        stat_val = sf.kolmogorov(im1b,im2b)
    else:
        print("Statistic not available, choose 'KS'")
        exit(0)
    return stat_val
    
def find_abmag(cnts, scl=16, flam=1.1845489E-19, plam=5887.4326):
    """ Good for Hubble ACS, see ACS Handbook pg 132.
        Defaults are for conversion of sources, scl = 1 for lens
    """
    zpt = -2.5*np.log10(flam)-21.10-5*np.log10(plam) + 18.6921
    return -2.5*np.log10(cnts/scl) + zpt
    
def convert_sersic(Io,Ro,n,direction='m2y'):
    """ Converts between Matt and Yiping parameters.  Direction goes as such:
        m2y - Matt's to Yiping's
        y2m - Yiping's to Matt's
    """
    if n >= 0.36: # from Ciotti & Bertin 1999, truncated to n^-3
        k=2.0*n-1./3+4./(405.*n)+46./(25515.*n**2.)+131./(1148175.*n**3.)
    else: # from MacArthur et al. 2003
        k=0.01945-0.8902*n+10.95*n**2.-19.67*n**3.+13.43*n**4.
    bn = 0.868*n-0.142 ## from Caon et al. 1993
    if direction == 'm2y':
        If = Io*10**(bn)
        Rf = Ro*(bn*np.log(10)/k)**(-n)
        return If, Rf, n
    elif direction == 'y2m':
        If = Io*10**(-bn)
        Rf = Ro*(bn*np.log(10)/k)**(n)
        return If, Rf, n
    else:
        print("direction must be either 'm2y' or 'y2m'")
        exit(0)
        
def px_to_pc(rpx,arcsec_per_px,z, Om=0.274, Ol=0.726, h=0.7, inv=False):
    """ Finds the parsec equivalent distance of one pixel for an image at
        a given z and a given cosmology.
        If you already have arsecs, then put rpx = rarcsec
        and set arcsec_per_px = 1
        If inv, then input rpx is assumed to be r in pc
    """
    if not inv:
        reff = rpx*arcsec_per_px*1/3600*np.pi/180
        return sf.angular_dia_dist(z,Om=Om,Ol=Ol, h=h)*reff
    else:
        reff = rpx/sf.angular_dia_dist(z, Om=Om,Ol=Ol, h=h)
        return reff*180/np.pi*3600/arcsec_per_px
    
def pdf_fit(x,y,yinv,p0=None,form='gaussian',method='lmfit'):
    """ Fits pdf of x vs. y.  Form input gives pdf to fit (and adjusts return
        values). Custom is allowed.  Method is lmfit or emcee.
        Right now p0 lists must be very specific, see below for format
    """
    if method == 'lmfit':
        params = lmfit.Parameters()
        args = (x,y,yinv)
        if form=='gaussian':
            if p0 is None:
                xc0 = np.mean(x)
                sig0 = np.std(x)
                h0 = np.max(y)
                p0 = [xc0, sig0, h0]
            params.add('mean', value = p0[0])
            params.add('sigma', value = p0[1], min = 0)
            params.add('hght', value = p0[2], min = 0)
            results = lmfit.minimize(sf.gauss_residual,params,args=args)
            return results.params['mean'].value, results.params['sigma'].value, results.params['hght'].value
        elif form == 'weibull':
            if p0 is None:
                k = 1.5
                lam = np.median(x)
                h = np.max(y)*lam
                p0 = [k, lam, h]
            params.add('k', value = p0[0])
            params.add('lam', value = p0[1])
            params.add('h', value = p0[2])
            results = lmfit.minimize(sf.weibull_residual,params,args=args)
            return results.params['k'].value, results.params['lam'].value, results.params['h'].value
        elif form == 'exponential':
            if p0 is None:
                tau = np.median(x)
                h = np.max(y)
                xc = np.min(x)
                p0 = [tau, h, xc]
            params.add('tau', value = p0[0])
            params.add('h', value = p0[1])
            params.add('xc', value = p0[2])
            results = lmfit.minimize(sf.exponential_residual,params,args=args)
            return results.params['tau'].value, results.params['h'].value, results.params['xc'].value
        elif form == 'cauchy':
            if p0 is None:
                xc = np.median(x)
                a = 1/np.std(x)
                h = np.max(y)
                p0 = [xc, a, h]
            params.add('xc', value = p0[0])
            params.add('a', value = p0[1])
            params.add('h', value = p0[2])
            results = lmfit.minimize(sf.cauchy_residual,params,args=args)
            return results.params['xc'].value, results.params['a'].value, results.params['h'].value
        else:
            print("Not a valid form ({}).  Choose from:\n  gaussian\n  exponential\n  weibull\n cauchy".format(form))
            exit(0)
    else:
        print("Not a valid method")
        exit(0)
            
def dist_params(dist,dist_err=None,bins=10,p0=None,form='gaussian', method='lmfit', view_plot=False,plot_label=None):
    if view_plot:
        plt.figure(plot_label)
        plt.ion()
        plt.hist(dist,bins=bins)
    y, xedge = np.histogram(dist,bins=bins)
    x = np.zeros((len(y)))
    for i in range(len(x)):
        x[i] = (xedge[i]+xedge[i+1])/2
    if dist_err is None:
        yinv = np.ones((len(y)))
    params = pdf_fit(x,y,yinv,p0=p0,form=form,method=method)
    xplt = np.linspace(x[0],x[-1],200)
    if view_plot:
        kws = {'color': 'k', 'linewidth':'2'}
        if form=='weibull':
            plt.plot(xplt,params[2]*sf.weibull(xplt,params[0],params[1]),**kws)
        elif form=='gaussian':
            plt.plot(xplt,sf.gaussian(xplt,params[1], center=params[0], height=params[2]),**kws)
        elif form=='exponential':
            plt.plot(xplt,params[1]*np.exp(-(xplt-params[2])/params[0]),**kws)
        elif form == 'cauchy':
            plt.plot(xplt,params[2]*sf.cauchy(xplt,params[0],params[1]),**kws)
        plt.show()
    return params
  
def get_norm(x, params, func, index_cnt=None, return_mesh=False, max_factor=1.2, kwargs=dict()):
    ndim = x.shape[0]
    if ndim > 10: ## Assume needs to be transposed
        x = x.reshape((1,len(x)))
        ndim = 1
    mx_pnts = 10**6
    try:
        if kwargs['idx'] == 999:
            max_factor = 1.0
    except:
        pass
    shrt_pnts = 100
#    max_factor = 1.5
    pnts_per_dim = min(int((mx_pnts)**(1/(ndim))),shrt_pnts)
    axis_arrays = []
    spacing = np.zeros((ndim))
    xaxes = np.zeros((ndim,pnts_per_dim))
    for i in range(ndim):
        ### The following assumes zero truncation...
        xl = np.min(x[i])
        try:
            if kwargs['idx'] == 999:
                xl = 0.1
        except:
            pass
        xaxes[i] = np.linspace(max_factor*xl, max_factor*np.max(x[i]),pnts_per_dim)
        axis_arrays.append(xaxes[i])
        spacing[i] = xaxes[i][1] - xaxes[i][0]
    mesh = np.meshgrid(*tuple(axis_arrays),indexing='xy')
    mesh = np.asarray(mesh)
    if type(func) == list:
        vals = np.ones(mesh.shape[1:])
        for f in func:
            vals *= f(params, mesh, index_cnt)
    else:
        vals = func(params, mesh, **kwargs)
    ### Remove weird values that ruin norm
    vals[vals==np.inf] = 0
    vals[vals==np.nan] = 0
#    xl, xh = np.min(axis_arrays[0]), np.max(axis_arrays[0])
#    yl, yh = np.min(axis_arrays[1]), np.max(axis_arrays[1])
#    print xl, xh, yl, yh
#    plt.imshow(vals,extent=[xl,xh,yl,yh],origin='lower')
#    plt.show()
#    plt.close()
    norm = np.sum(vals)*np.prod(spacing)
    if return_mesh:
        return norm, mesh, spacing, xaxes
    else:
        return norm
    ### Simplistic integral
#    if vals.ndim == 1 or vals.shape[0] == 1:
#        vals_int = np.zeros((vals.size-1))
#        vals = np.resize(vals, (vals.size,))
#    else:
#        vals_int = np.zeros((vals.shape-np.ones(vals.ndim)))
#    if vals.ndim == 1:
#        for i in range(vals_int.size):
#            vals_int[i] = (vals[i] + vals[i+1])/2
#    elif vals.ndim == 2:
#        for i in range(vals_int.shape[0]):
#            for j in range(vals_int.shape[1]):
#                vals_int[i,j] = (vals[i,j] + vals[i+1,j] + vals[i,j+1] + vals[i+1,j+1])/4
#    elif vals.ndim == 3:
#        for i in range(vals_int.shape[0]):
#            for j in range(vals_int.shape[1]):
#                for k in range(vals_int.shape[2]):
#                    vals_int[i,j,k] = (vals[i,j,k] + vals[i+1,j,k] + vals[i,j+1,k] + vals[i+1,j+1,k] + vals[i,j,k+1] + vals[i+1,j,k+1] + vals[i,j+1,k+1] + vals[i+1,j+1,k+1])/8
#    else:
#        print("Can't yet handle ndim > 3")
#        exit(0)
    
#    if ndim == 1:
#        norm = np.sum(func(params, xaxes[0]))*spacing[0]
#    elif ndim == 2:
#        norm = 0
#        for i in range(len(xaxes[0])):
#            for j in range(len(xaxes[1])):
#                norm += func(params, xaxes[0][i], xaxes[1][j])
#        return norm*np.prod(spacing)
#    elif ndim == 3:
#        norm = 0
#        for i in range(len(xaxes[0])):
#            for j in range(len(xaxes[1])):
#                for k in range(len(xaxes[2])):
#                    norm += func(params, xaxes[0][i], xaxes[1][j], xaxes[2][k])
#        return norm*np.prod(spacing)
#    elif ndim == 4:
#        norm = 0
#        for i in range(len(xaxes[0])):
#            for j in range(len(xaxes[1])):
#                for k in range(len(xaxes[2])):
#                    for l in range(len(xaxes[3])):
#                        norm += func(params, xaxes[0][i], xaxes[1][j], xaxes[2][k], xaxes[3][k])
#        return norm*np.prod(spacing)
#    else:
#        print("can't handle 5+ dimensions right now")
    
        
def distribution_nd_lmfit(params, x, func_list, index_cnt, rotate=False):
    """ params is lmfit object with the parameter names and values for each func
        x is the array of quantities to fit
        func_list is a list of normalized pdfs ndim long
        index_cnt is an array saying which index to use in each function
        Params will have names according to function, with repeats for
        multiply used fcts and angle params theta0, theta1, ...
    """
    ndim = len(func_list)
#    ndim = x.shape[0]
#    if ndim > 10: ## Assume needs to be transposed
#        x = x.reshape((1,len(x)))
#        ndim = 1
    ### Build rotation matrices:
    if rotate:
        nangles = comb(ndim, 2)
        if nangles == 1:
            thta = params['theta0'].value
            R = np.array(([np.cos(thta), np.sin(thta)], [-np.sin(thta), np.cos(thta)]))
        elif nangles == 3:
            thta0 = params['theta0'].value
            thta1 = params['theta1'].value
            thta2 = params['theta2'].value
            Rmat = np.zeros((3,3,3))
            Rmat[0] = np.array(([1, 0 , 0], [0, np.cos(thta0), -np.sin(thta0)], [0, np.sin(thta0), np.cos(thta0)]))
            Rmat[1] = np.array(([np.cos(thta1), 0, np.sin(thta1)], [0, 1, 0], [-np.sin(thta1), 0, np.cos(thta1)]))
            Rmat[2] = np.array(([np.cos(thta2), -np.sin(thta2), 0], [np.sin(thta2), np.cos(thta2), 0], [0, 0, 1]))
            R = np.dot(Rmat[2], np.dot(Rmat[1], Rmat[0]))
        else:
            print("Cannot presently accommodate 4D PDFs")
            exit(0)
        if len(x.shape) == 3:
            x_prime = np.zeros(x.shape)
            x_prime[0] = x[0] * np.cos(thta) + x[1] * np.sin(thta)
            x_prime[1] = -x[0] * np.sin(thta) + x[1] * np.cos(thta)
    #        x1 = x[0,:,0]
    #        x2 = x[1,0,:]
    #        xstack = np.vstack((x1,x2))
    #        x_prime = np.dot(R,xstack)
    #        x_prime = np.meshgrid(x_prime[0], x_prime[1], indexing='ij')
    #        x_prime = np.asarray(x_prime)
        else:
            x_prime = np.dot(R,x)
    else:
        x_prime = x
#    print x_prime[0]
#    plt.imshow(x_prime[0], interpolation='none')
#    plt.show()
#    plt.close()
#    print x_prime[1]
#    plt.imshow(x_prime[1], interpolation='none')
#    plt.show()
#    plt.close()
    prob = np.ones((x.shape[1:]))
    ### Product in rotated space...
    if func_list[0] == sf.cauchy_nd_lmfit:
#        if len(x_prime.shape) == 2:
#            for i in range(3):
#                x_in = x_prime[i].reshape(1,len(x_prime[i]))
#                prob_tmp = sf.cauchy_nd_lmfit(params, x_in, idx=i)
#                prob *= prob_tmp
#        else:
        prob_tmp = func_list[0](params, x_prime)
        prob *= prob_tmp
    else:
        for i in range(ndim):
    #        axes = np.arange(ndim)
    #        axes = tuple(axes[axes==i])
    #        x_fit = np.mean(x_prime[i], axes)
            if len(x_prime.shape) == 2:
                prob_tmp = func_list[i](params, x_prime[i], idx=index_cnt[i])
            else:
                prob_tmp = func_list[i](params, x_prime, idx=index_cnt[i])
    #        plt.imshow(prob_tmp[::-1], interpolation='none')
    #        plt.show()
    #        plt.close()
            prob *= prob_tmp
    return prob
        
def fit_nd_dist(Pdf, index_cnt=None, method='lmfit', save_name=None, verbose=False):
    """ Returns best fit params for a 2d distribution (two quantities).
        Intended for lmfit, but can also use MCMC later if desired
        params_0 is an lmfit.Parameters() object
        x is an n by m array, n: number of parameters, m: number of values
        if a save_name is included, will save the params object
    """
    x = Pdf.data
    try:
        invar = Pdf.invar
    except:
        invar = np.ones(x.shape)
    params_0 = Pdf.params
    paramarray = Pdf.paramarray
    param_bounds = Pdf.bounds
    normalized = Pdf.normalized
    func = Pdf.dist
    ### right now kwargs is built exclusively for sf.gen_central2d
    kwargs = dict()
    try:
        forms = Pdf.kwargs
        kwargs['dists'] = forms
    except:
        forms = None
    if method == 'lmfit' or method == 'opt_minimize':
        pass
    else:
        print("Right now you can only use method='lmfit', or 'opt_minimize'")
        exit(0)
    if type(func) == list:
        if verbose:
            print("Fitting {} with {}".format([f.__name__ for f in func],method))
    else:
        if verbose:
            print("Fitting {} with {}".format(func.__name__,method))
    t0 = time.time()
    def residuals_fct(params, x, invar, func, method, normalized, kwargs, index_cnt=None):
        ndim = x.shape[0]
        if ndim > 3:
            print("warning: norm resolution very limited for high dimensional arrays")
        ### Now get the norm (using very rough method right now)
        if type(func) == list:
            vals = np.ones(x[0].shape)
            idx = 0
            for f in func:
                vals *= f(params, x, index_cnt)
                tmp = f(params, x, index_cnt)
                print np.max(x[idx])
                print f.__name__
                plt.plot(x[idx], tmp, 'k.')
                plt.show()
                plt.close()
                idx += 1
        else:
            if func == sf.cauchy3_irq:
                vals = func(params, x, normalized=normalized)
            else:
                vals = func(params, x, **kwargs)
        if not normalized:
            norm = get_norm(x, params, func, kwargs=kwargs)
        else:
            norm = 1
#        plt.plot(x[0],x[1],'b.')
#        plt.imshow(mvals/norm,extent=[np.min(x[0]),np.max(x[0]),np.min(x[1]),np.max(x[1])],origin='lower')
#        plt.show()
#        plt.close()
#        print params
#        print vals
#        print -np.log(np.prod(vals/norm)) + abs(np.log(norm))
#        print np.max(vals/norm)
#        rtn = -np.log(vals)+np.log(norm)#-np.log(invar)
#        print np.log(norm)**2
        rtn = (-np.log(vals)+np.log(norm))*(invar) + abs(np.log(norm))
#        print np.sum(rtn)
        if method == 'opt_minimize':
#            rtn = np.sum(rtn)
            rtn = -np.log(np.prod(vals/norm)) + abs(np.log(norm))
            if rtn == np.inf or rtn == np.nan:
                rtn = 999999
        elif method == 'lmfit':
            rtn = np.ravel(rtn)
        return rtn

    def residuals_wrap(params, x, invar, func, method, normalized, kwargs, index_cnt=None):
        """ wrapper around residuals_fct to sum multiple components
        """
#        n_combs = sp.binom(x.shape[0],2) ### mix of 2d funcs
        ### Specific to 3 choose 2...opt_minimize, etc...
        nvars = x.shape[0]
        if nvars == 3:
            res = 0
            mnr = params[0]
            mni = params[1]
            mnd = params[2]
            scr = params[3]
            sci = params[4]
            scd = params[5]
            th1 = params[6]
            th2 = params[7]
            th3 = params[8]
            rpr, ipr, dpr = get_rots(x[0]-mnr, x[1]-mni, x[2]-mnd, th1, th2, th3)
#            plt.figure('rpr')
#            plt.hist(rpr,14)
#            plt.figure('ipr')
#            plt.hist(ipr,14)
#            plt.figure('dpr')
#            plt.hist(dpr,14)
#            plt.show()
#            plt.close()
            vals = kwargs['dists']
            for i in range(3):
                if i == 0:
                    paramsn = np.array(([scr, sci]))
                    X = np.vstack((rpr,ipr))
                    W = np.vstack((invar[0], invar[1]))
                    kwtmp = dict()
                    kwtmp['dists'] = [vals[0], vals[1]]
                elif i == 1:
                    paramsn = np.array(([sci, scd]))
                    X = np.vstack((ipr, dpr))
                    W = np.vstack((invar[1], invar[2]))
                    kwtmp = dict()
                    kwtmp['dists'] = [vals[1], vals[2]]
                elif i == 2:
                    paramsn = np.array(([scr, scd]))
                    X = np.vstack((rpr,dpr))
                    W = np.vstack((invar[0], invar[2]))
                    kwtmp = dict()
                    kwtmp['dists'] = [vals[0], vals[2]]
                rtmp = residuals_fct(paramsn, X, W, func, method, normalized, kwtmp)
                res += rtmp
#        elif nvars == 2:
#            mnr = params[0]
#            mni = params[1]
#            scr = params[3]
#            sci = params[4]
#            th1 = params[6]
#            rpr = (x[0] - mnr)*np.cos(th1) - (x[1]-mni)*np.sin(th1)
#            ipr = (x[0] - mnr)*np.sin(th1) + (x[1]-mni)*np.cos(th1)
#            vals = kwargs['dists']
#            res = residuals_fct(paramsn, X, W, func, method, normalized, kwtmp)
#            for i in range(3):
#                if i == 0:
#                    paramsn = np.array(([scr, sci]))
#                    X = np.vstack((rpr,ipr))
#                    W = np.vstack((invar[0], invar[1]))
#                    kwtmp = dict()
#                    kwtmp['dists'] = [vals[0], vals[1]]
#        print res
        return res
                              
    if method == 'lmfit':
        fit_kws = dict()
        fit_kws['maxfev'] = 4000
        results = lmfit.minimize(residuals_fct,params_0,args=(x, invar, func, method, normalized, kwargs),**fit_kws)
        fit_params = results.params
        t1 = time.time()
        print("Lmfit took {}s".format(t1-t0))
    elif method == 'opt_minimize':
        if func is sf.gen_central2d and x.shape[0] > 2:
            res = opt.minimize(residuals_wrap,paramarray,args=(x, invar, func, method, normalized, kwargs),bounds=param_bounds)
        else:
            ### Hack to widen Qgal range
            idx = (999 if Pdf.name == 'qgale' else None)
            kwargs['idx'] = idx
            res = opt.minimize(residuals_fct,paramarray,args=(x, invar,func, method, normalized, kwargs),bounds=param_bounds)
        fit_params = res.x
        t1 = time.time()
        if verbose:
            print "results:"
            print res
            print("opt.minimize took {}s".format(t1-t0))
    if save_name is not None:
        if method == 'lmfit':
            Pdf.params = fit_params
        elif method == 'opt_minimize':
            if verbose:
                print "fit_params:", fit_params
            Pdf.paramarray = fit_params
        np.save('/home/matt/software/matttest/results/pdf_params_{}'.format(save_name), Pdf)
    return fit_params        
        
def plot_1d_projections(Pdf, index_cnt=None, plot_2d=False, use_array=False, alt_data=None):
    x = Pdf.data
    if alt_data is not None:
        x = alt_data
    if use_array:
        params = Pdf.paramarray
    else:
        params = Pdf.params
    funcp = Pdf.dist
    funcn = Pdf.name
    ndim = x.shape[0]
    rscl = (np.sqrt((0.5)**(1/(1-3/2))-1)) ##convert a to r1/2
    exptime = 1#626 #varies - need to key on this
    photflam = 1.1845489#*10**-19 #constant
    iscl = photflam/exptime#*10**19 to get on scale [0, 1]
    if ndim > 5: ## Assume needs to be transposed
        x = x.reshape((1,len(x)))
        ndim = 1
    kwargs = dict()
    try:
        forms = Pdf.kwargs
        kwargs['dists'] = forms
    except:
        forms = None
    if funcp == sf.gen_central2d and x.shape[0] > 2:
        func = sf.gen_central3d
    else:
        func = funcp
    mf = 1.0
    norm, mesh_arr, spacing, xaxes = get_norm(x, params, func, return_mesh=True, max_factor = mf, kwargs=kwargs)
    if type(func) == list:
        vals = np.ones((mesh_arr.shape[1:]))
        for j in range(len(func)):
            vals *= func[j](params, mesh_arr, index_cnt)
    else:
        vals = func(params, mesh_arr, index_cnt, **kwargs)
    vals /= norm
    font = FontProperties()
    font.set_family('serif')
    font.set_size(18)
    ifont = font.copy()
    ifont.set_size(14)
    for i in range(ndim):
        xaxis = xaxes[i]
        if ndim > 1:
#            names = ["$radius\ (kpc)$", "$flux\ (ergs/cm^{2}/s/\AA\ *\ 10^{22})$", "$axis\ ratio (b/a)$"]
#            snms = ["rad", "int", "ax_rat"]
#            scl = [rscl, iscl, 1]
            names = ["r$_{\mathrm{c}}$ (kpc)", r"i$_{\mathrm{c}}$ (ergs cm$^{-2}$ s$^{-1}$ $\AA^{-1}$ arcsec$^{-2}$$\times$10$^{-19})$", "d$_{\mathrm{c}}$ (kpc)"]
            snms = ["rad", "int", "dist"]
            scl = [rscl, iscl, 1]
            if i == 0:
                if ndim == 3:
                    axes = (0,2)
                else:
                    axes = 0
            elif i == 1:
                if ndim == 3:
                    axes = (1,2)
                else:
                    axes = 1
            elif i == 2:
                axes = (0,1)
#            axes = np.arange(ndim)
#            axes = tuple(axes[axes!=i])
#            proj = vals
            proj = np.sum(vals, axis=axes)
            spp = np.arange(ndim)
            sp_tmp = spacing[spp[spp!=i]]
#            if i == 0:
#                si = 1
#            elif i == 1:
#                si = 0
            proj *= np.prod(sp_tmp)
        else:
#            names = ["$distance (kpc)$"]
#            snms = ["dist"]
            names = ["q$_{\mathrm{c}}$ (axis ratio)", "q$_{\mathrm{c}}$ (axis ratio)", "r$_{\mathrm{c}}$ (kpc)", r"i$_{\mathrm{c}}$ (ergs cm$^{-2}$ s$^{-1}$ $\AA^{-1}$ arcsec$^{-2}$$\times$10$^{-19})$", "d$_{\mathrm{c}}$ (kpc)"]
            snms = ["gal_ax_rat", "ax_rat","rad", "int", "dist"]
            scl = [1]
            proj = vals.reshape((vals.size,))
#            si = i
        hcolor = [0.3, 0.3, 0.3]
        plt.hist(x[i]*scl[i], bins=14, normed=True, color=hcolor)
        plt.plot(xaxis*scl[i], proj/scl[i],  'k',  linewidth = 2)
#        print Pdf.name
#        xlab = raw_input("Enter xlabel name:")
        if funcn == 'qgale':
            idx = 0
        elif funcn == 'qb':
            idx = 1
        elif funcn == 'rc':
            idx = 2
        elif funcn == 'ic':
            idx = 3
        elif funcn == 'rsepe':
            idx = 4
        else:
            print "don't recognize distribution parameter {}".format(funcn)
            idx = 0
            snms[idx] = 'test'
        plt.xlabel(names[idx], fontproperties=font)
        plt.ylabel("Probability", fontproperties=font)
        plt.savefig('/home/matt/software/matttest/results/figs/{}_hist.eps'.format(snms[idx]), bbox_inches='tight')
        plt.show()
        plt.close()
    if plot_2d:
        num_plots = 0
#        names = ["$radius\ (kpc)$", "$flux\ (ergs/cm^{2}/s/\AA\ *\ 10^{22})$", "$axis\ ratio (b/a)$"]
#        snms = ["rad", "int", "ax_rat"]
        names = ["r$_{\mathrm{c}}$ (kpc)", r"i$_{\mathrm{c}}$ (ergs cm$^{-2}$ s$^{-1}$ $\AA^{-1}$ arcsec$^{-2}$$\times$10$^{-19})$", "d$_{\mathrm{c}}$ (kpc)"]
        snms = ["rad", "int", "dist"]
        for i in range(ndim):
            num_plots += i
        if ndim == 2:
            num_plots = 1
        if num_plots != 0:
            for j in range(num_plots):
                ### Not sure how to generalize, doing 3d manually for now
                if ndim == 2:
                    plt.plot(x[0], x[1],'b.')
                    plt.imshow(np.log(vals.T[::-1]), extent=[np.min(x[0]), np.max(x[0])*mf, np.min(x[1]), np.max(x[1])*mf], interpolation='none')
                    plt.show()
                    plt.close()
                elif ndim == 3:
                    if j == 0:
                        idx1 = 0
                        idx2 = 1
                        axx = 2
#                        tp = True
                    elif j == 1:
                        idx1 = 0
                        idx2 = 2
                        axx = 1
#                        tp = True
                    elif j == 2:
                        idx1 = 1
                        idx2 = 2
                        axx = 0
#                        tp = True
                    print("Plotting x[{}] vs x[{}]".format(idx1,idx2))
                    plt.plot(x[idx1], x[idx2], 'b.')
                    plt.xlabel(names[idx1], fontproperties=font)
                    plt.ylabel(names[idx2], fontproperties=font)
                    img = np.sum(vals, axis = axx)* spacing[j]
                    xax = xaxes[idx1]
                    yax = xaxes[idx2]
                    implt = img.T
                    plt.imshow(np.log(implt[::-1,:]), extent=[np.min(x[idx1]), np.max(x[idx1])*mf, np.min(x[idx1]), np.max(x[idx2])*mf], interpolation='none')
                    plt.savefig('/home/matt/software/matttest/results/figs/{}_vs_{}.pdf'.format(snms[idx1], snms[idx2]))
                    plt.show()
                    plt.close()
                else:
                    print "Right now can only use plot_2d for 2 or 3 params"
                    exit(0)
                    
   
#def get_rot(x, y, thta):
#    cc = np.cos(thta)
#    ss = np.sin(thta)
#    R = np.array(([cc, -ss], [ss, cc]))
##    Rrev = np.array(([cc, ss], [-ss, cc]))
#    Xin = np.vstack((x,y))#,re))
#    Xrot = np.dot(R,Xin)
#    xrot = Xrot[0]
#    yrot = Xrot[1]
#    return xrot, yrot
     
def min_rotation(x, y, thta0=0):
    """ find a theta value that minimizes Pearson's (linear) correlation
        between x and y
        Right now using brute force search for lack of a better idea.
    """
    def get_corr(x, y, thta):
        xrot, yrot = get_rot(x, y, thta)
        return sf.correlation_coeff(xrot, yrot)
    xrot, yrot = get_rot(x, y, thta0)
    cp0 = sf.correlation_coeff(xrot, yrot)
    xroth, yroth = get_rot(x, y, thta0+0.01)
    xrotl, yrotl = get_rot(x, y, thta0-0.01)
    cph = sf.correlation_coeff(xroth, yroth)    
    cpl = sf.correlation_coeff(xrotl, yrotl)
    if abs(cph) < abs(cpl):
        sign = 1
    elif abs(cpl) > abs(cph):
        sign = -1
    else:
        print("sign can't be determined in 'min_rotation'")
        exit(0)
#    cpold = 10
    thresh = 0.01
    thetalow = thta0
    thetahigh = thta0 + sign*np.pi/2 #max of 90 degree rotation
    ### Loop through, extrapolating to get close to zero
    while abs(cp0) > thresh:
        cpl = get_corr(x, y, thetalow)
        cph = get_corr(x, y, thetahigh)
        [m, b] = np.polyfit([thetalow, thetahigh], [cpl, cph], 1)
        thetazero = -b/m
        cpz = get_corr(x, y, thetazero)
        if abs(cph) > abs(cpl):
            cph = cpz
            thetahigh = thetazero
        else:
            cpl = cpz
            thetalow = thetazero
        cp0 = cpz
    ### Return the closest to zero
    if abs(cph) < abs(cpl):
        return thetahigh
    else:
        return thetalow
        
def resample(nb, re, ie, rsep, nsers, qb, PAb, Qgal, PAgal):
    """ resample with replacement for bootstrapping
    """
#    params = [nb, re, ie, rsep, nsers, qb, PAb, Qgal, PAgal]
    rs = np.random.randint(len(re), size = (len(re),))
    rg = np.random.randint(len(nb), size = (len(nb),))
    new_params = [nb[rg], re[rs], ie[rs], rsep[rs], nsers[rs], qb[rs], PAb[rs], Qgal[rg], PAgal[rg]]
    return new_params
    
def circle_tophat(shape,center,radius):
        """ Makes an image with 1 for values within radius, zero without
        """
        circle = np.zeros(shape)
        rad = sf.make_rarr(np.arange(shape[0]), np.arange(shape[1]), center[0], center[1])
        circle[rad<=radius] = 1
        return circle
        
def get_ew(filename,inds):
    #1. Estimate Lya line flux from SDSS spectra
    root, junk = os.path.split(filename)
    fname = glob.glob(os.path.join(root,'spec-*.fits'))
    spec = pyfits.open(fname[0])
    sd = spec[1].data
    flux = np.zeros(len(sd))
    wave = np.zeros(len(sd))
    for i in range(len(sd)):
        flux[i] = sd[i][0]
        wave[i] = 10**sd[i][1]
#    plt.plot(wave,flux)
#    plt.show()
    idx1, idx2 = inds
    ### very rough estimate
    bg = (flux[idx2]+flux[idx1])/2
    dw = np.mean(np.ediff1d(wave[idx1:idx2]))
    Flya = (np.sum(flux[idx1:idx2])-bg)*dw*10**(-17)

    #2. Estimate UV continuum level from HST image and model of SDSS fiber aperature
    cont = pyfits.open(filename)
    uv_img = cont[7].data
    dimx, dimy = uv_img.shape
    xc, yc = np.floor(dimx/2), np.floor(dimy/2)
    rpx = 1/0.04*100
    circle = circle_tophat((dimx,dimy),[xc,yc], rpx)
    fwhm = 1.5/0.04/100 ## seeing estimate - 1.5" - need to double check on this
    sig = fwhm/(2*np.sqrt(2*np.log(2)))
    arr = np.arange(200)
    gauss = sf.gauss2d(arr, arr, sig, sig, xcenter = 100, ycenter=100)
    gauss /= np.sum(gauss)
#    plt.plot(gauss[:,100])
#    plt.show()
    conv_img = signal.convolve2d(uv_img,gauss,mode='same')
#    plt.imshow(conv_img, interpolation='none')
#    plt.show()
    cnt_uv = np.sum(conv_img*circle)
    fuv = cnt_uv*photflam
    print Flya/fuv
    return Flya/fuv
    
def calc_bb_array(ie, re):
    return eff_flux(np.sqrt(1/np.pi),ie, re)
    
def calc_selection_fct(bb_array, trials=1000):
    nc_mx = 10
    sf_array = np.zeros((len(bb_array),nc_mx,2)) #nsample by nclumps by found
    for i in range(trials):    
        nclumps = int((10*np.random.rand()+1))
        draws = np.floor(np.random.rand(nclumps)*len(bb_array)).astype(int)
        ### append the "drawn" array
        sf_array[draws,nclumps-1,1] += 1
        bb_sub = bb_array[draws]
        ### results are highly dependent on noise model...
        noise_std = np.random.randn()*0.002 + 0.005
#        noise = 0.0125*np.max(bb_sub)
        bb_msk = bb_sub > noise_std
        fnd = draws[bb_msk]
        ### append the "found" array
        sf_array[fnd,nclumps-1,0] += 1
    return sf_array
        
def k_correction(z, zref=2.6):
    """ Assumes simple linear form for k-corrections based on literature...
        van der Berg 2010, Figure 7
    """
    k_mag = (z-zref)*(-0.055)
    k_flx = 10**(-0.4*k_mag)
    return k_flx
    
def corr_coeff_bootstrap(nb, re, ie, rsep, nsers, qb, PAb, Qgal, PAgal, n=1000, sig=1):
    """ Checks variation of correlation coefficient with bootstraip to try to
        evaluate significance level of correlations
    """
    ccs = np.zeros((6,n))
    for i in range(n):
        [nn, rn, ien, rsn, nsn, qn, PAn, Qn, PAgn] = resample(nb, re, ie, rsep, nsers, qb, PAb, Qgal, PAgal)
        ### six correlation coeffs to check
        ccs[0,i] = sf.correlation_coeff(rn, ien)
        ccs[1,i] = sf.correlation_coeff(rn, qn)
        ccs[2,i] = sf.correlation_coeff(qn, ien)
        ccs[3,i] = sf.correlation_coeff(rn, rsn)
        ccs[4,i] = sf.correlation_coeff(rsn, ien)
        ccs[5,i] = sf.correlation_coeff(rsn, qn)
    ccvals = np.zeros((6,3))
    if sig == 1:
        lo, hi = [16, 84]    
    elif sig == 2:
        lo, hi = [2.5, 97.5]
    else:
        print "Please choose sig = 1 or 2."
        exit(0)
    for j in range(6):
        ccvals[j,0] = np.median(ccs[j])
        ### Adam's preferred metrics
#        ccvals[j,1] = abs(np.percentile(ccs[j], lo)-np.percentile(ccs[j], hi))
#        ccvals[j,2] = np.median(ccs[j])/abs(np.percentile(ccs[j], lo)-np.percentile(ccs[j], hi))
        ccvals[j,1] = np.percentile(ccs[j], lo)
        ccvals[j,2] = np.percentile(ccs[j], hi)
    return ccvals
        
def pdf_param_guesses(name, idx=0, return_array=False):
    """ Manually enter guesses here - may later modify to read guesses
        from an external text file
    """
    params_0 = lmfit.Parameters()
    if name == 'nb' or name == 'nbe':
#        lam = 3.5
        # for uniform
        xmin = 1
        xmax = 9
        param_names = ['xmin{}'.format(idx), 'xmax{}'.format(idx),]
        param_vals = [xmin, xmax]
        param_mins = [xmin, xmax]
        param_maxes = [xmin, xmax]
        param_fixed = [0, 0]
#    elif name == 'qb' or name == 'qgal':
#        
    elif name == 'rsep' or name == 'rsepe':
#        mn = 1.1
#        scl = 1.0
#        param_names = ['c{}'.format(idx), 's{}'.format(idx)]
#        param_vals = [mn, scl]
#        param_mins = [0.5*mn, 0.1*scl]
#        param_maxes = [1.5*mn, 10*scl]
#        param_fixed = [1, 1]
        scl = 1.1
        mn = 1.1
        param_names = ['c{}'.format(idx), 's{}'.format(idx)]
        param_vals = [mn, scl]
        param_mins = [0.05, 0.01]
        param_maxes = [4, 10]
        param_fixed = [1, 1]
    elif name == 'logie':
        mn = -1.2
        scl = 1.0
        param_names = ['mn{}'.format(idx), 'sig{}'.format(idx)]
        param_vals = [mn, scl]
        param_mins = [mn-1, 0.1*scl]
        param_maxes = [mn+1, 10*scl]
        param_fixed = [1, 1]
    elif name == 'logre':
        mn = -0.55
        scl = 0.7
        param_names = ['mn{}'.format(idx), 'sig{}'.format(idx)]
        param_vals = [mn, scl]
        param_mins = [mn-1, 0.1*scl]
        param_maxes = [mn+1, 10*scl]
        param_fixed = [1, 1]
    elif name == 'nsers':
        mn = 0.5
        scl = 0.5
        param_names = ['c{}'.format(idx), 's{}'.format(idx)]
        param_vals = [mn, scl]
        param_mins = [mn-1, 0.1*scl]
        param_maxes = [mn+1, 10*scl]
        param_fixed = [1, 1]
    elif name == 'qgal':
        mn = 0.45
        scl = 0.2
        param_names = ['mn{}'.format(idx), 'sig{}'.format(idx)]
        param_vals = [mn, scl]
        param_mins = [mn-1, 0.1*scl]
        param_maxes = [mn+1, 10*scl]
        param_fixed = [1, 1]
    elif name == 'qgale':
        mn = 0.5
        scl = 0.15
        param_names = ['c{}'.format(idx), 's{}'.format(idx)]
        param_vals = [mn, scl]
        param_mins = [0, 0.1*scl]
        param_maxes = [1, 10*scl]
        param_fixed = [1, 1]
    elif name == 'qb':
        mn = 0.5
        scl = 0.2
        param_names = ['c{}'.format(idx), 's{}'.format(idx)]
        param_vals = [mn, scl]
        param_mins = [mn-0.4, 0.001*scl]
        param_maxes = [mn+0.4, 100*scl]
        param_fixed = [1, 1]
    elif name == 'ic':
        #weibull
        mn = 0.2
        scl = 1.1
        #cauchy
        #mn = -0.2
        #scl = 1e-4
        #cauchy w/ mn >=0
        #mn = 0.01
        #scl = 0.1
        param_names = ['c{}'.format(idx), 's{}'.format(idx)]
        param_vals = [mn, scl]
        #weibull
        param_mins = [0.01, 0.01]
        param_maxes = [2, 100]
        #cauchy
#        param_mins = [0, 1e-6]
#        param_maxes = [0.05, 1]
        param_fixed = [1, 1]
    elif name == 'rc':
        mn = 0.2
        scl = 0.2
        param_names = ['c{}'.format(idx), 's{}'.format(idx)]
        param_vals = [mn, scl]
        param_mins = [0.03, 0.03]
        param_maxes = [2, 0.99]
        param_fixed = [1, 1]
    elif name == 're_rsep':
        k = 2
        ks = -0.1
        lam = 0.4
        lams = 0.1
        param_names = ['k{}'.format(idx), 'ks{}'.format(idx), 'lam{}'.format(idx), 'lams{}'.format(idx)]
        param_vals = [k, ks, lam, lams]
        param_mins = [1, -1, 0.05, -1]
        param_maxes = [8, 1, 0.5, 1]
        param_fixed = [1, 1, 1, 1]
#    elif name == 'logie_logre_nsers':
#        mn0, scl0, mn1, scl1, mn2, scl2, gam = -1.2, 1.0, -0.55, 0.7, 0.5, 0.5, 0.3
#        param_names = ['c0', 's0', 'c1', 's1', 'c2', 's2', 'gam0']
#        param_vals = [mn0, scl0, mn1, scl1, mn2, scl2, gam]
#        param_mins = [mn0-1, scl0/10, mn1-1, scl1/10, mn2-1, scl2/10, gam/10]
#        param_maxes = [mn0+1, scl0*10, mn1+1, scl1*10, mn2+1, scl2*10, gam*10]
#        param_fixed = np.ones((len(param_vals)))
    elif name == 'ie_re_nsers':
#        mnr, sigr, lamn, lamsr, kn, ksr, taui, tausn = 0.27, 0.17, 0.4, 0.01, 1.4, 0.01, 0.05, 0.3
        mnr, sigr, lamn, lamsr, kn, ksr, taui, tausn = 0.27, 0.17, 0.4, 0.5, 1.4, -0.1, 0.1, 0.01
        param_names = ['mnr', 'sigr', 'lamn', 'lamsr', 'kn', 'ksr', 'taui', 'tausn']
        param_vals = [mnr, sigr, lamn, lamsr, kn, ksr, taui, tausn]
        param_mins = [mnr-0.1, sigr/2, 0.1, -1, 1, -1, 0.05, -1]
        param_maxes = [mnr+0.1, sigr*2, lamn+1, 1, 10, 1, 0.4, 1]
        param_fixed = np.ones((len(param_vals)))
    elif name == 'ie_re_nsers_comb':
        mnr = 0.27
        mnn = 0.5
        mni = 0
        scr = 0.2
        scn = 0.5
        sci = 0.2
        th1 = np.pi/6
        th2 = np.pi/6
        th3 = 0
#        gam = 1
        param_names = ['mnr', 'mni', 'mnn', 'scr', 'sci', 'scn', 'th1', 'th2', 'th3']#, 'gam']
        param_vals = [mnr, mni, mnn, scr, sci, scn, th1, th2, th3]#, gam]
        param_mins = [mnr*0.5, mni-1, mnn*0.5, scr*0.1, sci*0.01, scn*0.1, -1000*np.pi, -1000*np.pi, -1000*np.pi]#, 0.1]
        param_maxes = [mnr*2, mni+0.2, mnn*4, scr*2, sci*2, scn*2, 1000*np.pi, 1000*np.pi, 1000*np.pi]#, 10]
        param_fixed = np.ones((len(param_vals)))
    elif name == 'eff_joint':
#        mnr = 0.27
#        mni = 0
#        mnq = 0.44
#        scr = 0.3
#        sci = 0.2
#        scq = 0.2
#        th1 = -np.pi/8
#        th2 = np.pi/6
#        th3 = -np.pi/8
        mnr = 0.253#*1.3
        mni = -0.146
        mnq = 0.48
        scr = 0.097#*1.3
        sci = 0.021#*2
        scq = 0.145
        th1 = -0.27
        th2 = 0.051
        th3 = -0.075
#        mnr = 0.25#*1.3
#        mni = 0
#        mnq = 0.5
#        scr = 0.097#*1.3
#        sci = 0.021#*2
#        scq = 0.145
#        th1 = np.pi/16
#        th2 = np.pi/8
#        th3 = np.pi/16
#        gam = 1
        param_names = ['mnr', 'mni', 'mnq', 'scr', 'sci', 'scq', 'th1', 'th2', 'th3']#, 'gam']
        param_vals = [mnr, mni, mnq, scr, sci, scq, th1, th2, th3]#, gam]
        param_mins = [mnr*0.5, mni-1, mnq*0.5, scr*0.01, sci*0.01, scq*0.1, -np.pi/4, -np.pi/4, -np.pi/4]
        param_maxes = [mnr*2, 0.05, mnq*4, scr*4, sci*4, scq*4, np.pi/4, np.pi/4, np.pi/4]
        param_fixed = np.ones((len(param_vals)))
    elif name == 'ird':
        ### Final
        mnr = 0.2#0.214
#        mni = -0.127
        mni = 0.2
        mnd = 0.3
        scr = 0.133#*1.3
#        sci = 0.0013#*2
        sci = 0.2
        scd = 0.864
        th1 = 0#np.pi/6#-0.005
        th2 = 0#np.pi/6#-0.0044
        th3 = 0#np.pi/6#-0.013
        ### Div10
#        mnr = 0.15
#        mni = -0.15
#        mnd = 1.3
#        scr = 0.133#*1.3
#        sci = 0.0013#*2        
#        scd = 0.864
#        th1 = -0.005
#        th2 = -0.0044
#        th3 = -0.013
        ### Tim10
#        mnr = 0.15#0.214
#        mni = -0.127
#        mnd = 1.3
#        scr = 0.133#*1.3
#        sci = 0.0013#*2        
#        scd = 0.864
#        th1 = -0.005
#        th2 = -0.0044
#        th3 = -0.013
        ### Gaussian radius
#        mnr = 0.054#*1.3
#        mni = -0.14
#        mnd = 1.34
#        scr = 0.295#*1.3
#        sci = 0.0013#*2        
#        scd = 0.845
#        th1 = -0.0058
#        th2 = -0.0044
#        th3 = -0.013
        param_names = ['mnr', 'mni', 'mnd', 'scr', 'sci', 'scd', 'th1', 'th2', 'th3']#, 'gam']
        param_vals = [mnr, mni, mnd, scr, sci, scd, th1, th2, th3]#, gam]
        param_mins = [mnr*0.01, -mni, 0, scr*0.1, sci/1e6, scd*0.1, -np.pi/4, -np.pi/4, -np.pi/4]
        param_maxes = [mnr*2, 0.5, mnd*2, scr*4, sci*10, scd*4, np.pi/4, np.pi/4, np.pi/4]
        param_fixed = np.ones((len(param_vals)))
    elif name == 'irj':
        ### Final
        mnr = 0.3#0.15#0.214
        mni = -0.2#-0.127
        qir = 0.5#0.7
        PAr = np.pi/6
        sigir = 0.15
#        scr = 0.133#*1.3
#        sci = 0.0013#*2        
#        th1 = -np.pi/6
        ### Div10
#        mnr = 0.15
#        mni = -0.15
#        mnd = 1.3
#        scr = 0.133#*1.3
#        sci = 0.0013#*2        
#        scd = 0.864
#        th1 = -0.005
#        th2 = -0.0044
#        th3 = -0.013
        ### Tim10
#        mnr = 0.15#0.214
#        mni = -0.127
#        mnd = 1.3
#        scr = 0.133#*1.3
#        sci = 0.0013#*2        
#        scd = 0.864
#        th1 = -0.005
#        th2 = -0.0044
#        th3 = -0.013
        ### Gaussian radius
#        mnr = 0.054#*1.3
#        mni = -0.14
#        mnd = 1.34
#        scr = 0.295#*1.3
#        sci = 0.0013#*2        
#        scd = 0.845
#        th1 = -0.0058
#        th2 = -0.0044
#        th3 = -0.013
#        param_names = ['mnr', 'mni', 'scr', 'sci', 'th1']#, 'gam']
#        param_vals = [mnr, mni, scr, sci, th1]#, gam]
#        param_mins = [mnr/4, -0.3, scr*0.1, 1e-21, -np.pi/4]
#        param_maxes = [mnr*4, 0.3, scr*4, sci*100, np.pi/4]
#        param_fixed = np.ones((len(param_vals)))
        param_names = ['mnr', 'mni', 'qir', 'PAr', 'sigir']#, 'gam']
        param_vals = [mnr, mni, qir, PAr, sigir]#, gam]
        param_mins = [mnr/4, -0.3, 0.1, -1000, 1e-10]
        param_maxes = [mnr*4, 0.3, 10, 1000, 1]
        param_fixed = np.ones((len(param_vals)))
    elif name == 'idj':
        ### Final
        mni = -0.1#0.15#0.214
        mnd = 0.3#-0.127
        sci = 0.2#0.7
        scd = 0.8
        th = -np.pi/6
        param_names = ['mni', 'mnd', 'sci', 'scd', 'th']#, 'gam']
        param_vals = [mni, mnd, sci, scd, th]#, gam]
        param_mins = [-0.3, -0.2, sci*0.01, scd/3, -np.pi/4]
        param_maxes = [0., mnd*4, sci*10, scd*3, np.pi/4]
        param_fixed = np.ones((len(param_vals)))
#        param_names = ['mnr', 'mni', 'qir', 'PAr', 'sigir']#, 'gam']
#        param_vals = [mnr, mni, qir, PAr, sigir]#, gam]
#        param_mins = [mnr/4, -0.3, 0.1, -1000, 1e-10]
#        param_maxes = [mnr*4, 0.3, 10, 1000, 1]
#        param_fixed = np.ones((len(param_vals)))
    
    if return_array:
        bounds = [0]*len(param_vals)
        for i in range(len(param_vals)):
            bounds[i] = ((param_mins[i], param_maxes[i]))
        return param_vals, bounds
    else:
        params_0 = sf.array_to_Parameters(params_0, param_vals, arraynames=param_names, minarray = param_mins, maxarray = param_maxes, fixed=param_fixed)
        return params_0
        