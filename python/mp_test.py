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