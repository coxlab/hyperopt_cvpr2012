#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" v1s_math module

Utility math functions.

"""

import scipy as N

def fastnorm(x):
    """ Fast Euclidean Norm (L2)

    This version should be faster than numpy.linalg.norm if 
    the dot function uses blas.

    Inputs:
      x -- numpy array

    Output:
      L2 norm from 1d representation of x
    
    """    
    xv = x.ravel()
    return N.dot(xv, xv)**(1/2.)

def fastsvd(M):
    """ Fast Singular Value Decomposition
    
    Inputs:
      M -- 2d numpy array

    Outputs:
      U,S,V -- see scipy.linalg.svd    

    """
    
    h, w = M.shape
    
    # -- thin matrix
    if h >= w:
        # subspace of M'M
        U, S, V = N.linalg.svd(N.dot(M.T, M))
        U = N.dot(M, V.T)
        # normalize
        for i in xrange(w):
            S[i] = fastnorm(U[:,i])
            U[:,i] = U[:,i] / S[i]
            
    # -- fat matrix
    else:
        # subspace of MM'
        U, S, V = N.linalg.svd(N.dot(M, M.T))
        V = N.dot(U.T, M)
        # normalize
        for i in xrange(h):
            S[i] = fastnorm(V[i])
            V[i,:] = V[i] / S[i]
            
    return U, S, V

def multigabor2d(glist,weights=None):

    env = [np.fft.fft2(gabor2d(*z)) for z in glist]
    
    if weights is None:
        weights = [np.abs(e).max()**2 for e in env]
        
    weights = np.array(weights)
    env = [w*e for (w,e) in zip(weights,env)]

    gabor = np.fft.fft2(reduce(lambda x,y: x + y, env))
    gabor -= gabor.mean()
    gabor /= fastnorm(gabor)
    return np.abs(gabor)

def gabor2d(gw, gh, gx0, gy0, wfreq, worient, wphase, shape):
    """ Generate a gabor 2d array
    
    Inputs:
      gw -- width of the gaussian envelope
      gh -- height of the gaussian envelope
      gx0 -- x indice of center of the gaussian envelope
      gy0 -- y indice of center of the gaussian envelope
      wfreq -- frequency of the 2d wave
      worient -- orientation of the 2d wave
      wphase -- phase of the 2d wave
      shape -- shape tuple (height, width)

    Outputs:
      gabor -- 2d gabor with zero-mean and unit-variance

    """
    
    height, width = shape
    y, x = N.mgrid[0:height, 0:width]
    
    X = x * N.cos(worient) * wfreq
    Y = y * N.sin(worient) * wfreq
	
    env = N.exp( -N.pi * ( ((x-gx0)**2./gw**2.) + ((y-gy0)**2./gh**2.) ) )
    wave = N.exp( 1j*(2*N.pi*(X+Y) + wphase) )
    gabor = N.real(env * wave)
    
    gabor -= gabor.mean()
    gabor /= fastnorm(gabor)
    
    return gabor


def gabor3d(gw, gh, gd, gx0, gy0, gz0, wfreq, worients, wphase, shape):
    """ Generate a gabor 2d array
    
    Inputs:
      gw -- width of the gaussian envelope
      gh -- height of the gaussian envelope
      gx0 -- x indice of center of the gaussian envelope
      gy0 -- y indice of center of the gaussian envelope
      wfreq -- frequency of the 2d wave
      worient -- orientation of the 2d wave
      wphase -- phase of the 2d wave
      shape -- shape tuple (height, width)

    Outputs:
      gabor -- 2d gabor with zero-mean and unit-variance

    """
    
    if not hasattr(wfreq,'__iter__'):
        wfreq = (wfreq,wfreq,wfreq)
    else:
        wfreq = tuple(wfreq)
    wf1,wf2,wf3 = wfreq
    
    height, width, depth = shape
    y, x, z = N.mgrid[0:height, 0:width, 0:depth]
    wphi,wpsi = worients
    
    X = x * N.cos(wphi) * N.sin(wpsi) * wf1
    Y = y * N.sin(wphi) * N.sin(wpsi) * wf2
    Z = z * N.cos(wpsi) * wf3
	
    env = N.exp( -N.pi * ( ((x-gx0)**2./gw**2.) + ((y-gy0)**2./gh**2.) + ((z-gz0)**2./gd**2.) ) )
    wave = N.exp( 1j*(2*N.pi*(X+Y+Z) + wphase) )
    gabor = N.real(env * wave)
    
    gabor -= gabor.mean()
    gabor /= fastnorm(gabor)
    
    return gabor