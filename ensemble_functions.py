#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 10:42:10 2019

@author: rlt1917
"""

'''
New file to execute ensemble experiments in PARALLEL
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as ss
import scipy.constants as constant
import math
from scipy.sparse import diags
import argparse
import copy

import warnings
#warnings.filterwarnings("error")
# SARAH: had to comment this line out, was screwing with input outputs, and
# giving unclosed file errors in dask

def LossRate(L,J1=None,losstype='Shprits2007-eq4'):
    """
    Function which calculates the electron lifetime and hence the loss rate for
    each cell.

    TO DO: May want to have this take L, J1 and call the coordinate transform
    from here

    NOTE: mu is already used here so say J1 is the first adiabatic invariant 

    Inputs:
        - L : Grid in space
        - J1 : first adiabatic invariant
        - losstype: which loss model to use

    Calulates
        - E : kinetic energy 
    
    Outputs:
        - tau: electron loss timescale

    """
    # some constants
    c = constant.c # speed of light in a vacuum 
    m0 = constant.m_e # mass of electron
    Beq = 2*(3.12 * 10**(-5)) / (L**3) #equatorial B for each L 

    if J1 is None:
    # set first adiabatic invariant mu to that of 2MeV electron at L=5
        example_E = 2*constant.mega*constant.physical_constants["electron volt"][0]
        example_Beq = 2*(3.12 * 10**(-5)) / (5**3)
        J1 = example_E*(example_E + 2* m0*c**2)/(2*m0*c**2 * example_Beq)

    # calculate kinetic energy at each L
    E = -m0 * c**2 + np.sqrt(m0**2 * c**4 + 2*J1* m0 * c**2 * Beq)


    # in mega electron volts
    E = E / (constant.mega*constant.physical_constants["electron volt"][0])

    
    if losstype == 'Shprits2007-eq4':
        # use KE to find lifetime at each L, in days
        tau = 1.2 * np.multiply(E**2, L**(-1))
       
        # now in seconds
        tau = tau*24*60*60

    else:
        raise NameError('loss type not implemented')

    assert np.all( tau > 1 )

    return tau



def Diff_Coeff(L,Kp, full=False,method='Ozeke'):
    """
    Function which calculates the diffusion coefficient at an L-Shell Value
    L and Kp index Kp, in units of days^{-1}
    """
    if method == 'Ozeke':

        DLL_B = 6.62 * 10**(-13) * L**8 * 10**(-0.0327*(L**2) + 0.625*L \
                 - 0.0108*(Kp**2) + 0.499*Kp)
    
        DLL_E = 2.16 * 10**(-8) * (L**6) * 10**(0.217*L + 0.461*Kp)
    
        D = (DLL_B + DLL_E)
   
        temp = ( 6.62 * 10**(-13) * 10**(-0.0327*(L**2) + 0.625*L \
                  - 0.0108*(Kp**2) + 0.499*Kp) ) + (\
                 2.16 * 10**(-8) * 10**(0.217*L + 0.461*Kp) )


        if full==True:
            return DLL_B, DLL_E
        else:
            return D

    elif method == 'L^6':
        
        # the D_0 factor here (taken from teh DELL formula at L=5 and Kp =4 )
        # containts the scaling to make this in units of days^{-1}
        D_0 = 2.16 * 10**(-8) * 10**(0.217*5 + 0.461*4)
        D = D_0*np.power(L,6)

        return D

    else:
        raise Error("Unknown diffusion coefficient method")

    
def Crank_Nicolson(dt,nt,dL,L,f,Dlist,Q,lbc=None,rbc=None,ltype='d',rtype='d',
    f_return=1,loss=None,plasmapause=5):
    '''
    Code for running the Modified Crank Nicolson numerical scheme for radial diffusion as in (Welling et al, 2012).
    Currently functionality requires Diffusion array to extend for one timestep beyond the final time
    (TO DO: Input temporal boundary condition at final timestep)
    
    A source term Q can be provided with the same shape as D. 
    
    TO DO: Input functionality for loss term.
    
    If no BC's are given we assume Dirichlet with constant left and right boundary.
    
    Boundary conditions allowed on left/right are Dirichlet ('d') and Neumann ('n'). Custom BC values for each timestep
    are allowed. Neumann BCs are calculated at order (dL)^2 (might put in functionality for lower order)
    
    Inputs:
        - dt: timestep (seconds)
        - nt: number of time steps
        - dL: space step
        - L: Grid in space
        - f: Initial conditions
        - D: Diffusion coefficient for each time step (list of nt diffusion coefficients)
        - Q: Source term to be applied in each timestep (list of nt source terms)
        - lbc: Left boundary condition (list of nt values for each time step)
        - rbc: Right boundary condition (list of nt values for each time step)
        - ltype: Type of left boundary condition. Default Dirichlet ('d') can also handle Neumann ('n')
        - rtype: Type of right boundary condition. Default Dirichlet ('d') can also handle Neumann ('n')
        - f_return: Which PSDs to include in seconds in result variable res
        - loss: What loss timescale model to use. 
        - plasmapause: where is plasmapause boundary. Default is L=5. (single
          value)
        
       Outputs:
        - Final PSD T
        - Array of PSD results at timesteps f_return
        
    
    '''
    T = f.copy()

    if lbc is None:
        lbc = [T[0]]*nt
    if rbc is None:
        rbc = [T[-1]]*nt
    
    
    lossterm = np.zeros(np.shape(T)) 
    if loss is not None:
        tau = LossRate(L,losstype=loss)

    s = (0.5*dt/dL**2)    
 
    res = []
    res.append(f)
    for n in range(nt):
    
        if loss is not None:
            lossterm = -dt*np.divide(T,tau)
            lossterm[L>=plasmapause] = 0
        

        D = Dlist[n]
        Dplus = Dlist[n+1]
      
        
        Dl = np.array([L[i]**2 * 0.5*((D[i] + D[i-1])/2 + (Dplus[i] + Dplus[i-1])/2)/(L[i]-0.5*dL)**2 \
               for i in range(1,len(L))])
        Dr = np.array([L[i]**2 * 0.5*((D[i] + D[i+1])/2 + (Dplus[i] + Dplus[i+1])/2)/(L[i]+0.5*dL)**2 \
               for i in range(len(L)-1)])

        Dc = np.array([x+y for x,y in zip(Dl[:-1],Dr[1:])])
        
        Qnew = (Q[n]+Q[n+1])/2
        
        if ltype=='d' and rtype=='d':
        
            A = diags([-s*Dl[1:], 1+s*Dc, -s*Dr[1:]], [-1, 0, 1], shape=(len(L)-2, len(L)-2)).toarray() 
            B1 = diags([s*Dl, 1-s*Dc, s*Dr[1:]], [0, 1, 2], shape=(len(L)-2, len(L))).toarray()
         
            #Input Dirichlet boundary for first row
            b = np.zeros(len(L)-2)
            b[0] = s*Dl[0] * lbc[n]
            b[-1] = s*Dr[-1] * rbc[n]
            B = np.add(np.add(np.dot(B1,T),b+ dt*(Qnew[1:-1])),lossterm[1:-1])
            T[1:-1] = np.linalg.solve(A,B)
            T[0] = lbc[n]
            T[-1] = rbc[n]
           
            
        elif ltype=='d' and rtype=='n':
            A = np.zeros([len(L)-1,len(L)-1])
            B1 = np.zeros([len(L)-1,len(L)])
            
            A[:-1,:] = diags([-s*Dl[1:], 1+s*Dc, -s*Dr[1:]], [-1, 0, 1], shape=(len(L)-2, len(L)-1)).toarray() 
            B1[:-1,:] = diags([s*Dl, 1-s*Dc, s*Dr[1:]], [0, 1, 2], shape=(len(L)-2, len(L))).toarray()
            
            A[-1,-1] = 1 + (2 * s * Dl[-1])
            A[-1,-2] = -2 * s * Dl[-1]
            B1[-1,-1] = 1 - (2 * s * Dl[-1])
            B1[-1,-2] = 2 * s * Dl[-1]
             
            b = np.zeros(len(L)-1)
            b[0] = s*Dl[0] * lbc[n]
            b[-1] = 4* s * Dl[-1] * rbc[n] * dL
            B = np.add(np.add(np.dot(B1,T),b+ dt*(Qnew[1:])),lossterm[1:])
            T[1:] = np.linalg.solve(A,B)
            T[0] = lbc[n]
          
            
        elif ltype=='n' and rtype=='d':
            A = np.zeros([len(L)-1,len(L)-1])
            B1 = np.zeros([len(L)-1,len(L)])
            
            A[1:,:] = diags([-s*Dl, 1+s*Dc, -s*Dr[1:]], [0, 1, 2], shape=(len(L)-2, len(L)-1)).toarray() 
            B1[1:,:] = diags([s*Dl, 1-s*Dc, s*Dr[1:]], [0, 1, 2], shape=(len(L)-2, len(L))).toarray()
            
            A[0,0] = 1 + (2 * s * Dr[0])
            A[0,1] = -2 * s * Dr[0]
            B1[0,0] = 1 - (2 * s * Dr[0])
            B1[0,1] = 2 * s * Dr[0]

            b = np.zeros(len(L)-1)
            b[-1] = s*Dr[0] * rbc[n]
            b[0] = -4* s * Dr[0] * lbc[n] * dL
            B = np.add(np.add(np.dot(B1,T),b+ dt*(Qnew[:-1])),lossterm[:-1])
            T[:-1] = np.linalg.solve(A,B)
            T[-1] = rbc[n]
           
            
        elif ltype=='n' and rtype=='n':
            A = np.zeros([len(L),len(L)])
            B1 = np.zeros([len(L),len(L)])
            
            A[1:-1,:] = diags([-s*Dl, 1+s*Dc, -s*Dr[1:]], [0, 1, 2], shape=(len(L)-2, len(L))).toarray() 
            B1[1:-1,:] = diags([s*Dl, 1-s*Dc, s*Dr[1:]], [0, 1, 2], shape=(len(L)-2, len(L))).toarray()
            
            A[0,0] = 1 + (2 * s * Dr[0])
            A[0,1] = -2 * s * Dr[0]
            B1[0,0] = 1 - (2 * s * Dr[0])
            B1[0,1] = 2 * s * Dr[0]
            
            A[-1,-1] = 1 + (2 * s * Dl[-1])
            A[-1,-2] = -2 * s * Dl[-1]
            B1[-1,-1] = 1 - (2 * s * Dl[-1])
            B1[-1,-2] = 2 * s * Dl[-1]
            
            b = np.zeros(len(L))
            b[-1] = 4* s * Dl[-1] * rbc[n] * dL
            b[0] = -4* s * Dr[0] * lbc[n] * dL
            B = np.add(np.add(np.dot(B1,T),b+ dt*(Qnew)),lossterm)
            T = np.linalg.solve(A,B)
        

        if n % f_return == 0:
            if n != 0:
                res.append(T.copy())


    return T,res


def PSD(L, A=9*10**4, B=0.05, mu=4, sig=0.38, gamma=5):
    """
    Function to calculate the initial phase space density profile over L-Shell
    space L.
    
    Inputs:
        L
        A
        B
        mu
        sigma
        gamma
    
    Outputs:
        f -- Initial Phase Space Density Profile
      
    """
    f = np.zeros(len(L))
    
    for i in range(len(L)):
        
        f[i] = A*np.exp(-(L[i]-mu)**2/(2*sig**2)) + 0.5*A*B*(math.erf(gamma*(L[i]-mu))+1)
    
    return f

#def experiment_temporal(nom):
    
#if __name__ == '__main__':
    
