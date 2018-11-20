#!/usr/bin/env python

import scipy.special as spec
import sys,os
import numpy as np
import heapq
import vegas
from scipy.special import jv, jn_zeros, yv
from scipy.optimize import fsolve
from scipy.integrate import quad, fixed_quad
import warnings
warnings.filterwarnings('ignore')

def ogata(f,h,N, nu):
    zeros=jn_zeros(nu,N)
    xi=zeros/np.pi
    Jp1=jv(nu+1,np.pi*xi)
    w=yv(nu, np.pi * xi) / Jp1
    get_psi=lambda t: t*np.tanh(np.pi/2*np.sinh(t))
    get_psip=lambda t:np.pi*t*(-np.tanh(np.pi*np.sinh(t)/2)**2 + 1)*np.cosh(t)/2 + np.tanh(np.pi*np.sinh(t)/2)
    knots=np.pi/h*get_psi(h*xi)
    Jnu=jv(nu,knots)
    psip=get_psip(h*xi)
    F=f(knots)
    psip[np.isnan(psip)]=1.0
    val=np.pi*np.sum(w*F*Jnu*psip)
    #print w,F,Jnu
    return val,np.pi*w[-1]*F[-1]*Jnu[-1]*psip[-1]

def trident(w,q,c,mid):
    fracup = (c+1.0)/2.0/q
    fracdown = (c+1.0)/2.0/q/c
    vallist = [w(fracdown),mid,w(fracup)]
    down,mid,up = vallist
    if max(vallist) == mid:
        bc = 1/q
    elif max(vallist) == up:
        bc = fracup
    elif max(vallist) == down:
        bc = fracdown
    return bc

def get_bc(w,Q,val=0,sign=0):
    q = Q
    c=2.0
    vallist=[w(1/c/q),w(1/q),w(c/q),val]
    half,mid,dub,value = vallist
    if max(vallist) == mid:
        bc = trident(w,q,c,mid)
    elif max(vallist) == dub and sign != -1:
        q = Q*c**3
        value = dub
        sig = 1
        bc =get_bc(w,q,val=value,sign = sig)
    elif max(vallist) == dub and sign == -1:
        bc = trident(w,2*Q,c,dub)
    elif max(vallist) == half and sign != 1:
        sig = -1
        q = Q*c**(-3)
        value = half
        bc = get_bc(w,q,val=value,sign=sig)
    elif max(vallist) == half and sign == 1:
        bc = trident(w,Q/2,c,half)
    elif max(vallist) == value:
        bc = trident(w,q*c**(2*sign),c,mid)
    return bc

def get_ogata_params_b(w, bmin, bmax, qT, nu):
    zero1 = jn_zeros(nu, 1)[0]
    h = fsolve(lambda h: bmin-zero1/qT*np.tanh(np.pi/2*np.sinh(h/np.pi*zero1)), bmin)[0]
    k = fsolve(lambda k: bmax-np.pi/qT*k*np.tanh(np.pi/2*np.sinh(h*k)), bmax)[0]
    if k<0:
        k = fsolve(lambda k: bmax-np.pi/qT*k*np.tanh(np.pi/2*np.sinh(h*k)), -k)[0]
    N = int(k+1)
    return h,N

def get_h(nu,h,N):
    zeron = jn_zeros(nu,N)[-1]/np.pi
    zero2n= jn_zeros(nu,2*N)[-1]/np.pi
    #print zeron,zero2n
    return 1.0/zero2n*np.arcsinh(2.0/np.pi*np.arctanh(zeron/zero2n*np.tanh(np.pi/2.0*np.sinh(h*zeron))))

def get_b_vals(bc, ib, it):
    ccut = 2.0
    b_min_half,b_min_dub=[bc*(ccut)**(-ib-1),bc*(ccut)**(it+1)]
    return b_min_half,b_min_dub

def get_psi(t):
    return t*np.tanh(np.pi/2*np.sinh(t))

def get_psip(t):
    return np.pi*t*(-np.tanh(np.pi*np.sinh(t)/2)**2 + 1)*np.cosh(t)/2 + np.tanh(np.pi*np.sinh(t)/2)

def compare(w, b_min_half, b_max_dub,nu,storage,peak,bc,qT,ib,it,h,N):
    peakval=peak
    bcc=bc
    if h> 0.01:
        return True,storage,peakval,bcc
    else:
        zero1 = jn_zeros(nu, 1)[0]
        h1= get_h(nu,h,N)
        bot1=w(np.pi/h1*get_psi(h1/np.pi*jn_zeros(nu, N)[0])/qT)
        bot2=w(np.pi/h *get_psi(h /np.pi*jn_zeros(nu, N)[0])/qT)
        top1=w(np.pi/h1*get_psi(h1/np.pi*jn_zeros(nu, N)[0])/qT)
        top2=w(np.pi/h *get_psi(h /np.pi*jn_zeros(nu, N)[0])/qT)
        bot=w(np.pi/h1*get_psi(h1/np.pi*jn_zeros(nu, N)[0])/qT)
        val1=abs(h1*np.exp(-2*np.pi**2/h1)*bot1+top1*jv(nu,np.pi/h1*get_psi(h1/np.pi*jn_zeros(nu, N)[-1])))
        val2=abs(h *np.exp(-2*np.pi**2/h )*bot2+top2*jv(nu,np.pi/h *get_psi(h /np.pi*jn_zeros(nu, N)[-1])))
        return val1<val2,storage,peakval,bcc

def adog(w, qT, nu, Nmax, Q, ib = 1, it = 1,iterations=1,stor={},bc=0,peakval=0,h=0,N=0):
    storage = stor
    if bc == 0:
        bcc=1/Q
        peak=w(bcc)
        b_min_half, b_max_dub =  get_b_vals(bcc, ib, it)
        h1,N1=get_ogata_params_b(w, b_min_half*2, b_max_dub/2, qT, nu)
        cut_bool,storage1,peak1,bcc1= compare(w, b_min_half, b_max_dub,nu,storage,peak,bcc,qT,ib,it,h1,N1)
        if cut_bool == True:
            return  adog(w, qT, nu, Nmax, Q, ib = ib+1, it = it+0,iterations=1,stor=storage1,bc=bcc1,peakval=peak1,h=h1,N=N1)
        elif cut_bool == False:
            return  adog(w, qT, nu, Nmax, Q, ib = ib+0, it = it+1,iterations=1,stor=storage1,bc=bcc1,peakval=peak1,h=h1,N=N1)
    else:
        bcc = bc
        b_min_half, b_max_dub =  get_b_vals(bcc, ib, it)
        peak = peakval
        cut_bool,storage1,peak1,bcc1=compare(w, b_min_half, b_max_dub,nu,storage,peak,bcc,qT,ib,it,h,N)
        #print cut_bool
        N1=2*N
        if 2*N1<Nmax:
            if cut_bool == True:
                h1= get_h(nu,h,N)
                return  adog(w,qT,nu,Nmax,Q,ib=ib+1,it=it+0,iterations=1,stor=storage1,bc=bcc1,peakval=peak1,h=h1,N=N1)
            elif cut_bool == False:
                h1=h
                return  adog(w,qT,nu,Nmax,Q,ib=ib+0,it=it+1,iterations=1,stor=storage1,bc=bcc1,peakval=peak1,h=h1,N=N1)
        else:
            result =  ogata(lambda x: w(x/qT)/qT,h/2,4*N1, nu)
            return 1/(2*np.pi)*result[0],1/(2*np.pi)*result[1] 

def compare3(w,nu,qT,h,N):
    h1= get_h(nu,h,N)
    bot1=w(np.pi/h1*get_psi(h1/np.pi*jn_zeros(nu, N)[0])/qT)
    bot2=w(np.pi/h *get_psi(h /np.pi*jn_zeros(nu, N)[0])/qT)
    top1=w(np.pi/h1*get_psi(h1/np.pi*jn_zeros(nu, N)[0])/qT)
    top2=w(np.pi/h *get_psi(h /np.pi*jn_zeros(nu, N)[0])/qT)
    #bot=w(np.pi/h1*get_psi(h1/np.pi*jn_zeros(nu, N)[0])/qT)
    val1=abs(h1*bot1+top1*jv(nu,np.pi/h1*get_psi(h1/np.pi*jn_zeros(nu, N)[-1])))
    val2=abs(h *bot2+top2*jv(nu,np.pi/h *get_psi(h /np.pi*jn_zeros(nu, N)[-1])))
    return val1<val2

#np.exp(-2*np.pi**2/h1)*
#np.exp(-2*np.pi**2/h )

def adog3(w, qT, nu, Nmax, Q):
    h1,N1=get_ogata_params_b(w, 1/2.0/Q, 2/Q, qT, nu)
    h2=np.heaviside(0.05-h1,0.5)*h1+np.heaviside(h1-0.05,0.5)*0.05
    its = np.log(Nmax/N1)/np.log(2)
    print its
    i=0
    N3=N1
    h4=h2
    while i<its:
        #print h4,N3
        cut_bool = compare3(w,nu,qT,h4,N3)
        bol = int(cut_bool == True)
        #print bol
        N3  = 2*N3
        h3  = get_h(nu,h2,N3)
        h4  = np.heaviside(bol-0.5,0.5)*h3+np.heaviside(0.5-bol,0.5)*h2
        i+=1
        #print N3
    result =  ogata(lambda x: w(x/qT)/qT,h4/32,2*N3, nu)
    return 1/(2*np.pi)*result[0],1/(2*np.pi)*result[1] 



def Wtilde(bT,Q,sigma):
    M=1.0/Q
    V=1/sigma**2
    b=0.5*(-M+np.sqrt(M**2+4*V))
    a=V/b**2
    return bT**(a-1)*np.exp(-bT/b)/b**a/spec.gamma(a)

Q=10.0
qT=10.0
sigma=0.8
w=lambda b: Wtilde(b,Q,sigma)
nu=0
Nmax=2000
def W(qT, Q, sigma, nu):
    M=1.0/Q
    V=1/sigma**2
    b=0.5*(-M+np.sqrt(M**2+4*V))
    a=V/b**2
    return 1/(2*np.pi)*spec.gamma(a+nu)/spec.gamma(a)*(b*qT/2.0)**nu*spec.hyp2f1((a+nu)/2.0, (a+nu+1.0)/2.0, nu+1.0, -qT**2.0*b**2.0)/spec.gamma(nu+1.0)
print adog3(w, qT, nu, Nmax, Q)[0]/W(qT, Q, sigma, nu)
