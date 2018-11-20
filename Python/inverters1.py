#!/usr/bin/env python

import sys,os
import numpy as np
import heapq
import vegas
from scipy.special import jv, jn_zeros, yv
from scipy.optimize import fsolve
from scipy.integrate import quad, fixed_quad
import warnings
warnings.filterwarnings('ignore')

class AdOg:

    def ogata(self, f,h,N, nu):
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

    def trident(self,w,q,c,mid):
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

    def get_bc(self,w,Q,val=0,sign=0):
        q = Q
        c=2.0
        vallist=[w(1/c/q),w(1/q),w(c/q),val]
        half,mid,dub,value = vallist
        if max(vallist) == mid:
            bc = self.trident(w,q,c,mid)
        elif max(vallist) == dub and sign != -1:
            q = Q*c**3
            value = dub
            sig = 1
            bc = self.get_bc(w,q,val=value,sign = sig)
        elif max(vallist) == dub and sign == -1:
            bc = self.trident(w,2*Q,c,dub)
        elif max(vallist) == half and sign != 1:
            sig = -1
            q = Q*c**(-3)
            value = half
            bc = self.get_bc(w,q,val=value,sign=sig)
        elif max(vallist) == half and sign == 1:
            bc = self.trident(w,Q/2,c,half)
        elif max(vallist) == value:
            bc = self.trident(w,q*c**(2*sign),c,mid)
        return bc

    def get_ogata_params_b(self, w, bmin, bmax, qT, nu):
        zero1 = jn_zeros(nu, 1)[0]
        h = fsolve(lambda h: bmin-zero1/qT*np.tanh(np.pi/2*np.sinh(h/np.pi*zero1)), bmin)[0]
        k = fsolve(lambda k: bmax-np.pi/qT*k*np.tanh(np.pi/2*np.sinh(h*k)), bmax)[0]
        if k<0:
            k = fsolve(lambda k: bmax-np.pi/qT*k*np.tanh(np.pi/2*np.sinh(h*k)), -k)[0]
        N = int(k+1)
        return h,N

    def get_b_vals(self, bc, ib, it):
        ccut = 2.0
	b_min_half,b_min_dub=[bc*(ccut)**(-ib-1),bc*(ccut)**(it+1)]
        return b_min_half,b_min_dub

    def compare(self, w, b_min_half, b_max_dub,nu,storage,peak,bc,qT,ib,it):
        zero1 = jn_zeros(nu, 1)[0]
        get_psi=lambda t: t*np.tanh(np.pi/2*np.sinh(t))
        get_psip=lambda t:np.pi*t*(-np.tanh(np.pi*np.sinh(t)/2)**2 + 1)*np.cosh(t)/2 + np.tanh(np.pi*np.sinh(t)/2)
        try:
            top = stor[b_max_dub]
        except:
            top = w(b_max_dub)
            storage[b_max_dub] = top
        try:
            bot = stor[b_min_half]
        except:
            bot = w(b_min_half)
            storage[b_min_half] = bot
        if bot > peak:
            peakval = bot
            bcc = b_min_half
        elif top > peak:
            peakval = top
            bcc = b_max_dub
        else:
            peakval = peak
            bcc = bc
        if (b_min_half/np.pi/zero1**2*qT)> 0.005:
            return True,storage,peakval,bcc
        else:
            return (b_min_half/np.pi/zero1**2*qT)*bot>top,storage,peakval,bcc
        h0=b_min_half/np.pi/zero1**2
        n0=2.0**(ib+it-1)
        xin0=jn_zeros(nu, n0)[0]/np.pi
        val1=w(2*np.pi/h0*get_psi(h0/2*xin0))*jv(nu,2*np.pi/h0*get_psi(h0/2*xin0))*get_psip(2*np.pi/h0*get_psi(h0/2*xin0))
        val2=w(  np.pi/h0*get_psi(h0  *xin0))*jv(nu,  np.pi/h0*get_psi(h0  *xin0))*get_psip(  np.pi/h0*get_psi(h0  *xin0))
        return val1<val2,storage,peakval,bcc

    def get_h(self,nu,h,N):
        zeron = jn_zeros(nu,N)[0]
        zero2n= jn_zeros(nu,2*N)[0]
        return 1/zero2n*np.arcsinh(2/np.pi*np.arctanh(zeron/zero2n*np.tanh(np.pi/2*sinh(h*zeron))))

    def adog(self, w, qT, nu, Nmax, Q, ib = 1, it = 1,iterations=1,stor={},bc=0,peakval=0,h=0,N=0):
        storage = stor
        b_min_half, b_max_dub = self.get_b_vals(bcc, ib, it)
        cut_bool,storage1,peak1,bcc1=self.compare(w, b_min_half, b_max_dub,nu,storage,peak,bcc,qT,ib,it)
        if bc == 0:
            bcc=1/Q
            peak=w(1/Q)
            h1,N1=self.get_ogata_params_b(w, b_min_half*2, b_max_dub/2, qT, nu,h=h1,N=N1)
            if cut_bool == True:
                return self.adog(w, qT, nu, Nmax, Q, ib = ib+1, it = it+0,iterations=its,stor=storage1,bc=bcc1,peakval=peak1,h=h1,N=N1)
            elif cut_bool == False:
                return self.adog(w, qT, nu, Nmax, Q, ib = ib+0, it = it+1,iterations=its,stor=storage1,bc=bcc1,peakval=peak1,h=h1,N=N1)
        else:
            bcc = bc
            N1=2*N
            peak = peakval
            if 2*N1<Nmax:
                if cut_bool == True:
                    h1=self.get_h(nu,h,N)
                    return self.adog(w,qT,nu,Nmax,Q,ib=ib+1,it=it+0,iterations=its,stor=storage1,bc=bcc1,peakval=peak1,h=h1,N=N1)
                elif cut_bool == False:
                    h1=h
                    return self.adog(w,qT,nu,Nmax,Q,ib=ib+0,it=it+1,iterations=its,stor=storage1,bc=bcc1,peakval=peak1,h=h1,N=N1)
            else:
                result = self.ogata(lambda x: w(x/qT)/qT,h,N1, nu)
                return 1/(2*np.pi)*result[0],1/(2*np.pi)*result[1] 

class OGATA:

    def __init__(self,xmin,xmax,nu):

        zero1 = jn_zeros(nu, 1)[0]
        h = fsolve(lambda h: xmin-zero1*np.tanh(np.pi/2*np.sinh(h/np.pi*zero1)), xmin)[0]
        k = fsolve(lambda k: xmax-np.pi*k*np.tanh(np.pi/2*np.sinh(h*k)), xmax)[0]
        N = int(k)
        #print
        #print '\nogata N=',N
        #sys.exit()
        zeros=jn_zeros(nu,N)
        xi=zeros/np.pi
        Jp1=jv(nu+1,np.pi*xi)
        self.w=yv(nu, np.pi * xi) / Jp1
        get_psi=lambda t: t*np.tanh(np.pi/2*np.sinh(t))
        get_psip=lambda t:np.pi*t*(-np.tanh(np.pi*np.sinh(t)/2)**2 + 1)*np.cosh(t)/2 + np.tanh(np.pi*np.sinh(t)/2)
        self.knots=np.pi/h*get_psi(h*xi)
        self.Jnu=jv(nu,self.knots)
        self.psip=get_psip(h*xi)

    def invert(self,w, qT):
        F=w(self.knots/qT)/qT
        return 0.5*np.sum(self.w*F*self.Jnu*self.psip)

class Quad:
    def quadinv(self, w, q, nu, eps):
        quadreturn = quad(lambda bT: jv(nu,q*bT)*w(bT),0.0,5.0, epsabs = 0.0, epsrel = eps)
        return 1/(2*np.pi)*quadreturn[0], 1/(2*np.pi)*quadreturn[1]

class Fix_Quad:
    def fix_quadinv(self, w, q, nu, num):
        quadreturn = fixed_quad(lambda bT: jv(nu,q*bT)*w(bT),0.0,2.0, n = num)
        return 1/(2*np.pi)*quadreturn[0]#, 1/(2*np.pi)*quadreturn[1]

class Vegas:
    def transform(self, f, p):
        return f(np.tan(p))*(1/np.cos(p))**2
    def MCinv(self, f, q, nu, m):
        integ = vegas.Integrator([[0, np.pi/2.0]])
        result = integ(lambda p: 1/(2*np.pi)*self.transform(f, p)*jv(nu, q*np.tan(p)), nitn=10, neval=int(m))[0]
        lst = str(result).replace('+-', '(').replace(')', '(').split('(')
        num_zeros = 0
        if str(lst[0][0]) == '-':
            num_zeros = -1
        if len(lst[1]) == 3:
            integral = float(lst[0])
            error = float(lst[1])
        else:
            num_zeros = num_zeros+len(lst[0])-4
            lst[1] = '0.'+lst[1].zfill(num_zeros+2)
            integral = float(lst[0])
            error = float(lst[1])
        return integral, error



if __name__== "__main__":
    ogata = AdOg()
    adquad = Quad()
    fixquad=Fix_Quad()
    VEGAS=Vegas()
    print VEGAS.MCinv(lambda x: x*np.exp(-x**2), 1.0, 0, 100)
    print fixquad.fix_quadinv(lambda x: x*np.exp(-x**2), 1.0, 0, 100)
    print ogata.adog(lambda x: x*np.exp(-x**2), 1.0, 0, 10, 2.0)
    print adquad.quadinv(lambda x: x*np.exp(-x**2), 1.0, 0,1e-5)
