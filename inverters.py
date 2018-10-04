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
        return np.pi*np.sum(w*F*Jnu*psip)

    def get_ogata_params_b(self, w, bmin, bmax, qT, nu):
        zero1 = jn_zeros(nu, 1)[0]
        h = fsolve(lambda h: bmin-zero1/qT*np.tanh(np.pi/2*np.sinh(h/np.pi*zero1)), bmin)[0]
        k = fsolve(lambda k: bmax-np.pi/qT*k*np.tanh(np.pi/2*np.sinh(h*k)), bmax)[0]
        if k<0:
            k = fsolve(lambda k: bmax-np.pi/qT*k*np.tanh(np.pi/2*np.sinh(h*k)), -k)[0]
        N = int(k+1)
        return h,N

    def get_b_vals(self, bc, ib, it):
        ccut = 4.0
	bmin,bmin1,bmin2=[bc/((ccut)**ib), bc/((ccut)**ib), bc/((ccut)**ib)/ccut]
	bmax,bmax1,bmax2=[bc*ccut**it, bc*ccut**it*ccut, bc*ccut**it]
        return bmin, bmax, bmin1, bmax1, bmin2, bmax2

    def compare(self, w, b_min_half, b_max_dub):
        return w(b_min_half)>w(b_max_dub)

    def adog(self, w, qT, nu, Nmax, Q, ib = 1, it = 1,iterations=1):
        bc = 1/Q
        if iterations == 1:  its=int(round(1.0/np.log(2)*np.log(Nmax/1.0)))
        else: its=iterations
        b_min, b_max, b_min_half, b_max_half, b_min_dub, b_max_dub = self.get_b_vals(bc, ib, it)
        if ib+it-2<iterations:
            cut_bool = self.compare(w, b_min_half, b_max_dub)
            if cut_bool == True:
                return self.adog(w, qT, nu, Nmax, Q, ib = ib+1, it = it+0,iterations=its)
            elif cut_bool == False:
                return self.adog(w, qT, nu, Nmax, Q, ib = ib+0, it = it+1,iterations=its)
        else:
            h,N=self.get_ogata_params_b(w, 2*b_min, b_max/2, qT, nu)
            return 1/(2*np.pi)*self.ogata(lambda x: w(x/qT)/qT,h,N, nu)

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
