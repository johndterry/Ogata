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

    def get_h(self,nu,h,N):
        zeron = jn_zeros(nu,N)[-1]/np.pi
        zero2n= jn_zeros(nu,2*N)[-1]/np.pi
        #print zeron,zero2n
        return 1.0/zero2n*np.arcsinh(2.0/np.pi*np.arctanh(zeron/zero2n*np.tanh(np.pi/2.0*np.sinh(h*zeron))))

    def get_b_vals(self, bc, ib, it):
        ccut = 2.0
	b_min_half,b_min_dub=[bc*(ccut)**(-ib-1),bc*(ccut)**(it+1)]
        return b_min_half,b_min_dub

    def compare(self, w, b_min_half, b_max_dub,nu,storage,peak,bc,qT,ib,it,h,N):
        #print h,N
        peakval=peak
        bcc=bc
        if h> 0.01:
            return True,storage,peakval,bcc
        else:
            zero1 = jn_zeros(nu, 1)[0]
            get_psi=lambda t: t*np.tanh(np.pi/2*np.sinh(t))
            get_psip=lambda t:np.pi*t*(-np.tanh(np.pi*np.sinh(t)/2)**2 + 1)*np.cosh(t)/2 + np.tanh(np.pi*np.sinh(t)/2)
            #try:
            #    top = stor[b_max_dub]
            #except:
            #    top = w(b_max_dub)
            #    storage[b_max_dub] = top
            #try:
            #    bot = stor[b_min_half]
            #except:
            #    bot = w(b_min_half)
            #    storage[b_min_half] = bot
            #if bot > peak:
            #    peakval = bot
            #    bcc = b_min_half
            #elif top > peak:
            #    peakval = top
            #    bcc = b_max_dub
            #print  h,abs(np.exp(-1/h)*bot),abs(top*jv(nu,np.pi/h*get_psi(h/np.pi*jn_zeros(nu, N)[-1]))),np.exp(-1/h)*bot>top*jv(nu,np.pi/h*get_psi(h/np.pi*jn_zeros(nu, N)[-1]))
            h1=self.get_h(nu,h,N)
            bot1=w(np.pi/h1*get_psi(h1/np.pi*jn_zeros(nu, N)[0])/qT)
            bot2=w(np.pi/h *get_psi(h /np.pi*jn_zeros(nu, N)[0])/qT)
            top1=w(np.pi/h1*get_psi(h1/np.pi*jn_zeros(nu, N)[-1])/qT)
            top2=w(np.pi/h *get_psi(h /np.pi*jn_zeros(nu, N)[-1])/qT)
            #bot=w(np.pi/h1*get_psi(h1/np.pi*jn_zeros(nu, N)[0])/qT)
            val1=abs(h1*np.exp(-2*np.pi**2/h1)*bot1+top1*jv(nu,np.pi/h1*get_psi(h1/np.pi*jn_zeros(nu, N)[-1])))
            val2=abs(h *np.exp(-2*np.pi**2/h )*bot2+top2*jv(nu,np.pi/h *get_psi(h /np.pi*jn_zeros(nu, N)[-1])))
            #print 'adog1', val1,val2,val1<val2
            return val1<val2,storage,peakval,bcc
        #h0=b_min_half/np.pi/zero1**2
        #n0=2.0**(ib+it-1)
        #xin0=jn_zeros(nu, n0)[0]/np.pi
        #return (b_min_half/np.pi/zero1**2)**2*bot>top,storage,peakval,bcc
        #val1=w(2*np.pi/h0*get_psi(h0/2*xin0))*jv(nu,2*np.pi/h0*get_psi(h0/2*xin0))*get_psip(2*np.pi/h0*get_psi(h0/2*xin0))
        #val2=w(  np.pi/h0*get_psi(h0  *xin0))*jv(nu,  np.pi/h0*get_psi(h0  *xin0))*get_psip(  np.pi/h0*get_psi(h0  *xin0))
        #return val1<val2,storage,peakval,bcc

    def adog(self, w, qT, nu, Nmax, Q, ib = 1, it = 1,iterations=1,stor={},bc=0,peakval=0,h=0,N=0):
        storage = stor
        if bc == 0:
            bcc=1/Q
            peak=w(bcc)
            b_min_half, b_max_dub = self.get_b_vals(bcc, ib, it)
            h1,N1=self.get_ogata_params_b(w, b_min_half*2, b_max_dub/2, qT, nu)
            cut_bool,storage1,peak1,bcc1=self.compare(w, b_min_half, b_max_dub,nu,storage,peak,bcc,qT,ib,it,h1,N1)
            h2=self.get_h(nu,h1,2*N1)
            if cut_bool == True:
                return self.adog(w, qT, nu, Nmax, Q, ib = ib+1, it = it+0,iterations=1,stor=storage1,bc=bcc1,peakval=peak1,h=h1,N=2*N1)
            elif cut_bool == False:
                return self.adog(w, qT, nu, Nmax, Q, ib = ib+0, it = it+1,iterations=1,stor=storage1,bc=bcc1,peakval=peak1,h=h1,N=2*N1)
        else:
            bcc = bc
            b_min_half, b_max_dub = self.get_b_vals(bcc, ib, it)
            peak = peakval
            cut_bool,storage1,peak1,bcc1=self.compare(w, b_min_half, b_max_dub,nu,storage,peak,bcc,qT,ib,it,h,N)
            #print cut_bool, h,N
            N1=2*N
            if 2*N1<Nmax:
                if cut_bool == True:
                    h1=self.get_h(nu,h,N)
                    return self.adog(w,qT,nu,Nmax,Q,ib=ib+1,it=it+0,iterations=1,stor=storage1,bc=bcc1,peakval=peak1,h=h1,N=N1)
                elif cut_bool == False:
                    h1=h
                    return self.adog(w,qT,nu,Nmax,Q,ib=ib+0,it=it+1,iterations=1,stor=storage1,bc=bcc1,peakval=peak1,h=h1,N=N1)
            else:
                #print 'adog1', h,N1
                result = self.ogata(lambda x: w(x/qT)/qT,h/2,4*N1, nu)
                return 1/(2*np.pi)*result[0],1/(2*np.pi)*result[1] 

    def adog2(self, w, qT, nu, Nmax, Q, ib = 1, it = 1,iterations=1,stor={},bc=0,peakval=0):
        print Nmax
        if bc == 0:
            bcc=1/Q
            peak=w(1/Q)
        else:
            bcc = bc
            peak = peakval      
        storage = stor
        if iterations == 1:  its=int(round(1.0/np.log(1.5)*np.log(Nmax/2.0)))
        else: its=iterations
        b_min_half, b_max_dub = self.get_b_vals(bcc, ib, it)
        if ib+it-2<iterations:
            cut_bool,storage1,peak1,bcc1=self.compare(w, b_min_half, b_max_dub,nu,storage,peak,bcc,qT,ib,it)
            if cut_bool == True:
                return self.adog(w, qT, nu, Nmax, Q, ib = ib+1, it = it+0,iterations=its,stor=storage1,bc=bcc1,peakval=peak1)
            elif cut_bool == False:
                return self.adog(w, qT, nu, Nmax, Q, ib = ib+0, it = it+1,iterations=its,stor=storage1,bc=bcc1,peakval=peak1)
        else:
            h,N=self.get_ogata_params_b(w, b_min_half*2, b_max_dub/2, qT, nu)
            #print h,N
            result = self.ogata(lambda x: w(x/qT)/qT,h,N, nu)
            return 1/(2*np.pi)*result[0],1/(2*np.pi)*result[1] 

    def get_psi(self,t):
        return t*np.tanh(np.pi/2*np.sinh(t))

    def get_psip(self,t):
        return np.pi*t*(-np.tanh(np.pi*np.sinh(t)/2)**2 + 1)*np.cosh(t)/2 + np.tanh(np.pi*np.sinh(t)/2)

    def compare3(self,w,nu,qT,h,N,h1):
        #h1= self.get_h(nu,h,N)
        #print h,N
        xbot1=np.pi/h1*self.get_psi(h1/np.pi*jn_zeros(nu, N)[0])/qT
        xbot2=np.pi/h *self.get_psi(h /np.pi*jn_zeros(nu, N)[0])/qT
        bot1=w(xbot1)
        bot2=w(xbot2)
        xtop1=np.pi/h1*self.get_psi(h1/np.pi*jn_zeros(nu, N)[-1])/qT
        xtop2=np.pi/h *self.get_psi(h /np.pi*jn_zeros(nu, N)[-1])/qT
        #print jv(nu,xtop1*qT),jv(nu,xtop2*qT)
        #print h1*jn_zeros(nu, 1000)[-1]#h1,np.pi/h1*self.get_psi(h1/np.pi*jn_zeros(nu, 1000)[-1])
        top1=w(xtop1)
        top2=w(xtop2)
        ##print h
        #val1=np.exp(-np.pi/h)*w(xbot2)#*jv(nu,xbot2*qT)
        #val2=  w(xtop2)*jv(nu,xbot2*qT)
        val1=abs(top1*jv(nu,xtop1*qT))+abs(np.exp(-np.pi**2/2/h1)*bot1)
        val2=abs(top2*jv(nu,xtop2*qT))+abs(np.exp(-np.pi**2/2/h)*bot2)
        print 'adog3', abs(top1*jv(nu,xtop1*qT)),abs(top2*jv(nu,xtop2*qT)),abs(np.exp(-np.pi**2/2/h1)),abs(np.exp(-np.pi**2/2/h))
        #print val1>val2
        #print val1<val2
        return val1<val2
    
    def adog3(self,w, qT, nu, Nmax, Q):
        h1,N1=self.get_ogata_params_b(w, 1.0/2.0/Q, 2.0/Q, qT, nu)
        its = np.log(float(Nmax)/N1)/np.log(2.0)
        i=0
        N3=N1
        h4=h1
        while i<its-1:
            #print h4
            h1 = self.get_h(nu,h4,N3)
            N3=2*N3
            if h4>0.05:
                h4=h1
            else:
                if h4*N3<2.0:
                    cut_bol=self.compare3(w,nu,qT,h4,N3,h1)
                    #print cut_bol
                    bol= int(cut_bol == True)
                    h4  = np.heaviside(bol-0.5,0.5)*h1+np.heaviside(0.5-bol,0.5)*h4
            #h1 = self.get_h(nu,h4,N3)
            #bol=self.compare3(w,nu,qT,h4,N3,h1)
            #h4  = np.heaviside(bol-0.5,0.5)*h1+np.heaviside(0.5-bol,0.5)*h4
            #if h4*N3>1.0:
            #    h1 = self.get_h(nu,h4,N3)
            #    cut_bol=self.compare3(w,nu,qT,h4,N3,h1)
            #    bol= int(cut_bol == True)
            #    h4  = np.heaviside(bol-0.5,0.5)*h4+np.heaviside(0.5-bol,0.5)*h1
            #else:
            #    if h4>0.05:
            #        h4 = self.get_h(nu,h4,N3)
            i+=1
        result =  self.ogata(lambda x: w(x/qT)/qT,h4,N3, nu)
        return 1/(2*np.pi)*result[0],1/(2*np.pi)*result[1]

#    def adog(self, w, qT, nu, Nmax, Q, ib = 1, it = 1):
#        bc = 1/Q
#        b_min, b_max, b_min_half, b_max_half, b_min_dub, b_max_dub = self.get_b_vals(bc, ib, it)
#        napprox = b_max_dub*qT
#        print qT,napprox
#        if napprox<Nmax:
#            cut_bool = self.compare(w, b_min_half, b_max_dub)
#            if cut_bool == True:
#                return self.adog(w, qT, nu, Nmax, Q, ib = ib+1, it = it+0)
#            elif cut_bool == False:
#                return self.adog(w, qT, nu, Nmax, Q, ib = ib+0, it = it+1)
#        else:
#            h,N=self.get_ogata_params_b(w, b_min, b_max, qT, nu)
#            print N
#            return 1/(2*np.pi)*self.ogata(lambda x: w(x/qT)/qT,h,N, nu)

#    def get_h(self,w,Q,epsilon,nu,qT):
#        i=1
#        c=2.0
#        peakval=w(1/Q)
#        bot=w(1/c/Q)
#        while bot>epsilon*peakval:
#             i = i+1
#             bot=w(1/c**i/Q)
#             #print 'rat is', epsilon*peakval/bot
#        bmin=1/(c**i*Q)
#        print 'peakval is', peakval,bot,epsilon
#        zero1 = jn_zeros(nu, 1)[0]
#        h = fsolve(lambda h: bmin-zero1/qT*np.tanh(np.pi/2*np.sinh(h/np.pi*zero1)), bmin/np.pi/zero1**2)[0]
#        return h,bot

#    def get_N(self,w,Q,epsilon,nu,qT,h,bot):
#        i=0
#        c=2.0
#        top=w(c/Q)
#        while top>np.exp(-np.pi**2/2/h)*bot:
#             i = i+1
#             top=w(c**i*Q)
#        bmax=c**i*Q
#        print 'bmax/bc is', bmax
#        N = int(abs(fsolve(lambda n: bmax-np.pi*n/qT*np.tanh(np.pi/2*np.sinh(h*n)), bmax)[0]))
#        return N

#    def adog2(self,w,qT,nu,Q,epsilon=0.1,ib = 1,it = 1):
#        bc = 1/Q
#        h,bot = self.get_h(w,Q,epsilon,nu,qT)
#        N=self.get_N(w,Q,epsilon,nu,qT,h,bot)
#        print h,N
#        return 1/(2*np.pi)*self.ogata(lambda x: w(x/qT)/qT,h,N, nu)

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
    og = AdOg()
    import scipy.special as spec
    #def adog(w, qT, nu, Nmax, Q):
    #    return ogata.adog3(w, qT, nu, Nmax, Q)

    def Wtilde(bT,Q,sigma):
        M=1.0/Q
        V=1/sigma**2
        b=0.5*(-M+np.sqrt(M**2+4*V))
        a=V/b**2
        return bT**(a-1)*np.exp(-bT/b)/b**a/spec.gamma(a)

    Q=50.0
    qT=50.0
    sigma=0.8
    w=lambda b: Wtilde(b,Q,sigma)
    nu=0
    Nmax=20
    def W(qT, Q, sigma, nu):
      M=1.0/Q
      V=1/sigma**2
      b=0.5*(-M+np.sqrt(M**2+4*V))
      a=V/b**2
      return 1/(2*np.pi)*spec.gamma(a+nu)/spec.gamma(a)*(b*qT/2.0)**nu*spec.hyp2f1((a+nu)/2.0, (a+nu+1.0)/2.0, nu+1.0, -qT**2.0*b**2.0)/spec.gamma(nu+1.0)
    print abs((W(qT, Q, sigma, nu)-og.adog(w, qT, nu, Nmax, Q)[0])/W(qT, Q, sigma, nu))
    print abs((W(qT, Q, sigma, nu)-og.adog3(w, qT, nu, Nmax, Q)[0])/W(qT, Q, sigma, nu))
