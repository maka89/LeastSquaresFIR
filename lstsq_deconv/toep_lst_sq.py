## Preconditioners and some of the other code is copied from
## From "An Introduction to Iterative Toeplitz Solvers" by Raymond Hon-Fu Chan and Xiao-Qing Jin
##

import numpy as np
from .utils import get_cb

def Tdot(gev, reg, x, n):
    m=len(x)
    y = np.zeros(2*n)
    y[0:m]=x
    y = np.fft.irfft(np.fft.rfft(y)*gev[0],n=2*n)
    y[n::]=0.0#=y[0:n]
    y = np.fft.irfft(np.fft.rfft(y)*gev[1],n=2*n)
    y=y[0:m]
    y += reg*x
    return y.real
def get_b(gev,y,m):
    y2=np.zeros(2*len(y))
    y2[0:len(y)] = y
    Y = np.fft.rfft(y2)
    return np.fft.irfft(gev[1]*Y,n=2*len(y))[0:m]
def circ_solve(C,y):
    if C is None:
        return y
    else:
        y=np.fft.irfft(np.fft.rfft(y)/C,n=len(y))
        return y.real
    
def get_precond_ev_gev(t,r,method,implen):
    n=len(t)
    t1  = np.conj(r[1::][::-1])
    tmp = np.concatenate([t,np.array([0]),t1])
    gev  = [np.fft.rfft(tmp)]
    
    t1  = np.conj(t[1::][::-1])
    tmp =np.concatenate([r,np.array([0]),t1])
    gev.append(np.fft.rfft(tmp))
    
    description=""
    
    tmp = np.fft.rfft(t)
    t=  np.fft.irfft(tmp.conj()*tmp)[0:implen]
    n=len(t)
    t1  = np.conj(t[1::][::-1])
    if method==0:
        description="No Preconditioner"
        ev=None
        
    if method == 1:
        description="T.Chan's Preconditioner"
        coef =np.linspace(1.0/n,1.0-1/n,n-1)
        x= np.concatenate([ t[[0]], (1.0-coef)*t[1::]+coef*t1])
        ev = np.fft.rfft(x)#.real

    if method == 2:
        description="Strang's Preconditioner"
        
        m=n//2
        x=np.concatenate([t[0:m], np.array([0]), t[1:m][::-1]])
        ev = np.fft.rfft(x)#.real
    
    if method == 3:
        description="R.Chan's Preconditioner"
        x = np.concatenate([t[[0]],t[1::]+t1])
        ev = np.fft.rfft(x)#.real
    
    
    if method == 12:
        description="Superoptimal Preconditioner"
        h = np.zeros(n) # h: first column of the circulant part
        s = np.zeros(n) # s: first column of the skew-circulant part
        h[0] = 0.5*t[0]
        s[0] = h[0]
        
        h[1::] = 0.5*(t[1::] + t1);
        s[1::] = t[1::]-h[1::];
        ev1 = np.fft.rfft(h)
        coef = np.linspace(1.0,-1.0+2.0/n,n)
        
        c = coef*s
        # first column of T. Chan’s preconditioner
        # for the skew-circulant part
        ev2 = np.fft.rfft(c);
        
        tmp=np.linspace(0,1.0-1.0/n,n)
        
        d = np.exp(tmp)#(0:1/n:1-1/n)*pi*i)
        s = s*d;
        sev = np.fft.rfft(s) # % eigenvalues of the skew-circulant part
        sev = sev*np.conj(sev)
        s = np.fft.irfft(s,n=n)
        s = np.conj(d)*s;
        h = coef*s;
        ev3 = np.fft.rfft(h)
        ev = (np.abs(ev1)**2 + 2*ev1*ev2+ev3)/ev1
        
    return ev,gev,description

    
    
def pcg(gev,ev,b,ig,reg,tol,it_max,n,disp=False,atol=False):

    x=ig
    r=b-Tdot(gev,reg,x,n)
    
    err0 = np.linalg.norm(r)
    err=err0
    
    z=circ_solve(ev,r)
    p=z
    k=0
    
    if atol:
        tol /=err0
    if disp:
        print("\n at step {0:}, residual={1:}".format(k,err))
        
    while True:
        Ap = Tdot(gev,reg,p,n)
        fac = np.dot(r,z)
        alpha = fac/np.dot(p,Ap)
        x=x+alpha*p
        r = r-alpha*Ap
        
        err = np.linalg.norm(r)
        if err/err0 < tol or k >= it_max:
            break
        
        z=circ_solve(ev,r)
        beta = np.dot(r,z)/fac
        p=z+beta*p
        k+=1
        
        if disp:
            print("\n at step {0:}, relative residual = {1:}".format(k,err/err0))
    if k >= it_max and disp:
        print("\n Maximum iterations reached. relative residual={0:}".format(err/err0))
    info = {"iter":k,"relative_error":err/err0, "error":err}
    
    return x,info
 

def lstsquares_toep_cg(c_r,y,xinit=None,imp_length=None,reg=0.0,precond=None,tol=1e-4,it_max=1000,disp=False,atol=True):
    if imp_length is None and xinit is not None:
        imp_length = len(xinit)
    if xinit is None and imp_length is not None:
        xinit = np.zeros(imp_length)
    if imp_length is None and xinit is None:
        imp_length = len(y)
        xinit = np.zeros(imp_length)
        
    
    if precond is None:
        precond = 0
        if len(y)//imp_length >= 3:
            precond =1
    
    ev,gev,precond_descrition = get_precond_ev_gev(c_r[0],c_r[1],precond,imp_length)
    b=get_b(gev,y,imp_length)
    x,info=pcg(gev,ev,b,xinit,reg,tol,it_max,len(y),disp=disp,atol=atol)
    info["Preconditioner"]=precond_descrition
    return x,info