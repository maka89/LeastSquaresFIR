## Preconditioners and some of the other code is copied from
## From "An Introduction to Iterative Toeplitz Solvers" by Raymond Hon-Fu Chan and Xiao-Qing Jin
##

import numpy as np
from utils import get_cb

def Tdot(gev,x):
    n=len(x)
    y = np.zeros(2*n)
    y[0:n]=x
    y = np.fft.irfft(np.fft.rfft(y)*gev,n=2*n)
    y=y[0:n]
    
    return y.real
def circ_solve(C,y):
    y=np.fft.irfft(np.fft.rfft(y)/C,n=len(y))
    return y.real
    
def get_precond_ev_gev(t,method):
    n=len(t)
    t1  = np.conj(t[1::][::-1])
    tmp =np.concatenate([t,np.array([0]),t1])
    gev  = np.fft.rfft(tmp)
    
    description=""
    
    if method==0:
        description="No Preconditioner"
        x=np.zeros(n)
        x[0]=1.0
        ev=np.fft.rfft(x)
        
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

    
    
def pcg(gev,ev,b,ig,tol,it_max,disp=False,atol=False):

    x=ig
    r=b-Tdot(gev,x)
    
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
        Ap = Tdot(gev,p)
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
 

def toeplitz_solve_cg(c,b,xinit=None,precond=1,tol=1e-4,it_max=1000,disp=False,atol=True):
    if xinit is None:
        xinit = b*0.0
    ev,gev,precond_descrition = get_precond_ev_gev(c,precond)
    x,info=pcg(gev,ev,b,xinit,tol,it_max,disp=disp,atol=atol)
    info["Preconditioner"]=precond_descrition
    return x,info
    
def least_squares_fir_cg(x,y,impulse_length,reg=0.0,xinit=None,precond=1,tol=1e-4,it_max=1000,disp=False,atol=True):
    assert(len(x)==len(y))
    c,b = get_cb(x,y,impulse_length,reg)
    return toeplitz_solve_cg(c,b,xinit=xinit,precond=precond,tol=tol,it_max=it_max,disp=disp,atol=atol)
    
