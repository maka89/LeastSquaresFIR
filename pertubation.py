import numpy as np
from utils import get_cb,calc_err
from cg_utils import toeplitz_solve_cg
from scipy.linalg import solve_toeplitz,matmul_toeplitz

def matmul_real_circulant(c,x):
    C= np.fft.rfft(c)
    X=np.fft.rfft(x)
    return np.fft.irfft(C*X,n=len(c))

def circ_to_toeplitz(c):
    col=np.copy(c)
    row = np.zeros_like(c)
    row[1::]=c[1::][::-1]
    row[0]=col[0]
    return col,row
def get_Delta(c):
    c,r = circ_to_toeplitz(c)
    r[0]=0.0
    r*=-1.0
    return r*0.0,r
    
def mult_P2(x,cr_c,cr_d):
    c_c,r_c = cr_c
    c_d,r_d = cr_d
    
    n=len(c_c)
    z = np.zeros(n)
    z[0:len(x)] = x
    
    z1 = matmul_toeplitz((c_d,r_d),z)
    z1 = matmul_toeplitz((r_c,c_c),z1)
    
    z2 = matmul_toeplitz((c_c,r_c),z) #matmul_real_circulant()
    z2 = matmul_toeplitz((r_d,c_d),z2)
    
    out = z1+z2
    return out[0:len(x)]

def mult_P3(x,cr_c,cr_d):
    c_d,r_d = cr_d
    
    n=len(c_d)
    z = np.zeros(n)
    z[0:len(x)] = x
    
    z1=matmul_toeplitz((c_d,r_d),z)
    z2=matmul_toeplitz((r_d,c_d),z1)
    return z2[0:len(x)]
def rh1(y,cr_d,m):
    c_d,r_d = cr_d
    z=matmul_toeplitz((r_d,c_d),y)
    return z[0:m]

def lsq_fir_linear_pert(x,y,imp_length,reg=0.0,order=2,check=True):
    
    assert(len(x)==len(y))
        
    c,b = get_cb(x,y,imp_length,reg)
    h0=solve_toeplitz( c, b)
    
    if check:
        err0=calc_err(x,y,h0,reg,linear=True)
    
    cr_c = circ_to_toeplitz(np.copy(x))
    cr_d = get_Delta(np.copy(x))
    htot=h0
    if order >= 1:
        b1 = rh1(y,cr_d,imp_length) - mult_P2(h0,cr_c,cr_d)
        h1 = solve_toeplitz(c,b1)
        htot += h1
        
        hm = np.copy(h1)
        hmm = np.copy(h0)
        for i in range(2,order+1):
            bn = - mult_P2(hm,cr_c,cr_d) - mult_P3(hmm,cr_c,cr_d)
            hnew= solve_toeplitz(c,bn)
            htot += hnew
            
            hmm = np.copy(hm)
            hm=np.copy(hnew)
    if check:
        errn=calc_err(x,y,htot,reg,linear=True)
        return htot,(errn<err0)
    else:
        return htot,None


def lsq_fir_linear_pert_cg(x,y,imp_length,reg=0.0,order=2,check=True,cg_xinit=None,cg_precond=1,cg_itmax=1000,cg_tol=1e-16,cg_disp=False,cg_atol=True):
    
    assert(len(x)==len(y))
        
    c,b = get_cb(x,y,imp_length,reg)
    h0,info=toeplitz_solve_cg( c, b, xinit=cg_xinit, precond=cg_precond,tol=cg_tol,it_max=cg_itmax,disp=cg_disp,atol=cg_atol)
    if check:
        err0=calc_err(x,y,h0,reg,linear=True)
        assert(info["iter"] < cg_itmax)
    
    cr_c = circ_to_toeplitz(np.copy(x))
    cr_d = get_Delta(np.copy(x))
    htot=h0
    if order >= 1:
        b1 = rh1(y,cr_d,imp_length) - mult_P2(h0,cr_c,cr_d)
        h1,info = toeplitz_solve_cg(c, b1, xinit=None, precond=cg_precond,tol=cg_tol,it_max=cg_itmax,disp=cg_disp,atol=cg_atol)
        if check:
            assert(info["iter"] < cg_itmax)
        htot += h1
        
        hm = np.copy(h1)
        hmm = np.copy(h0)
        for i in range(2,order+1):
            bn = - mult_P2(hm,cr_c,cr_d) - mult_P3(hmm,cr_c,cr_d)
            hnew, info = toeplitz_solve_cg(c, bn, xinit=None, precond=cg_precond,tol=cg_tol,it_max=cg_itmax,disp=cg_disp,atol=cg_atol)
            if check:
                assert(info["iter"] < cg_itmax)
            htot += hnew
            
            hmm=np.copy(hm)
            hm=np.copy(hnew)

    if check:
        errn=calc_err(x,y,htot,reg,linear=True)
        return htot,(errn<err0)
    else:
        return htot,None

        