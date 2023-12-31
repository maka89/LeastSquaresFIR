import numpy as np
from scipy.linalg import solve_toeplitz

    
def fwd(x,h,n=None,linear=False):

    if linear:
        k = n+len(h)
        h2 = np.zeros(k)
        h2[0:len(h)]=h
        x2=np.zeros(k)
        x2[0:n]=x
        
        X=np.fft.rfft(x2)
        H=np.fft.rfft(h2)
        return np.fft.irfft(X*H,n=k)[0:n]
    
    else:
        h2 = np.zeros(n)
        h2[0:len(h)]=h
        X=np.fft.rfft(x)
        H=np.fft.rfft(h2)
        return np.fft.irfft(X*H,n=n)
        

def calc_err(x,y,h,reg,linear=False):
    yp=fwd(x,h,n=len(y),linear=linear)
    return np.sum((yp-y)**2)+np.sum((np.sqrt(reg)*h)**2)
 
 
#Get Toeplitz matrix and vector b for normal equation linear system.
def get_cb(x,y,imp_length,reg):
    n=len(x)
    X = np.fft.rfft(x)
    Y = np.fft.rfft(y)

    b = np.fft.irfft(np.conj(X)*Y,n)[0:imp_length] # Normal equation vec b
     
    tmp_a = np.fft.irfft(np.conj(X)*X,n) # Normal Equation matrix A is first (imp_length x imp_length) elements of the circulant matrix made from this vector.

    c = tmp_a[0:imp_length]
    c[0]+=reg
    
    return c,b

######
##
## Algorithm that assumes 
## a fixed length impulse response (length imp_length)
## and calculates the least-squares solution of the 
## convolution equation y = convolve(x, w). With optional l2-regularization.
##
##
## Arguments:
## reg - l2 regulariztion strength (optional).
## x - input signal length N
## y - output signal length N
## imp_length - desired impulse response length.
##
## Returns:
## w - Learned impulse response (length imp_length)
def least_squares_fir(x,y,imp_length,reg=0.0):
    
    assert(len(x)==len(y))
        
    c,b = get_cb(x,y,imp_length,reg)
    h=solve_toeplitz( c, b)
    return h
    
def get_fir_fft(x,y,imp_length):
    X=np.fft.rfft(x)
    Y=np.fft.rfft(y)
    return np.fft.irfft(Y/X,n=len(x))[0:imp_length]

