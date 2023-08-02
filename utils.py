import numpy as np
from scipy.linalg import solve_toeplitz

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
 

def get_cb(x,y,imp_length,reg):
    n=len(x)
    X = np.fft.rfft(x)
    Y = np.fft.rfft(y)

    b = np.fft.irfft(np.conj(X)*Y,n)[0:imp_length] # Normal equation vec b
     
    tmp_a = np.fft.irfft(np.conj(X)*X,n) # Normal Equation matrix A is first (imp_length x imp_length) elements of the circulant matrix made from this vector.

    c = tmp_a[0:imp_length]
    c[0]+=reg
    
    return c,b
    
def least_squares_fir(x,y,imp_length,reg=0.0):

    c,b = get_cb(x,y,imp_length,reg)
    return solve_toeplitz( (c,c), b)
    

