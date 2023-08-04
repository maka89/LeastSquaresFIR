from scipy.optimize import fmin_l_bfgs_b
import numpy as np


#For gradient checking
#from convolve_torch import ConvolveTorch

class FFT:
    def forward(x):
        t = np.fft.rfft(x)
        return t
        
    def backward(x,n):
        return IFFT.forward(x.conj(),n).conj()/n
        
FFT.forward = staticmethod(FFT.forward)
FFT.backward = staticmethod(FFT.backward)

class IFFT:
    def forward(x,n):
        return np.fft.irfft(x,n)
    def backward(x):
        return FFT.forward(x.conj()).conj()*len(x)
        
IFFT.forward = staticmethod(IFFT.forward)
IFFT.backward = staticmethod(IFFT.backward)

class Convolve:
    def __init__(self,x,y,m,reg,linear=True):
        self.m = m
        self.linear=linear
        if linear:
            self.m2 = self.m
        else:
            self.m2 = 0
        self.n = len(y)
        
        assert(len(x) == self.n)
        self.y = y
        
        x2 = np.zeros(self.n+self.m2)
        x2[0:len(x)] = x
        self.X2 = FFT.forward(x2)
        
        h2 = np.zeros(self.n+self.m2)
        self.reg = np.sqrt(reg)
        
    def forward(self,h):
    
        h2 = np.zeros(self.n+self.m2)
        h2[0:self.m]=h
        H=FFT.forward(h2)
        T=H*self.X2
        T=IFFT.forward(T,self.n+self.m2).real
        return T[0:self.n]
        
    def backward(self,err):
        err = np.concatenate((err,np.zeros(self.m2)))
        err = IFFT.backward(err)
        err = err*self.X2
        err = FFT.backward(err,self.n+self.m2).real
        return err[0:self.m]
        
    def err(self,h):
        yp=self.forward(h)
        err = np.sum((yp-self.y)**2)
        err += np.sum((self.reg*h)**2)
        
        grad = self.backward(2.0*(yp-self.y))
        grad += 2.0*self.reg*h*self.reg
        return err, np.copy(grad) 
        
def post_opt(x,y,h,reg=0.0,linear=True,disp=False,maxiter=100):
    
    m=len(h)
    model = Convolve(x,y,m,reg,linear=linear)

    def err(p,model):
        err,grad = model.err(p)
        
        ## Gradient checking
        #err2,grad2 = model2.err(p)
        #print(grad)
        #print(grad2)
        #assert(np.allclose(grad2,grad))
        
        return err,grad
    
    x,f,d = fmin_l_bfgs_b(err,h,args=[model],disp=disp,maxiter=maxiter)
    return x,f
    