#For gradient checking
import torch
torch.set_default_dtype(torch.float64)

class ConvolveTorch:
    def __init__(self,x,y,m,reg,linear=True):
        torch.set_default_dtype(torch.float64)
        self.m = m
        self.linear=linear
        if linear:
            self.m2 = self.m
        else:
            self.m2 = 0
        self.n = len(y)
        
        assert(len(x) == self.n)
        self.y = torch.tensor(y)
        
        x2 = torch.zeros(self.n+self.m2)
        x2[0:len(x)] = torch.tensor(x)
        self.X2 = torch.fft.fft(x2)
        
        self.reg = torch.sqrt(torch.tensor(reg))
        
    def forward(self,h):
    
        h2 = torch.zeros(self.n+self.m2)
        h2[0:self.m]=h
        H=torch.fft.fft(h2)
        T=H*self.X2
        T=torch.fft.ifft(T).real
        return T[0:self.n]
        
        
    def err(self,h):
        h=torch.autograd.Variable(torch.tensor(h),requires_grad=True)
        yp=self.forward(h)
        
        err = torch.sum((yp-self.y)**2)
        err += torch.sum((self.reg*h)**2)
        
        err.backward()
        grad=h.grad.detach().numpy()
        
        return err.item(), grad