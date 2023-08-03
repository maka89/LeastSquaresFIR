from utils import least_squares_fir,calc_err
from cg_utils import least_squares_fir_cg
from post_opt import post_opt
import numpy as np

np.random.seed(0)
x=np.random.randn(10000)
y=np.random.randn(10000)
impulse_length = 100

reg=1e5
fir1 = least_squares_fir(x,y,impulse_length,reg=reg)
print(calc_err(x,y,fir1,reg,linear=False))
fir2, info = least_squares_fir_cg(x,y,impulse_length,reg=reg,tol=1e-9,atol=True,precond=1)
#print(info)
print(calc_err(x,y,fir2,reg,linear=False))

#More complex regularization
reg = np.ones(impulse_length)*reg

#linear=True uses linear convolution instead of circular
fir3,f = post_opt(x,y,fir1,reg=reg,linear=False)
print(f)
print(calc_err(x,y,fir3,reg,linear=False))