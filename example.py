from utils import least_squares_fir,calc_err
from cg_utils import least_squares_fir_cg
from post_opt import post_opt
from pertubation import lsq_fir_linear_pert,lsq_fir_linear_pert_cg

import numpy as np
import time

np.random.seed(0)
x=np.random.randn(10000)
y=np.random.randn(10000)
impulse_length = 1000
linear_bool = True
reg=1e4

# Circular Convolution
fir1 = least_squares_fir(x,y,impulse_length,reg=reg)
print("Direct",calc_err(x,y,fir1,reg,linear=linear_bool))

# Circular convolution CG
fir2, info = least_squares_fir_cg(x,y,impulse_length,reg=reg,tol=1e-9,atol=True,precond=1)
print("PCG",calc_err(x,y,fir2,reg,linear=linear_bool))


# More complex regularization
reg2 = np.ones(impulse_length)*reg

#linear=True uses linear convolution instead of circular
fir3,f = post_opt(x,y,fir1,reg=reg2,linear=linear_bool)

print("Postopt",f)
print("Postopt",calc_err(x,y,fir3,reg,linear=linear_bool))


#Pertubation Theory CG
fir4,chck=lsq_fir_linear_pert_cg(x,y,impulse_length,reg=reg,order=8)
print("PertubationCG",calc_err(x,y,fir4,reg,linear=linear_bool))

#Pertubation Theory Direct
fir5,chck=lsq_fir_linear_pert(x,y,impulse_length,reg=reg,order=8)
print("Pertubation",calc_err(x,y,fir5,reg,linear=linear_bool))
