from lstsq_deconv import least_squares_fir, calc_err,get_fir_fft
from lstsq_deconv import least_squares_fir_cg
from lstsq_deconv import lsq_fir_linear_pert_cg, lsq_fir_linear_pert
from lstsq_deconv import post_opt
from lstsq_deconv.toep_lst_sq import lstsquares_toep_cg
import numpy as np
import time

np.random.seed(0)
x=np.random.randn(60*44100)
y=np.random.randn(60*44100)
impulse_length = 20*44100
linear_bool = True
reg=1e-3

fir0 = get_fir_fft(x,y,impulse_length,reg)
print("FFT",calc_err(x,y,fir0,reg,linear=linear_bool))

# Circular Convolution
#fir1 = least_squares_fir(x,y,impulse_length,reg=reg)
#print("Direct",calc_err(x,y,fir1,reg,linear=linear_bool))


# More complex regularization
reg2 = np.ones(impulse_length)*reg

# Circular convolution CG
fir2, info = least_squares_fir_cg(x,y,impulse_length,xinit=fir0,reg=reg2,tol=1e-9,atol=True,precond=1,it_max=50)
print(info)
print("PCG",calc_err(x,y,fir2,reg,linear=linear_bool))


# More complex regularization
reg2 = np.ones(impulse_length)*reg

#linear=True uses linear convolution instead of circular
#fir3,f = post_opt(x,y,fir2,reg=reg2,linear=linear_bool)

#err = calc_err(x,y,fir3,reg,linear=linear_bool)
#assert(np.allclose(err,f))
#print("Postopt",err)


#Pertubation Theory CG
#t0 = time.time()
#fir4,chck=lsq_fir_linear_pert_cg(x,y,impulse_length,reg=reg2,order=8)
#print("PertubationCG",calc_err(x,y,fir4,reg,linear=linear_bool),time.time()-t0)

#Pertubation Theory Direct
#t0 = time.time()
#fir5,chck=lsq_fir_linear_pert(x,y,impulse_length,reg=reg,order=8)
#print("Pertubation",calc_err(x,y,fir5,reg,linear=linear_bool),time.time()-t0)

r=np.zeros_like(x)
r[0] = x[0]
fir6,info=lstsquares_toep_cg((x,r),y,xinit=fir2,precond=1,imp_length=impulse_length,reg=reg,tol=1e-8*np.sum(y**2),it_max=20000,disp=True,atol=True)
print(info)
print("Toeplitz LstSq PCG",calc_err(x,y,fir6,reg,linear=linear_bool))