# LeastSquaresFIR
Given sequences x and y, find the impulse response that gives the least-squares error .

## Usage

### Circular Convolution
- least_squares_fir() that directly solves the least-squares equation using a direct toeplitz solver.
- least_squares_fir_cg() that can be used for very high impulse lengths. Uses Preconditioned Conjugate Gradient method to solve the same equation.

### Linear Convolution
  #### Pertubation Theory Solver
  The following solvers assumes linear convolution, using circular convolution as the "unperturbed" system. Expected to work if impulse_length << data_length.
  
  - lsq_fir_linear_pert() The terms in solution is calculated using a direct toeplitz solver. I.e has to solve "order" toeplitz systems.
  - lsq_fir_linear_pert_cg() The terms in the solution is calculated using the Preconditioned Conjugate Gradient method.

  #### Post-optimiztion of Circular Convolution
  - post_opt() routine can find the least_squares solution to linear convolution by scalar minimization using l-bfgs-b. It can handle more complex regularization. Can use the solution from one of the "circular convoliton" methods as initial guess.

After finding the impulse_response, it can be "tuned" with the post_opt() routine. This allows for using linear convolution and more complex regularization. Uses regular scalar minimization using l-bfgs-b.

```python
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



```
