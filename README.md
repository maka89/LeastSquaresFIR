# LeastSquaresFIR
Given sequences x and y, find the impulse response that gives the least-squares error .

## Usage

### Circular Convolution
- least_squares_fir() that directly solves the least-squares equation using a direct toeplitz solver.
- least_squares_fir_cg() that can be used for very high impulse lengths. Uses Preconditioned Conjugate Gradient method to solve the same equation.

### Linear Convolution
  #### Pertubation Theory Solver
  The following solvers assumes linear convolution, using circular convolution as the "unperturbed" system. Expected to work if impulse_length << data_length.
  
  -lsq_fir_linear_pert() The terms in solution is calculated using a direct toeplitz solver. I.e has to solve "order" toeplitz systems.
  -lsq_fir_linear_pert_cg() The terms in the solution is calculated using the Preconditioned Conjugate Gradient method.

  #### Post-optimiztion of Circular Convolution
  - post_opt() routine can find the least_squares solution to linear convolution by scalar minimization using l-bfgs-b. It can handle more complex regularization. Can use the solution from one of the "circular convoliton" methods as initial guess.

After finding the impulse_response, it can be "tuned" with the post_opt() routine. This allows for using linear convolution and more complex regularization. Uses regular scalar minimization using l-bfgs-b.

```python
from utils import least_squares_fir
from cg_utils import least_squares_fir_cg
from post_opt import post_opt

x=np.random.randn(10000)
y=np.random.randn(10000)
impulse_length = 100

reg=1e-10
fir1 = least_squares_fir(x,y,impulse_length,reg=reg)
fir2, info = least_squares_fir_cg(x,y,impulse_length,reg=reg)

#More complex regularization
reg = np.ones(impulse_length)*reg
reg[impulse_length//2:impulse_length]*=2.0

#linear=True uses linear convolution instead of circular
fir1 = post_opt(x,y,fir1,reg=reg,linear=True)


```
