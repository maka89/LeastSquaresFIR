# LeastSquaresFIR
Given sequences x and y, find the impulse response that gives the least-squares error .

## Usage
Code has two main methods: 
- least_squares_fir() that directly solves the least-squares equation using a direct toeplitz solver.
- lest_squares_fir_cg() that can be used for very high impulse lengths. Uses Preconditioned Conjugate Gradient method to solve the same equation.

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
