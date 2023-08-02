# LeastSquaresFIR
Given sequences x and y, find the impulse response that gives the least-squares error .

## Usage
Code has two main methods: 
- least_squares_fir() that directly solves the least-squares equation using a direct toeplitz solver.
- lest_squares_fir_cg() that can be used for very high impulse lengths. Uses Preconditioned Conjugate Gradient method to solve the same equation. 
```python
from utils import least_squares_fir
from cg_utils import least_squares_fir_cg

x=np.random.randn(10000)
y=np.random.randn(10000)
impulse_length = 100

fir1 = least_squares_fir(x,y,impulse_length)
fir2, info = least_squares_fir_cg(x,y,impulse_length)
```
