from __future__ import division, print_function
import numpy as np
try:
  from numba import njit, prange
except ImportError:
  print('Numba not found')
  pass



@njit(parallel=True)
def test(r):
    N = 2
    print('N = ', N)
    x = r.reshape(N, 3)
    return 0



if __name__ == '__main__':
    print('# Start')
    N = 2
    r = np.random.rand(N, 3)
    x = test(r)
    print('x = ', x)
    print('# End')
