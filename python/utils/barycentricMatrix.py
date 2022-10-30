'''
Build the barycentric resampling matrices P_{N,-m}.
See reference: Rectangular spectral collocation, T.A. Driscoll and N. Hale,
IMA Journal of Numerical Analysis 36, 108 (2016).

P_{N,-m}: samples a function defined at N points x_k to a (N-m)
          points at y_k. The parameter m can be positive (down-sampling)
          or negative (up-sampling).
'''
import numpy as np

def barycentricMatrix(x, y):
  '''
  Return resampling matrix P_{N,-m}.
  Inputs:
  x = numpy array, N points x_k.
  y = numpy array, N-m points.

  Note that m can be positive (down-sampling) or negative (up-sampling).
  '''
  # Flatten arrays
  x = x.flatten()
  y = y.flatten()

  # Get problem size
  N = x.size

  # Calculate barycentric weights 
  w = np.ones(N)
  w[1::2] = -1.0
  w[0] = 0.5
  w[-1] = -0.5 * (-1)**N

  # Difference between sets and sum
  with np.errstate(divide='ignore', invalid='ignore'):
    yx = y[:,None] - x
    w_yx = w[None, :] / yx
    sum_w_yx = np.sum(w_yx, axis=1)

    # Build matrix
    P = w_yx / sum_w_yx[:,None]

  # Remove NaNs and return
  P[(yx == 0)] = 1.0
  return P


def barycentricMatrix_loops(x, y):
  '''
  Return resampling matrix P_{N,-m}.
  Inputs:
  x = numpy array, N points x_k.
  y = numpy array, N-m points.

  Note that m can be positive (down-sampling) or negative (up-sampling).
  '''
  # Flatten arrays
  x = x.flatten()
  y = y.flatten()

  # Get problem size
  N = x.size
  M = y.size
  m = N - M

  # Calculate barycentric weights 
  w = np.ones(N)
  w[1::2] = -1.0
  w[0] = 0.5
  w[-1] = -0.5 * (-1)**N

  # Build matrix
  P = np.empty((M,N))
  for j in range(M):
    S = 0.0
    for k in range(N):
      S += w[k] / (y[j] - x[k])
    for k in range(N): 
      if np.abs(y[j] - x[k]) > np.finfo(float).eps:
        P[j,k] = w[k] / (y[j] - x[k]) / S
      else:
        P[j,k] = 1.0
  return P


