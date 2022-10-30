import numpy as np


def finite_diff(s, M, n_s):
  '''
  Following the paper Calculation of weights in finite different formulas,
  Bengt Fornberg, SIAM Rev. 40 (3), 685 (1998).

  Inputs:
  s = grid points
  M = order of the highest derivative to compute
  n_s = support to compute the derivatives

  Ouputs:
  D_s = M+1 matrices to compute the first M derivatives
  '''
  N = s.size - 1
  D_s = np.zeros((s.size, s.size, M+1))
  n_s_half = (n_s - 1) // 2
  n_s = n_s - 1

  
  # Loop over rows of D_s
  for xi, si in enumerate(s):
    # Find xs around x[xi] to compute coefficients
    if xi < n_s_half:
      x = s[0:n_s+1]
    elif xi > (s.size - n_s_half-2):
      x = s[-n_s-1:]
    else:
      x = s[xi - n_s_half : xi - n_s_half + n_s + 1]

    # Computer coefficients of differential matrices
    c1 = 1.0
    c4 = x[0] - si
    c = np.zeros((n_s+1, M+1))
    c[0,0] = 1.0
    for i in range(1, n_s+1):
      mn = np.minimum(i, M)
      c2 = 1.0
      c5 = c4
      c4 = x[i] - si
      for j in range(i):
        c3 = x[i] - x[j]
        c2 = c2 * c3
        if j == i-1:
          for k in range(mn, 0, -1):
            c[i,k] = c1 * (k * c[i-1,k-1] - c5 * c[i-1, k]) / c2
          c[i,0] = -c1 * c5 * c[i-1, 0] / c2
        for k in range(mn,0,-1):
          c[j,k] = (c4 * c[j,k] - k*c[j,k-1]) / c3
        c[j,0] = c4 * c[j,0] / c3
      c1 = c2
    
    # Copy c to D
    if xi < n_s_half:
      D_s[xi, 0 : n_s+1] = c[:,:]
    elif xi > (s.size - n_s_half-2):
      D_s[xi, -n_s-1:] = c[:,:]
    else:
      D_s[xi, xi - n_s_half : xi - n_s_half + n_s + 1,:] = c[:,:]      

  return D_s
