import numpy as np


def cheb(N):
  '''
  Compute the differentiation matrix D and the Chebyshev extrema points x
  for a polynomial of degree N.

  Inputs:
  N = degree of Chebyshev polynomial
  Outputs:
  D = (N+1) x (N+1) differentiation matrix.
  x = (N+1) extrema points

  Translated cheb.m to python.
  '''
  if N == 0:
    D = 0
    x = 0
    return (D, x)   
  else:
    s = np.linspace(0, N, N+1)
    x = np.reshape(np.cos(np.pi * s / N), (N+1,1))
    c = (np.hstack(([2.], np.ones(N-1), [2.])) * (-1)**s).reshape(N+1,1)
    X = np.tile(x,(1,N+1))
    dX = X - X.T
    D = np.dot(c, 1./c.T) / (dX + np.eye(N+1))
    D -= np.diag(np.sum(D.T, axis=0))    
    return D, x.reshape(N+1)

def cheb_parameter_space(N):
  '''
  Output Chebyshev extrama points for a polynomial of degree N

  Inputs:
  N = degree of Chebyshev polynomial
  Outputs:
  x = (N+1) extrema points

  Translated cheb.m to python.
  '''
  if N == 0:
    x = 0
    return x   
  else:
    s = np.linspace(0, N, N+1)
    x = np.reshape(np.cos(np.pi * s / N), (N+1,1))
    return  x.reshape(N+1)

def clencurt(N):
  '''
  Return weights w for Clenshaw-Curtis quadrature.
  Weights for the Lobatto points (Chebyshev extrema).
  
  Translated from matlab clencurt.m.
  '''
  theta = (np.pi / N) * np.reshape(np.arange(N+1), (N+1,1)) 
  x = np.cos(theta)
  w = np.zeros(N+1)
  ii = np.arange(1, N)
  v = np.ones((N-1,1))

  if (N % 2) == 0:
    w[0] = 1.0 / (N**2 - 1.0) 
    w[N] = w[0]
    for k in range(1, N // 2):
      v = v - 2.0 * np.cos(2.0 * k * theta[ii]) / (4.0 * k**2 - 1)
    v = v - np.cos(N * theta[ii]) / (N**2 - 1)
  else:
    w[0] = 1.0 / N**2
    w[N] = w[0]
    for k in range(1, (N-1) // 2 + 1):
      v = v - 2.0 * np.cos(2.0 * k * theta[ii]) / (4.0 * k**2 - 1)     
  w[ii] = 2.0 * np.reshape(v, v.size) / N
  return w 


def gauss_weights(N):
  '''
  Compute Legendre points and weights for Gauss quadrature.
  From Spectral Methods in MATLAB, Trefethen 2000.
  '''
  s = np.arange(1, N)
  beta = 0.5 / np.sqrt(1.0 - 1.0 / (2 * s)**2)
  T = np.diag(beta, k=1) + np.diag(beta, k=-1)
  eig_values, eig_vectors = np.linalg.eigh(T)
  w = 2 * eig_vectors[0,:]**2
  return eig_values, w


def cheb_extrema_points(N):
  '''
  Compute the Chebyshev extrema points x for a polynomial of degree N.
  Also known as: Chebyshev points of the second kind or Chebyshev-Lobatto points.

  Inputs:
  N = degree of Chebyshev polynomial
  Outputs:
  x = (N+1) extrema points
  '''
  if N == 0:
    x = 1
  else:
    s = np.linspace(0, N, N+1)
    x = np.cos(np.pi * s / N)
  return x


def cheb_root_points(N):
  '''
  Compute the Chebyshev roots x for a polynomial of degree N.
  Also known as: Chebyshev points of the first kind, Chebyshev nodes
  or Chebyshev-Gauss points.

  Inputs:
  N = degree of Chebyshev polynomial
  Outputs:
  x = (N+1) extrema points
  '''
  if N == 0:
    x = None
  else:
    s = np.linspace(0, N-1, N)
    x = np.cos(np.pi * (s+0.5) / N)
  return x


def cheb_calc_coef_loops(x):
  '''
  Compute the Chebyshev coefficients with the values of a function
  evaluated at the Chebyshev extrema points.

  Slow function, only for tests.
  '''
  cheb_coef = np.zeros(x.size)
  c = np.ones(x.size)
  c[0] = 2.0
  c[-1] = 2.0
  for j in range(cheb_coef.size):
    summ = 0.0
    for k in range(cheb_coef.size):
      summ += x[k] * np.cos(np.pi * j * k / (cheb_coef.size - 1)) / c[k]
    cheb_coef[j] = (2.0 / (cheb_coef.size - 1)) * summ / c[j]
  return cheb_coef


def cheb_calc_coef(x):
  '''
  Compute the Chebyshev coefficients with the values of a function
  evaluated at the Chebyshev extrema points.
  '''
  c = np.ones(x.size)
  c[0] = 2.0
  c[-1] = 2.0
  j = np.arange(x.size)
  jk = np.outer(j, j)
  summ = np.cos((np.pi / (x.size - 1))* jk) * (x[None,:] / c[None,:])
  return (2.0 / (x.size - 1)) * np.sum(summ, axis=1) / c

def chebfft(v):
  '''
  Compute Chebyshev differentiation via FFT
  assumes grid is [-1,1]
  TODO: Analytical expressions for higher order derivatives are needed
  '''

  num_points = v.size-1
  x = np.cos(np.linspace(0,num_points,num_points+1)*np.pi/num_points)

  ii = np.linspace(0,num_points-1,num_points)
  V = np.hstack((v,np.flipud(v[1:num_points])))
  U = np.real(np.fft.fft(V))

  W = np.real(np.fft.ifft(np.multiply(np.hstack((np.hstack((ii,0)), np.linspace(0,num_points-2,num_points-1)-num_points+1))*1j,U)))

  w = np.zeros(num_points+1)

  w[1:num_points] = -np.divide(W[1:num_points],np.sqrt(1-x[1:num_points]**2))
  w[0] = np.sum(ii**2*U[ii.astype('int')])/num_points+0.5*num_points*U[num_points]
  w[num_points] = np.sum(np.multiply(np.multiply(np.power(-1,ii+1),ii**2), U[ii.astype('int')]))/num_points + 0.50*(-1)**(num_points+1)*num_points*U[num_points]

  return (-1.)*w

def cheb_real_to_reciprocal(N):
  '''
  Compute the matrix that  transfer information from real space 
  (defined at Chebyshev extrema points) to reciprocal (Fourier) space.
  '''
  c = np.ones(N)
  c[0] = 2.0
  c[-1] = 2.0
  j = np.arange(N)
  jk = np.outer(j, j)

  P = (2.0 / (N - 1)) * np.cos((np.pi / (N - 1)) * jk) / c[None,:]
  P[0, :] *= 0.5
  P[-1, :] *= 0.5
  return P


def cheb_eval_loops(alpha, cheb_coef, order = 0):
  '''
  Evaluate Chebyshev polynomial at the points x with a naive method.

  order = 0: Chebyshev grid = [1, ..., -1]
  order = 1: Chebyshev grid = [-1, ..., 1]

  Slow function, only for tests.
  TODO: implement Clenshaw's recurrence formula.
  '''
  if order == 1:
    alpha = -1.0 * alpha
  x = np.zeros(alpha.size)
  for k in range(cheb_coef.size):
    x += cheb_coef[k] * np.cos(k * np.arccos(alpha))
  return x 


def cheb_eval(alpha, cheb_coef, order = 0):
  '''
  Evaluate Chebyshev polynomial at the points x with a naive method.

  order = 0: Chebyshev grid = [1, ..., -1]
  order = 1: Chebyshev grid = [-1, ..., 1]

  TODO: implement Clenshaw's recurrence formula.
  '''
  if order == 1:
    alpha = -1.0 * alpha
  k = np.arange(cheb_coef.size)
  return np.sum(cheb_coef[:, None] * np.cos(k[:,None] * np.arccos(alpha)), axis=0)


def cheb_reciprocal_to_real(N):
  '''
  Compute the matrix that  transfer information from reciprocal 
  (Fourier) space to real space (defined at Chebyshev extrema points).
  '''
  N = N - 1
  k = np.arange(N + 1)
  alpha = np.cos(np.pi * k / N)
  return np.cos(k[:,None] * np.arccos(alpha))



  # Clenshaw's formula does not seem to help
  # x = np.zeros(alpha.size)
  # for k in range(cheb_coef.size):
  #   d = np.zeros(alpha.size + 1)
  #   for i in range(cheb_coef.size - 2, 0, -1):
  #     d[i] = 2 * alpha[k] * d[i+1] - d[i+2] + cheb_coef[i]
  #   x[k] = alpha[k] * d[1] - d[2] + cheb_coef[0]
  # return x


