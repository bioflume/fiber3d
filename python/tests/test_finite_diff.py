from __future__ import division, print_function
import numpy as np
import sys
sys.path.append('../')

from utils import cheb
from utils import finite_diff



if __name__ == '__main__':

  num_points = 32
  fd_points = 11

  # Get high order derivatives, enforce D * vector_ones = 0
  D_1, s = cheb.cheb(num_points - 1)
  s = np.flipud(s)
  D_1 = np.flipud(np.flipud(D_1.T).T)
  D_2 = np.dot(D_1, D_1)
  D_2 -= np.diag(np.sum(D_2.T, axis=0))
  D_3 = np.dot(D_2, D_1)
  D_3 -= np.diag(np.sum(D_3.T, axis=0))
  D_4 = np.dot(D_2, D_2)
  D_4 -= np.diag(np.sum(D_4.T, axis=0))
    
  # Finite differences matrices
  D = finite_diff.finite_diff(s, 4, fd_points)
  FD_0 = D[:,:,0]
  FD_1 = D[:,:,1]
  FD_2 = D[:,:,2]
  FD_3 = D[:,:,3]
  FD_4 = D[:,:,4]

  print('\n\n\n')
  # Look at the difference between matrices
  print('|I   - FD_0| = ', np.linalg.norm(np.eye(num_points) - FD_0) / np.linalg.norm(np.eye(num_points)), np.max(np.eye(num_points) - FD_0))
  print('|D_1 - FD_1| = ', np.linalg.norm(D_1 - FD_1) / np.linalg.norm(D_1), np.max(D_1 - FD_1))
  print('|D_2 - FD_2| = ', np.linalg.norm(D_2 - FD_2) / np.linalg.norm(D_2), np.max(D_2 - FD_2))
  print('|D_3 - FD_3| = ', np.linalg.norm(D_3 - FD_3) / np.linalg.norm(D_3), np.max(D_3 - FD_3))
  print('|D_4 - FD_4| = ', np.linalg.norm(D_4 - FD_4) / np.linalg.norm(D_4), np.max(D_4 - FD_4))
  print('\n')

  # Compute derivatives of function
  # f = np.exp(-s**2) + s
  # f_1 = -2 * s * np.exp(-s**2) + 1.0
  # f_2 = (-2 + 4*s**2) * np.exp(-s**2)
  # f_3 = (12*s - 8*s**3) * np.exp(-s**2)
  # f_4 = (12 - 48*s**2 + 16*s**4) * np.exp(-s**2)
  f = np.cos(s * np.pi) + s + np.exp(2 * s)
  f_1 = -np.sin(s * np.pi) * np.pi + 1.0 + np.exp(2 * s) * 2
  f_2 = -np.cos(s * np.pi) * np.pi**2 + np.exp(2*s) * 4
  f_3 = np.sin(s * np.pi) * np.pi**3 + np.exp(2*s) * 8
  f_4 = np.cos(s * np.pi) * np.pi**4 + np.exp(2*s) * 16
  f_th = np.column_stack((s, f, f_1, f_2, f_3, f_4))
  np.savetxt('data/theory.dat', f_th)
  
  # Chebyshev
  f_1 = np.dot(D_1, f)
  f_2 = np.dot(D_2, f)
  f_3 = np.dot(D_3, f)
  f_4 = np.dot(D_4, f)
  f_cheb = np.column_stack((s, f, f_1, f_2, f_3, f_4))
  np.savetxt('data/cheb.dat', f_cheb)

  # Finite diff
  f_1 = np.dot(FD_1, f)
  f_2 = np.dot(FD_2, f)
  f_3 = np.dot(FD_3, f)
  f_4 = np.dot(FD_4, f)
  f_fd = np.column_stack((s, f, f_1, f_2, f_3, f_4))
  np.savetxt('data/finite_differences.dat', f_fd)

  # Print differences with theory
  print('|f_1 - f_1_cheb| = ', np.linalg.norm(f_th[:,2] - f_cheb[:,2]) / np.linalg.norm(f_th[:,2]), np.max(f_th[:,2] - f_cheb[:,2]))
  print('|f_1 - f_1_cheb| = ', np.linalg.norm(f_th[:,3] - f_cheb[:,3]) / np.linalg.norm(f_th[:,3]), np.max(f_th[:,3] - f_cheb[:,3]))
  print('|f_1 - f_1_cheb| = ', np.linalg.norm(f_th[:,4] - f_cheb[:,4]) / np.linalg.norm(f_th[:,4]), np.max(f_th[:,4] - f_cheb[:,4]))
  print('|f_1 - f_1_cheb| = ', np.linalg.norm(f_th[:,5] - f_cheb[:,5]) / np.linalg.norm(f_th[:,5]), np.max(f_th[:,5] - f_cheb[:,5]))
  print(' ')
  print('|f_1 - f_1_fd| = ', np.linalg.norm(f_th[:,2] - f_fd[:,2]) / np.linalg.norm(f_th[:,2]), np.max(f_th[:,2] - f_fd[:,2]))
  print('|f_1 - f_1_fd| = ', np.linalg.norm(f_th[:,3] - f_fd[:,3]) / np.linalg.norm(f_th[:,3]), np.max(f_th[:,3] - f_fd[:,3]))
  print('|f_1 - f_1_fd| = ', np.linalg.norm(f_th[:,4] - f_fd[:,4]) / np.linalg.norm(f_th[:,4]), np.max(f_th[:,4] - f_fd[:,4]))
  print('|f_1 - f_1_fd| = ', np.linalg.norm(f_th[:,5] - f_fd[:,5]) / np.linalg.norm(f_th[:,5]), np.max(f_th[:,5] - f_fd[:,5]))



  print('# End')
