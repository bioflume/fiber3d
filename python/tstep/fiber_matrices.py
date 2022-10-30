
# Standard imports
from __future__ import print_function
import numpy as np
import scipy.linalg
import scipy.sparse.linalg as scspli 
import scipy.optimize as scop
from functools import partial
import sys
sys.path.append('../')

# Local imports
from utils import cheb
from utils import barycentricMatrix as bary
from utils import finite_diff

class fiber_matrices(object):
  '''
  Small class to compute and store fiber matrices for various resolutions
  Call this for various num_points (8, 16, 32, 64, 96) and save them
  '''
  def __init__(self, 
               num_points = 32,
               num_points_finite_diff = 4,
               uprate_poten = 2):

    
    self.num_points = num_points
    self.num_points_finite_diff = num_points
    if num_points_finite_diff > 0:
      self.num_points_finite_diff = num_points_finite_diff
    self.uprate_poten = uprate_poten # upsampling ratio to compute potentials
      
    return



  def compute_matrices(self):
    ''' 
    Need to call this whenever num_points changes
    '''
    

    num_points = self.num_points

    
    self.alpha = np.linspace(-1, 1, self.num_points)
    
    num_points_roots = num_points - 4
    self.alpha_roots = 2*(0.5+np.linspace(0,num_points_roots-1,num_points_roots))/num_points_roots - 1
    
    num_points_tension = num_points - 2
    self.alpha_tension = 2*(0.5+np.linspace(0,num_points_tension-1,num_points_tension))/num_points_tension - 1

    # this is the order of the finite differencing
    # 2nd order scheme: 3 points for 1st der, 4 points for 2nd, 5 points for 3rd, 6 points for 4th
    # 4th order scheme: 5 points for 1st der, 6 points for 2nd, 7 points for 3rd, 8 points for 4th
    num_points_finite_diff = self.num_points_finite_diff

    D = finite_diff.finite_diff(self.alpha, 1, num_points_finite_diff+1)
    self.D_1_0 = D[:,:,1]
    D = finite_diff.finite_diff(self.alpha, 2, num_points_finite_diff+2)
    self.D_2_0 = D[:,:,2]
    D = finite_diff.finite_diff(self.alpha, 3, num_points_finite_diff+3)
    self.D_3_0 = D[:,:,3]
    D = finite_diff.finite_diff(self.alpha, 4, num_points_finite_diff+4)
    self.D_4_0 = D[:,:,4]

    # Compute weights for trapezoidal rule
    self.weights_0 = np.ones_like(self.alpha) * 2.0
    self.weights_0[0] = 1.0
    self.weights_0[-1] = 1.0
    self.weights_0 *= 1.0 / ((num_points-1))

    # Compute weights for simpsons rule (num_points have to be odd)
    self.weights_simp_0 = np.ones_like(self.alpha) 
    self.weights_simp_0[1::2] = 4.0
    self.weights_simp_0[2::2] = 2.0
    self.weights_simp_0[0] = 1.0
    self.weights_simp_0[-1] = 1.0
    self.weights_simp_0 *= 2.0 / (3.0*(num_points-1))
    
    # Compute weights for trapezoidal rule on the upsampled grid (always 2*num_points)
    self.alpha_kerUp = np.linspace(-1, 1, self.uprate_poten*num_points)
    self.weights_0_up = np.ones_like(self.alpha_kerUp) * 2.0
    self.weights_0_up[0] = 1.0
    self.weights_0_up[-1] = 1.0
    self.weights_0_up *= 1.0 / ((self.uprate_poten*num_points)-1)
    
    #self.alpha_kerUp = np.flipud(cheb.cheb_parameter_space(self.uprate_poten*self.num_points-1))
    
    self.P_kerUp = bary.barycentricMatrix(self.alpha, self.alpha_kerUp)
    self.P_kerDn = bary.barycentricMatrix(self.alpha_kerUp,self.alpha)

    # Get matrices to transform from Chebyshev points of second and first kind
    self.P_X = bary.barycentricMatrix(self.alpha, self.alpha_roots)
    self.P_T = bary.barycentricMatrix(self.alpha, self.alpha_tension)
    self.P_cheb_representations_all_dof = np.zeros((4*num_points - 14, 4*num_points))
    self.P_cheb_representations_all_dof[0*num_points-0  : 1*num_points-4,  0*num_points : 1*num_points] = self.P_X
    self.P_cheb_representations_all_dof[1*num_points-4  : 2*num_points-8,  1*num_points : 2*num_points] = self.P_X
    self.P_cheb_representations_all_dof[2*num_points-8  : 3*num_points-12, 2*num_points : 3*num_points] = self.P_X
    self.P_cheb_representations_all_dof[3*num_points-12 : 4*num_points-14, 3*num_points : 4*num_points] = self.P_T

    return



  def get_matrices(self, length, num_points_up, which_matrix):

    '''
    Returns the matrix asked in which_matrix given num_points_up and length
    '''


    # bring up the output
    output_1, output_2, output_3, output_4 = [], [], [], []

    if which_matrix == 'alpha_and_s':
      output_1 = self.alpha
      output_2 = (1.0 + self.alpha) * (length / 2.0)

    elif which_matrix == 'Ds': # scaled D0s
      output_1 = self.D_1_0 * (2.0 / length)
      output_2 = self.D_2_0 * (2.0 / length)**2
      output_3 = self.D_3_0 * (2.0 / length)**3
      output_4 = self.D_4_0 * (2.0 / length)**4

    elif which_matrix == 'D0s':
      output_1, output_2, output_3, output_4 = self.D_1_0, self.D_2_0, self.D_3_0, self.D_4_0

    elif which_matrix == 'alpha_roots_tension':
      output_1, output_2 = self.alpha_roots, self.alpha_tension

    elif which_matrix == 'weights_all': # together with upsampled
      output_1 = self.weights_0 * (length / 2.0)
      output_2 = self.weights_0_up * (length / 2.0)

    elif which_matrix == 'simpsons': # for integration
      output_1 = self.weights_simp_0 * length / 2.0

    elif which_matrix == 'P_kernel': # P_kerUp and P_kerDn
      output_1, output_2 = self.P_kerUp, self.P_kerDn

    elif which_matrix == 'alpha_and_s_kernel': # s_kerUp and alpha_kerUp
      output_1 = self.alpha_kerUp
      output_2 = (1.0 + self.alpha_kerUp) * (length / 2.0)

    elif which_matrix == 'PX_PT_Pcheb': # P_T, P_X, P_cheb_rep
      output_1, output_2, output_3 = self.P_X, self.P_T, self.P_cheb_representations_all_dof



    return output_1, output_2, output_3, output_4










