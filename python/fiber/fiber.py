'''
Small class to handle a single fiber. The two main references are
1.
2.

We use a linearized implicit penalty method to enforce inextensibility

X_s * X_st = (\tau / dt) * (1 - X^n * X^{n+1})

We use a local mobility derived from slender body theory, see Ref 1 or 2.

M = c_0 * (1 + X_s * X_s) + c_1 * (1 - X_s * X_2)

with
c_0 = -log(e * epsilon**2) / (8 * pi * viscosity)
c_1 = 2 / (8 * pi * viscosity)
'''
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
from utils import timer
from utils import finite_diff
from utils import barycentricMatrix as bary

class fiber(object):
  '''
  Small class to handle a single fiber.
  '''
  def __init__(self,
               num_points = 32,
               num_points_finite_diff = 7,
               length = 1.0,
               fiber_ds = 0.5/16,
               epsilon = 1e-03,
               E = 1.0,
               viscosity = 1.0,
               dt = 1e-03,
               adaptive_num_points = 1,
               num_points_max = 96,
               BC_start_0 = 'force',
               BC_start_1 = 'torque',
               BC_end_0 = 'force',
               BC_end_1 = 'torque',
               BC_start_vec_0 = np.zeros(3),
               BC_start_vec_1 = np.zeros(3),
               BC_end_vec_0 = np.zeros(3),
               BC_end_vec_1 = np.zeros(3),
               inonlocal = 0,
               ireparam = 0,
               reparam_degree = 4,
               reparam_iter = 20,
               tstep_order = 1,
               beta_tstep = 1,
               growing = 0,
               penalty_param = 500):
    # Store some parameters
    self.fiber_ds = fiber_ds
    self.num_points = num_points
    self.num_points_up = num_points
    self.num_points_finite_diff = num_points
    if num_points_finite_diff > 0:
      self.num_points_finite_diff = num_points_finite_diff

    self.uprate_poten = 2 # upsampling ratio to compute potentials

    # Maximum upsampling rate
    self.maxUp = 6
    self.num_points_maxUp = self.num_points*self.maxUp
    self.length_new = length
    self.length = length
    self.length_previous = length
    self.epsilon = epsilon
    self.E = E
    self.viscosity = viscosity
    self.dt = dt
    self.c_0 = -np.log(np.e * epsilon**2) / (8.0 * np.pi * viscosity)
    self.c_1 = 2.0 / (8.0 * np.pi * viscosity)
    self.penalty_param = penalty_param
    # self.beta = 8 * np.pi / (-np.log(np.e * epsilon**2)) * np.minimum(np.maximum(1.0**4 / (E * dt), 1.0), 1e+04) * 4000.0
    self.beta_0 = 8 * np.pi / (-np.log(np.e * epsilon**2)) * self.penalty_param
    self.beta = self.beta_0
    #self.beta = self.beta_0 / self.dt
    self.BC_start_0 = BC_start_0
    self.BC_start_1 = BC_start_1
    self.BC_end_0 = BC_end_0
    self.BC_end_1 = BC_end_1
    self.BC_start_vec_0 = BC_start_vec_0
    self.BC_start_vec_1 = BC_start_vec_1
    self.BC_end_vec_0 = BC_end_vec_0
    self.BC_end_vec_1 = BC_end_vec_1
    self.growing = growing
    self.stop_growing = False
    self.v_growth = 0.0
    self.v_shrink = 0.0
    self.v_length = 0.0
    self.rate_catastrophe = 0.0
    self.rate_rescue = 0.0
    self.rate_seed = 0.0
    self.rate_catastrophe_stall = 0.0
    self.force_stall = 1.0
    self.length_min = 0.0
    self.length_before_die = None
    self.inonlocal = inonlocal # whether include non-local part in self-mobility
    self.ireparam = ireparam # reparametrizing fibers
    self.reparam_degree = reparam_degree # attenuation factor, degree
    self.reparam_iter = reparam_iter # maximum number of reparam. iteration
    self.hinged_tip = None
    self.tip_position_old = None
    self.tip_position_new = None
    self.velocity = None
    self.adaptive_num_points = adaptive_num_points
    self.num_points_max = num_points_max # Maximum number of points that we can use for a fiber
    self.force_on_body = None
    self.torque_on_body = None
    
    self.num_points_finite_diff = self.num_points
    if num_points_finite_diff > 0:
      self.num_points_finite_diff = num_points_finite_diff

    # Uniformly spaced points along fiber
    alpha = np.linspace(-1,1,num_points)

    if self.adaptive_num_points:
      self.max_spacing = self.length/self.num_points
      self.length_0, self.num_points_0 = self.length, self.num_points
      # upsampling or downsampling matrix needed when resolution changes
      self.P_adap = np.zeros((num_points,num_points))


    # Create fiber configuration, straight fiber along x
    self.x = np.zeros((num_points, 3)) # current time step, n
    self.x[:,0] = -alpha * (self.length / 2.0)
    self.s = (1.0 + alpha) * (self.length / 2.0)

    # history of fiber configs, max. fourth order is considered, [:,0]: earliest, [:,-1]: last step
    self.xhist = np.zeros((num_points,3,tstep_order))
    self.x_old = np.zeros((num_points, 3))
    self.x_new = np.zeros((num_points, 3)) # new time step, n+1
    self.tension = np.zeros(num_points)
    self.tension_new = np.zeros(num_points)

    self.tstep_order = tstep_order
    self.beta_tstep = beta_tstep # multiplies x(n+1) when forming matrices
    self.x_rhs = None # extrapolated for RHS for tstep_order > = 2
    self.x_ext = None # extrapolated for building matrices for tstep_order > = 2


    self.x_modes = np.zeros_like(self.x)
    self.xs_modes = np.zeros_like(self.x)
    self.force = np.zeros((num_points, 3))
    self.force_motors = np.zeros((num_points, 3))

    # Derivatives
    self.xs = np.zeros_like(self.x)
    self.xss = np.zeros_like(self.x)
    self.xsss = np.zeros_like(self.x)
    self.xssss = np.zeros_like(self.x)

    self.x_up = np.zeros((self.num_points_up, 3))
    self.xs_up = np.zeros_like(self.x_up)
    self.xss_up = np.zeros_like(self.x_up)
    self.xsss_up = np.zeros_like(self.x_up)
    self.xssss_up = np.zeros_like(self.x_up)

    self.time_left = None
    self.justHinged = False
    self.iReachSurface = False
    self.iReachSurface_fake = False
    self.hinged_point = None
    self.attached_to_body = None
    self.nuc_site_idx = None
    self.nuc_site_idx_in_active = None
    self.not_functional = False

    return


  def compute_derivatives_at_extrap(self, length, fib_mat):

    # Compute derivatives at extrapolated points needed for time stepping order > 1

    D_1, D_2, D_3, D_4 = fib_mat.get_matrices(length, self.num_points_up, 'Ds')
    D_1_up, D_2_up, D_3_up, D_4_up = fib_mat.get_matrices(length, self.num_points_up, 'D_ups')
    P_up, P_down, out3, out4 = fib_mat.get_matrices(length, self.num_points_up, 'P_upsample')


    self.xs_ext= np.dot(D_1, self.x_ext)
    self.xss_ext = np.dot(D_2, self.x_ext)
    self.xsss_ext = np.dot(D_3, self.x_ext)
    self.xssss_ext = np.dot(D_4, self.x_ext)

    self.x_up_ext = np.dot(P_up, self.x_ext)
    self.xs_up_ext = np.dot(D_1_up, self.x_up_ext)
    self.xss_up_ext = np.dot(D_2_up, self.x_up_ext)
    self.xsss_up_ext = np.dot(D_3_up, self.x_up_ext)
    self.xssss_up_ext = np.dot(D_4_up, self.x_up_ext)


    return




  def update_length(self, F_end):
    '''
    Update the fiber length and rescale the differential matrices.
    '''
    # Poisson process to catastrophe or rescue
    self.length_previous = self.length
    r = np.random.rand(1)
    if self.growing:
      if self.v_growth != 0.:
        F = -np.dot(self.xs[-1], F_end)
        #vg = np.minimum(self.v_growth, self.v_growth * np.exp(-F / self.force_stall))
        #rate_catastrophe = self.rate_catastrophe_stall / (1.0 + (self.rate_catastrophe_stall / self.rate_catastrophe - 1.0) * vg / self.v_growth)
        #self.v_length = vg
        # HARD CODED NO CATASTROPHE, GROWTH ALL THE TIME
        rate_catastrophe = 0
        self.v_length = self.v_growth
        if r > np.exp(-self.dt * rate_catastrophe):
          self.growing = 0
          self.v_length = -self.v_shrink
          if self.length <= self.length_min:
            self.v_length = 0.0
      else:
        self.v_length = 0.
    else:
      rate_rescue = self.rate_rescue
      if self.length <= self.length_min:
        self.v_length = 0.0
        rate_rescue = self.rate_seed
      if r > np.exp(-self.dt * rate_rescue):
        self.growing = 1
        F = -np.dot(self.xs[-1], F_end)
        if self.force_stall != 0:
          vg = np.maximum(0.0, self.v_growth * np.exp(-F / self.force_stall))
          self.v_length = vg
        else:
          self.v_length = 0.

    # Grow or shrink MT
    self.length += self.v_length * self.dt
    if self.length <= self.length_min:
      self.length = self.length_min

  def update_resolution(self, fib_mat):

    # Make sure that the maximum spacing is conserved
    new_num_points = min(max(int(2**np.floor(np.log2(self.num_points_0 * np.sqrt(self.length/self.length_0)))),16), self.num_points_max)
    #new_num_points = min(max(16, (int(self.length/self.fiber_ds)+1)&~3), self.num_points_max)

    if new_num_points != self.num_points:
      alpha_old, s, out3, out4 = fib_mat.get_matrices(self.length, self.num_points_up, 'alpha_and_s')

      alpha = np.linspace(-1,1,new_num_points)
      self.P_adap = bary.barycentricMatrix(alpha_old, alpha)

      # Update variables
      xhist_new = np.zeros((new_num_points,3,self.tstep_order))
      for iorder in range(self.tstep_order):
        xhist_new[:,:,iorder] = np.dot(self.P_adap, self.xhist[:,:,iorder])

      self.xhist = np.copy(xhist_new)
      self.x = np.dot(self.P_adap, self.x)
      self.x_new = np.dot(self.P_adap, self.x_new)
      self.tension = np.dot(self.P_adap, self.tension)
      self.force = np.dot(self.P_adap, self.force)
      self.force_motors = np.dot(self.P_adap, self.force_motors)

      self.x_modes = np.zeros_like(self.x)
      self.xs_modes = np.zeros_like(self.x)

      # then update matrices
      self.num_points = new_num_points
    return



  def set_BC(self,
             BC_start_0 = 'force',
             BC_start_1 = 'torque',
             BC_end_0 = 'force',
             BC_end_1 = 'torque',
             BC_start_vec_0 = np.zeros(3),
             BC_start_vec_1 = np.zeros(3),
             BC_end_vec_0 = np.zeros(3),
             BC_end_vec_1 = np.zeros(3)):
    '''
    Set Boundary Conditions options. For each end of the fiber
    (labeled start and end) we need to provide two boundary conditions,
    one for the translational degrees of freedom and other for the orientations.

    options for translation:
    force = it applies a force to the end of the fiber.
    position = it prescribes the position of the fiber's end.

    options for rotation:
    torque = it applies a torque to the end of the fiber.
    angle = it enforces an orientation of the fiber's end.

    '''
    self.BC_start_0 = BC_start_0
    self.BC_start_1 = BC_start_1
    self.BC_end_0 = BC_end_0
    self.BC_end_1 = BC_end_1
    self.BC_start_vec_0 = BC_start_vec_0
    self.BC_start_vec_1 = BC_start_vec_1
    self.BC_end_vec_0 = BC_end_vec_0
    self.BC_end_vec_1 = BC_end_vec_1
    return


  def form_linear_operator(self, fib_mat, inextensibility = 'penalty'):
    '''
    Returns the linear operator A that define the linear system
    ONLY 1st ORDER, USES PRECOMPUTED AND STORED MATRICES

    A * (X^{n+1}, T^{n+1}) = RHS
    '''

    # Compute material derivatives at time

    num_points = self.num_points
    num_points_up = self.num_points
    num_points_down = self.num_points

    xs = self.xs
    xss = self.xss
    xsss = self.xsss
    xssss = self.xssss

    D_1, D_2, D_3, D_4 = fib_mat.get_matrices(self.length, num_points_up,'Ds')


    # Allocate memory for matrix
    A = np.zeros((4 * num_points_up, 4 * num_points_down))
    I = np.eye(num_points)
    I_vec = np.diag(I)

    # Build submatrices to couple coordinates to coordinates
    A_XX = self.beta_tstep * (I / self.dt) \
           + (self.E * self.c_0) * (D_4.T * (I_vec + xs[:,0]**2)).T \
           + (self.E * self.c_1) * (D_4.T * (I_vec - xs[:,0]**2)).T
    A_XY = (self.E * self.c_0) * (D_4.T * (xs[:,0]*xs[:,1])).T - (self.E * self.c_1) * (D_4.T * (xs[:,0]*xs[:,1])).T
    A_XZ = (self.E * self.c_0) * (D_4.T * (xs[:,0]*xs[:,2])).T - (self.E * self.c_1) * (D_4.T * (xs[:,0]*xs[:,2])).T
    A_YY = self.beta_tstep * (I / self.dt) \
           + (self.E * self.c_0) * (D_4.T * (I_vec + xs[:,1]**2)).T \
           + (self.E * self.c_1) * (D_4.T * (I_vec - xs[:,1]**2)).T
    A_YZ = (self.E * self.c_0) * (D_4.T * (xs[:,1]*xs[:,2])).T - (self.E * self.c_1) * (D_4.T * (xs[:,1]*xs[:,2])).T
    A_ZZ = self.beta_tstep * (I / self.dt) \
           + (self.E * self.c_0) * (D_4.T * (I_vec + xs[:,2]**2)).T \
           + (self.E * self.c_1) * (D_4.T * (I_vec - xs[:,2]**2)).T

    # Build submatrices to couple tension to coordinates
    A_XT = -self.c_0 * (2.0 * (D_1.T * xs[:,0]).T + np.diag(xss[:,0])) - self.c_1 * np.diag(xss[:,0])
    A_YT = -self.c_0 * (2.0 * (D_1.T * xs[:,1]).T + np.diag(xss[:,1])) - self.c_1 * np.diag(xss[:,1])
    A_ZT = -self.c_0 * (2.0 * (D_1.T * xs[:,2]).T + np.diag(xss[:,2])) - self.c_1 * np.diag(xss[:,2])

    # Build submatrices coordinates to tension
    beta = 0
    if inextensibility == 'penalty':
      beta = self.beta

    # NEW FORMULATION (GOKBERK)
    A_TX = -(self.c_1 + 7.0 * self.c_0) * self.E * (D_4.T * xss[:,0]).T - 6.0 * self.c_0 * self.E * (D_3.T * xsss[:,0]).T  - self.penalty_param * (D_1.T * xs[:,0]).T

    A_TY = -(self.c_1 + 7.0 * self.c_0) * self.E * (D_4.T * xss[:,1]).T - 6.0 * self.c_0 * self.E * (D_3.T * xsss[:,1]).T - self.penalty_param * (D_1.T * xs[:,1]).T

    A_TZ = -(self.c_1 + 7.0 * self.c_0) * self.E * (D_4.T * xss[:,2]).T - 6.0 * self.c_0 * self.E * (D_3.T * xsss[:,2]).T - self.penalty_param * (D_1.T * xs[:,2]).T

    # Build submatrices tension to tension
    A_TT = -2.0 * self.c_0 * D_2 + (self.c_1 + self.c_0) * np.diag(xss[:,0]**2  + xss[:,1]**2 + xss[:,2]**2)


    if False:
      c = np.log(np.e * self.epsilon**2);
      A_TX = (self.E * 6.0 * c) * (D_3.T * xsss[:,0]).T - \
             (self.E * (2.0 - 7.0 * c)) * (D_4.T * xss[:,0]).T - \
              beta * (D_1.T * xs[:,0]).T
      A_TY = (self.E * 6.0 * c) * (D_3.T * xsss[:,1]).T - \
             (self.E * (2.0 - 7.0 * c)) * (D_4.T * xss[:,1]).T - \
             beta * (D_1.T * xs[:,1]).T
      A_TZ = (self.E * 6.0 * c) * (D_3.T * xsss[:,2]).T - \
             (self.E * (2.0 - 7.0 * c)) * (D_4.T * xss[:,2]).T - \
             beta * (D_1.T * xs[:,2]).T

      # Build submatrices tension to tension
      A_TT = (2.0 * c) * D_2 + (2.0 - c) * np.diag(xss[:,0]**2  + xss[:,1]**2 + xss[:,2]**2)


    if self.inonlocal:
      # Add nonlocal part of self mobility
      K_XX, K_XY, K_XZ, K_YY, K_YZ, K_ZZ = self.self_mobility_nonlocal(fib_mat)

      A_XX_K = -np.dot(K_XX, self.E * D_4)
      A_XY_K = -np.dot(K_XY, self.E * D_4)
      A_XZ_K = -np.dot(K_XZ, self.E * D_4)
      A_YY_K = -np.dot(K_YY, self.E * D_4)
      A_YZ_K = -np.dot(K_YZ, self.E * D_4)
      A_ZZ_K = -np.dot(K_ZZ, self.E * D_4)

      A_XX += -A_XX_K
      A_XY += -A_XY_K
      A_XZ += -A_XZ_K
      A_YY += -A_YY_K
      A_YZ += -A_YZ_K
      A_ZZ += -A_ZZ_K

      A_XT_K = np.dot(K_XX, np.diag(xss[:,0]) + (D_1.T * xs[:,0]).T) + \
               np.dot(K_XY, np.diag(xss[:,1]) + (D_1.T * xs[:,1]).T) + \
               np.dot(K_XZ, np.diag(xss[:,2]) + (D_1.T * xs[:,2]).T)

      A_YT_K = np.dot(K_XY, np.diag(xss[:,0]) + (D_1.T * xs[:,0]).T) + \
               np.dot(K_YY, np.diag(xss[:,1]) + (D_1.T * xs[:,1]).T) + \
               np.dot(K_YZ, np.diag(xss[:,2]) + (D_1.T * xs[:,2]).T)

      A_ZT_K = np.dot(K_XZ, np.diag(xss[:,0]) + (D_1.T * xs[:,0]).T) + \
               np.dot(K_YZ, np.diag(xss[:,1]) + (D_1.T * xs[:,1]).T) + \
               np.dot(K_ZZ, np.diag(xss[:,2]) + (D_1.T * xs[:,2]).T)

      A_XT += -A_XT_K
      A_YT += -A_YT_K
      A_ZT += -A_ZT_K

      A_TX += -np.dot((D_1.T * xs[:,0]).T, A_XX_K + A_XY_K + A_XZ_K)
      A_TY += -np.dot((D_1.T * xs[:,1]).T, A_XY_K + A_YY_K + A_YZ_K)
      A_TZ += -np.dot((D_1.T * xs[:,2]).T, A_XZ_K + A_YZ_K + A_ZZ_K)

      A_TT += -np.dot((D_1.T * xs[:,0]).T, A_XT_K) - np.dot((D_1.T * xs[:,1]).T, A_YT_K) -\
              np.dot((D_1.T * xs[:,2]).T, A_ZT_K)


    # Collect all block matrices
    A = np.vstack((np.hstack((A_XX, A_XY, A_XZ, A_XT)),
                   np.hstack((A_XY, A_YY, A_YZ, A_YT)),
                   np.hstack((A_XZ, A_YZ, A_ZZ, A_ZT)),
                   np.hstack((A_TX, A_TY, A_TZ, A_TT))))
    return A



  def form_linear_operator_tension(self):
    '''
    Returns the linear operator A that define the linear system

    A * (T^{n+1}) = RHS

    assuming X^{n+1} = X^n, i.e. dt=0.
    '''
    # Compute material derivatives at time
    xss = self.xss
    out1, D_2, out3, out4 = fib_mat.get_matrices(self.length, self.num_points_up, 'Ds')


    # Allocate memory for matrix
    c = np.log(np.e * self.epsilon**2)

    # Build submatrices tension to tension
    A_TT = (2.0 * c) * D_2 + (2.0 - c) * np.diag(xss[:,0]**2  + xss[:,1]**2 + xss[:,2]**2)

    return A_TT


  def compute_RHS(self, fib_mat = None, force_external = None, flow = None, inextensibility = 'penalty'):
    '''
    Compute the Right Hand Side for the linear system with upsampling
    A * (X^{n+1}, T^{n+1}) = RHS

    with
    RHS = (X^n / dt + flow + Mobility * force_external, ...)

    Note that the internal force contributions (flexibility and in extensibility)
    force_internal includes flexibility and inextensibility contributions have been
    included in the linear operator A.

    ONLY ORDER 1
    '''


    num_points = self.num_points
    num_points_up = self.num_points
    num_points_down = self.num_points

    D_1, D_2, D_3, D_4 = fib_mat.get_matrices(self.length, num_points_up, 'Ds')

    x = self.x
    xs = self.xs
    xss = self.xss
    xsss = self.xsss
    xssss = self.xssss
    alpha, s, out3, out4 = fib_mat.get_matrices(self.length, num_points_up, 'alpha_and_s')



    I_vec = np.ones(num_points)

    # Build RHS
    RHS = np.zeros(4 * num_points)

    # TODO (GK): xs should be calculated at x_rhs when polymerization term is added to the rhs
    RHS[0:num_points]              = x[:,0] / self.dt + (alpha + 1.0) * (self.v_length * 0.5) * xs[:,0]
    RHS[num_points:2*num_points]   = x[:,1] / self.dt + (alpha + 1.0) * (self.v_length * 0.5) * xs[:,1]
    RHS[2*num_points:3*num_points] = x[:,2] / self.dt + (alpha + 1.0) * (self.v_length * 0.5) * xs[:,2]
    RHS[3*num_points:]             = -self.penalty_param

    # Add background flow contribution
    if flow is not None:

      RHS[0:num_points]              += flow[:,0]
      RHS[num_points:2*num_points]   += flow[:,1]
      RHS[2*num_points:3*num_points] += flow[:,2]

      # NEW FORMULATION (GOKBERK)
      RHS[3*num_points:] +=  (xs[:,0] * np.dot(D_1, flow[:,0]) + xs[:,1] * np.dot(D_1, flow[:,1]) +  xs[:,2] * np.dot(D_1, flow[:,2]))

      if False:
        RHS[3*num_points:] +=  8 * np.pi * self.viscosity * (xs[:,0] * np.dot(D_1, flow[:,0]) + xs[:,1] * np.dot(D_1, flow[:,1]) +  xs[:,2] * np.dot(D_1, flow[:,2]))



    # Add external force contribution
    if force_external is not None:

      fs = np.dot(D_1, force_external)
      RHS[0:num_points] += self.c_0 * ((I_vec + xs[:,0]**2)        * force_external[:,0]) + \
                           self.c_0 * ((        xs[:,0] * xs[:,1]) * force_external[:,1]) + \
                           self.c_0 * ((        xs[:,0] * xs[:,2]) * force_external[:,2]) + \
                           self.c_1 * ((I_vec - xs[:,0]**2)        * force_external[:,0]) + \
                           self.c_1 * ((      - xs[:,0] * xs[:,1]) * force_external[:,1]) + \
                           self.c_1 * ((      - xs[:,0] * xs[:,2]) * force_external[:,2])
      RHS[num_points:2*num_points] += self.c_0 * ((        xs[:,1] * xs[:,0]) * force_external[:,0]) + \
                                      self.c_0 * ((I_vec + xs[:,1]**2)        * force_external[:,1]) + \
                                      self.c_0 * ((        xs[:,1] * xs[:,2]) * force_external[:,2]) + \
                                      self.c_1 * ((      - xs[:,1] * xs[:,0]) * force_external[:,0]) + \
                                      self.c_1 * ((I_vec - xs[:,1]**2)        * force_external[:,1]) + \
                                      self.c_1 * ((      - xs[:,1] * xs[:,2]) * force_external[:,2])
      RHS[2*num_points:3*num_points] += self.c_0 * ((      xs[:,2] * xs[:,0]) * force_external[:,0]) + \
                                        self.c_0 * ((      xs[:,2] * xs[:,1]) * force_external[:,1]) + \
                                        self.c_0 * ((I_vec + xs[:,2]**2)        * force_external[:,2]) + \
                                        self.c_1 * ((      - xs[:,2] * xs[:,0]) * force_external[:,0]) + \
                                        self.c_1 * ((      - xs[:,2] * xs[:,1]) * force_external[:,1]) + \
                                        self.c_1 * ((I_vec - xs[:,2]**2)        * force_external[:,2])
      RHS[3*num_points:] += 2 * self.c_0 * (xs[:,0] * fs[:,0] + xs[:,1] * fs[:,1] + xs[:,2] * fs[:,2]) + (self.c_0 - self.c_1) * (xss[:,0] * force_external[:,0] + xss[:,1] * force_external[:,1] + xss[:,2] * force_external[:,2])


      if False:
        beta = self.beta
        c = np.log(np.e * self.epsilon**2);
        RHS[3*num_points:] += -2.0 * c * (xs[:,0] * fs[:,0] + xs[:,1] * fs[:,1] + xs[:,2] * fs[:,2]) + (2.0 - c) * (xss[:,0] * force_external[:,0] + xss[:,1] * force_external[:,1] + xss[:,2] * force_external[:,2])

        # Implicit penalty term
        RHS[3*num_points:]   += -beta


      if self.inonlocal:
        # Add non-local part of self-mobility
        K_XX, K_XY, K_XZ, K_YY, K_YZ, K_ZZ = self.self_mobility_nonlocal(fib_mat)

        RHSx = np.dot(K_XX, force_external[:,0]) + np.dot(K_XY, force_external[:,1]) + \
               np.dot(K_XZ, force_external[:,2])
        RHSy = np.dot(K_XY, force_external[:,0]) + np.dot(K_YY, force_external[:,1]) + \
               np.dot(K_YZ, force_external[:,2])
        RHSz = np.dot(K_XZ, force_external[:,0]) + np.dot(K_YZ, force_external[:,1]) + \
               np.dot(K_ZZ, force_external[:,2])

        RHS[0:num_points] += RHSx
        RHS[num_points:2*num_points] += RHSy
        RHS[2*num_points:3*num_points] += RHSz

        RHS[3*num_points:] += xs[:,0] * np.dot(D_1, RHSx) + xs[:,1] * np.dot(D_1, RHSy) + \
                              xs[:,2] * np.dot(D_1, RHSz)

    return RHS


  def compute_RHS_tension(self, force_external = None, flow = None):
    '''
    GK: upsampling added

    Compute the Right Hand Side for the linear system
    A * (T^{n+1}) = RHS

    Assuming X^{n+1} = X^n, i.e. dt = 0.
    '''
    # Compute material derivatives at time n
    xs = self.xs
    xss = self.xss
    xsss = self.xsss
    xssss = self.xssss
    num_points = self.num_points



    I_vec = np.ones(num_points)

    # Build RHS
    c = np.log(np.e * self.epsilon**2);
    RHS = np.zeros(num_points)
    RHS = (2.0 - 7.0 * c) * (xss[:,0]*xssss[:,0] + xss[:,1]*xssss[:,1] + xss[:,2]*xssss[:,2]) - \
          6.0 * c * (xsss[:,0]*xsss[:,0] + xsss[:,1]*xsss[:,1] + xsss[:,2]*xsss[:,2]) - \
          self.beta * (1.0 - (xs[:,0]*xs[:,0] + xs[:,1]*xs[:,1] + xs[:,2]*xs[:,2]))
    return RHS


  def apply_BC(self, A, RHS):
    '''
    Modify linear operator A and RHS to obey boundary conditions.
    '''
    D_1, D_2, D_3, D_4 = fib_mat.get_matrices(self.length, self.num_points_up, 'Ds')
    # Compute material derivatives at time n
    xs = self.xs
    xss = self.xss

    I = np.eye(self.num_points)
    num = self.num_points

    # Apply BC at one end
    if self.BC_start_0 == 'force':
      # Apply force
      A[0,:]               = 0
      A[0,0:num]           = self.E * D_3[0,:]
      A[0,3*num:]          = -xs[0,0] * I[0,:]
      A[num,:]             = 0
      A[num,num:2*num]     = self.E * D_3[0,:]
      A[num,3*num:]        = -xs[0,1] * I[0,:]
      A[2*num,:]           = 0
      A[2*num,2*num:3*num] = self.E * D_3[0,:]
      A[2*num,3*num:]      = -xs[0,2] * I[0,:]
      A[3*num,:]           = 0
      A[3*num,0:num]       = -self.E * D_2[0,:] * xss[0,0]
      A[3*num,num:2*num]   = -self.E * D_2[0,:] * xss[0,1]
      A[3*num,2*num:3*num] = -self.E * D_2[0,:] * xss[0,2]
      A[3*num,3*num:]      = -I[0,:]

      RHS[0]     = self.BC_start_vec_0[0]
      RHS[num]   = self.BC_start_vec_0[1]
      RHS[2*num] = self.BC_start_vec_0[2]
      RHS[3*num] = np.dot(self.BC_start_vec_0, xs[0,:])
    elif self.BC_start_0 == 'position':
      # Enforce position
      A[0,:] = 0
      A[0,0:num]           = I[0,:] / self.dt
      A[num,:] = 0
      A[num,num:2*num]     = I[0,:] / self.dt
      A[2*num,:] = 0
      A[2*num,2*num:3*num] = I[0,:] / self.dt
      A[3*num,:] = 0
      A[3*num,0:num]       = self.E * 3 * xss[0,0] * D_3[0,:]
      A[3*num,num:2*num]   = self.E * 3 * xss[0,1] * D_3[0,:]
      A[3*num,2*num:3*num] = self.E * 3 * xss[0,2] * D_3[0,:]
      A[3*num,3*num:]      = D_1[0,:]

      RHS[0]     = self.BC_start_vec_0[0] / self.dt
      RHS[num]   = self.BC_start_vec_0[1] / self.dt
      RHS[2*num] = self.BC_start_vec_0[2] / self.dt
      RHS[3*num] = 0
    elif self.BC_start_0 == 'velocity':
      # Enforce velocity
      A[0,:] = 0
      A[0,0:num]           = I[0,:] / self.dt
      A[num,:] = 0
      A[num,num:2*num]     = I[0,:] / self.dt
      A[2*num,:] = 0
      A[2*num,2*num:3*num] = I[0,:] / self.dt
      A[3*num,:] = 0
      A[3*num,0:num]       = (6.0*self.E*(self.c_0+self.c_1)) * xss[0,0] * D_3[0,:]
      A[3*num,num:2*num]   = (6.0*self.E*(self.c_0+self.c_1)) * xss[0,1] * D_3[0,:]
      A[3*num,2*num:3*num] = (6.0*self.E*(self.c_0+self.c_1)) * xss[0,2] * D_3[0,:]
      A[3*num,3*num:]      = (2.0*self.c_0) * D_1[0,:]

      RHS[0]     = self.x[0,0] / self.dt
      RHS[num]   = self.x[0,1] / self.dt
      RHS[2*num] = self.x[0,2] / self.dt
      RHS[3*num] = 0
    if self.BC_start_1 == 'torque':
      # Apply torque
      offset = 1
      A[offset,:]                 = 0
      A[offset,0:num]             = D_2[offset-1,:]
      A[offset+num,:]             = 0
      A[offset+num,num:2*num]     = D_2[offset-1,:]
      A[offset+2*num,:]           = 0
      A[offset+2*num,2*num:3*num] = D_2[offset-1,:]

      RHS[offset]       = self.BC_start_vec_1[0]
      RHS[offset+num]   = self.BC_start_vec_1[1]
      RHS[offset+2*num] = self.BC_start_vec_1[2]
    elif self.BC_start_1 == 'angle':
      # Enforce orientation
      offset = 1
      A[offset,:] = 0
      A[offset,0:num] = D_1[offset-1,:]
      A[offset+num,:] = 0
      A[offset+num,num:2*num] = D_1[offset-1,:]
      A[offset+2*num,:] = 0
      A[offset+2*num,2*num:3*num] = D_1[offset-1,:]

      RHS[offset] = self.BC_start_vec_1[0]
      RHS[offset+num] = self.BC_start_vec_1[1]
      RHS[offset+2*num] = self.BC_start_vec_1[2]
    elif self.BC_start_1 == 'angular_velocity':
      # Enforce orientation
      offset = 1
      A[offset,:] = 0
      A[offset,0:num] = D_1[offset-1,:] / self.dt
      A[offset+num,:] = 0
      A[offset+num,num:2*num] = D_1[offset-1,:] / self.dt
      A[offset+2*num,:] = 0
      A[offset+2*num,2*num:3*num] = D_1[offset-1,:] / self.dt

      RHS[offset] = xs[0,0] / self.dt
      RHS[offset+num] = xs[0,1] / self.dt
      RHS[offset+2*num] = xs[0,2] / self.dt

    # Apply BC at the other end
    if self.BC_end_0 == 'force':
      # Apply force
      offset = num - 1
      A[offset,:]                 = 0
      A[offset,0:num]             = -self.E * D_3[offset,:]
      A[offset,3*num:]            = xs[offset,0] * I[offset,:]
      A[offset+num,:]             = 0
      A[offset+num,num:2*num]     = -self.E * D_3[offset,:]
      A[offset+num,3*num:]        = xs[offset,1] * I[offset,:]
      A[offset+2*num,:]           = 0
      A[offset+2*num,2*num:3*num] = -self.E * D_3[offset,:]
      A[offset+2*num,3*num:]      = xs[offset,2] * I[offset,:]
      A[offset+3*num,:]           = 0
      A[offset+3*num,0:num]       = self.E * D_2[offset,:] * xss[offset,0]
      A[offset+3*num,num:2*num]   = self.E * D_2[offset,:] * xss[offset,1]
      A[offset+3*num,2*num:3*num] = self.E * D_2[offset,:] * xss[offset,2]
      A[offset+3*num,3*num:]      = I[offset,:]

      RHS[offset]       = self.BC_end_vec_0[0]
      RHS[offset+num]   = self.BC_end_vec_0[1]
      RHS[offset+2*num] = self.BC_end_vec_0[2]
      RHS[offset+3*num] = np.dot(self.BC_end_vec_0, xs[offset,:])
    elif self.BC_end_0 == 'position':
      # Enforce position
      offset = num - 1
      A[offset,:]                 = 0
      A[offset,0:num]             = I[offset,:] / self.dt
      A[offset+num,:]             = 0
      A[offset+num,num:2*num]     = I[offset,:] / self.dt
      A[offset+2*num,:]           = 0
      A[offset+2*num,2*num:3*num] = I[offset,:] / self.dt
      A[offset+3*num,:]           = 0
      A[offset+3*num,0:num]       = self.E * 3 * xss[offset,0] * D_3[offset,:]
      A[offset+3*num,num:2*num]   = self.E * 3 * xss[offset,1] * D_3[offset,:]
      A[offset+3*num,2*num:3*num] = self.E * 3 * xss[offset,2] * D_3[offset,:]
      A[offset+3*num,3*num:]      = D_1[offset,:]

      RHS[offset]       = self.BC_end_vec_0[0] / self.dt
      RHS[offset+num]   = self.BC_end_vec_0[1] / self.dt
      RHS[offset+2*num] = self.BC_end_vec_0[2] / self.dt
      RHS[offset+3*num] = 0
    if self.BC_end_1 == 'torque':
      # Apply torque
      offset = num - 2
      A[offset,:]                 = 0
      A[offset,0:num]             = D_2[offset+1,:]
      A[offset+num,:]             = 0
      A[offset+num,num:2*num]     = D_2[offset+1,:]
      A[offset+2*num,:]           = 0
      A[offset+2*num,2*num:3*num] = D_2[offset+1,:]

      RHS[offset]       = self.BC_end_vec_1[0]
      RHS[offset+num]   = self.BC_end_vec_1[1]
      RHS[offset+2*num] = self.BC_end_vec_1[2]
    elif self.BC_end_1 == 'angle':
      # Enforce orientation
      offset = num - 2
      A[offset,:] = 0
      A[offset,0:num] = D_1[offset+1,:]
      A[offset+num,:] = 0
      A[offset+num,num:2*num] = D_1[offset+1,:]
      A[offset+2*num,:] = 0
      A[offset+2*num,2*num:3*num] = D_1[offset+1,:]

      RHS[offset] = self.BC_end_vec_1[0]
      RHS[offset+num] = self.BC_end_vec_1[1]
      RHS[offset+2*num] = self.BC_end_vec_1[2]
    return A, RHS


  def apply_BC_tension(self, A, RHS):
    '''
    Modify linear operator A and RHS to obey boundary conditions
    to the tension linear system. Assume free fiber.
    '''
    num = self.num_points
    I = np.eye(self.num_points)
    A[0,0:]      = -I[0,:]
    RHS[0] = 0.0
    offset = num - 1
    A[offset,0:]      = I[offset,:]
    RHS[offset] = 0.0
    return A, RHS


  def apply_BC_rectangular(self, A, RHS, fib_mat, v_on_fiber, force_on_fiber):
    '''
    GK: APPLYING BCs WITH UPSAMPLING
    NOT SURE ABOUT VELOCITY BCs FOR HIGHER ORDER
    Modify linear operator A and RHS to obey boundary conditions.
    We use here the rectangular spectral collocation method of Driscoll and
    Hale.

    ONLY ORDER 1
    '''

    # Compute material derivatives at time n
    num = self.num_points
    x = self.x
    x_rhs = self.x
    xs_rhs = self.xs
    xs = xs_rhs
    xss = self.xss
    D_1, D_2, D_3, out4 = fib_mat.get_matrices(self.length, self.num_points_up, 'Ds')

    out1, out2, P_cheb_representations_all_dof, out4 = fib_mat.get_matrices(self.length, self.num_points_up, 'PX_PT_Pcheb')

    I = np.eye(num)
    A_rect = np.dot(P_cheb_representations_all_dof, A)
    RHS_rect = np.dot(P_cheb_representations_all_dof, RHS)

    B = np.zeros((14, 4 * num))
    B_RHS = np.zeros(14)

    # Apply BC at one end
    if self.BC_start_0 == 'force':
      # Apply force
      B[0,0:num]           = self.E * D_3[0,:]
      B[0,3*num:]          = -xs[0,0] * I[0,:]
      B[1,num:2*num]       = self.E * D_3[0,:]
      B[1,3*num:]          = -xs[0,1] * I[0,:]
      B[2,2*num:3*num]     = self.E * D_3[0,:]
      B[2,3*num:]          = -xs[0,2] * I[0,:]
      B[3,0:num]           = -self.E * D_2[0,:] * xss[0,0]
      B[3,num:2*num]       = -self.E * D_2[0,:] * xss[0,1]
      B[3,2*num:3*num]     = -self.E * D_2[0,:] * xss[0,2]
      B[3,3*num:]          = -I[0,:]

      B_RHS[0] = self.BC_start_vec_0[0]
      B_RHS[1] = self.BC_start_vec_0[1]
      B_RHS[2] = self.BC_start_vec_0[2]
      B_RHS[3] = np.dot(self.BC_start_vec_0, xs[0,:])
    elif self.BC_start_0 == 'velocity':
      # Enforce velocity
      B[0,0:num]       = self.beta_tstep * I[0,:] / self.dt
      B[1,num:2*num]   = self.beta_tstep * I[0,:] / self.dt
      B[2,2*num:3*num] = self.beta_tstep * I[0,:] / self.dt
      B[3,0:num]       = (6.0*self.E*(self.c_0+0*self.c_1)) * xss[0,0] * D_3[0,:]
      B[3,num:2*num]   = (6.0*self.E*(self.c_0+0*self.c_1)) * xss[0,1] * D_3[0,:]
      B[3,2*num:3*num] = (6.0*self.E*(self.c_0+0*self.c_1)) * xss[0,2] * D_3[0,:]
      B[3,3*num:]      = (2.0*self.c_0) * D_1[0,:]

      B_RHS[0] = x_rhs[0,0] / self.dt + self.BC_start_vec_0[0]
      B_RHS[1] = x_rhs[0,1] / self.dt + self.BC_start_vec_0[1]
      B_RHS[2] = x_rhs[0,2] / self.dt + self.BC_start_vec_0[2]
      B_RHS[3] = 0

      #if self.BC_start_vec_0.any():
      #  B_RHS[3] += xs[0,0]*self.BC_start_vec_0[0] + xs[0,1]*self.BC_start_vec_0[1] + xs[0,2]*self.BC_start_vec_0[2]
      if v_on_fiber is not None:
        B_RHS[3] += -(v_on_fiber[0,0]*xs[0,0] + v_on_fiber[0,1]*xs[0,1] + v_on_fiber[0,2]*xs[0,2])
      if force_on_fiber is not None:
        B_RHS[3] += -2 * self.c_0 * (force_on_fiber[0,0]*xs[0,0] + force_on_fiber[0,1]*xs[0,1] + force_on_fiber[0,2]*xs[0,2])

    if self.BC_start_1 == 'torque':
      # Apply torque
      B[4,0:num]       = D_2[0,:]
      B[5,num:2*num]   = D_2[0,:]
      B[6,2*num:3*num] = D_2[0,:]

      B_RHS[4] = self.BC_start_vec_1[0]
      B_RHS[5] = self.BC_start_vec_1[1]
      B_RHS[6] = self.BC_start_vec_1[2]
    elif self.BC_start_1 == 'angular_velocity':
      # Enforce orientation
      B[4,0:num]       = self.beta_tstep * D_1[0,:] / self.dt
      B[5,num:2*num]   = self.beta_tstep * D_1[0,:] / self.dt
      B[6,2*num:3*num] = self.beta_tstep * D_1[0,:] / self.dt
      
      link_dir = x[0,:]/np.sqrt(x[0,0]**2 + x[0,1]**2 + x[0,2]**2)
      omega = self.BC_start_vec_1
      B_RHS[4] = xs_rhs[0,0] / self.dt + omega[1]*link_dir[2] - omega[2]*link_dir[1]
      B_RHS[5] = xs_rhs[0,1] / self.dt + omega[2]*link_dir[0] - omega[0]*link_dir[2]
      B_RHS[6] = xs_rhs[0,2] / self.dt + omega[0]*link_dir[1] - omega[1]*link_dir[0]

    # Apply BC at the other end
    if self.BC_end_0 == 'force':
      # Apply force
      B[7,:]            = 0
      B[7,0:num]        = -self.E * D_3[-1,:]
      B[7,3*num:]       = xs[-1,0] * I[-1,:]
      B[8,:]            = 0
      B[8,num:2*num]    = -self.E * D_3[-1,:]
      B[8,3*num:]       = xs[-1,1] * I[-1,:]
      B[9,:]            = 0
      B[9,2*num:3*num]  = -self.E * D_3[-1,:]
      B[9,3*num:]       = xs[-1,2] * I[-1,:]
      B[10,:]           = 0
      B[10,0:num]       = self.E * D_2[-1,:] * xss[-1,0]
      B[10,num:2*num]   = self.E * D_2[-1,:] * xss[-1,1]
      B[10,2*num:3*num] = self.E * D_2[-1,:] * xss[-1,2]
      B[10,3*num:]      = I[-1,:]

      B_RHS[7]  = self.BC_end_vec_0[0]
      B_RHS[8]  = self.BC_end_vec_0[1]
      B_RHS[9]  = self.BC_end_vec_0[2]
      B_RHS[10] = np.dot(self.BC_end_vec_0, xs[-1,:])

    if self.BC_end_1 == 'torque':
      # Apply torque
      B[11,:]           = 0
      B[11,0:num]       = D_2[-1,:]
      B[12,:]           = 0
      B[12,num:2*num]   = D_2[-1,:]
      B[13,:]           = 0
      B[13,2*num:3*num] = D_2[-1,:]

      B_RHS[11] = self.BC_end_vec_1[0]
      B_RHS[12] = self.BC_end_vec_1[1]
      B_RHS[13] = self.BC_end_vec_1[2]

    if self.BC_end_0 == 'velocity':
      # Enforce velocity
      B[7,0:num]       = self.beta_tstep * I[-1,:] / self.dt
      B[8,num:2*num]   = self.beta_tstep * I[-1,:] / self.dt
      B[9,2*num:3*num] = self.beta_tstep * I[-1,:] / self.dt
      B[10,0:num]       = (6.0*self.E*(self.c_0+0*self.c_1)) * xss[-1,0] * D_3[-1,:]
      B[10,num:2*num]   = (6.0*self.E*(self.c_0+0*self.c_1)) * xss[-1,1] * D_3[-1,:]
      B[10,2*num:3*num] = (6.0*self.E*(self.c_0+0*self.c_1)) * xss[-1,2] * D_3[-1,:]
      B[10,3*num:]      = (2.0*self.c_0) * D_1[-1,:]

      B_RHS[7] = x_rhs[-1,0] / self.dt + self.BC_end_vec_0[0]
      B_RHS[8] = x_rhs[-1,1] / self.dt + self.BC_end_vec_0[1]
      B_RHS[9] = x_rhs[-1,2] / self.dt + self.BC_end_vec_0[2]
      B_RHS[10] = 0

      if v_on_fiber is not None:
        B_RHS[10] += -(v_on_fiber[-1,0]*xs[-1,0] + v_on_fiber[-1,1]*xs[-1,1] + v_on_fiber[-1,2]*xs[-1,2])
      if force_on_fiber is not None:
        B_RHS[10] += -2 * self.c_0 * (force_on_fiber[-1,0]*xs[-1,0] + force_on_fiber[-1,1]*xs[-1,1] + force_on_fiber[-1,2]*xs[-1,2])

    AA = np.vstack((A_rect, B))
    RHS_all = np.zeros(4*num)
    RHS_all[0:4*num-14] = RHS_rect
    RHS_all[4*num-14:] = B_RHS

    return AA, RHS_all


  def self_mobility(self):
    '''
    Build the self-mobility for one fiber.

    GK: Check if non-local part of self-mobility works.
        force has to be multiplied with weights in the non-local part
    '''

    # Compute material derivatives at time
    xs = self.xs
    num_points = self.num_points

    # Allocate memory for matrix
    I = np.eye(num_points)

    # Build submatrices to couple coordinates to coordinates
    M_XX = self.c_0 * (I + np.diag(xs[:,0]**2)     ) + self.c_1 * (I - np.diag(xs[:,0]**2))
    M_XY = self.c_0 * (    np.diag(xs[:,0]*xs[:,1])) + self.c_1 * (  - np.diag(xs[:,0]*xs[:,1]))
    M_XZ = self.c_0 * (    np.diag(xs[:,0]*xs[:,2])) + self.c_1 * (  - np.diag(xs[:,0]*xs[:,2]))
    M_YY = self.c_0 * (I + np.diag(xs[:,1]**2)     ) + self.c_1 * (I - np.diag(xs[:,1]**2))
    M_YZ = self.c_0 * (    np.diag(xs[:,1]*xs[:,2])) + self.c_1 * (  - np.diag(xs[:,1]*xs[:,2]))
    M_ZZ = self.c_0 * (I + np.diag(xs[:,2]**2)     ) + self.c_1 * (I - np.diag(xs[:,2]**2))


    # Collect all block matrices
    M = np.vstack((np.hstack((M_XX, M_XY, M_XZ)),
                   np.hstack((M_XY, M_YY, M_YZ)),
                   np.hstack((M_XZ, M_YZ, M_ZZ))))


    return M

  def self_mobility_nonlocal(self, fib_mat, reg = 5e-3, eps_distance = 1e-3):

    X = self.x
    xs = self.xs
    s = self.s
    num_points = self.num_points
    weights, out2, out3, out4 = fib_mat.get_matrices(self.length, self.num_points_up, 'weights_all')

    x, y, z = X[:,0], X[:,1], X[:,2]
    rx = x - x[:,None]
    ry = y - y[:,None]
    rz = z - z[:,None]
    ds = s - s[:,None]

    dr2 = rx*rx + ry*ry + rz*rz
    norm_r = np.sqrt(dr2)
    norm_r_reg = np.sqrt(dr2 + reg*reg)
    norm_ds = np.sqrt(ds*ds + reg*reg)

    rx_hat = (rx / norm_r).flatten()
    rx_hat[0::num_points+1] = xs[:,0]
    rx_hat = np.reshape(rx_hat, (num_points, num_points))
    ry_hat = (ry / norm_r).flatten()
    ry_hat[0::num_points+1] = xs[:,1]
    ry_hat = np.reshape(ry_hat, (num_points, num_points))
    rz_hat = (rz / norm_r).flatten()
    rz_hat[0::num_points+1] = xs[:,2]
    rz_hat = np.reshape(rz_hat, (num_points, num_points))

    fs = (self.c_1/2.0) * np.sum(weights[None,:] / norm_ds, axis = 1)

    K_XX = (0.500*self.c_1 * (1 / norm_r_reg + rx_hat * rx_hat / norm_r_reg)) * weights[None,:] - np.diag(fs * (1 + xs[:,0]**2))
    K_XY = (0.500*self.c_1 * (                 rx_hat * ry_hat / norm_r_reg)) * weights[None,:] - np.diag(fs *      xs[:,0] * xs[:,1])
    K_XZ = (0.500*self.c_1 * (                 rx_hat * rz_hat / norm_r_reg)) * weights[None,:] - np.diag(fs *      xs[:,0] * xs[:,2])
    K_YY = (0.500*self.c_1 * (1 / norm_r_reg + ry_hat * ry_hat / norm_r_reg)) * weights[None,:] - np.diag(fs * (1 + xs[:,1]**2))
    K_YZ = (0.500*self.c_1 * (                 ry_hat * rz_hat / norm_r_reg)) * weights[None,:] - np.diag(fs *      xs[:,1] * xs[:,2])
    K_ZZ = (0.500*self.c_1 * (1 / norm_r_reg + rz_hat * rz_hat / norm_r_reg)) * weights[None,:] - np.diag(fs * (1 + xs[:,2] * xs[:,2]))


    return K_XX, K_XY, K_XZ, K_YY, K_YZ, K_ZZ



  def self_residual_full(self, xT, fib_mat):
    '''
    Compute the residual of the nonlinear function.
    '''
    D_1, D_2, D_3, D_4 = fib_mat.get_matrices(self.length, self.num_points_up, 'Ds')
    # Compute material derivatives
    xs = np.dot(D_1, xT[:,0:3])
    xssss = np.dot(D_4, xT[:,0:3])

    # Allocate memory for matrix
    I = np.eye(self.num_points)

    # Build submatrices to couple coordinates to coordinates
    M_XX = self.c_0 * (I + np.diag(xs[:,0]**2)     ) + self.c_1 * (I - np.diag(xs[:,0]**2))
    M_XY = self.c_0 * (    np.diag(xs[:,0]*xs[:,1])) + self.c_1 * (  - np.diag(xs[:,0]*xs[:,1]))
    M_XZ = self.c_0 * (    np.diag(xs[:,0]*xs[:,2])) + self.c_1 * (  - np.diag(xs[:,0]*xs[:,2]))
    M_YY = self.c_0 * (I + np.diag(xs[:,1]**2)     ) + self.c_1 * (I - np.diag(xs[:,1]**2))
    M_YZ = self.c_0 * (    np.diag(xs[:,1]*xs[:,2])) + self.c_1 * (  - np.diag(xs[:,1]*xs[:,2]))
    M_ZZ = self.c_0 * (I + np.diag(xs[:,2]**2)     ) + self.c_1 * (I - np.diag(xs[:,2]**2))

    # Collect all block matrices
    M = np.vstack((np.hstack((M_XX, M_XY, M_XZ)),
                   np.hstack((M_XY, M_YY, M_YZ)),
                   np.hstack((M_XZ, M_YZ, M_ZZ))))

    # Compute force
    fx = -self.E * xssss[:,0] + np.dot(D_1, (xs[:,0] * xT[:,3]))
    fy = -self.E * xssss[:,1] + np.dot(D_1, (xs[:,1] * xT[:,3]))
    fz = -self.E * xssss[:,2] + np.dot(D_1, (xs[:,2] * xT[:,3]))
    f = np.concatenate([fx, fy, fz])

    # Compute nonlinear functions
    Mf_aux = np.dot(M, f)
    Mf = np.empty((xT.shape[0], 3))
    Mf[:,0] = Mf_aux[0*self.num_points : 1*self.num_points]
    Mf[:,1] = Mf_aux[1*self.num_points : 2*self.num_points]
    Mf[:,2] = Mf_aux[2*self.num_points : 3*self.num_points]
    xout = (xT[:,0:3] - self.x)  - self.dt * Mf
    Tout = np.ones(self.num_points) - (xs[:,0]**2 + xs[:,1]**2 + xs[:,2]**2)
    out = np.empty_like(xT)
    out[:,0] = xout[:,0]
    out[:,1] = xout[:,1]
    out[:,2] = xout[:,2]
    out[:,3] = Tout
    return out


  def self_residual_force(self, x, fib_mat):
    '''
    Compute the residual of the nonlinear function.
    '''
    D_1, D_2, D_3, D_4 = fib_mat.get_matrices(self.length, self.num_points_up, 'Ds')
    # Take coordinates
    xT = np.empty((self.num_points, 4))
    xT[:,0] = x[0*self.num_points:1*self.num_points]
    xT[:,1] = x[1*self.num_points:2*self.num_points]
    xT[:,2] = x[2*self.num_points:3*self.num_points]
    xT[:,3] = x[3*self.num_points:4*self.num_points]

    # Compute material derivatives
    xs = np.dot(D_1, xT[:,0:3])
    xssss = np.dot(D_4, xT[:,0:3])

    # Allocate memory for matrix
    I = np.eye(self.num_points)
    M = self.self_mobility(self.x)

    # Compute force
    fx = -self.E * xssss[:,0] + np.dot(D_1, (xs[:,0] * xT[:,3]))
    fy = -self.E * xssss[:,1] + np.dot(D_1, (xs[:,1] * xT[:,3]))
    fz = -self.E * xssss[:,2] + np.dot(D_1, (xs[:,2] * xT[:,3]))
    f = np.concatenate([fx, fy, fz])

    # Compute new position
    Mf_aux = np.dot(M, f)
    xout = (x[0:3*self.num_points] - self.x.flatten(order='F')) - self.dt * Mf_aux
    # Multiplier for C_F
    xout[0*self.num_points] += x[4*self.num_points + 0]
    xout[1*self.num_points] += x[4*self.num_points + 1]
    xout[2*self.num_points] += x[4*self.num_points + 2]
    xout[1*self.num_points-1] += x[4*self.num_points + 3]
    xout[2*self.num_points-1] += x[4*self.num_points + 4]
    xout[3*self.num_points-1] += x[4*self.num_points + 5]
    # Multiplier for C_L
    xout[0*self.num_points+1] += x[4*self.num_points + 8] - x[4*self.num_points + 7]
    xout[1*self.num_points+1] += x[4*self.num_points + 6] - x[4*self.num_points + 8]
    xout[2*self.num_points+1] += x[4*self.num_points + 7] - x[4*self.num_points + 6]
    xout[1*self.num_points-2] += x[4*self.num_points + 11] - x[4*self.num_points + 10]
    xout[2*self.num_points-2] += x[4*self.num_points + 9]  - x[4*self.num_points + 11]
    xout[3*self.num_points-2] += x[4*self.num_points + 10] - x[4*self.num_points + 9]

    # Compute constraints
    F_ext_start = np.array([0.0, 0.0, 0.0])
    F_ext_end = np.array([0.0, 0.0, 0.0])
    L_ext_start = np.array([0.0, 0.0, 0.0])
    L_ext_end = np.array([0.0, 0.0, 0.0])
    Tout = np.ones(self.num_points) - (xs[:,0]**2 + xs[:,1]**2 + xs[:,2]**2)
    C_F_start = F_ext_start + (-self.E * np.dot(D_3[0],  xT[:,0:3]) + x[3*self.num_points]   * xs[0])
    C_F_end   = F_ext_end   - (-self.E * np.dot(D_3[-1], xT[:,0:3]) + x[4*self.num_points-1] * xs[-1])
    xss_start = np.dot(D_2[0],  xT[:,0:3])
    xss_end   = np.dot(D_2[-1], xT[:,0:3])
    C_L_start = L_ext_start + self.E * np.cross(xss_start, xs[0])
    C_L_end   = L_ext_end   - self.E * np.cross(xss_end, xs[-1])
    res = np.concatenate([xout.flatten(), Tout.flatten(), C_F_start.flatten(), C_F_end.flatten(), C_L_start.flatten(), C_L_end.flatten()])

    return res


  def self_jacobian_force(self, x, fib_mat):
    '''
    Jacobian for the nonlinear solver.
    '''
    # Take coordinates
    D_1, D_2, D_3, D_4 = fib_mat.get_matrices(self.length, self.num_points_up, 'Ds')
    xT = np.empty((self.num_points, 4))
    xT[:,0] = x[0*self.num_points:1*self.num_points]
    xT[:,1] = x[1*self.num_points:2*self.num_points]
    xT[:,2] = x[2*self.num_points:3*self.num_points]
    xT[:,3] = x[3*self.num_points:4*self.num_points]

    # Compute material derivatives
    xs = np.dot(D_1, xT[:,0:3])
    xssss = np.dot(D_4, xT[:,0:3])

    # Allocate memory for matrix
    I = np.eye(self.num_points)
    M = self.self_mobility(self.x)
    M_XX = M[0*self.num_points:1*self.num_points, 0*self.num_points:1*self.num_points]
    M_XY = M[0*self.num_points:1*self.num_points, 1*self.num_points:2*self.num_points]
    M_XZ = M[0*self.num_points:1*self.num_points, 2*self.num_points:3*self.num_points]
    M_YY = M[1*self.num_points:2*self.num_points, 1*self.num_points:2*self.num_points]
    M_YZ = M[1*self.num_points:2*self.num_points, 2*self.num_points:3*self.num_points]
    M_ZZ = M[2*self.num_points:3*self.num_points, 2*self.num_points:3*self.num_points]

    # Factor force
    dFX_dX = -self.E * D_4 + np.dot(D_1, np.dot(np.diag(xT[:,3]), D_1))
    J_XX = I - self.dt * np.dot(M_XX, dFX_dX)
    J_XY =   - self.dt * np.dot(M_XY, dFX_dX)
    J_XZ =   - self.dt * np.dot(M_XZ, dFX_dX)
    J_YY = I - self.dt * np.dot(M_YY, dFX_dX)
    J_YZ =   - self.dt * np.dot(M_YZ, dFX_dX)
    J_ZZ = I - self.dt * np.dot(M_ZZ, dFX_dX)

    dFX_dT = np.dot(D_1, np.diag(xs[:,0]))
    dFY_dT = np.dot(D_1, np.diag(xs[:,1]))
    dFZ_dT = np.dot(D_1, np.diag(xs[:,2]))
    J_XT = - self.dt * (np.dot(M_XX, dFX_dT) + np.dot(M_XY, dFY_dT) + np.dot(M_XZ, dFZ_dT))
    J_YT = - self.dt * (np.dot(M_XY, dFX_dT) + np.dot(M_YY, dFY_dT) + np.dot(M_YZ, dFZ_dT))
    J_ZT = - self.dt * (np.dot(M_XZ, dFX_dT) + np.dot(M_YZ, dFY_dT) + np.dot(M_ZZ, dFZ_dT))
    J_TX = np.dot(np.diag(-2.0 * xs[:,0]), D_1)
    J_TY = np.dot(np.diag(-2.0 * xs[:,1]), D_1)
    J_TZ = np.dot(np.diag(-2.0 * xs[:,2]), D_1)
    J_TT = np.zeros((self.num_points, self.num_points))

    # Constraints at the ends
    num_constraints = 12
    J_XC = np.zeros((self.num_points, num_constraints))
    J_XC[0, 0]  = 1.0
    J_XC[-1,3]  = 1.0
    J_XC[1, 8]  =  1.0
    J_XC[1, 7]  = -1.0
    J_XC[-2,11] =  1.0
    J_XC[-2,10] = -1.0
    J_YC = np.zeros((self.num_points, num_constraints))
    J_YC[0, 1]  = 1.0
    J_YC[-1,4]  = 1.0
    J_YC[1, 6]  =  1.0
    J_YC[1, 8]  = -1.0
    J_YC[-2,9]  =  1.0
    J_YC[-2,11] = -1.0
    J_ZC = np.zeros((self.num_points, num_constraints))
    J_ZC[0, 2]  = 1.0
    J_ZC[-1,5]  = 1.0
    J_ZC[1, 7]  =  1.0
    J_ZC[1, 6]  = -1.0
    J_ZC[-2,10] =  1.0
    J_ZC[-2,9]  = -1.0
    J_TC = np.zeros((self.num_points, num_constraints))

    xss_start = np.dot(D_2[0],  xT[:,0:3])
    xss_end   = np.dot(D_2[-1], xT[:,0:3])

    J_CX = np.zeros((num_constraints, self.num_points))
    J_CX[0,:]  =  (-self.E * D_3[0]  + x[3*self.num_points]   * D_1[0])
    J_CX[3,:]  = -(-self.E * D_3[-1] + x[4*self.num_points-1] * D_1[-1])
    J_CX[7,:]  =  self.E * (xss_start[2] * D_1[0]  - D_2[0]  * xs[0,2])
    J_CX[10,:] = -self.E * (xss_end[2]   * D_1[-1] - D_2[-1] * xs[-1,2])
    J_CX[8,:]  =  self.E * (D_2[0]  * xs[0,1]  - xss_start[1] * D_1[0])
    J_CX[11,:] = -self.E * (D_2[-1] * xs[-1,1] - xss_end[1]   * D_1[-1])
    J_CY = np.zeros((num_constraints, self.num_points))
    J_CY[1,:]  =  (-self.E * D_3[0]  + x[3*self.num_points]   * D_1[0])
    J_CY[4,:]  = -(-self.E * D_3[-1] + x[4*self.num_points-1] * D_1[-1])
    J_CY[8,:]  =  self.E * (xss_start[0] * D_1[0]  - D_2[0]  * xs[0,0])
    J_CY[11,:] = -self.E * (xss_end[0]   * D_1[-1] - D_2[-1] * xs[-1,0])
    J_CY[6,:]  =  self.E * (D_2[0]  * xs[0,2]  - xss_start[2] * D_1[0])
    J_CY[9,:]  = -self.E * (D_2[-1] * xs[-1,2] - xss_end[2]   * D_1[-1])
    J_CZ = np.zeros((num_constraints, self.num_points))
    J_CZ[2,:]  =  (-self.E * D_3[0]  + x[3*self.num_points]   * D_1[0])
    J_CZ[5,:]  = -(-self.E * D_3[-1] + x[4*self.num_points-1] * D_1[-1])
    J_CZ[6,:]  =  self.E * (xss_start[1] * D_1[0]  - D_2[0]  * xs[0,1])
    J_CZ[9,:]  = -self.E * (xss_end[1]   * D_1[-1] - D_2[-1] * xs[-1,1])
    J_CZ[7,:]  =  self.E * (D_2[0]  * xs[0,0]  - xss_start[0] * D_1[0])
    J_CZ[10,:] = -self.E * (D_2[-1] * xs[-1,0] - xss_end[0]   * D_1[-1])

    J_CT = np.zeros((num_constraints, self.num_points))
    J_CT[0,0]  =  xs[0, 0]
    J_CT[3,-1] = -xs[-1,0]
    J_CT[1,0]  =  xs[0, 1]
    J_CT[4,-1] = -xs[-1,1]
    J_CT[2,0]  =  xs[0, 2]
    J_CT[5,-1] = -xs[-1,2]
    J_CC = np.zeros((num_constraints, num_constraints))

    # Collect all block matrices
    J = np.vstack((np.hstack((J_XX, J_XY, J_XZ, J_XT, J_XC)),
                   np.hstack((J_XY, J_YY, J_YZ, J_YT, J_YC)),
                   np.hstack((J_XZ, J_YZ, J_ZZ, J_ZT, J_ZC)),
                   np.hstack((J_TX, J_TY, J_TZ, J_TT, J_TC)),
                   np.hstack((J_CX, J_CY, J_CZ, J_CT, J_CC))))
    return J


  def preconditioner_jacobian(self, xT, J):
    '''
    Use LU as a preconditioner for one fiber.
    '''
    # Compute Jacobian if necessary
    if J is None:
      J = self.self_jacobian_force(xT)

    # Factorize the Jacobian
    if False:
      LU = scipy.linalg.lu_factor(J)
      LU_partial = partial(scipy.linalg.lu_solve, LU)
      return scspli.LinearOperator(J.shape, matvec=LU_partial)
    elif False:
      try:
        Pinv = np.linalg.pinv(J)
      except np.linalg.linalg.LinAlgError as err:
        Pinv= np.dot(J, np.linalg.pinv(np.dot(J.T, J))).T

      def product(x):
        return np.dot(Pinv, x)
      return scspli.LinearOperator(J.shape, matvec=product)
    else:
      try:
        def product_svd(x, U, Sinv, V):
          return np.dot(V.T, (Sinv * np.dot(U.T, x)))

        U, S, V = np.linalg.svd(J)
        Sinv = [1.0 / x if abs(x) > S[0] * 1e-15 else 0.0 for x in S]
        product_partial = partial(product_svd, U = U, Sinv = Sinv, V = V)
        return scspli.LinearOperator(J.shape, matvec=product_partial)
      except np.linalg.linalg.LinAlgError as err:
        def product_svd(x, J, U, Sinv, V):
          return np.dot(J, np.dot(V.T, (Sinv * np.dot(U.T, x))))
        JJ = np.dot(J.T, J)
        U, S, V = np.linalg.svd(JJ)
        Sinv = [1.0 / x if abs(x) > S[0] * 1e-15 else 0.0 for x in S]
        product_partial = partial(product_svd, J = J, U = U, Sinv = Sinv, V = V)
        return scspli.LinearOperator(J.shape, matvec=product_partial)


  def filter(self,p,fib_mat):
    '''
    Set to zero all the modes with q > p.
    '''
    modes = np.empty_like(self.x)
    modes[:, 0] = cheb.cheb_calc_coef(self.x[:, 0])
    modes[:, 1] = cheb.cheb_calc_coef(self.x[:, 1])
    modes[:, 2] = cheb.cheb_calc_coef(self.x[:, 2])
    modes_tension = cheb.cheb_calc_coef(self.tension)

    modes[p:,:] = 0.0
    modes_tension[p:] = 0.0

    alpha, out2, out3, out4 = fib_mat.get_matrices(self.length, self.num_points_up, 'alpha_and_s')
    # GK: SHOULD WE PASS 1 TO cheb_eval, ALPHA IS [-1, ..., 1]
    self.x[:, 0] = cheb.cheb_eval(alpha, modes[:, 0], order = 1)
    self.x[:, 1] = cheb.cheb_eval(alpha, modes[:, 1], order = 1)
    self.x[:, 2] = cheb.cheb_eval(alpha, modes[:, 2], order = 1)
    self.tension = cheb.cheb_eval(alpha, modes_tension)


  def correct(self, fib_mat):
    '''
    Solve a nonlinear system to fix the inextensibility violations.
    Solve for x:
    min ||x - x_new||_2
    subject to D_1*x - 1 = 0
    '''

    def constraint(x, D_1, D_2):
      timer.timer('apply_constraint')
      x3 = x.reshape((x.size //3, 3))
      xs = np.dot(D_1, x3)
      # C_0 = xs[:,0]**2 + xs[:,1]**2 + xs[:,2]**2 - 1.0
      C_0 = xs[1:-1,0]**2 + xs[1:-1,1]**2 + xs[1:-1,2]**2 - 1.0
      # C_1 = np.dot(D_2[0,:], x3)
      C_2 = np.dot(D_2[-1,:], x3)
      timer.timer('apply_constraint')
      # return np.concatenate([C_0, C_1, C_2])
      return np.concatenate([C_0, C_2])

    D_1, D_2, out3, out4 = fib_mat.get_matrices(self.length, self.num_points_up, 'Ds')
    constraint_partial = partial(constraint, D_1=D_1, D_2=D_2)

    def jacobian_const(x, D_1):
      timer.timer('apply_jac_const')
      xs = np.dot(D_1, x.reshape(x.size // 3, 3))
      J_TX = np.dot(np.diag(2.0 * xs[:,0]), D_1)
      J_TY = np.dot(np.diag(2.0 * xs[:,1]), D_1)
      J_TZ = np.dot(np.diag(2.0 * xs[:,2]), D_1)
      timer.timer('apply_jac_const')
      return np.hstack((J_TX, J_TY, J_TZ))
    jacobian_const_partial = partial(jacobian_const, D_1=D_1)
    jacobian_const_partial = None

    eq_const = {'type': 'eq',
                'fun': constraint_partial,
                'jac': jacobian_const_partial}

    def func(x, x_new):
      timer.timer('apply_func')
      F = sum((x - x_new.flatten())**2)
      timer.timer('apply_func')
      return F
    func_partial = partial(func, x_new=self.x_new)

    def jacobian(x, x_new):
      timer.timer('apply_jac')
      J = 2.0 * (x.flatten() - x_new.flatten()) # + np.ones(x.size) * 1e-10
      timer.timer('apply_jac')
      return J
    jacobian_partial = partial(jacobian, x_new=self.x_new)

    timer.timer('minimize')
    res = scop.minimize(func_partial, self.x_new, method='SLSQP', jac=jacobian_partial, constraints=eq_const, options={'ftol': 1e-8, 'disp': True, 'maxiter': 360})
    timer.timer('minimize')
    self.x = res.x.reshape(res.x.size // 3, 3)

  def compute_modes(self):
    '''

    '''
    # if np.isfinite(self.x).all() is False:
    #   print('numpy.isfinite(self.x).all() !!!')
    #   sys.exit()
    # if np.isfinite(self.xs).all() is False:
    #   print('numpy.isfinite(self.xs).all() !!!')
    #   sys.exit()
    self.x_modes[:, 0] = cheb.cheb_calc_coef(self.x[:, 0])
    self.x_modes[:, 1] = cheb.cheb_calc_coef(self.x[:, 1])
    self.x_modes[:, 2] = cheb.cheb_calc_coef(self.x[:, 2])
    self.xs_modes[:, 0] = cheb.cheb_calc_coef(self.xs[:, 0])
    self.xs_modes[:, 1] = cheb.cheb_calc_coef(self.xs[:, 1])
    self.xs_modes[:, 2] = cheb.cheb_calc_coef(self.xs[:, 2])



  def force_operator(self, fib_mat):
    '''
    f = -E * X_ssss + (T*X_s)_s
    f = -E * X_ssss + T_s * X_s + T * X_ss
    '''
    # -self.c_0 * (2.0 * (D_1.T * xs[:,0]).T + np.diag(xss[:,0])) - self.c_1 * np.diag(xss[:,0])


    xs = self.xs
    xss = self.xss
    D_1, out2, out3, D_4 = fib_mat.get_matrices(self.length, self.num_points_up, 'Ds')
    num_points = self.num_points
    num_points_down = self.num_points


    Z = np.zeros((num_points_down, num_points_down))
    A_XX = -self.E * D_4
    A_YY = -self.E * D_4
    A_ZZ = -self.E * D_4

    A_XT = np.diag(xss[:,0])
    A_YT = np.diag(xss[:,1])
    A_ZT = np.diag(xss[:,2])

    A_XT += (D_1.T * xs[:,0]).T
    A_YT += (D_1.T * xs[:,1]).T
    A_ZT += (D_1.T * xs[:,2]).T


    A = np.vstack((np.hstack((A_XX, Z,    Z,    A_XT)),
                   np.hstack((Z,    A_YY, Z,    A_YT)),
                   np.hstack((Z,    Z   , A_ZZ, A_ZT))))
    return A



  def force_torque_link_operators(self, fib_mat):
    '''
    F[0] = -E * X_sss[0] - X_s[0] * T[0]
    F[0] = -E * X_sss[0] + X_s[0] * T[0]

    tau[0] = E * X_ss \cross X_s
    '''
    #self.E * self.D_3[0,:]
    #  A[0,3*num:]          = -xs[0,0] * I[0,:]
    D_1, D_2, D_3, D_4 = fib_mat.get_matrices(self.length, self.num_points_up, 'Ds')
    I = np.eye(self.num_points)
    Z = np.zeros(self.num_points)
    FXX = -self.E * D_3[0,:]
    FYY = -self.E * D_3[0,:]
    FZZ = -self.E * D_3[0,:]
    FXT = self.xs[0,0] * I[0,:]
    FYT = self.xs[0,1] * I[0,:]
    FZT = self.xs[0,2] * I[0,:]
    F_0 = np.copy(np.vstack((np.hstack((FXX, Z, Z, FXT)),
                             np.hstack((Z, FYY, Z, FYT)),
                             np.hstack((Z, Z, FZZ, FZT)))))
    FXX = -self.E * D_3[-1,:]
    FYY = -self.E * D_3[-1,:]
    FZZ = -self.E * D_3[-1,:]
    FXT = self.xs[-1,0] * I[-1,:]
    FYT = self.xs[-1,1] * I[-1,:]
    FZT = self.xs[-1,2] * I[-1,:]
    F_1 = np.copy(np.vstack((np.hstack((FXX, Z, Z, FXT)),
                             np.hstack((Z, FYY, Z, FYT)),
                             np.hstack((Z, Z, FZZ, FZT)))))
    tau_0 = np.zeros_like(F_0)
    tau_1 = np.zeros_like(F_1)

    return F_0, F_1, tau_0, tau_1
