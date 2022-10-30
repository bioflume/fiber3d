'''
Nonlinear integrator for multiple fibers.

'''
from __future__ import division, print_function
import numpy as np
from functools import partial
import scipy.linalg as scla
import scipy.sparse as scsp
import scipy.sparse.linalg as scspla
import sys
import multi
from bpm_utilities import gmres

from utils import nonlinear 
from utils import timer
from utils import miscellaneous
from fiber import fiber
#from multi import multi
from quaternion import quaternion
from forces import forces

#from lib import periodic_fmm as fmm


class integrator(object):
  '''
  Collection of integrators.
  '''
  def __init__(self, scheme, fibers, tolerance, verbose, p_fmm, eta, bodies):
    self.scheme = scheme
    self.fibers = fibers
    self.tolerance = tolerance
    self.verbose = verbose
    self.p_fmm = p_fmm
    self.eta = eta
    self.linear_operator = None
    self.bodies = bodies
    self.molecular_motor = None
    self.As_dok_BC = None
    self.force_fg_0 = 0
    self.force_fg_1 = 0
    self.force_fg_2 = 0
    self.force_fg_3 = 0
    self.force_fg_4 = 0
    self.filtering = fibers[0].filtering

    self.Nfibers_markers = sum([x.num_points for x in fibers])

    return


  def advance_time_step(self, dt, *args, **kwargs):
    '''
    Advance time step with integrator self.scheme
    '''
    return getattr(self, self.scheme)(dt, *args, **kwargs)


  def nonlinear_solver(self, dt):
    '''
    Solve nonlinear system to find new configuration.
    '''
    for fib in self.fibers:
      x = np.zeros(4 * fib.num_points + 12)
      x[0*fib.num_points:1*fib.num_points] = fib.x[:,0]
      x[1*fib.num_points:2*fib.num_points] = fib.x[:,1]
      x[2*fib.num_points:3*fib.num_points] = fib.x[:,2]
      (sol, res_norm, it, nonlinear_evaluations, gmres_iterations) = nonlinear.nonlinear_solver(fib.self_residual_force,
                                                                                                x,
                                                                                                fib.self_jacobian_force,
                                                                                                M = fib.preconditioner_jacobian,
                                                                                                verbose = self.verbose,
                                                                                                gmres_verbose = False,
                                                                                                max_outer_it = 20,
                                                                                                max_inner_it = 20,
                                                                                                tol = self.tolerance,
                                                                                                gmres_restart = 100)
      fib.x[:,0] = sol[0*fib.num_points : 1*fib.num_points]
      fib.x[:,1] = sol[1*fib.num_points : 2*fib.num_points]
      fib.x[:,2] = sol[2*fib.num_points : 3*fib.num_points]      
      fib.tension = np.copy(sol[3 * fib.num_points : 4 * fib.num_points])
    return



  def linear_solver(self, dt):
    '''
    Use linear solve to find new configuration.
    '''
    # Copy current state
    system_size = len(self.bodies) * 6 + self.Nfibers_markers * 4
    x0 = np.zeros(system_size)
    offset = np.zeros(len(self.fibers) + 1, dtype=int)
    offset_bodies = len(self.bodies) * 6
    for k, fib in enumerate(self.fibers):
      x0[offset_bodies + offset[k] * 4 + fib.num_points * 0 : offset_bodies + offset[k] * 4 + fib.num_points * 1 ] = fib.x[:,0]
      x0[offset_bodies + offset[k] * 4 + fib.num_points * 1 : offset_bodies + offset[k] * 4 + fib.num_points * 2 ] = fib.x[:,1]
      x0[offset_bodies + offset[k] * 4 + fib.num_points * 2 : offset_bodies + offset[k] * 4 + fib.num_points * 3 ] = fib.x[:,2]
      offset[k + 1] = offset[k] + fib.num_points

    # Set fibers boundaries conditions (spring to bodies)
    for k, b in enumerate(self.bodies):
      links_force = np.zeros_like(b.links_location)
      links_torque = np.zeros_like(b.links_location)
      
      # Get links location
      rotation_matrix = b.orientation.rotation_matrix()
      links_location = np.array([np.dot(rotation_matrix, vec) for vec in b.links_location])
      links_location += b.location

      # Get links axis orientation
      links_axis = np.array([np.dot(rotation_matrix, vec) for vec in b.links_axis])

      # Loop over attached fibers
      offset_links = b.links_first_fibers
      for i, link in enumerate(links_location):
        # Compute harmonic forces 
        force = b.links_spring_constant[i] * (self.fibers[offset_links+i].x[0] - link) 
        
        # Compute harmonic torque
        # xs = np.dot(self.fibers[offset_links+i].D_1[0,:], self.fibers[offset_links+i].x)
        # xs = xs / np.linalg.norm(xs)
        # xss = np.dot(self.fibers[offset_links+i].D_2[0,:], self.fibers[offset_links+i].x)
        # xss = xss / np.linalg.norm(xss)
        # theta = np.arccos(np.dot(links_axis[i], xs))
        # torque = (b.links_spring_constants_angle[i] * (theta - 0.0)) * np.cross(xss, xs)

        # Compute torque
        xs = np.dot(self.fibers[offset_links+i].D_1[0,:], self.fibers[offset_links+i].x)
        xss = np.dot(self.fibers[offset_links+i].D_2[0,:], self.fibers[offset_links+i].x)
        # torque = self.fibers[offset_links+i].E * np.cross(xss, xs) 
        # torque = self.fibers[offset_links+i].E * np.cross(xss, links_axis[i]) 
        torque = -self.fibers[offset_links+i].E * np.cross(xs, links_axis[i]) 

        # Save forces and torques and apply BC
        links_force[i] = force
        links_torque[i] = torque
        self.fibers[offset_links+i].set_BC(BC_start_0 = 'force', BC_start_vec_0 = force, 
                                           BC_start_1 = 'torque', BC_start_vec_1 = torque,
                                           BC_end_0 = 'force', BC_end_vec_0 = np.array([1.0, 0.0, 0.0]))
        # self.fibers[offset_links+i].set_BC(BC_start_0 = 'force', BC_start_vec_0 = force, 
        #                                    BC_start_1 = 'angle', BC_start_vec_1 = links_axis[i],
        #                                    BC_end_0 = 'force', BC_end_vec_0 = np.array([1.0, 0.0, 0.0]))
        print('link      = ', link)
        print('x_0       = ', self.fibers[offset_links+i].x[0])
        print('force     = ', force)
        print('link_axis = ', links_axis[i])
        # print('xss       = ', xss)
        print('xs        = ', xs)
        print('dot       = ', np.dot(links_axis[i], xs))
        # print('theta     = ', theta)
        print('torque    = ', torque)
        # print('\n')

      # Compute force torque acting on the body
      K = b.calc_K_matrix(r_vectors = links_location, Nblobs = len(links_location))
      force_torque = np.dot(K.T, links_force.flatten())
      force_torque[3:] += sum(links_torque)
      b.force_torque = np.copy(force_torque)
      print('F         = ', force_torque[0:3])
      print('T         = ', force_torque[3:])
        
    # Build the linear system
    timer.timer('Build_linear_system')
    A_all = []
    RHS_all = np.zeros(system_size)
    for k, fib in enumerate(self.fibers):
      # Set linear operator
      A = fib.form_linear_operator()
      
      # Set RHS
      RHS = fib.compute_RHS()

      # Apply BC
      A, RHS = fib.apply_BC(A, RHS)

      # Save data
      A_all.append(A)
      RHS_all[offset_bodies + offset[k] * 4 : offset_bodies + (offset[k] + fib.num_points) * 4] = RHS
    timer.timer('Build_linear_system')
      
    # Set RHS for rigid bodies
    for k, b in enumerate(self.bodies):
      RHS_all[6*k : 6*(k+1)] = b.force_torque

    # Compute Preconditioners 
    timer.timer('PC_init')
    LU_all = [] 
    P_all = [] 
    for k, fib in enumerate(self.fibers): 
      (LU, P) = scla.lu_factor(A_all[k]) 
      P_all.append(P) 
      LU_all.append(LU) 
    def P_inv(x, LU, P, offset, offset_bodies): 
      timer.timer('PC_apply')
      y = np.empty_like(x)
      # PC for bodies is the identity matrix
      y[0 : offset_bodies] = x[0 : offset_bodies]
      for i in range(len(LU)): 
        y[offset_bodies + offset[i] * 4 : offset_bodies + offset[i+1] * 4] = scla.lu_solve((LU[i], P[i]), 
                                                                                           x[offset_bodies+offset[i]*4 : offset_bodies+offset[i+1]*4])
      timer.timer('PC_apply')
      return y 
    P_inv_partial = partial(P_inv, LU = LU_all, P = P_all, offset = offset, offset_bodies = offset_bodies) 
    P_inv_partial_LO = scspla.LinearOperator((system_size, system_size), matvec = P_inv_partial, dtype='float64') 
    timer.timer('PC_init')

    # Transform to sparse matrix 
    As = scsp.block_diag(A_all) 
    
    # Prepare linear operator
    linear_operator_partial = partial(self.linear_operator, 
                                      As = As, 
                                      fibers = self.fibers, 
                                      Nmarkers = self.Nfibers_markers, 
                                      eta = self.eta, 
                                      dt = dt, 
                                      p_fmm = self.p_fmm,
                                      offset = offset,
                                      bodies = self.bodies)   
    A = scspla.LinearOperator((system_size, system_size), matvec = linear_operator_partial, dtype='float64')

    # Solve linear system 
    counter = gmres_counter(print_residual = self.verbose) 
    (sol, info_precond) = scspla.gmres(A, RHS_all, x0=x0, tol=self.tolerance, M=P_inv_partial_LO, maxiter=1000, restart=150, callback=counter) 

    # Update bodies configuration
    for k, b in enumerate(self.bodies):
      # print('body             = ', k)
      # print('velocity         = ', sol[k*6 : k*6 + 3])
      # print('angular velocity = ', sol[k*6+3 : k*6 + 6])
      b.location = b.location + sol[k*6 : k*6 + 3] * dt
      quaternion_dt = quaternion.Quaternion.from_rotation(sol[k*6+3 : k*6 + 6] * dt)
      b.orientation = quaternion_dt * b.orientation

    # Update fiber configuration
    for k, fib in enumerate(self.fibers):
      fib.x[:,0] = sol[offset_bodies + offset[k] * 4 + 0 * fib.num_points : offset_bodies + offset[k] * 4 + 1 * fib.num_points]
      fib.x[:,1] = sol[offset_bodies + offset[k] * 4 + 1 * fib.num_points : offset_bodies + offset[k] * 4 + 2 * fib.num_points]
      fib.x[:,2] = sol[offset_bodies + offset[k] * 4 + 2 * fib.num_points : offset_bodies + offset[k] * 4 + 3 * fib.num_points]
      fib.tension = np.copy(sol[offset_bodies + offset[k] * 4 + 3 * fib.num_points : offset_bodies + offset[k] * 4 + 4 * fib.num_points])
    return






























  def linear_solver_new(self, dt, *args, **kwargs):
    '''
    Solve nonlinear system to find new configuration.
    '''
    # 0. Precomputation 
    # Array x0 has all the degrees of freedom, first bodies and then fibers
    # x0 = (body_0, body_1, ..., fiber_0, fiber_1, ...)
    # 
    # offset[k] = has the number of fiber points before fiber k in the array x0
    timer.timer('initialization_step')
    system_size = len(self.bodies) * 6 + self.Nfibers_markers * 4
    offset = np.zeros(len(self.fibers) + 1, dtype=int)
    offset_bodies = len(self.bodies) * 6
    x0 = np.zeros(system_size)
    for k, b in enumerate(self.bodies):
      x0[6*k:6*k+3] = b.location
    for k, fib in enumerate(self.fibers):
      offset[k + 1] = offset[k] + fib.num_points
      x0[offset_bodies+4*offset[k]+0 : offset_bodies+4*offset[k]+0+fib.num_points*4 : 4] = fib.x[:,0]
      x0[offset_bodies+4*offset[k]+1 : offset_bodies+4*offset[k]+1+fib.num_points*4 : 4] = fib.x[:,1]
      x0[offset_bodies+4*offset[k]+2 : offset_bodies+4*offset[k]+2+fib.num_points*4 : 4] = fib.x[:,2]

      # Compute derivatives along the fiber
      fib.xs = np.dot(fib.D_1, fib.x)
      fib.xss = np.dot(fib.D_2, fib.x)
      fib.xsss = np.dot(fib.D_3, fib.x)
      fib.xssss = np.dot(fib.D_4, fib.x)
    timer.timer('initialization_step')

    # 1. Compute external forces
    force_fib = np.zeros((self.Nfibers_markers, 3))
    force_bodies, force_fibers = forces.compute_external_forces(self.bodies, 
                                                                self.fibers, 
                                                                x0, 
                                                                self.Nfibers_markers,
                                                                offset, 
                                                                offset_bodies, 
                                                                *args, **kwargs)
    # Concatenate fibers
    num_fibers = len(self.fibers)
    xf = np.concatenate([fib.x for fib in self.fibers])
    array_num_points = np.array([fib.num_points for fib in self.fibers])
    offset_particles = offset[0:-1]

    # Compute material derivative
    xs = []
    for fib in self.fibers:
      xs.append( np.dot(fib.D_1, fib.x) )
    xs = np.array(xs)
    xs = np.reshape(np.array(xs), (xs.size // 3, 3))
 
    # Create external density force
    # 1. Unbind microtubules
    # timer.timer('fg_release')
    # self.force_generator.release_particle(xf, array_num_points, offset_particles, dt)
    # timer.timer('fg_release')
    # # 2. Capture microtubules
    # timer.timer('fg_trap')
    # self.force_generator.trap_particle_pycuda(xf, num_fibers, array_num_points, offset_particles)
    # timer.timer('fg_trap')
    # # 3. Compute active and and passive force
    # timer.timer('fg_force')
    # force_fg = self.force_generator.compute_force_pycuda(xf, xs, num_fibers, array_num_points, offset_particles)
    # # force_fg = np.zeros_like(force_fibers)
    # timer.timer('fg_force')
    # # print('norm_fg = ', np.linalg.norm(force_fg))
    # print('norm_f  = ', np.linalg.norm(force_fibers))
    # force_fg = np.zeros(xs.size)

    # Smooth force in time
    # force_fibers += (force_fg + 2.0 * self.force_fg_2 + 2.0 * self.force_fg_1 + self.force_fg_0) / 6.0
    # force_fibers += (0.5 * force_fg + self.force_fg_4 + self.force_fg_3 + self.force_fg_2 + self.force_fg_1 + 0.5 * self.force_fg_0) / 5.0
    # force_fibers = (0.5 * force_fibers + self.force_fg_4 + self.force_fg_3 + self.force_fg_2 + self.force_fg_1 + 0.5 * self.force_fg_0) / 5.0
    #self.force_fg_0 = self.force_fg_1
    #self.force_fg_1 = self.force_fg_2
    #self.force_fg_2 = self.force_fg_3
    #self.force_fg_3 = self.force_fg_4
    #self.force_fg_4 = force_fg
    # self.force_fg_4 = force_fibers

    # torque_fibers = forces.compute_torque_hinges(self.bodies, 
    #                                              self.fibers, 
    #                                              x0, 
    #                                              self.Nfibers_markers,
    #                                              offset, 
    #                                              offset_bodies, 
    #                                              *args, **kwargs)

    # 1b. Update fibers length
    for k, fib in enumerate(self.fibers):
      fib.update_length(force_fibers[offset[k]+fib.num_points-1])


    # 2. Build Block-diagonal matrices and RHS for fibers
    timer.timer('Build_linear_system_fibers')
    A_all = []
    RHS_all = np.zeros(system_size)
    for k, fib in enumerate(self.fibers):
      # Set linear operator and RHS
      A = fib.form_linear_operator()
      RHS = fib.compute_RHS(force_external = force_fibers[offset[k]:offset[k+1]])

      # Apply BC
      A, RHS = fib.apply_BC(A, RHS)

      # Save data
      A_all.append(A)
      RHS_all[offset_bodies + offset[k] * 4 : offset_bodies + (offset[k] + fib.num_points) * 4] = RHS
    # Transform to sparse matrix 
    if len(A_all) > 0:
      As_fibers_block = scsp.block_diag(A_all) 
      As_fibers = scsp.csr_matrix(As_fibers_block)
    else:
      As_fibers = None
    timer.timer('Build_linear_system_fibers')


    # 3. Build PC for fibers
    timer.timer('PC_init_fibers')
    LU_all = [] 
    P_all = [] 
    for k, fib in enumerate(self.fibers): 
      (LU, P) = scla.lu_factor(A_all[k]) 
      P_all.append(P) 
      LU_all.append(LU) 
    def P_inv(x, LU, P, offset, offset_bodies): 
      timer.timer('PC_apply_fibers')
      y = np.empty_like(x)
      # PC for bodies is the identity matrix
      y[0 : offset_bodies] = x[0 : offset_bodies]
      # For fibers is block diagonal
      for i in range(len(LU)): 
        y[offset_bodies + offset[i] * 4 : offset_bodies + offset[i+1] * 4] = scla.lu_solve((LU[i], P[i]), 
                                                                                           x[offset_bodies+offset[i]*4 : offset_bodies+offset[i+1]*4])
      timer.timer('PC_apply_fibers')
      return y 
    P_inv_partial = partial(P_inv, LU = LU_all, P = P_all, offset = offset, offset_bodies = offset_bodies) 
    P_inv_partial_LO = scspla.LinearOperator((system_size, system_size), matvec = P_inv_partial, dtype='float64') 
    timer.timer('PC_init_fibers')
    

    # 4. Build RHS 
    timer.timer('Build_sparse_BC')
    # this is already done for fibers 
    As_dok_BC = scsp.dok_matrix((system_size, system_size))
    for k, b in enumerate(self.bodies):
      # Get links location
      if b.links_location is not None:
        rotation_matrix = b.orientation.rotation_matrix()
        links_loc = np.array([np.dot(rotation_matrix, vec) for vec in b.links_location])
        offset_links = b.links_first_fibers
      else:
        links_loc = []

      # Loop over attached fibers
      for i, link in enumerate(links_loc):
        offset_fiber = offset_links + i
        offset_point = offset[offset_fiber] * 4 + offset_bodies
        fib = self.fibers[offset_fiber]
        
        # Matrix A_body_fiber, for position
        As_dok_BC[k*6 + 0, offset_point+fib.num_points*0 : offset_point+fib.num_points*1] = fib.E * fib.D_3[0,:] 
        As_dok_BC[k*6 + 1, offset_point+fib.num_points*1 : offset_point+fib.num_points*2] = fib.E * fib.D_3[0,:] 
        As_dok_BC[k*6 + 2, offset_point+fib.num_points*2 : offset_point+fib.num_points*3] = fib.E * fib.D_3[0,:] 
        As_dok_BC[k*6 + 0, offset_point+fib.num_points*3] = -fib.xs[0,0] 
        As_dok_BC[k*6 + 1, offset_point+fib.num_points*3] = -fib.xs[0,1] 
        As_dok_BC[k*6 + 2, offset_point+fib.num_points*3] = -fib.xs[0,2] 

        As_dok_BC[k*6 + 3, offset_point+fib.num_points*1 : offset_point+fib.num_points*2] = -fib.E * links_loc[i,2] * fib.D_3[0,:] 
        As_dok_BC[k*6 + 3, offset_point+fib.num_points*2 : offset_point+fib.num_points*3] =  fib.E * links_loc[i,1] * fib.D_3[0,:] 
        As_dok_BC[k*6 + 4, offset_point+fib.num_points*0 : offset_point+fib.num_points*1] =  fib.E * links_loc[i,2] * fib.D_3[0,:] 
        As_dok_BC[k*6 + 4, offset_point+fib.num_points*2 : offset_point+fib.num_points*3] = -fib.E * links_loc[i,0] * fib.D_3[0,:] 
        As_dok_BC[k*6 + 5, offset_point+fib.num_points*0 : offset_point+fib.num_points*1] = -fib.E * links_loc[i,1] * fib.D_3[0,:] 
        As_dok_BC[k*6 + 5, offset_point+fib.num_points*1 : offset_point+fib.num_points*2] =  fib.E * links_loc[i,0] * fib.D_3[0,:] 

        As_dok_BC[k*6 + 3, offset_point+fib.num_points*3] = (-links_loc[i,2]*fib.xs[0,1]+links_loc[i,1]*fib.xs[0,2]) 
        As_dok_BC[k*6 + 4, offset_point+fib.num_points*3] = ( links_loc[i,2]*fib.xs[0,0]-links_loc[i,0]*fib.xs[0,2]) 
        As_dok_BC[k*6 + 5, offset_point+fib.num_points*3] = (-links_loc[i,1]*fib.xs[0,0]+links_loc[i,0]*fib.xs[0,1]) 

        # Matrix A_fiber_body, for position 
        As_dok_BC[offset_point+fib.num_points*0, k*6 + 0] = -1.0 
        As_dok_BC[offset_point+fib.num_points*1, k*6 + 1] = -1.0 
        As_dok_BC[offset_point+fib.num_points*2, k*6 + 2] = -1.0 
        As_dok_BC[offset_point+fib.num_points*3, k*6 + 0] = -fib.xs[0,0] 
        As_dok_BC[offset_point+fib.num_points*3, k*6 + 1] = -fib.xs[0,1] 
        As_dok_BC[offset_point+fib.num_points*3, k*6 + 2] = -fib.xs[0,2] 

        As_dok_BC[offset_point+fib.num_points*0, k*6 + 4] = -links_loc[i,2] 
        As_dok_BC[offset_point+fib.num_points*0, k*6 + 5] =  links_loc[i,1] 
        As_dok_BC[offset_point+fib.num_points*1, k*6 + 3] =  links_loc[i,2] 
        As_dok_BC[offset_point+fib.num_points*1, k*6 + 5] = -links_loc[i,0] 
        As_dok_BC[offset_point+fib.num_points*2, k*6 + 3] = -links_loc[i,1] 
        As_dok_BC[offset_point+fib.num_points*2, k*6 + 4] =  links_loc[i,0] 

        As_dok_BC[offset_point+fib.num_points*3, k*6 + 3] = ( fib.xs[0,1]*links_loc[i,2] - fib.xs[0,2]*links_loc[i,1]) 
        As_dok_BC[offset_point+fib.num_points*3, k*6 + 4] = (-fib.xs[0,0]*links_loc[i,2] + fib.xs[0,2]*links_loc[i,0]) 
        As_dok_BC[offset_point+fib.num_points*3, k*6 + 5] = ( fib.xs[0,0]*links_loc[i,1] - fib.xs[0,1]*links_loc[i,0]) 

        # # Matrix A_body_fiber, for angle 
        # As_dok_BC[k*6 + 3, offset_point+fib.num_points*1 : offset_point+fib.num_points*2] +=  fib.E * fib.xs[0,2] * fib.D_2[0,:] 
        # As_dok_BC[k*6 + 3, offset_point+fib.num_points*2 : offset_point+fib.num_points*3] += -fib.E * fib.xs[0,1] * fib.D_2[0,:] 
        # As_dok_BC[k*6 + 4, offset_point+fib.num_points*0 : offset_point+fib.num_points*1] += -fib.E * fib.xs[0,2] * fib.D_2[0,:] 
        # As_dok_BC[k*6 + 4, offset_point+fib.num_points*2 : offset_point+fib.num_points*3] +=  fib.E * fib.xs[0,0] * fib.D_2[0,:] 
        # As_dok_BC[k*6 + 5, offset_point+fib.num_points*0 : offset_point+fib.num_points*1] +=  fib.E * fib.xs[0,1] * fib.D_2[0,:] 
        # As_dok_BC[k*6 + 5, offset_point+fib.num_points*1 : offset_point+fib.num_points*2] += -fib.E * fib.xs[0,0] * fib.D_2[0,:] 

        # # Matrix A_fiber_body, for angle 
        # As_dok_BC[offset_point+fib.num_points*0+1, k*6 + 4] = -links_loc[i,2] 
        # As_dok_BC[offset_point+fib.num_points*0+1, k*6 + 5] =  links_loc[i,1] 
        # As_dok_BC[offset_point+fib.num_points*1+1, k*6 + 3] =  links_loc[i,2] 
        # As_dok_BC[offset_point+fib.num_points*1+1, k*6 + 5] = -links_loc[i,0] 
        # As_dok_BC[offset_point+fib.num_points*2+1, k*6 + 3] = -links_loc[i,1] 
        # As_dok_BC[offset_point+fib.num_points*2+1, k*6 + 4] =  links_loc[i,0] 


    As_BC = scsp.csr_matrix(As_dok_BC)
    timer.timer('Build_sparse_BC')
    # 5. Define linear operator 
    timer.timer('Build_A')
    def A_body_fiber(x, As_fibers, As_BC, bodies, offset_bodies, offset): 
      # Create solution and body mobility
      timer.timer('Apply_A')
      y = np.empty_like(x)
      M = np.eye(6)
      radius = 1.0
      M[0:3, 0:3] = M[0:3, 0:3] * (6.0 * np.pi * self.eta * radius)
      M[3:6, 3:6] = M[3:6, 3:6] * (8.0 * np.pi * self.eta * radius**3)

      # a. Multiply by block diagonal matrices of bodies
      for k, b in enumerate(bodies):
        y[k*6 : (k+1)*6] = np.dot(M, x[k*6 : (k+1)*6])

      # b. Multiply by block diagonal matrices of fibers
      if As_fibers is not None:
        y[offset_bodies:] = As_fibers.dot(x[offset_bodies:])      

      # c. Add BC
      y += As_BC.dot(x)

      # d. Add far field flow
      timer.timer('Apply_A')
      return y
    A_body_fiber_partial = partial(A_body_fiber, 
                                   As_fibers=As_fibers, 
                                   As_BC=As_BC, 
                                   bodies=self.bodies, 
                                   offset_bodies=offset_bodies, 
                                   offset=offset)
    A_body_fiber_LO = scspla.LinearOperator((system_size, system_size), matvec = A_body_fiber_partial, dtype='float64') 
    timer.timer('Build_A')

    # 7. Call GMRES 
    timer.timer('GMRES')
    counter = gmres_counter(print_residual = self.verbose) 
    (sol, info_precond) = gmres.gmres(A_body_fiber_LO, 
                                      RHS_all, 
                                      tol=self.tolerance, 
                                      atol=0.0,
                                      M=P_inv_partial_LO, 
                                      maxiter=60, 
                                      restart=150, 
                                      callback=counter)

    if info_precond != 0:
      print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
      print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
      print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
      print('gmres info_precond = ', info_precond)
      # sys.exit()    
    timer.timer('GMRES')

    # 9. Update bodies configuration
    timer.timer('update')
    for k, b in enumerate(self.bodies):
      b.location = b.location + sol[k*6 : k*6 + 3] * dt
      quaternion_dt = quaternion.Quaternion.from_rotation(sol[k*6+3 : k*6 + 6] * dt)
      b.orientation = quaternion_dt * b.orientation
    
    # 10. Update fiber configuration
    for k, fib in enumerate(self.fibers):
      fib.tension = np.copy(sol[offset_bodies + offset[k] * 4 + 3 * fib.num_points : offset_bodies + offset[k] * 4 + 4 * fib.num_points])
      if True:
        timer.timer('correct')
        fib.x_mid[:,0] = sol[offset_bodies + offset[k] * 4 + 0 * fib.num_points : offset_bodies + offset[k] * 4 + 1 * fib.num_points]
        fib.x_mid[:,1] = sol[offset_bodies + offset[k] * 4 + 1 * fib.num_points : offset_bodies + offset[k] * 4 + 2 * fib.num_points]
        fib.x_mid[:,2] = sol[offset_bodies + offset[k] * 4 + 2 * fib.num_points : offset_bodies + offset[k] * 4 + 3 * fib.num_points]
        fib.correct()
        timer.timer('correct')
      else:
        fib.x[:,0] = sol[offset_bodies + offset[k] * 4 + 0 * fib.num_points : offset_bodies + offset[k] * 4 + 1 * fib.num_points]
        fib.x[:,1] = sol[offset_bodies + offset[k] * 4 + 1 * fib.num_points : offset_bodies + offset[k] * 4 + 2 * fib.num_points]
        fib.x[:,2] = sol[offset_bodies + offset[k] * 4 + 2 * fib.num_points : offset_bodies + offset[k] * 4 + 3 * fib.num_points]

    # 8. Fix shift 
    for k, b in enumerate(self.bodies):
      # Get links location
      if b.links_location is not None:
        rotation_matrix = b.orientation.rotation_matrix()
        links_loc = np.array([np.dot(rotation_matrix, vec) for vec in b.links_location])
        offset_links = b.links_first_fibers
      else:
        links_loc = []

      # Loop over attached fibers
      for i, link in enumerate(links_loc):
        offset_fiber = offset_links + i
        offset_point = offset[offset_fiber] + offset_bodies
        fib = self.fibers[offset_fiber]
        #xs0 = np.dot(fib.D_1[0,:], fib.x) / np.linalg.norm(np.dot(fib.D_1[0,:], fib.x))
        #link_s = link / np.linalg.norm(link)
        dx = fib.x[0] - link - b.location
        #dxs = xs0 - link_s
        fib.x -= dx

        # print('++++++++++++++++ ', i)
        # print('error ========== ', np.linalg.norm(dx))
        # print('error_s ======== ', np.linalg.norm(dxs))

        if False:
          ab = np.cross(xs0, link_s) 
          theta = (ab / np.maximum(np.linalg.norm(ab), 1e-15)) * np.arcsin(np.linalg.norm(ab) / np.linalg.norm(xs0))
          q = quaternion.Quaternion.from_rotation(theta)
          rotation_matrix = q.rotation_matrix()
          fib.x = np.array([np.dot(rotation_matrix, x-fib.x[0]) for x in fib.x]) + link + b.location
       
        # dx = fib.x[0] - link - b.location
        # dxs = np.dot(fib.D_1[0,:], fib.x) / np.linalg.norm(np.dot(fib.D_1[0,:], fib.x)) - link_s        
        # print('error ========== ', np.linalg.norm(dx))
        # print('error_s ======== ', np.linalg.norm(dxs))
        
    timer.timer('update')
        

    return















  def linear_solver_fibers(self, dt, *args, **kwargs):
    '''
    Solve nonlinear system to find new configuration.
    '''
    # 0. Precomputation 
    # Array x0 has all the degrees of freedom of fibers
    # x0 = (fiber_0, fiber_1, ...)
    # 
    # offset[k] = has the number of fiber points before fiber k in the array x0
    timer.timer('initialization_step')
    system_size = self.Nfibers_markers * 4
    offset = np.zeros(len(self.fibers) + 1, dtype=int)
    x0 = np.zeros(system_size)
    xs = []
    for k, fib in enumerate(self.fibers):
      fib.dt = dt
      offset[k + 1] = offset[k] + fib.num_points
      x0[4*offset[k]+0 : 4*offset[k]+0+fib.num_points*4 : 4] = fib.x[:,0]
      x0[4*offset[k]+1 : 4*offset[k]+1+fib.num_points*4 : 4] = fib.x[:,1]
      x0[4*offset[k]+2 : 4*offset[k]+2+fib.num_points*4 : 4] = fib.x[:,2]

      # Compute derivatives along the fiber
      fib.xs = np.dot(fib.D_1, fib.x)
      fib.xss = np.dot(fib.D_2, fib.x)
      fib.xsss = np.dot(fib.D_3, fib.x)
      fib.xssss = np.dot(fib.D_4, fib.x)
      
      if fib.filtering:
        fib.x_up = np.dot(fib.P_up, fib.x)
        fib.xs_up = np.dot(fib.D_1_up, fib.x_up)
        fib.xss_up = np.dot(fib.D_2_up, fib.x_up)
        fib.xsss_up = np.dot(fib.D_3_up, fib.x_up)
        fib.xssss_up = np.dot(fib.D_4_up, fib.x_up)
      
      xs.append(fib.xs)
    xs = np.array(xs)
    xs = np.reshape(np.array(xs), (xs.size // 3, 3))
    timer.timer('initialization_step')

    # 1. Compute external forces
    force_bodies, force_fibers = forces.compute_external_forces(self.bodies, 
                                                                self.fibers, 
                                                                x0, 
                                                                self.Nfibers_markers,
                                                                offset, 
                                                                0, 
                                                                *args, **kwargs)
    force_fibers[:,:] = 0.0
    
    # 1b. Update fibers length
    #for k, fib in enumerate(self.fibers):
    #  fib.update_length(force_fibers[offset[k]+fib.num_points-1])


    # 2. Build Block-diagonal matrices and RHS for fibers
    timer.timer('Build_linear_system_fibers')
    A_all = []
    RHS_all = np.zeros(system_size)
    for k, fib in enumerate(self.fibers):
      # Set linear operator and RHS
      A = fib.form_linear_operator()
      RHS = fib.compute_RHS(force_external = force_fibers[offset[k]:offset[k+1]])

      # Apply BC
      # fib.set_BC(BC_start_0 = 'force', BC_start_vec_0 = np.array([0.,0.,-2.0]))
      # A, RHS = fib.apply_BC(A, RHS)
      A, RHS = fib.apply_BC_rectangular(A, RHS)

      # Save data
      A_all.append(A)
      RHS_all[offset[k] * 4 : (offset[k] + fib.num_points) * 4] = RHS
    # Transform to sparse matrix 
    if len(A_all) > 0:
      As_fibers_block = scsp.block_diag(A_all) 
      As_fibers = scsp.csr_matrix(As_fibers_block)
    else:
      As_fibers = None
    timer.timer('Build_linear_system_fibers')


    # 3. Build PC for fibers
    timer.timer('PC_init_fibers')
    if True:
      LU_all = [] 
      P_all = [] 
      for k, fib in enumerate(self.fibers): 
        (LU, P) = scla.lu_factor(A_all[k]) 
        P_all.append(P) 
        LU_all.append(LU) 
      def P_inv(x, LU, P, offset): 
        timer.timer('PC_apply_fibers')
        y = np.empty_like(x)
        # For fibers is block diagonal
        for i in range(len(LU)): 
          y[offset[i] * 4 : offset[i+1] * 4] = scla.lu_solve((LU[i], P[i]), 
                                                             x[offset[i]*4 : offset[i+1]*4])
        timer.timer('PC_apply_fibers')
        return y 
      P_inv_partial = partial(P_inv, LU = LU_all, P = P_all, offset = offset) 
      P_inv_partial_LO = scspla.LinearOperator((system_size, system_size), matvec = P_inv_partial, dtype='float64') 
    else:
      Q_all = []
      R_all = []
      for k, fib in enumerate(self.fibers):
        Q, R = scla.qr(A_all[k], check_finite=False)
        Q_all.append(Q)
        R_all.append(R)
      def P_inv(x, Q, R, offset):
        timer.timer('PC_apply_fibers')
        y = np.empty_like(x)
        for i in range(len(Q)):
          y[offset[i] * 4 : offset[i+1] * 4] = scla.solve_triangular(R[i], np.dot(Q[i].T, x[offset[i]*4 : offset[i+1]*4]), check_finite=False)
        timer.timer('PC_apply_fibers')
        return y 
      P_inv_partial = partial(P_inv, Q = Q_all, R = R_all, offset = offset) 
      P_inv_partial_LO = scspla.LinearOperator((system_size, system_size), matvec = P_inv_partial, dtype='float64') 
    timer.timer('PC_init_fibers')
    
    # 5. Define linear operator 
    timer.timer('Build_A')
    def A_fiber(x, As_fibers, offset): 
      # Create solution and body mobility
      timer.timer('Apply_A')
      y = np.empty_like(x)

      # b. Multiply by block diagonal matrices of fibers
      if As_fibers is not None:
        y = As_fibers.dot(x)      

      # d. Add far field flow
      timer.timer('Apply_A')
      return y
    A_fiber_partial = partial(A_fiber, 
                              As_fibers=As_fibers, 
                              offset=offset)
    A_fiber_LO = scspla.LinearOperator((system_size, system_size), matvec = A_fiber_partial, dtype='float64') 
    timer.timer('Build_A')

    # 7. Call GMRES 
    timer.timer('GMRES')
    counter = gmres_counter(print_residual = self.verbose) 
    (sol, info_precond) = gmres.gmres(A_fiber_LO, 
                                      RHS_all, 
                                      tol=self.tolerance, 
                                      atol=0,
                                      M=P_inv_partial_LO, 
                                      maxiter=60, 
                                      restart=150, 
                                      callback=counter)

    if info_precond != 0:
      print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
      print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
      print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
      print('gmres info_precond = ', info_precond)
      # sys.exit()    
    timer.timer('GMRES')


    timer.timer('update')   
    # 10. Update fiber configuration
    for k, fib in enumerate(self.fibers):
      fib.tension = np.copy(sol[offset[k] * 4 + 3 * fib.num_points : offset[k] * 4 + 4 * fib.num_points])
      fib.x_new[:,0] = sol[offset[k] * 4 + 0 * fib.num_points : offset[k] * 4 + 1 * fib.num_points]
      fib.x_new[:,1] = sol[offset[k] * 4 + 1 * fib.num_points : offset[k] * 4 + 2 * fib.num_points]
      fib.x_new[:,2] = sol[offset[k] * 4 + 2 * fib.num_points : offset[k] * 4 + 3 * fib.num_points]      
    timer.timer('update')
        

    return












  def linear_solver_fibers_and_bodies(self, dt, *args, **kwargs):
    '''
    Solve nonlinear system to find new configuration.
    '''
    # 0. Precomputation 
    # Array x0 has all the degrees of freedom, first bodies and then fibers
    # x0 = (body_0, body_1, ..., fiber_0, fiber_1, ...)
    # 
    # offset[k] = has the number of fiber points before fiber k in the array x0
    timer.timer('initialization_step')
    system_size = len(self.bodies) * 6 + self.Nfibers_markers * 4
    offset = np.zeros(len(self.fibers) + 1, dtype=int)
    offset_bodies = len(self.bodies) * 6
    x0 = np.zeros(system_size)
    xs = np.zeros(self.Nfibers_markers * 3)

    for k, b in enumerate(self.bodies):
      x0[6*k:6*k+3] = b.location
    for k, fib in enumerate(self.fibers):
      fib.dt = dt
      offset[k + 1] = offset[k] + fib.num_points
      x0[offset_bodies+4*offset[k]+0 : offset_bodies+4*offset[k]+0+fib.num_points*4 : 4] = fib.x[:,0]
      x0[offset_bodies+4*offset[k]+1 : offset_bodies+4*offset[k]+1+fib.num_points*4 : 4] = fib.x[:,1]
      x0[offset_bodies+4*offset[k]+2 : offset_bodies+4*offset[k]+2+fib.num_points*4 : 4] = fib.x[:,2]

      # Compute derivatives along the fiber
      fib.xs = np.dot(fib.D_1, fib.x)
      fib.xss = np.dot(fib.D_2, fib.x)
      fib.xsss = np.dot(fib.D_3, fib.x)
      fib.xssss = np.dot(fib.D_4, fib.x)

      if fib.filtering:
        fib.x_up = np.dot(fib.P_up, fib.x)
        fib.xs_up = np.dot(fib.D_1_up, fib.x_up)
        fib.xss_up = np.dot(fib.D_2_up, fib.x_up)
        fib.xsss_up = np.dot(fib.D_3_up, fib.x_up)
        fib.xssss_up = np.dot(fib.D_4_up, fib.x_up)
      xs[3*offset[k]+0 : 3*offset[k] + 0 + fib.num_points*3 : 3] = fib.xs[:,0]
      xs[3*offset[k]+1 : 3*offset[k] + 1 + fib.num_points*3 : 3] = fib.xs[:,1]
      xs[3*offset[k]+2 : 3*offset[k] + 2 + fib.num_points*3 : 3] = fib.xs[:,2]      
    xs = np.array(xs)
    xs = np.reshape(np.array(xs), (xs.size // 3, 3))
    timer.timer('initialization_step')

    # 1. Compute external forces
    force_bodies, force_fibers = forces.compute_external_forces(self.bodies, 
                                                                self.fibers, 
                                                                x0, 
                                                                self.Nfibers_markers,
                                                                offset, 
                                                                offset_bodies, 
                                                                *args, **kwargs)

    if False:
      # Concatenate fibers
      num_fibers = len(self.fibers)
      xf = np.concatenate([fib.x for fib in self.fibers])
      array_num_points = np.array([fib.num_points for fib in self.fibers])
      offset_particles = offset[0:-1]

      # Compute material derivative
      xs = []
      for fib in self.fibers:
        xs.append( np.dot(fib.D_1, fib.x) )
      xs = np.array(xs)
      xs = np.reshape(np.array(xs), (xs.size // 3, 3))
 
      # Create external density force
      # 1. Unbind microtubules
      timer.timer('fg_release')
      self.force_generator.release_particle(xf, array_num_points, offset_particles, dt)
      timer.timer('fg_release')
      # 2. Capture microtubules
      timer.timer('fg_trap')
      self.force_generator.trap_particle_pycuda(xf, num_fibers, array_num_points, offset_particles)
      timer.timer('fg_trap')
      # 3. Compute active and and passive force
      timer.timer('fg_force')
      force_fg = self.force_generator.compute_force_pycuda(xf, xs, num_fibers, array_num_points, offset_particles)
      timer.timer('fg_force')
      # print('norm_f  = ', np.linalg.norm(force_fibers))

      # Smooth force in time
      force_fibers += force_fg
      # force_fibers += (0.5 * force_fg + self.force_fg_4 + self.force_fg_3 + self.force_fg_2 + self.force_fg_1 + 0.5 * self.force_fg_0) / 5.0
      self.force_fg_0 = self.force_fg_1
      self.force_fg_1 = self.force_fg_2
      self.force_fg_2 = self.force_fg_3
      self.force_fg_3 = self.force_fg_4
      self.force_fg_4 = force_fg

    elif True and self.molecular_motor is not None:
      timer.timer('molecular_motor')
      # Compute fiber modes
      timer.timer('molecular_motor.modes')
      for fib in self.fibers:
        fib.compute_modes()
      timer.timer('molecular_motor.modes')

      # Compute x and xs
      timer.timer('molecular_motor.find_x_xs_and_length_MT')
      self.molecular_motor.find_x_xs_and_length_MT(self.fibers)
      timer.timer('molecular_motor.find_x_xs_and_length_MT')

      # Compute force
      timer.timer('molecular_motor.compute_force')
      self.molecular_motor.compute_force()
      timer.timer('molecular_motor.compute_force')

      # Spread force
      timer.timer('molecular_motor.spread_force')
      for k, fib in enumerate(self.fibers):
        fib.force_motors[:,:] = 0.0
      self.molecular_motor.spread_force(self.fibers)
      force_fg = np.zeros_like(force_fibers)
      for k, fib in enumerate(self.fibers):
        force_fg[offset[k] : offset[k]+fib.num_points] = fib.force_motors
      timer.timer('molecular_motor.spread_force')

      # Update links
      timer.timer('molecular_motor.update_links')
      self.molecular_motor.update_links_numba(dt, self.fibers)
      timer.timer('molecular_motor.update_links')
      
      # Walk and diffuse
      timer.timer('molecular_motor.walk')
      self.molecular_motor.walk(dt)
      timer.timer('molecular_motor.walk')
      timer.timer('molecular_motor.diffuse')
      self.molecular_motor.diffuse(dt)    
      timer.timer('molecular_motor.diffuse')
      timer.timer('molecular_motor')
    
      # Smooth force in time
      force_fibers += force_fg
      # force_fibers += (0.5 * force_fg + self.force_fg_4 + self.force_fg_3 + self.force_fg_2 + self.force_fg_1 + 0.5 * self.force_fg_0) / 5.0
      # self.force_fg_0 = self.force_fg_1
      # self.force_fg_1 = self.force_fg_2
      # self.force_fg_2 = self.force_fg_3
      # self.force_fg_3 = self.force_fg_4
      # self.force_fg_4 = force_fg




    # np.savetxt('force_fibers_non_smooth.dat', force_fibers)
    if False:
      for k, fib in enumerate(self.fibers):
        smooth = np.zeros((fib.num_points, fib.num_points))
        np.fill_diagonal(smooth, 0.25)
        np.fill_diagonal(smooth[1:,:], 0.25)
        np.fill_diagonal(smooth[:,1:], 0.25)
        np.fill_diagonal(smooth[2:,:], 0.125)
        np.fill_diagonal(smooth[:,2:], 0.125)
        smooth -= np.diag(np.sum(smooth.T, axis=0)) - np.eye(fib.num_points)
      
        force_fibers[offset[k] : offset[k] + fib.num_points] = np.dot(smooth, force_fibers[offset[k] : offset[k] + fib.num_points])
        force_fg[offset[k] : offset[k] + fib.num_points] = np.dot(smooth, force_fg[offset[k] : offset[k] + fib.num_points])

    # 1b. Update fibers length
    for k, fib in enumerate(self.fibers):
      fib.update_length(force_fibers[offset[k]+fib.num_points-1])
        
    # 2. Build Block-diagonal matrices and RHS for fibers
    timer.timer('Build_linear_system_fibers')
    A_all = []
    RHS_all = np.zeros(system_size)
    RHS_all[0:6*len(self.bodies):6] = force_bodies[:,0]
    RHS_all[1:6*len(self.bodies):6] = force_bodies[:,1]
    RHS_all[2:6*len(self.bodies):6] = force_bodies[:,2]
    for k, fib in enumerate(self.fibers):
      # Set linear operator and RHS
      A = fib.form_linear_operator()
      RHS = fib.compute_RHS(force_external = force_fibers[offset[k]:offset[k+1]])

      # Apply BC
      fib.set_BC(BC_start_0='velocity', 
                 BC_start_1='angular_velocity', 
                 # BC_start_1='torque', 
                 BC_end_0='force', 
                 BC_end_vec_0=force_fibers[offset[k] + fib.num_points - 1] * fib.weights[-1])
      # A, RHS = fib.apply_BC(A, RHS)
      A, RHS = fib.apply_BC_rectangular(A, RHS)

      # Save data
      A_all.append(A)
      RHS_all[offset_bodies + offset[k] * 4 : offset_bodies + (offset[k] + fib.num_points) * 4] = RHS
    # Transform to sparse matrix 
    if len(A_all) > 0:
      As_fibers_block = scsp.block_diag(A_all) 
      As_fibers = scsp.csr_matrix(As_fibers_block)
    else:
      As_fibers = None
    timer.timer('Build_linear_system_fibers')


    # 3. Build PC for fibers
    timer.timer('PC_init_fibers')
    if False:
      LU_all = [] 
      P_all = [] 
      for k, fib in enumerate(self.fibers): 
        (LU, P) = scla.lu_factor(A_all[k]) 
        P_all.append(P) 
        LU_all.append(LU) 
      def P_inv(x, LU, P, offset, offset_bodies): 
        timer.timer('PC_apply_fibers')
        y = np.empty_like(x)
        # PC for bodies is the identity matrix
        y[0 : offset_bodies] = x[0 : offset_bodies]
        # For fibers is block diagonal
        for i in range(len(LU)): 
          y[offset_bodies + offset[i] * 4 : offset_bodies + offset[i+1] * 4] = scla.lu_solve((LU[i], P[i]), 
                                                                                             x[offset_bodies+offset[i]*4 : offset_bodies+offset[i+1]*4],
                                                                                             check_finite=False)
        timer.timer('PC_apply_fibers')
        return y 
      P_inv_partial = partial(P_inv, LU = LU_all, P = P_all, offset = offset, offset_bodies = offset_bodies) 
      P_inv_partial_LO = scspla.LinearOperator((system_size, system_size), matvec = P_inv_partial, dtype='float64') 
    else:
      Q_all = []
      R_all = []
      for k, fib in enumerate(self.fibers):
        Q, R = scla.qr(A_all[k], check_finite=False)
        Q_all.append(Q)
        R_all.append(R)
      def P_inv(x, Q, R, offset, offset_bodies):
        timer.timer('PC_apply_fibers')
        y = np.empty_like(x)
        # PC for bodies is the identity matrix
        y[0 : offset_bodies] = x[0 : offset_bodies]
        # For fibers is block diagonal
        for i in range(len(Q)):
          y[offset_bodies + offset[i] * 4 : offset_bodies + offset[i+1] * 4] = scla.solve_triangular(R[i], 
                                                                                                     np.dot(Q[i].T, x[offset_bodies + offset[i]*4 : 
                                                                                                                      offset_bodies + offset[i+1]*4]), 
                                                                                                     check_finite=False)
        timer.timer('PC_apply_fibers')
        return y 
      P_inv_partial = partial(P_inv, Q = Q_all, R = R_all, offset = offset, offset_bodies = offset_bodies) 
      P_inv_partial_LO = scspla.LinearOperator((system_size, system_size), matvec = P_inv_partial, dtype='float64') 
    timer.timer('PC_init_fibers')


    # 4. Build Link Matrix (fibers' Boundary Cconditions)
    timer.timer('Build_sparse_BC')
    # this is already done for fibers 
    As_dok_BC = scsp.dok_matrix((system_size, system_size))
    for k, b in enumerate(self.bodies):
      # Get links location
      if b.links_location is not None:
        rotation_matrix = b.orientation.rotation_matrix()
        links_loc = np.array([np.dot(rotation_matrix, vec) for vec in b.links_location])
        offset_links = b.links_first_fibers
      else:
        links_loc = []

      # Loop over attached fibers
      for i, link in enumerate(links_loc):
        offset_fiber = offset_links + i
        offset_point = offset[offset_fiber] * 4 + offset_bodies
        fib = self.fibers[offset_fiber]

        if False:
          # Row elimination
          # Matrix A_body_fiber, for position
          As_dok_BC[k*6 + 0, offset_point+fib.num_points*0 : offset_point+fib.num_points*1] = fib.E * fib.D_3[0,:] 
          As_dok_BC[k*6 + 1, offset_point+fib.num_points*1 : offset_point+fib.num_points*2] = fib.E * fib.D_3[0,:] 
          As_dok_BC[k*6 + 2, offset_point+fib.num_points*2 : offset_point+fib.num_points*3] = fib.E * fib.D_3[0,:] 
          As_dok_BC[k*6 + 0, offset_point+fib.num_points*3] = -fib.xs[0,0] 
          As_dok_BC[k*6 + 1, offset_point+fib.num_points*3] = -fib.xs[0,1] 
          As_dok_BC[k*6 + 2, offset_point+fib.num_points*3] = -fib.xs[0,2] 
          
          As_dok_BC[k*6 + 3, offset_point+fib.num_points*1 : offset_point+fib.num_points*2] = -fib.E * links_loc[i,2] * fib.D_3[0,:] 
          As_dok_BC[k*6 + 3, offset_point+fib.num_points*2 : offset_point+fib.num_points*3] =  fib.E * links_loc[i,1] * fib.D_3[0,:] 
          As_dok_BC[k*6 + 4, offset_point+fib.num_points*0 : offset_point+fib.num_points*1] =  fib.E * links_loc[i,2] * fib.D_3[0,:] 
          As_dok_BC[k*6 + 4, offset_point+fib.num_points*2 : offset_point+fib.num_points*3] = -fib.E * links_loc[i,0] * fib.D_3[0,:] 
          As_dok_BC[k*6 + 5, offset_point+fib.num_points*0 : offset_point+fib.num_points*1] = -fib.E * links_loc[i,1] * fib.D_3[0,:] 
          As_dok_BC[k*6 + 5, offset_point+fib.num_points*1 : offset_point+fib.num_points*2] =  fib.E * links_loc[i,0] * fib.D_3[0,:] 
          
          As_dok_BC[k*6 + 3, offset_point+fib.num_points*3] = (-links_loc[i,2]*fib.xs[0,1]+links_loc[i,1]*fib.xs[0,2]) 
          As_dok_BC[k*6 + 4, offset_point+fib.num_points*3] = ( links_loc[i,2]*fib.xs[0,0]-links_loc[i,0]*fib.xs[0,2]) 
          As_dok_BC[k*6 + 5, offset_point+fib.num_points*3] = (-links_loc[i,1]*fib.xs[0,0]+links_loc[i,0]*fib.xs[0,1]) 

          # Matrix A_fiber_body, for position 
          As_dok_BC[offset_point+fib.num_points*0, k*6 + 0] = -1.0 
          As_dok_BC[offset_point+fib.num_points*1, k*6 + 1] = -1.0 
          As_dok_BC[offset_point+fib.num_points*2, k*6 + 2] = -1.0 
          As_dok_BC[offset_point+fib.num_points*3, k*6 + 0] = -fib.xs[0,0] 
          As_dok_BC[offset_point+fib.num_points*3, k*6 + 1] = -fib.xs[0,1] 
          As_dok_BC[offset_point+fib.num_points*3, k*6 + 2] = -fib.xs[0,2] 
          
          As_dok_BC[offset_point+fib.num_points*0, k*6 + 4] = -links_loc[i,2] 
          As_dok_BC[offset_point+fib.num_points*0, k*6 + 5] =  links_loc[i,1] 
          As_dok_BC[offset_point+fib.num_points*1, k*6 + 3] =  links_loc[i,2] 
          As_dok_BC[offset_point+fib.num_points*1, k*6 + 5] = -links_loc[i,0] 
          As_dok_BC[offset_point+fib.num_points*2, k*6 + 3] = -links_loc[i,1] 
          As_dok_BC[offset_point+fib.num_points*2, k*6 + 4] =  links_loc[i,0] 
          
          As_dok_BC[offset_point+fib.num_points*3, k*6 + 3] = ( fib.xs[0,1]*links_loc[i,2] - fib.xs[0,2]*links_loc[i,1]) 
          As_dok_BC[offset_point+fib.num_points*3, k*6 + 4] = (-fib.xs[0,0]*links_loc[i,2] + fib.xs[0,2]*links_loc[i,0]) 
          As_dok_BC[offset_point+fib.num_points*3, k*6 + 5] = ( fib.xs[0,0]*links_loc[i,1] - fib.xs[0,1]*links_loc[i,0]) 

          if True:
            # Matrix A_body_fiber, for angle 
            As_dok_BC[k*6 + 3, offset_point+fib.num_points*1 : offset_point+fib.num_points*2] +=  fib.E * fib.xs[0,2] * fib.D_2[0,:] 
            As_dok_BC[k*6 + 3, offset_point+fib.num_points*2 : offset_point+fib.num_points*3] += -fib.E * fib.xs[0,1] * fib.D_2[0,:] 
            As_dok_BC[k*6 + 4, offset_point+fib.num_points*0 : offset_point+fib.num_points*1] += -fib.E * fib.xs[0,2] * fib.D_2[0,:] 
            As_dok_BC[k*6 + 4, offset_point+fib.num_points*2 : offset_point+fib.num_points*3] +=  fib.E * fib.xs[0,0] * fib.D_2[0,:] 
            As_dok_BC[k*6 + 5, offset_point+fib.num_points*0 : offset_point+fib.num_points*1] +=  fib.E * fib.xs[0,1] * fib.D_2[0,:] 
            As_dok_BC[k*6 + 5, offset_point+fib.num_points*1 : offset_point+fib.num_points*2] += -fib.E * fib.xs[0,0] * fib.D_2[0,:] 
            
            # Matrix A_fiber_body, for angle 
            As_dok_BC[offset_point+fib.num_points*0+1, k*6 + 4] = -links_loc[i,2] 
            As_dok_BC[offset_point+fib.num_points*0+1, k*6 + 5] =  links_loc[i,1] 
            As_dok_BC[offset_point+fib.num_points*1+1, k*6 + 3] =  links_loc[i,2] 
            As_dok_BC[offset_point+fib.num_points*1+1, k*6 + 5] = -links_loc[i,0] 
            As_dok_BC[offset_point+fib.num_points*2+1, k*6 + 3] = -links_loc[i,1] 
            As_dok_BC[offset_point+fib.num_points*2+1, k*6 + 4] =  links_loc[i,0] 
        else:
          # Rectangular mathod, Driscoll and Hale
          # Matrix A_body_fiber, for position
          As_dok_BC[k*6 + 0, offset_point+fib.num_points*0 : offset_point+fib.num_points*1] = fib.E * fib.D_3[0,:] 
          As_dok_BC[k*6 + 1, offset_point+fib.num_points*1 : offset_point+fib.num_points*2] = fib.E * fib.D_3[0,:] 
          As_dok_BC[k*6 + 2, offset_point+fib.num_points*2 : offset_point+fib.num_points*3] = fib.E * fib.D_3[0,:] 
          As_dok_BC[k*6 + 0, offset_point+fib.num_points*3] = -fib.xs[0,0] 
          As_dok_BC[k*6 + 1, offset_point+fib.num_points*3] = -fib.xs[0,1] 
          As_dok_BC[k*6 + 2, offset_point+fib.num_points*3] = -fib.xs[0,2] 
          
          As_dok_BC[k*6 + 3, offset_point+fib.num_points*1 : offset_point+fib.num_points*2] = -fib.E * links_loc[i,2] * fib.D_3[0,:] 
          As_dok_BC[k*6 + 3, offset_point+fib.num_points*2 : offset_point+fib.num_points*3] =  fib.E * links_loc[i,1] * fib.D_3[0,:] 
          As_dok_BC[k*6 + 4, offset_point+fib.num_points*0 : offset_point+fib.num_points*1] =  fib.E * links_loc[i,2] * fib.D_3[0,:] 
          As_dok_BC[k*6 + 4, offset_point+fib.num_points*2 : offset_point+fib.num_points*3] = -fib.E * links_loc[i,0] * fib.D_3[0,:] 
          As_dok_BC[k*6 + 5, offset_point+fib.num_points*0 : offset_point+fib.num_points*1] = -fib.E * links_loc[i,1] * fib.D_3[0,:] 
          As_dok_BC[k*6 + 5, offset_point+fib.num_points*1 : offset_point+fib.num_points*2] =  fib.E * links_loc[i,0] * fib.D_3[0,:] 
          
          As_dok_BC[k*6 + 3, offset_point+fib.num_points*3] = (-links_loc[i,2]*fib.xs[0,1]+links_loc[i,1]*fib.xs[0,2]) 
          As_dok_BC[k*6 + 4, offset_point+fib.num_points*3] = ( links_loc[i,2]*fib.xs[0,0]-links_loc[i,0]*fib.xs[0,2]) 
          As_dok_BC[k*6 + 5, offset_point+fib.num_points*3] = (-links_loc[i,1]*fib.xs[0,0]+links_loc[i,0]*fib.xs[0,1]) 

          # Matrix A_fiber_body, for position 
          As_dok_BC[offset_point+fib.num_points*4-14, k*6 + 0] = -1.0 
          As_dok_BC[offset_point+fib.num_points*4-13, k*6 + 1] = -1.0 
          As_dok_BC[offset_point+fib.num_points*4-12, k*6 + 2] = -1.0 
          As_dok_BC[offset_point+fib.num_points*4-11, k*6 + 0] = -fib.xs[0,0] 
          As_dok_BC[offset_point+fib.num_points*4-11, k*6 + 1] = -fib.xs[0,1] 
          As_dok_BC[offset_point+fib.num_points*4-11, k*6 + 2] = -fib.xs[0,2] 
          
          As_dok_BC[offset_point+fib.num_points*4-14, k*6 + 4] = -links_loc[i,2] 
          As_dok_BC[offset_point+fib.num_points*4-14, k*6 + 5] =  links_loc[i,1] 
          As_dok_BC[offset_point+fib.num_points*4-13, k*6 + 3] =  links_loc[i,2] 
          As_dok_BC[offset_point+fib.num_points*4-13, k*6 + 5] = -links_loc[i,0] 
          As_dok_BC[offset_point+fib.num_points*4-12, k*6 + 3] = -links_loc[i,1] 
          As_dok_BC[offset_point+fib.num_points*4-12, k*6 + 4] =  links_loc[i,0] 
          
          As_dok_BC[offset_point+fib.num_points*4-11, k*6 + 3] = ( fib.xs[0,1]*links_loc[i,2] - fib.xs[0,2]*links_loc[i,1]) 
          As_dok_BC[offset_point+fib.num_points*4-11, k*6 + 4] = (-fib.xs[0,0]*links_loc[i,2] + fib.xs[0,2]*links_loc[i,0]) 
          As_dok_BC[offset_point+fib.num_points*4-11, k*6 + 5] = ( fib.xs[0,0]*links_loc[i,1] - fib.xs[0,1]*links_loc[i,0]) 

          if True:
            # Matrix A_body_fiber, for angle 
            As_dok_BC[k*6 + 3, offset_point+fib.num_points*1 : offset_point+fib.num_points*2] +=  fib.E * fib.xs[0,2] * fib.D_2[0,:] 
            As_dok_BC[k*6 + 3, offset_point+fib.num_points*2 : offset_point+fib.num_points*3] += -fib.E * fib.xs[0,1] * fib.D_2[0,:] 
            As_dok_BC[k*6 + 4, offset_point+fib.num_points*0 : offset_point+fib.num_points*1] += -fib.E * fib.xs[0,2] * fib.D_2[0,:] 
            As_dok_BC[k*6 + 4, offset_point+fib.num_points*2 : offset_point+fib.num_points*3] +=  fib.E * fib.xs[0,0] * fib.D_2[0,:] 
            As_dok_BC[k*6 + 5, offset_point+fib.num_points*0 : offset_point+fib.num_points*1] +=  fib.E * fib.xs[0,1] * fib.D_2[0,:] 
            As_dok_BC[k*6 + 5, offset_point+fib.num_points*1 : offset_point+fib.num_points*2] += -fib.E * fib.xs[0,0] * fib.D_2[0,:] 
            
            # Matrix A_fiber_body, for angle 
            As_dok_BC[offset_point+fib.num_points*4-10, k*6 + 4] = -links_loc[i,2] 
            As_dok_BC[offset_point+fib.num_points*4-10, k*6 + 5] =  links_loc[i,1] 
            As_dok_BC[offset_point+fib.num_points*4-9, k*6 + 3]  =  links_loc[i,2] 
            As_dok_BC[offset_point+fib.num_points*4-9, k*6 + 5]  = -links_loc[i,0] 
            As_dok_BC[offset_point+fib.num_points*4-8, k*6 + 3]  = -links_loc[i,1] 
            As_dok_BC[offset_point+fib.num_points*4-8, k*6 + 4]  =  links_loc[i,0] 

            
    As_BC = scsp.csr_matrix(As_dok_BC)
    timer.timer('Build_sparse_BC')
    # 5. Define linear operator 
    timer.timer('Build_A')
    def A_body_fiber(x, As_fibers, As_BC, bodies, offset_bodies, offset): 
      # Create solution and body mobility
      timer.timer('Apply_A')
      y = np.empty_like(x)
      M = np.eye(6)
      radius = 1.0
      M[0:3, 0:3] = M[0:3, 0:3] * (6.0 * np.pi * self.eta * radius)
      M[3:6, 3:6] = M[3:6, 3:6] * (8.0 * np.pi * self.eta * radius**3)

      # a. Multiply by block diagonal matrices of bodies
      for k, b in enumerate(bodies):
        y[k*6 : (k+1)*6] = np.dot(M, x[k*6 : (k+1)*6])

      # b. Multiply by block diagonal matrices of fibers
      if As_fibers is not None:
        y[offset_bodies:] = As_fibers.dot(x[offset_bodies:])      

      # c. Add BC
      y += As_BC.dot(x)

      # d. Add far field flow
      timer.timer('Apply_A')
      return y

    A_body_fiber_partial = partial(A_body_fiber, 
                                   As_fibers=As_fibers, 
                                   As_BC=As_BC, 
                                   bodies=self.bodies, 
                                   offset_bodies=offset_bodies, 
                                   offset=offset)
    A_body_fiber_LO = scspla.LinearOperator((system_size, system_size), matvec = A_body_fiber_partial, dtype='float64') 
    timer.timer('Build_A')

    # 7. Call GMRES 
    timer.timer('GMRES')
    counter = gmres_counter(print_residual = self.verbose) 
    (sol, info_precond) = gmres.gmres(A_body_fiber_LO, 
                                      RHS_all, 
                                      tol=self.tolerance, 
                                      atol=0,
                                      M=P_inv_partial_LO, 
                                      maxiter=60, 
                                      restart=150, 
                                      callback=counter)

    if info_precond != 0:
      print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
      print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
      print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
      print('gmres info_precond = ', info_precond)
      # sys.exit()    
    timer.timer('GMRES')


    timer.timer('update')   

    # 10.a Update fiber configuration
    for k, fib in enumerate(self.fibers):
      fib.tension = np.copy(sol[offset_bodies + offset[k] * 4 + 3 * fib.num_points : offset_bodies + offset[k] * 4 + 4 * fib.num_points])
      fib.x_new[:,0] = sol[offset_bodies + offset[k] * 4 + 0 * fib.num_points : offset_bodies + offset[k] * 4 + 1 * fib.num_points]
      fib.x_new[:,1] = sol[offset_bodies + offset[k] * 4 + 1 * fib.num_points : offset_bodies + offset[k] * 4 + 2 * fib.num_points]
      fib.x_new[:,2] = sol[offset_bodies + offset[k] * 4 + 2 * fib.num_points : offset_bodies + offset[k] * 4 + 3 * fib.num_points]      

      # 10.a.a Update # of points per fiber and upsampling rate
      if fib.adaptive_num_points:
        fib.update_resolution()
      # Upsampling rate
      fib.find_upsample_rate()

    # 10.b Update bodies configuration
    for k, b in enumerate(self.bodies):
      b.location_new = b.location + sol[k*6 : k*6 + 3] * dt
      quaternion_dt = quaternion.Quaternion.from_rotation(sol[k*6+3 : k*6 + 6] * dt)
      b.orientation_new = quaternion_dt * b.orientation   
    timer.timer('update')     

    # Update number of fiber markers
    self.Nfibers_markers = sum([x.num_points for x in self.fibers])

    return




  def linear_solver_fibers_hydro(self, dt, *args, **kwargs):
    '''
    Solve nonlinear system to find new configuration.
    GK: linear solver for fibers in a fluid
    '''

    # ------------------
    # 0. INITIALIZATION
    # ------------------
    timer.timer('initialization_step')
    # 0.1. Get number of particles and offsets
    num_bodies, num_fibers, offset_bodies, offset_fibers = multi.get_num_particles_and_offsets(self.bodies, self.fibers)
    
    # num. of fiber marker points, num. of unknowns
    Nfibers_markers, system_size = offset_fibers[-1], num_bodies * 6 + offset_fibers[-1] * 4  
    self.Nfibers_markers = Nfibers_markers
   
    # Given external forces
    external_forces = np.zeros((Nfibers_markers, 3))
    external_forces[:,2] = -5.0

    x0 = np.zeros(system_size) # particle configuration at the previous time step

    for k, b in enumerate(self.bodies):
      x0[6*k:6*k+3] = b.location
    for k, fib in enumerate(self.fibers):
      fib.force = external_forces[offset_fibers[k]:offset_fibers[k+1]]
      fib.dt = dt
      x0[offset_bodies+4*offset_fibers[k]+0 : offset_bodies+4*offset_fibers[k]+0+fib.num_points*4 : 4] = fib.x[:,0]
      x0[offset_bodies+4*offset_fibers[k]+1 : offset_bodies+4*offset_fibers[k]+1+fib.num_points*4 : 4] = fib.x[:,1]
      x0[offset_bodies+4*offset_fibers[k]+2 : offset_bodies+4*offset_fibers[k]+2+fib.num_points*4 : 4] = fib.x[:,2]

      # Compute derivatives along the fiber
      fib.xs = np.dot(fib.D_1, fib.x)
      fib.xss = np.dot(fib.D_2, fib.x)
      fib.xsss = np.dot(fib.D_3, fib.x)
      fib.xssss = np.dot(fib.D_4, fib.x)

      if fib.filtering:
        fib.find_upsample_rate()
        fib.x_up = np.dot(fib.P_up, fib.x)
        fib.xs_up = np.dot(fib.D_1_up, fib.x_up)
        fib.xss_up = np.dot(fib.D_2_up, fib.x_up)
        fib.xsss_up = np.dot(fib.D_3_up, fib.x_up)
        fib.xssss_up = np.dot(fib.D_4_up, fib.x_up)
    
    timer.timer('initialization_step')

    # ------------------------------------------
    # 1. EXTERNAL FORCES, MOTORS, (DE)POLYMERIZE 
    # ------------------------------------------
    timer.timer('external_forces')
    force_bodies, force_fibers = forces.compute_external_forces([], 
                                                                self.fibers, 
                                                                x0, 
                                                                self.Nfibers_markers,
                                                                offset_fibers, 
                                                                offset_bodies, 
                                                                *args, **kwargs)
    timer.timer('external_forces')
    
    # 1.0. MOLECULAR MOTORS
    if True and self.molecular_motor is not None:
      timer.timer('molecular_motor')
      # Compute fiber modes
      timer.timer('molecular_motor.modes')
      for fib in self.fibers:
        fib.compute_modes()
      timer.timer('molecular_motor.modes')

      # Compute x and xs
      timer.timer('molecular_motor.find_x_xs_and_length_MT')
      self.molecular_motor.find_x_xs_and_length_MT(self.fibers)
      timer.timer('molecular_motor.find_x_xs_and_length_MT')

      # Compute force
      timer.timer('molecular_motor.compute_force')
      self.molecular_motor.compute_force()
      timer.timer('molecular_motor.compute_force')

      # Spread force
      timer.timer('molecular_motor.spread_force')
      for k, fib in enumerate(self.fibers):
        fib.force_motors[:,:] = 0.0
      self.molecular_motor.spread_force(self.fibers)
      force_fg = np.zeros_like(force_fibers)
      for k, fib in enumerate(self.fibers):
        force_fg[offset_fibers[k] : offset_fibers[k]+fib.num_points] = fib.force_motors
      timer.timer('molecular_motor.spread_force')

      # Update links
      timer.timer('molecular_motor.update_links')
      self.molecular_motor.update_links_numba(dt, self.fibers)
      timer.timer('molecular_motor.update_links')
      
      # Walk and diffuse
      timer.timer('molecular_motor.walk')
      self.molecular_motor.walk(dt)
      timer.timer('molecular_motor.walk')
      timer.timer('molecular_motor.diffuse')
      self.molecular_motor.diffuse(dt)    
      timer.timer('molecular_motor.diffuse')
      timer.timer('molecular_motor')
    
      # Smooth force in time
      force_fibers += force_fg
      # force_fibers += (0.5 * force_fg + self.force_fg_4 + self.force_fg_3 + self.force_fg_2 + self.force_fg_1 + 0.5 * self.force_fg_0) / 5.0
      # self.force_fg_0 = self.force_fg_1
      # self.force_fg_1 = self.force_fg_2
      # self.force_fg_2 = self.force_fg_3
      # self.force_fg_3 = self.force_fg_4
      # self.force_fg_4 = force_fg


    # 1.1. UPDATE FIBERS' LENGTHS
    timer.timer('update_length')
    for k, fib in enumerate(self.fibers):
      if fib.igrowing:
        fib.update_length(force_fibers[offset_fibers[k]+fib.num_points-1])
      fib.force = force_fibers[offset_fibers[k]:offset_fibers[k+1]]
    timer.timer('update_length')

    # ---------------------------------------------------   
    # 2. BLOCK DIAGONAL MATRICES AND RHSs FOR FIBERS
    # ---------------------------------------------------

    # 2.0. Get source and target points (shape: num_points * nfibers * 3)
    xtrg = multi.gather_target_points([], self.fibers)

    # 2.1. Compute velocity due to fiber forces on other fibers (includes self flow)
    timer.timer('pairwise_interactions')
    v_external = multi.flow_fibers(force_fibers, xtrg, xtrg, self.fibers, offset_fibers, self.eta, self.filtering)
    timer.timer('pairwise_interactions')

    # 2.2. Subtract self-flow due to Stokeslet i
    timer.timer('subtract_self_interactions')
    v_external += multi.self_flow_fibers(force_fibers, offset_fibers, self.fibers, self.eta, self.filtering)
    timer.timer('subtract_self_interactions')

    timer.timer('Build_linear_system_fibers')
    # 2.3. Get fibers A and RHS:
    A_fibers, A_fibers_blocks, RHS_fibers = multi.get_fibers_matrices(self.fibers, offset_fibers, v_external)

    # 2.4. Get fibers force operator (Precompute and save matrices before GMRES)
    fibers_force_operator = multi.build_fibers_force_operator(self.fibers)

    timer.timer('Build_linear_system_fibers')

    # prepare indexing to reshape outputs in GMRES
    flat2mat = np.zeros((3*offset_fibers[-1],3),dtype = bool)
    flat2mat_vT = np.zeros((4*offset_fibers[-1],5),dtype = bool)
    P_cheb_all = []
    for k, fib in enumerate(self.fibers):
      flat2mat[3*offset_fibers[k]                   :3*offset_fibers[k] +   fib.num_points,0] = True
      flat2mat[3*offset_fibers[k] +   fib.num_points:3*offset_fibers[k] + 2*fib.num_points,1] = True
      flat2mat[3*offset_fibers[k] + 2*fib.num_points:3*offset_fibers[k] + 3*fib.num_points,2] = True
    
      flat2mat_vT[4*offset_fibers[k]                   :4*offset_fibers[k] +   fib.num_points,0] = True
      flat2mat_vT[4*offset_fibers[k] +   fib.num_points:4*offset_fibers[k] + 2*fib.num_points,1] = True
      flat2mat_vT[4*offset_fibers[k] + 2*fib.num_points:4*offset_fibers[k] + 3*fib.num_points,2] = True
      flat2mat_vT[4*offset_fibers[k] : 4*offset_fibers[k+1]-14,3] = True
      flat2mat_vT[4*offset_fibers[k] : 4*offset_fibers[k+1],4] = True

      P_cheb_all.append(fib.P_cheb_representations_all_dof)
    P_cheb_sprs = scsp.csr_matrix(scsp.block_diag(P_cheb_all))  


    # ---------------------------------------------------   
    # 3. DEFINE LINEAR OPERATOR
    # ---------------------------------------------------

    timer.timer('Build_A')
    def linear_operator(y, num_fibers, offset_fibers, trg, eta, A_fibers, fibers_force_operator):
      
      # Get degrees of freedom (fibers coordinates and tensions).
      XT = y # dimension: 4 * num_points
    
      # 1. Compute fibers density force (computed at high-res. then downsampled to num_points)
      timer.timer('computing_forces_inGMRES')
      force_fibers = fibers_force_operator.dot(XT)
      timer.timer('computing_forces_inGMRES')
    
      # 2. Compute fluid velocity due to fibers at trg
      # First, reorder forces
      fw = np.zeros((force_fibers.size // 3, 3))
      fw[:,0] = force_fibers[flat2mat[:,0]]
      fw[:,1] = force_fibers[flat2mat[:,1]]  
      fw[:,2] = force_fibers[flat2mat[:,2]]

      
      # Compute velocity due to force terms treated implicitly (bending and tension)
      timer.timer('pairwise_interactions')
      v = multi.flow_fibers(fw, trg, trg, self.fibers, offset_fibers, self.eta, self.filtering)
      timer.timer('pairwise_interactions')

      timer.timer('subtract_self_interactions')
      v += multi.self_flow_fibers(fw, offset_fibers, self.fibers, self.eta, self.filtering)
      timer.timer('subtract_self_interactions')

      # Copy flow to right format
      v = v.reshape((v.size // 3, 3))
      vT = np.zeros(offset_fibers[-1] * 4)
      vT[flat2mat_vT[:,0]] = v[:,0]
      vT[flat2mat_vT[:,1]] = v[:,1]
      vT[flat2mat_vT[:,2]] = v[:,2]
      vT[flat2mat_vT[:,3]] = P_cheb_sprs.dot(vT[flat2mat_vT[:,4]])

      # 3. Multiply by fiber matrices
      AfXT = A_fibers.dot(XT)

      return AfXT - vT
    


    system_size = offset_fibers[-1] * 4
    linear_operator_partial = partial(linear_operator,
                                    num_fibers=num_fibers,
                                    offset_fibers=offset_fibers,
                                    trg=xtrg,
                                    eta=self.eta,
                                    A_fibers=A_fibers,
                                    fibers_force_operator=fibers_force_operator)

    linear_operator_partial = scspla.LinearOperator((system_size, system_size), matvec = linear_operator_partial, dtype='float64')     
    
    timer.timer('Build_A')

    # Build PC for fibers
    timer.timer('Build_PC')
    LU_all = [] 
    P_all = [] 
    for k, fib in enumerate(self.fibers): 
      (LU, P) = scla.lu_factor(A_fibers_blocks[k]) 
      P_all.append(P) 
      LU_all.append(LU) 
    def P_inv(x, LU, P, offset_fibers): 
      y = np.empty_like(x)
      # For fibers is block diagonal
      for i in range(len(LU)): 
        y[offset_fibers[i] * 4 : offset_fibers[i+1] * 4] = scla.lu_solve((LU[i], P[i]),  x[offset_fibers[i]*4 : offset_fibers[i+1]*4])
      return y 
    
    P_inv_partial = partial(P_inv, LU = LU_all, P = P_all, offset_fibers = offset_fibers) 
    P_inv_partial_LO = scspla.LinearOperator((system_size, system_size), matvec = P_inv_partial, dtype='float64') 

    timer.timer('Build_PC')

    # -----------------
    # 4. GMRES to SOLVE
    # -----------------
    timer.timer('GMRES')
    counter = gmres_counter(print_residual = False) 
    (sol, info_precond) = gmres.gmres(linear_operator_partial, 
                                      RHS_fibers, 
                                      tol=self.tolerance, 
                                      atol=0,
                                      M=P_inv_partial_LO, 
                                      maxiter=60, 
                                      restart=150, 
                                      callback=counter)
    
    if info_precond != 0:
      print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
      print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
      print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
      print('gmres info_precond = ', info_precond)
      # sys.exit()
    else:
      print('GMRES converged in ', counter.niter, ' iterations.')
    timer.timer('GMRES')



    # -----------------------------
    # 5. UPDATE FIBER CONFIGURATION
    # -----------------------------
    niters = np.zeros(len(self.fibers), dtype = int)
    for k, fib in enumerate(self.fibers):
      fib.tension = np.copy(sol[offset_bodies + offset_fibers[k] * 4 + 3 * fib.num_points : offset_bodies + offset_fibers[k] * 4 + 4 * fib.num_points])
      fib.x_new[:,0] = sol[offset_bodies + offset_fibers[k] * 4 + 0 * fib.num_points : offset_bodies + offset_fibers[k] * 4 + 1 * fib.num_points]
      fib.x_new[:,1] = sol[offset_bodies + offset_fibers[k] * 4 + 1 * fib.num_points : offset_bodies + offset_fibers[k] * 4 + 2 * fib.num_points]
      fib.x_new[:,2] = sol[offset_bodies + offset_fibers[k] * 4 + 2 * fib.num_points : offset_bodies + offset_fibers[k] * 4 + 3 * fib.num_points]      
      
      # If correction algorithm is used, fib.x_new is going to assign to fib.x after correction
      fib.x = np.copy(fib.x_new)

      if fib.adaptive_num_points:
        # If using adaptive number of points, then update resolution
        timer.timer('updating_resolution')
        fib.update_resolution()
        timer.timer('updating_resolution')

      if fib.filtering:
        # If de-aliasing, then find upsampling rate
        timer.timer('finding_upsampling_rate')
        fib.find_upsample_rate()
        timer.timer('finding_upsampling_rate')

      if fib.ireparam:
        # reparameterize fiber
        timer.timer('reparameterization')
        niters[k] = fib.reparameterize(fib.reparam_iter, fib.reparam_degree)
        timer.timer('reparameterization')

    if self.fibers[0].ireparam:
      print('Maximum number of reparam. iterations is ', max(niters, key = abs))
  

    # Update number of fiber markers
    self.Nfibers_markers = sum([x.num_points for x in self.fibers])

    return
















  def linear_system_hyd():
    '''
    # Preprocess

    # External forces on fibers and bodies

    # Molecular motors

    # Fiber length

    # RHS, bodies

    # Bodies preconditioners????

    # Build fibers matrices and RHS

    # Fibers preconditioner

    # Sparse matrix, bodies-fibers coupling 

    # Linear operator

    # GMRES

    # Update
    '''
    # Preprocess

    # External forces on fibers and bodies

    # Molecular motors

    # Fiber length

    # RHS, bodies

    # Bodies preconditioners????

    # Build fibers matrices and RHS

    # Fibers preconditioner

    # Sparse matrix, bodies-fibers coupling 

    # Linear operator

    # GMRES

    # Update


    def A_body_fiber_hyd():
      # Equation for body surface velocities
      if True:
        # a. Double layer potential
        # b. Self-interaction correction
        # c. Geometric matrix K*U
        # d. Force fibers, generate flow
        # e. Flow generate by fibers
        pass

      # Equation for rigid body velocities
      if True:
        # a. Geometrix matrix, K^T*U
        # b. Identity matrix I*U
        pass

      # Equation for fibers
      if True:
        # a. Double layer from bodies, interpolate between Chebyshev grids
        # b. Boundary conditions from bodies
        # c. Flow from other fibers, interpolate between Chebyshev grids
        # d. Fiber self-interaction
        pass
      pass


      pass
    



    














class gmres_counter(object):
  '''
  Callback generator to count iterations. 
  '''
  def __init__(self, print_residual = False):
    self.print_residual = print_residual
    self.niter = 0
  def __call__(self, rk=None):
    self.niter += 1
    if self.print_residual is True:
      if self.niter == 1:
        print('gmres =  0 1')
      print('gmres = ', self.niter, rk)

