from __future__ import division, print_function
import numpy as np
from functools import partial
import scipy.linalg as scla
import scipy.sparse as scsp
import scipy.sparse.linalg as scspla
import sys
import time
import copy
import stkfmm
import os

from scipy.spatial import ConvexHull

# OUR CLASSES
from tstep import initialize
from tstep import fiber_matrices
from tstep import tstep_utils
from bpm_utilities import gmres
#from bpm_utilities import gmres
from utils import nonlinear
from utils import timer
from utils import miscellaneous
from fiber import fiber
from body import body
from quaternion import quaternion
from forces import forces
from shape_gallery import shape_gallery
from quadratures import Smooth_Closed_Surface_Quadrature_RBF
from periphery import periphery
from kernels import kernels
import generateGrid

from mpi4py import MPI

import _pickle as cpickle

try:
  from numba import njit, prange
  from numba.typed import List
except ImportError:
  print('Numba not found')

class tstep(object):
  '''

  Time stepping algorithms: fixed or adaptive 1st order schemes
  Two routines:
  1. all structures (fiber, rigid body, periphery) with hydro (WET)
  2. all structures without hydro (DRY)
  '''


  ##############################################################################################
  def __init__(self, prams, options, input_file, fibers, bodies, molecular_motors):
    self.options = options
    self.prams = prams
    self.attached_to_cortex = prams.attached_to_cortex
    self.output_name = options.output_name # output_name: saving files to
    self.output_txt_files = options.output_txt_files # save in txt files
    self.motor_left = prams.motor_left
    self.motor_right = prams.motor_right
    self.eta = prams.eta # suspending fluid viscosity
    self.adaptive_num_points = options.adaptive_num_points # flag for adaptive grid for fibers
    self.ireparam = options.ireparam # flag for reparameterization
    self.reparam_iter = options.reparam_iter # maximum # of reparameterization iteration
    self.reparam_degree = options.reparam_degree # power for attenuation factor in reparam
    self.iupsample = options.iupsample # upsampling for integration
    self.write_message('Is upsampling on?: ' + str(self.iupsample))
    self.integration = options.integration # integration scheme 'trapz' or 'simpsons'
    self.write_message('Integration scheme: ' + str(self.integration))
    self.repulsion = options.repulsion # flag for repulsion
    self.adaptive_time = options.adaptive_time # flag for adaptive scheme
    self.scheme = options.time_step_scheme # with or without hydro
    self.inextensibility = options.inextensibility
    self.write_message('Inextensibility scheme: ' + str(self.inextensibility))
    self.dt = options.dt # if not adaptive time stepping
    self.dt_min = options.dt_min # minimum time step allowed in adaptive scheme
    self.dt_max = options.dt_max # maximum time step allowed in adaptive scheme
    self.safety_factor = 0.9  # factor to avoid too large step size and errors
    self.beta_up = 1.1        # factor to increase time step size
    self.beta_down = 0.5      # factor to decrease time step size
    self.velMeasureRad = prams.velMeasureRad
    self.velP = prams.velMeasureP
    self.error_history = np.zeros(10)
    self.max_length = prams.max_length
    self.fiber_body_attached = prams.fiber_body_attached # whether body and fiber are attached in simulation
    self.write_message('Fiber and body attached?: ' + str(self.fiber_body_attached))
    self.final_time = prams.final_time # time horizon
    self.tol_tstep = options.tol_tstep # time stepping tolerance for adaptivity
    self.tol_gmres = options.tol_gmres # gmres tolerance
    self.write_message('GMRES tolerance: ' + str(self.tol_gmres))
    self.iCytoPulling = options.iCytoPulling # Cytoplasmic pulling flag
    self.write_message('Is cytoplasmic pulling on?' + str(self.iCytoPulling))
    self.cytoPull_Elongation = False
    self.minL = prams.minL
    self.time_max = prams.time_max
    self.iComputeVelocity = options.iComputeVelocity
    self.ncompute_vel = options.ncompute_vel
    self.iNoSliding = options.iNoSliding
    self.nucleus_radius = prams.nucleus_radius # radius in artificial repulsion force due to nucleus
    self.nucleus_position = prams.nucleus_position # position in artificial repulsion force due to nucleus
    self.scale_nuc2fib = prams.scale_nuc2fib # repulsion strength for nucleus-fiber interaction
    self.scale_nuc2bdy = prams.scale_nuc2bdy # repulsion strength for nucleus-body interaction
    self.scale_nuc2nuc = prams.scale_nuc2nuc # repulsion strength for nucleus-nucleus interaction
    self.len_nuc2fib = prams.len_nuc2fib # repulsion length scale for nucleus-fiber interaction
    self.len_nuc2bdy = prams.len_nuc2bdy # repulsion length scale for nucleus-body interaction
    self.len_nuc2nuc = prams.len_nuc2nuc # repulsion length scale for nucleus-nucleus interaction
    self.cortex_radius = prams.cortex_radius # artificial cortex radius
    self.n_save = options.n_save # save data after this many time steps
    self.iExternalForce = options.iExternalForce # this is for hydro scheme
    self.precompute_body_PC = options.precompute_body_PC # flag for precompute and apply rotation to body PC
    self.isaveForces = options.isaveForces # flag for saving forces on fibers and bodies
    self.iFixObjects = options.iFixObjects # flag for fixing objects until equilibrium fiber length reached
    self.irelease = options.irelease
    self.release_check = options.release_check
    self.release_condition = options.release_condition
    self.iPeripheralNucleation= options.iPeripheralNucleation
    self.useFMM = options.useFMM # flag for using FMM
    self.fmm_order = options.fmm_order
    self.fmm_max_pts = options.fmm_max_pts
    self.oseen_kernel_source_target_stkfmm_partial = None
    self.stresslet_kernel_source_target_stkfmm_partial = None
    self.save_file = options.save_file
    self.dynInstability = options.dynInstability
    self.resume_from_step = prams.resume_from_step
    self.site_idcs_dying = []
    self.site_idcs_nucleating = []
    self.site_idcs_hinged = []
    self.clamp_oscil_mag = prams.clamp_oscil_mag
    self.clamp_oscil_k = prams.clamp_oscil_k
    self.clamp_oscil_speed = prams.clamp_oscil_speed
    self.body_shape = prams.body_shape
    self.body_a = prams.body_a
    self.body_b = prams.body_b
    self.body_c = prams.body_c 
    self.body_r = prams.body_r

    if self.useFMM and self.scheme == 'time_step_hydro':
      self.write_message('FMM is in use:')
      self.write_message('FMM order: ' + str(options.fmm_order))
      self.write_message('FMM maximum points: ' + str(options.fmm_max_pts))
      # initialize FMM if in use
      timer.timer('FMM_oseen_init')
      pbc = stkfmm.PAXIS.NONE
      kernel = stkfmm.KERNEL.PVel
      kernels_index = stkfmm.KERNEL(kernel)
      fmm_PVel = stkfmm.STKFMM(self.fmm_order, self.fmm_max_pts, pbc, kernels_index)
      kdimSL, kdimDL, kdimTrg = stkfmm.getKernelDimension(fmm_PVel, kernel)
      self.oseen_kernel_source_target_stkfmm_partial = partial(kernels.oseen_kernel_source_target_stkfmm, fmm_PVel=fmm_PVel)
      timer.timer('FMM_oseen_init')

      timer.timer('FMM_stresslet_init')
      pbc = stkfmm.PAXIS.NONE
      kernel = stkfmm.KERNEL.PVel
      kernels_index = stkfmm.KERNEL(kernel)
      fmmStress_PVel = stkfmm.STKFMM(self.fmm_order, self.fmm_max_pts, pbc, kernels_index)
      kdimSL, kdimDL, kdimTrg = stkfmm.getKernelDimension(fmmStress_PVel, kernel)
      self.stresslet_kernel_source_target_stkfmm_partial = partial(kernels.stresslet_kernel_source_target_stkfmm, fmm_PVel=fmmStress_PVel)
      timer.timer('FMM_stresslet_init')

    if self.isaveForces:
      self.fibers_repulsion_force = []
      self.fibers_motor_force = []
      self.bodies_repulsion_force = []
      self.bodies_link_force_torque = []
      self.bodies_mm_attached_repulsion_force = []
      self.offset_fibers = []

    # Get time stepping coefficients
    self.beta, self.Xcoeff, self.rhsCoeff = 1.0, [1.0], [1.0]

    # INITIALIZE STRUCTURES
    if True:
      self.fibers, self.bodies, self.molecular_motors, self.fibers_names, self.body_names, self.fibers_types, self.body_types, self.f_fibers_ID, self.f_bodies_ID, self.f_molecular_motors_ID, self.f_time_system, self.f_fibers_forces_ID, self.f_bodies_forces_ID, self.MM_on_moving_surf, self.bodies_mm_attached, self.bodies_types_mm_attached, self.bodies_names_mm_attached, self.f_bodies_mm_attached_ID, self.f_bodies_mm_attached_forces_ID, self.f_mm_on_moving_surf_ID, self.f_body_vels_ID, self.f_body_density_ID, self.f_traction_ID, self.f_body_forces_ID, self.f_fibers_forces_ID = initialize.initialize_from_file(input_file,options,prams)
    else:
      # use given configurations
      self.fibers, self.bodies, self.molecular_motors, self.fibers_names, self.body_names, self.fibers_types, self.body_types, self.f_fibers_ID, self.f_bodies_ID, self.f_molecular_motors_ID, self.f_time_system, self.f_fibers_forces_ID, self.f_bodies_forces_ID = initialize.initialize_manually(fibers,bodies,molecular_motors,options,prams)
      self.MM_on_moving_surf, self.bodies_mm_attached, self.bodies_types_mm_attached, self.bodies_names_mm_attached, self.f_bodies_mm_attached_ID, self.f_bodies_mm_attached_forces_ID, self.f_mm_on_moving_surf_ID = [], [], [], [], [], [], []

    self.body_like_shell = []
  
    self.f_fiber_tips = None
    name = options.output_name + '_fiberTips.txt'
    self.f_fiber_tips = open(name, 'wb', buffering = 100)

    self.f_nhinged_time = None
    if options.iNoSliding:
      name = options.output_name + '_nhinged_time.txt'
      self.f_nhinged_time = open(name, 'wb', buffering = 100)

    # OPEN A FILE TO WRITE GRID POINTS AND VELOCITY FIELD
    #name = options.output_name + '_velocity_at_grid.txt'
    #self.f_grid_velocity = open(name, 'wb', buffering = 100)
    name = options.output_name + '_grid_points.txt'
    self.f_grid_points = open(name, 'wb', buffering = 100)
    self.time_now = 0

    # OPEN A FILE TO KEEP LOGS
    f_log = open(self.output_name + '.logFile', 'w+')

    # If Nblobs is given, then discretize rigid bodies with given Nblobs
    self.A_inv_bodies = []
    if options.Nblobs is not None and self.scheme == 'time_step_hydro':
      for k, b in enumerate(self.bodies):
        if isinstance(options.Nblobs, int):
          # if Nblobs is not an array than the same for all bodies
          Nblobs = options.Nblobs
        else:
          Nblobs = options.Nblobs(k)
        if prams.body_shape == 'sphere':
          b.discretize_body_surface(shape = prams.body_shape, Nblobs = Nblobs, radius = 1/1.04 * prams.body_r)
        elif prams.body_shape == 'ellipsoid':
          b.discretize_body_ellipsoid(shape = prams.body_shape, Nblobs = Nblobs, a = prams.body_a/1.04, b = prams.body_b/1.04, c = prams.body_c/1.04)

        body_config = b.get_r_vectors_surface()
        body_norms = b.get_normals()
        b.density = np.zeros((b.Nblobs,3))
        b.density_new = np.zeros((b.Nblobs,3))
        b.traction = np.zeros((b.Nblobs,3))
        dx = body_config[:,0] - body_config[:,0,None]
        dy = body_config[:,1] - body_config[:,1,None]
        dz = body_config[:,2] - body_config[:,2,None]
        dr = np.sqrt(dx**2 + dy**2 + dz**2)
        dquad = min(dr[0,1:])
        self.write_message('Quadrature spacing on body is ' + str(dquad))
        if b.dfilament is not None:
          self.write_message('Interfilament spacing is ' + str(b.dfilament))

        if self.output_txt_files: # Save configuration just to check how points are distributed
          name = self.output_name + '_' + str(k) + '_body_configuration.txt'
        else:
          name = self.output_name + '_' + str(k) + '.body_configuration'

        with open(name, 'w') as f:
          if self.output_txt_files:
            info = np.array([body_config.size // 3, 3, 0])
            np.savetxt(f, info[None, :])
          else:
            f.write(str(body_config.size // 3) + str(3) + '\n')
          np.savetxt(f, body_config)
          np.savetxt(f, body_norms)
          f.close()
        name = self.output_name + '_body_weights.txt'
        with open(name, 'w') as f:
          np.savetxt(f, b.quadrature_weights)
          f.close()

      # If we want to precompute preconditioner for rigid bodies:
      if self.precompute_body_PC and self.scheme == 'time_step_hydro':
        self.write_message('Precomputing rigid body preconditioner...')
        for k, b in enumerate(self.bodies):
          r_vectors = b.reference_configuration
          normals = b.reference_normals
          weights = b.quadrature_weights

          # Stresslet tensor
          M = kernels.stresslet_kernel_times_normal_numba(r_vectors, normals, eta = self.bodies[0].viscosity_scale*self.eta)
          # Singularity subtraction
          b.calc_vectors_singularity_subtraction(eta = self.bodies[0].viscosity_scale*self.eta, r_vectors = r_vectors, normals = normals)
          ex, ey, ez = b.ex.flatten(), b.ey.flatten(), b.ez.flatten()
          I = np.zeros((3*b.Nblobs, 3*b.Nblobs))
          for i in range(b.Nblobs):
            I[3*i:3*(i+1), 3*i+0] = ex[3*i:3*(i+1)] / weights[i]
            I[3*i:3*(i+1), 3*i+1] = ey[3*i:3*(i+1)] / weights[i]
            I[3*i:3*(i+1), 3*i+2] = ez[3*i:3*(i+1)] / weights[i]
          M -= I

          A = np.zeros((3*b.Nblobs+6, 3*b.Nblobs+6))
          K = b.calc_K_matrix()
          A[0:3*b.Nblobs, 0:3*b.Nblobs] = np.copy(M)
          A[0:3*b.Nblobs, 3*b.Nblobs:3*b.Nblobs+6] = -np.copy(K)
          A[3*b.Nblobs:3*b.Nblobs+6, 0:3*b.Nblobs] = -np.copy(K.T)
          A[3*b.Nblobs:3*b.Nblobs+6, 3*b.Nblobs:3*b.Nblobs+6] = np.eye(6)
          self.A_inv_bodies.append(np.linalg.inv(A))

          

    # BUILD FIBER MATRICES FOR GIVEN RESOLUTION
    self.fibers_gone = [] # fibers gone through catastrophe in the last step
    self.num_nucleated_fibers = [] # num. fibers nucleated recently

    self.fib_mats = []
    self.fib_mat_resolutions = np.array([])

    fib_mat = fiber_matrices.fiber_matrices(num_points = options.num_points, num_points_finite_diff = options.num_points_finite_diff)
    fib_mat.compute_matrices()
    self.fib_mats.append(fib_mat)
    self.fib_mat_resolutions = np.append(self.fib_mat_resolutions, options.num_points)


    for k, fib in enumerate(self.fibers):
      if fib.num_points not in self.fib_mat_resolutions:
        self.write_message(('Matrices for new resolution are being computed ...'))
        # If fib_mat class for this resolution has not been created, then create and store
        fib_mat = fiber_matrices.fiber_matrices(num_points = fib.num_points, num_points_finite_diff = options.num_points_finite_diff)
        fib_mat.compute_matrices()
        self.fib_mats.append(fib_mat)
        self.fib_mat_resolutions = np.append(self.fib_mat_resolutions, fib.num_points)


    # INITIALIZE PERIPHERY IF EXISTS
    self.shell, self.M_inv_periphery, self.normals_shell, self.trg_shell_surf = None, [], [], []
    self.periphery_radius = prams.periphery_radius
    self.periphery_a = prams.periphery_a
    self.periphery_b = prams.periphery_b
    self.periphery_c = prams.periphery_c
    if prams.periphery is not None and self.scheme == 'time_step_hydro':
      # Build shape
      if prams.periphery is 'sphere':
        nodes_periphery, normals_periphery, h_periphery, gradh_periphery = shape_gallery.shape_gallery(prams.periphery,
                                                                        options.Nperiphery, radius=prams.periphery_radius)
      else:
        nodes_periphery, normals_periphery, h_periphery, gradh_periphery = shape_gallery.shape_gallery(prams.periphery,
                                                                        options.Nperiphery, a = prams.periphery_a, b = prams.periphery_b, c = prams.periphery_c)
      # Normals are in the opposite direction to bodies' normals
      normals_periphery = -normals_periphery
      hull_periphery = ConvexHull(nodes_periphery)
      triangles_periphery = hull_periphery.simplices
      # Get quadratures
      quadrature_weights_periphery = Smooth_Closed_Surface_Quadrature_RBF.Smooth_Closed_Surface_Quadrature_RBF(nodes_periphery,
                                                                                                             triangles_periphery,
                                                                                                             h_periphery,
                                                                                                             gradh_periphery)
      # Build shell class
      self.shell = periphery.Periphery(np.array([0., 0., 0.]), quaternion.Quaternion([1.0, 0.0, 0.0, 0.0]),
                  nodes_periphery, normals_periphery, quadrature_weights_periphery)
      # Compute singularity subtraction vectors
      self.shell.get_singularity_subtraction_vectors(eta = self.eta)

      # Precompute shell's r_vectors and normals
      self.trg_shell_surf = self.shell.get_r_vectors()
      self.normals_shell = self.shell.get_normals()


      if self.output_txt_files: # Save configuration just to check how points are distributed
        name = self.output_name + '_periphery.txt'
      else:
        name = self.output_name + '.periphery'

      with open(name, 'w') as f:
        if self.output_txt_files:
          info = np.array([self.trg_shell_surf.size//3, 3, 0])
          np.savetxt(f,info[None,:])
        else:
          f.write(str(self.trg_shell_surf.size // 3) + str(3) + '\n')
        np.savetxt(f, self.trg_shell_surf)
        np.savetxt(f, self.normals_shell)
        f.close()

      # Build shell preconditioner
      weights = self.shell.quadrature_weights
      M = kernels.stresslet_kernel_times_normal_numba(self.trg_shell_surf, self.normals_shell, eta = self.eta)
      N = self.shell.Nblobs
      I = np.zeros((3*N, 3*N))
      for i in range(N):
        I[3*i:3*(i+1), 3*i+0] = self.shell.ex[3*i:3*(i+1)] / weights[i]
        I[3*i:3*(i+1), 3*i+1] = self.shell.ey[3*i:3*(i+1)] / weights[i]
        I[3*i:3*(i+1), 3*i+2] = self.shell.ez[3*i:3*(i+1)] / weights[i]
      I_vec = np.ones(N*3)
      I_vec[0::3] /= (1.0 * weights)
      I_vec[1::3] /= (1.0 * weights)
      I_vec[2::3] /= (1.0 * weights)
      M += -I - np.diag(I_vec)
      # Save shell's self interaction matrix, and apply it as running simulations, do not build again
      self.shell_stresslet = np.copy(M)
      # Similarly, save shell's complementary matrix
      self.shell_complementary = kernels.complementary_kernel(self.trg_shell_surf, self.normals_shell)
      M += self.shell_complementary
      # Preconditioner:
      #self.M_inv_periphery = np.linalg.inv(M)
      (LU, P) = scla.lu_factor(M)
      self.M_inv_periphery = []
      self.M_inv_periphery.append(LU)
      self.M_inv_periphery.append(P)

      # Singularity subtraction vectors, reshaped again
      self.shell.ex = self.shell.ex.reshape((N, 3))
      self.shell.ey = self.shell.ey.reshape((N, 3))
      self.shell.ez = self.shell.ez.reshape((N, 3))

    return # init
  ##############################################################################################

  def take_time_steps(self, *args, **kwargs):
    '''
    Main routine for taking time steps
    '''

    # Set the stage
    start_time = time.time()
    current_time, step, steps_rejected, max_error_at_step = 0.0, 0, 0, []


    if self.fibers:
      for k, fib in enumerate(self.fibers):
        # Update resolution if minimum spacing changes (scales with sqrt(L))
        if self.adaptive_num_points:
          timer.timer('updating_resolution')
          Nold = np.copy(fib.num_points)
          indx = np.where(self.fib_mat_resolutions == fib.num_points)
          indx = indx[0][0]
          fib.update_resolution(self.fib_mats[indx]) # DONE
          if fib.num_points != Nold:
            self.write_message(('Resolution has changed, N = ' + str(fib.num_points)))
            if fib.num_points not in self.fib_mat_resolutions:
              self.write_message(('Matrices for new resolution are being computed ...'))
              # If fib_mat class for this resolution has not been created, then create and store
              fib_mat = fiber_matrices.fiber_matrices(num_points = fib.num_points, num_points_finite_diff = fib.num_points_finite_diff)
              fib_mat.compute_matrices()
              self.fib_mats.append(fib_mat)
              self.fib_mat_resolutions = np.append(self.fib_mat_resolutions, fib.num_points)
          timer.timer('updating_resolution')

    timer.timer('Entire_simulation')


    # Save initial configuration
    self.write_to_file(current_time, step, steps_rejected)
    self.force_on_mtoc = np.zeros((len(self.bodies),6))
    self.force_on_fibers = np.zeros((len(self.fibers),3))

    # START TAKING TIME STEPS
    while current_time < self.final_time - 1e-10:

      # Make sure we land on exactly the final_time
      if current_time + self.dt > self.final_time:
        self.dt = self.final_time - current_time

      # Update step and write current time step
      step += 1
      self.write_message('stars')
      self.write_message('Step = ' + str(step) + ', time = ' + str(current_time + self.dt))
      self.write_message('Time step size is ' + str(self.dt))

      # Assign time step size fibers
      for k, fib in enumerate(self.fibers): fib.dt = self.dt

      # STEP 1: Take a time step
      a_step = time.time()
      self.time_now = current_time
      self.step_now = step
      self.fibers_gone = []

      getattr(self, self.scheme)(self.dt, *args, **kwargs)
      self.write_message('Time step took ' + str(time.time() - a_step) + ' seconds.')

      # STEP 2: Check inextensibility error to see if result is acceptible
      # note: if there is no fiber, then keep taking a step with fixed dt
      accept, dt_new, max_err = True, self.dt, 0.
      if self.fibers:
        accept, dt_new, max_err = self.check_error()
        max_error_at_step.append(max_err)

      # STEP 3: If solution is acceptable, then update the system
      # if not, then take another time step with smaller time step size
      if accept:
        self.write_message('Solution is ACCEPTED, maximum error is ' + str(max_err))

        # update time
        current_time += self.dt

        # if there are solid bodies, update bodies
        for k, b in enumerate(self.bodies):
          self.write_message('things are moving')
          b.location, b.orientation = np.copy(b.location_new), copy.copy(b.orientation_new)
          b.velocity, b.angular_velocity = np.copy(b.velocity_new), np.copy(b.angular_velocity_new)
          if hasattr(b,'density'): b.density = np.copy(b.density_new)

        self.fibers_types = np.zeros(len(self.bodies))

        for k, fib in enumerate(self.fibers):
          fib.x_old = np.copy(fib.x)
          fib.x = np.copy(fib.x_new)
          fib.tension = np.copy(fib.tension_new)
          
          # Fix the link location
          if fib.attached_to_body is not None:
            ib = fib.attached_to_body
            b = self.bodies[ib]
            self.fibers_types[ib] += 1
            rotation_matrix = b.orientation.rotation_matrix()
            link_loc = np.dot(rotation_matrix, b.nuc_sites[fib.nuc_site_idx])
            dx = fib.x[0] - link_loc - b.location
            fib.x -= dx

        self.write_message('There are ' + str(len(self.fibers)) + ' MTs')
        
        # Update shell's unknowns
        if self.shell is not None: self.shell.density = np.copy(self.shell.density_new)

        # SAVE DATA IF ...
        if np.remainder(step-steps_rejected, self.n_save) == 0:
          elapsed_time = time.time() - start_time
          self.write_to_file(current_time, step, steps_rejected)

        for k,fib in enumerate(self.fibers):

          # Update resolution if minimum spacing changes (scales with sqrt(L))
          if self.adaptive_num_points:
            timer.timer('updating_resolution')
            Nold = np.copy(fib.num_points)
            indx = np.where(self.fib_mat_resolutions == fib.num_points)
            indx = indx[0][0]
            fib.update_resolution(self.fib_mats[indx]) # DONE
            if fib.num_points != Nold:
              self.write_message(('Resolution has changed, N = ' + str(fib.num_points)))
              if fib.num_points not in self.fib_mat_resolutions:
                self.write_message(('Matrices for new resolution are being computed ...'))
                # If fib_mat class for this resolution has not been created, then create and store
                fib_mat = fiber_matrices.fiber_matrices(num_points = fib.num_points, num_points_finite_diff = fib.num_points_finite_diff)
                fib_mat.compute_matrices()
                self.fib_mats.append(fib_mat)
                self.fib_mat_resolutions = np.append(self.fib_mat_resolutions, fib.num_points)
            timer.timer('updating_resolution')

      else:
        for k, fib in enumerate(self.fibers):
          fib.length = np.copy(fib.length_previous)
        self.write_message('Solution is REJECTED, maximum error is ' + str(max_err))
        steps_rejected += 1
       

      self.write_message('New time step size is ' + str(dt_new))
      # Update time step size
      self.dt = dt_new
      if max_err >= 1:
        self.write_message('Maximum error is way too large! Terminating...')
        break

    
      if self.time_max is not None:
        if current_time >= self.time_max:
          self.write_message('Maximum time is reached, terminating...')
          self.write_to_file(current_time, step, steps_rejected)
          break
      #if self.shell is None and self.max_length is not None and N_Lmax >= 55:
      #   self.write_message('There are ' + str(N_Lmax) + ' long MTs, terminating')
      #   break


    # Close config files
    timer.timer('Entire_simulation')
    self.write_message('stars')
    self.write_message('Simulation has been completed, closing files...')
    self.write_message(str(step-steps_rejected) + ' time steps are accepted.')
    self.write_message(str(steps_rejected) + ' time steps are rejected.')


    if len(self.fibers_types) > 0:
      for f_ID in self.f_fibers_ID: f_ID.close()

    if len(self.body_types) > 0:
      for f_ID in self.f_bodies_ID: f_ID.close()

    # Write timings:
    self.write_message('Total CPU time is ' + str(time.time()-start_time) + 'sec')
    timer.timer(' ', print_all = True, output_file = self.output_name + '.timers')

    return # take_time_steps

  ##############################################################################################
  def time_step_hydro(self, dt, *args, **kwargs):
    '''
    Time step including hydrodynamics
    System can have rigid bodies, fibers, (stationary) molecular motors, confinement
    '''

    # Update fibers' lengths
    timer.timer('update_length')
    if self.periphery_a is None and self.periphery_radius is not None:
      periphery_a = self.periphery_radius
      periphery_b = self.periphery_radius
      periphery_c = self.periphery_radius
    else:
      periphery_a = self.periphery_a
      periphery_b = self.periphery_b
      periphery_c = self.periphery_c

    self.fibers_gone, self.num_nucleated_fibers, self.num_nucleated_imaginary = [], 0, 0
    self.sites_gone_imaginary, self.terminal_time_gone_imaginary = [], []
    # CORTICAL PUSHING

    istep = self.step_now
    
    timer.timer('update_length')

    # ------------------
    # 1. INITIALIZATION
    # ------------------
    timer.timer('initialization_step')

    # 1.1. Get number of particles and offsets
    num_bodies, num_fibers, offset_bodies, offset_fibers, system_size = tstep_utils.get_num_particles_and_offsets(self.bodies, self.fibers, self.shell, ihydro = True)
    self.offset_fibers = offset_fibers

    offset = 0
    if self.shell is not None:
      offset = 3*self.shell.Nblobs

    # Body part of the solution
    for k,b in enumerate(self.bodies):
      istart = offset + offset_bodies[k]*3 + 6*k

    # Fiber part of the solution, also compute derivatives of fiber configurations
    xfibers = np.zeros((offset_fibers[-1],3))
    x0 = np.zeros((offset_fibers[-1]*3))
    xEnds = np.zeros((len(self.fibers),3))
    trg_fib = np.zeros((offset_fibers[-1],3))
    xsDs_block, ysDs_block, zsDs_block = [], [], []
    for k,fib in enumerate(self.fibers):
      istart = offset + offset_bodies[-1]*3 + 6*len(self.bodies) + 4*offset_fibers[k]
      x0[3*offset_fibers[k] + 0 : 3*offset_fibers[k] + 3*fib.num_points + 0 : 3] = fib.x[:,0]
      x0[3*offset_fibers[k] + 1 : 3*offset_fibers[k] + 3*fib.num_points + 1 : 3] = fib.x[:,1]
      x0[3*offset_fibers[k] + 2 : 3*offset_fibers[k] + 3*fib.num_points + 2 : 3] = fib.x[:,2]
      xEnds[k] = fib.x[-1]
      # Find the index for fib_mats
      indx = np.where(self.fib_mat_resolutions == fib.num_points)
      # Get the class that has the matrices
      fib_mat = self.fib_mats[indx[0][0]]

      # Call differentiation matrices
      D_1, D_2, D_3, D_4 = fib_mat.get_matrices(fib.length_previous, fib.num_points_up, 'Ds')

      # Compute derivatives along the fiber (scale diff. matrices with current length?)
      fib.xs = np.dot(D_1, fib.x)
      fib.xss = np.dot(D_2, fib.x)
      fib.xsss = np.dot(D_3, fib.x)
      fib.xssss = np.dot(D_4, fib.x)

      xfibers[offset_fibers[k] : offset_fibers[k+1]] = fib.xs
      trg_fib[offset_fibers[k] : offset_fibers[k+1]] = fib.x

      xsDs_block.append((D_1.T * fib.xs[:,0]).T)
      ysDs_block.append((D_1.T * fib.xs[:,1]).T)
      zsDs_block.append((D_1.T * fib.xs[:,2]).T)
    if len(xsDs_block) > 0:
      xsDs = scsp.csr_matrix(scsp.block_diag(xsDs_block))
      ysDs = scsp.csr_matrix(scsp.block_diag(ysDs_block))
      zsDs = scsp.csr_matrix(scsp.block_diag(zsDs_block))

    # 1.3. External forces
    force_bodies = np.zeros((len(self.bodies),6))
    force_fibers = np.zeros((offset_fibers[-1],3))

    motor_force_fibers = np.zeros((offset_fibers[-1],3))
    if self.iCytoPulling:
      self.write_message('There is explicit cyto. pulling force on fibers')
      for k, fib in enumerate(self.fibers):
        # DYNEIN
        motor_force_fibers[offset_fibers[k]:offset_fibers[k+1]] = 0.40 * fib.xs

    # ---------------------------------------------------
    # 2. BLOCK DIAGONAL MATRICES AND RHSs FOR FIBERS
    # ---------------------------------------------------
    # 2.1. Get source and target points
    trg_bdy_cent = np.zeros((len(self.bodies),3))
    trg_bdy_surf = np.zeros((offset_bodies[-1],3))

    # Initialize K matrices and array to keep normals of points on bodies
    K_bodies, normals_blobs = [], np.empty((offset_bodies[-1],3))

    for k, b in enumerate(self.bodies):
      trg_bdy_cent[k] = b.location
      r_vec_surface = b.get_r_vectors_surface()
      trg_bdy_surf[offset_bodies[k]:offset_bodies[k+1]] = r_vec_surface
      # Normals
      normals_blobs[offset_bodies[k]:offset_bodies[k+1]] = b.get_normals()
      # Compute correction for singularity subtractions
      b.calc_vectors_singularity_subtraction(eta = self.bodies[0].viscosity_scale*self.eta, r_vectors = r_vec_surface, normals = normals_blobs[offset_bodies[k]:offset_bodies[k+1]])
      # Build K_matrix
      K_bodies.append(b.calc_K_matrix())


    trg_fib = trg_fib.flatten()
    trg_bdy_cent = trg_bdy_cent.flatten()
    trg_bdy_surf = trg_bdy_surf.flatten()


    # Build Stokeslet for fibers
    Gfibers = []
    if self.fibers:
      timer.timer('Build_Stokeslet')
      if self.useFMM:
        Gfibers = tstep_utils.get_self_fibers_FMMStokeslet(self.fibers, self.eta,
                                                  fib_mats = self.fib_mats,
                                                  fib_mat_resolutions = self.fib_mat_resolutions,
                                                  iupsample = self.iupsample)
      else:
        Gfibers = tstep_utils.get_self_fibers_Stokeslet(self.fibers, self.eta,
                                                  fib_mats = self.fib_mats,
                                                  fib_mat_resolutions = self.fib_mat_resolutions,
                                                  iupsample = self.iupsample)

      timer.timer('Build_Stokeslet')

    # 2.2. Compute velocity due to fiber forces on fibers, bodies and shell
    # Concatenate target points
    trg_all = np.concatenate((trg_fib,trg_bdy_surf), axis = 0)
    if self.shell is not None: trg_all = np.concatenate((trg_all,self.trg_shell_surf.flatten()), axis = 0)

    if force_fibers.any():
      timer.timer('fiber2all_explicit_interactions')
      vfib2all = tstep_utils.flow_fibers(force_fibers, trg_fib, trg_all,
        self.fibers, offset_fibers, self.eta, integration = self.integration, fib_mats = self.fib_mats,
        fib_mat_resolutions = self.fib_mat_resolutions, iupsample = self.iupsample,
        oseen_fmm = self.oseen_kernel_source_target_stkfmm_partial, fmm_max_pts = 500)
      timer.timer('fiber2all_explicit_interactions')
      vfib2fib = vfib2all[:3*offset_fibers[-1]]

      vfib2bdy = np.array([])
      if self.bodies:
        vfib2bdy = np.copy(vfib2all[3*offset_fibers[-1]:3*offset_fibers[-1]+3*offset_bodies[-1]])

      vfib2shell = np.array([])
      if self.shell is not None:
        vfib2shell = vfib2all[3*offset_fibers[-1]+3*offset_bodies[-1]:]


      # Subtract self-interaction which is approximated by SBT
      timer.timer('subtract_self_interactions')
      vfib2fib += tstep_utils.self_flow_fibers(force_fibers, offset_fibers, self.fibers, Gfibers, self.eta,
        integration = self.integration, fib_mats = self.fib_mats, fib_mat_resolutions = self.fib_mat_resolutions, iupsample = self.iupsample)
      timer.timer('subtract_self_interactions')
    else:
      vfib2fib, vfib2bdy = np.zeros(offset_fibers[-1]*3), np.zeros(offset_bodies[-1]*3)
      vfib2shell = np.array([])
      if self.shell is not None: vfib2shell = np.zeros(self.shell.Nblobs*3)


    # 2.3. Compute velocity due to body forces on fibers, bodies and shell
    if force_bodies.any(): # non-zero body force

      timer.timer('body2all_explicit_interactions')
      if self.useFMM and (trg_bdy_cent.size//3) * (trg_all.size//3) >= 500:
        vbdy2all = self.oseen_kernel_source_target_stkfmm_partial(trg_bdy_cent, trg_all, force_bodies[:,:3], eta = self.bodies[0].viscosity_scale*self.eta)
      else:
        vbdy2all = kernels.oseen_kernel_source_target_numba(trg_bdy_cent, trg_all, force_bodies[:,:3], eta = self.bodies[0].viscosity_scale*self.eta)
      vbdy2all += kernels.rotlet_kernel_source_target_numba(trg_bdy_cent, trg_all, force_bodies[:,3:], eta = self.bodies[0].viscosity_scale*self.eta)
      timer.timer('body2all_explicit_interactions')

      vbdy2bdy = vbdy2all[3*offset_fibers[-1]:3*offset_fibers[-1]+3*offset_bodies[-1]]
      vbdy2fib = np.array([])
      if self.fibers:
        vbdy2fib = vbdy2all[:3*offset_fibers[-1]]

      vbdy2shell = np.array([])
      if self.shell is not None:
        vbdy2shell = vbdy2all[3*offset_fibers[-1]+3*offset_bodies[-1]:]

    else:
      vbdy2fib, vbdy2bdy = np.zeros(offset_fibers[-1]*3), np.zeros(offset_bodies[-1]*3)
      vbdy2shell = np.array([])
      if self.shell is not None: vbdy2shell = np.zeros(self.shell.Nblobs*3)

    # 2.4. Indexing to reshape fiber outputs in GMRES
    flat2mat = np.zeros((3*offset_fibers[-1],3),dtype = bool)
    flat2mat_vT = np.zeros((4*offset_fibers[-1],6),dtype = bool)
    flat2mat_1st = np.zeros((offset_fibers[-1]), dtype = bool)
    flat2mat_tenBC = np.zeros((4*offset_fibers[-1]), dtype = bool)

    flat2mat_last = np.zeros((offset_fibers[-1]), dtype = bool)
    flat2mat_lastTenBC = np.zeros((4*offset_fibers[-1]), dtype = bool)

    P_cheb_all = []
    rot_mat = self.bodies[0].orientation.rotation_matrix()
    for k, fib in enumerate(self.fibers):
      flat2mat[3*offset_fibers[k]                   :3*offset_fibers[k] +   fib.num_points,0] = True
      flat2mat[3*offset_fibers[k] +   fib.num_points:3*offset_fibers[k] + 2*fib.num_points,1] = True
      flat2mat[3*offset_fibers[k] + 2*fib.num_points:3*offset_fibers[k] + 3*fib.num_points,2] = True

      flat2mat_vT[4*offset_fibers[k]                   :4*offset_fibers[k] +   fib.num_points,0] = True
      flat2mat_vT[4*offset_fibers[k] +   fib.num_points:4*offset_fibers[k] + 2*fib.num_points,1] = True
      flat2mat_vT[4*offset_fibers[k] + 2*fib.num_points:4*offset_fibers[k] + 3*fib.num_points,2] = True
      flat2mat_vT[4*offset_fibers[k] + 3*fib.num_points:4*offset_fibers[k] + 4*fib.num_points,3] = True
      flat2mat_vT[4*offset_fibers[k] : 4*offset_fibers[k+1]-14,4] = True
      flat2mat_vT[4*offset_fibers[k] : 4*offset_fibers[k+1],5] = True

      indx = np.where(self.fib_mat_resolutions == fib.num_points)
      indx = indx[0][0]
      out1, out2, P_cheb_representations_all_dof, out4 = self.fib_mats[indx].get_matrices(fib.length, fib.num_points_up, 'PX_PT_Pcheb')
      P_cheb_all.append(P_cheb_representations_all_dof)

      # (-) end is clamped
      BC_start_0, BC_start_1 = 'velocity', 'angular_velocity'
      
      # (+) end of fiber
      BC_end_0, BC_end_1 = 'force','torque'
      
      # Clamped end has a time-dependent orientation with respect to the normal
      # dXs / dt = Aw sin(k nu - wt) along the long axis of the body
      # find the location of the fiber with respect to body's COM
      
      xyz = self.bodies[0].nuc_sites[fib.nuc_site_idx]
      rfib = np.sqrt(xyz[0]**2 + xyz[1]**2 + xyz[2]**2)
      nu = np.arccos(xyz[2]/rfib)
      waveSpeed = self.clamp_oscil_mag * self.clamp_oscil_speed*np.sin(self.clamp_oscil_k*nu-self.clamp_oscil_speed*self.time_now)
      
      nvec = np.zeros_like(xyz)
      nvec[0] = 2*xyz[0]/self.body_a**2
      nvec[1] = 2*xyz[1]/self.body_b**2
      nvec[2] = 2*xyz[2]/self.body_c**2
      nvec = nvec / np.linalg.norm(nvec)
      vec_ref = np.cross(np.array([0, 0, 1]), nvec) * waveSpeed
      BC_start_vec_1 = np.dot(rot_mat, vec_ref)
      fib.set_BC(BC_start_0 = BC_start_0,
                BC_start_1 = BC_start_1,
                BC_end_0 = BC_end_0,
                BC_end_1 = BC_end_1,
                BC_start_vec_1 = BC_start_vec_1)

      if fib.BC_start_0 is 'velocity':
        flat2mat_1st[offset_fibers[k]] = True
        flat2mat_tenBC[4*offset_fibers[k]+4*fib.num_points-11] = True

      if fib.BC_end_0 is 'velocity':
        flat2mat_last[offset_fibers[k]+fib.num_points-1] = True
        flat2mat_lastTenBC[4*offset_fibers[k]+4*fib.num_points-4] = True

    if P_cheb_all: P_cheb_sprs = scsp.csr_matrix(scsp.block_diag(P_cheb_all))
    # ---------------------------------------------------
    # 3. DEFINE LINEAR OPERATOR
    # ---------------------------------------------------

    # 3.1. Build RHS (x,y,z) of point 1, then (x,y,z) of point 2. This is the order
    timer.timer('Build_linear_system_fibers')
    As_fibers, A_fibers_blocks, RHS_all = tstep_utils.get_fibers_and_bodies_matrices(self.fibers, self.bodies, self.shell,
      system_size, offset_fibers, offset_bodies, force_fibers, motor_force_fibers, force_bodies,
      vfib2fib+vbdy2fib, vbdy2bdy+vfib2bdy, vfib2shell+vbdy2shell, self.fib_mats, self.fib_mat_resolutions, inextensibility = self.inextensibility,ihydro = True)
    timer.timer('Build_linear_system_fibers')

    # 3.2. Fibers' force operator (sparse)
    fibers_force_operator = []
    if self.fibers:
      timer.timer('fibers_force_operator')
      fibers_force_operator = tstep_utils.build_fibers_force_operator(self.fibers, self.fib_mats, self.fib_mat_resolutions)
      timer.timer('fibers_force_operator')

    # 3.3. Build link matrix (fibers' boundary conditions)
    As_BC = []
    if self.fibers and self.bodies:
      timer.timer('Build_sparse_BC')
      As_BC = tstep_utils.build_link_matrix(4*offset_fibers[-1]+6*len(self.bodies), self.bodies,self.fibers, offset_fibers, 6*len(self.bodies),
        self.fib_mats, self.fib_mat_resolutions)
      timer.timer('Build_sparse_BC')

    # 3.4. Preconditioner for fibers
    LU_fibers, P_fibers = [], []
    if self.fibers:
      timer.timer('Build_PC_fibers') # this can be avoided if factorization is done when building the matrix A_fibers_blocks
      LU_fibers, P_fibers = tstep_utils.build_block_diagonal_lu_preconditioner_fiber(self.fibers, A_fibers_blocks)
      timer.timer('Build_PC_fibers')


    # 3.5. Preconditioner for bodies, if not precomputed
    LU_bodies, P_bodies = [], []
    if not self.precompute_body_PC and self.bodies:
      LU_bodies, P_bodies = tstep_utils.build_block_diagonal_preconditioner_body(self.bodies, self.bodies[0].viscosity_scale*self.eta, K_bodies = K_bodies)

    # 3.6. Preconditioner for whole system
    def P_inv(x, A_inv_bodies, LU_bodies, P_bodies, bodies, offset_bodies, shell, Minv, LU_fibers, P_fibers, offset_fibers):
      y = np.empty_like(x)

      # Shell part of the solution
      if shell is not None:
        offset = 3*shell.Nblobs
        #y[:offset] = np.dot(Minv, x[:offset])
        LU_shell, P_shell = Minv[0], Minv[1]
        y[:offset] = scla.lu_solve((LU_shell, P_shell),x[:offset])
      else:
        offset = 0

      # Body part of the solution
      for k, b in enumerate(bodies):
        istart = offset + 3*offset_bodies[k] + 6*k
        iend = offset + 3*offset_bodies[k+1] + 6*(k+1)
        xb = x[istart : iend]

        if LU_bodies:
          yb = scla.lu_solve((LU_bodies[k], P_bodies[k]), x[istart:iend])
        else:
          # Rotate vectors to body frame
          rotation_matrix = b.orientation.rotation_matrix().T
          xb = xb.reshape((b.Nblobs+2,3))
          xb_body = np.array([np.dot(rotation_matrix, vec) for vec in xb]).flatten()

          # Apply PC
          yb_body = np.dot(A_inv_bodies[k], xb_body)
          # Rotate vectors to laboratory frame
          yb_body = yb_body.reshape((b.Nblobs+2, 3))
          yb = np.array([np.dot(rotation_matrix.T, vec) for vec in yb_body]).flatten()

        y[istart:iend] = yb

      # Fiber part of the solution
      for k in range(len(LU_fibers)):
        istart = offset + offset_bodies[-1]*3 + len(bodies)*6 + offset_fibers[k]*4
        iend = offset + offset_bodies[-1]*3 + len(bodies)*6 + offset_fibers[k+1]*4
        y[istart:iend] = scla.lu_solve((LU_fibers[k], P_fibers[k]), x[istart:iend])

      return y

    P_inv_partial = partial(P_inv, A_inv_bodies = self.A_inv_bodies , LU_bodies = LU_bodies,
      P_bodies = P_bodies, bodies = self.bodies, offset_bodies = offset_bodies,
      shell = self.shell, Minv = self.M_inv_periphery, LU_fibers = LU_fibers, P_fibers = P_fibers, offset_fibers = offset_fibers)

    P_inv_partial_LO = scspla.LinearOperator((system_size, system_size), matvec = P_inv_partial, dtype='float64')




    # 3.7. Time Matrix-Vector multiplication
    timer.timer('Build_A')
    def A_body_fiber_hyd(x_all, bodies, shell,
                        trg_bdy_surf, trg_bdy_cent, trg_shell_surf, trg_all,
                        normals_blobs, normals_shell,
                        As_fibers, As_BC, Gfibers, fibers, trg_fib,
                        fibers_force_operator, xfibers, K_bodies = None):

      '''
      | -0.5*I + T   -K   {G,R}Cbf + Gf         | |w*mu|   |   - G*F - R*L|
      |     -K^T      I        0                | | U  | = |      0       |
      |    -QoT      Cfb    A_ff - Qo{G,R} Cbf  | | Xf |   |     RHSf     |
      '''
      # Extract shell density
      offset = 0
      if shell is not None:
        offset = 3*shell.Nblobs
        shell_density = np.reshape(x_all[:offset], (shell.Nblobs, 3))

      # Extract body density, velocity and fiber's position and tension from x_all
      res = np.zeros_like(x_all) # residual
      body_densities = np.zeros((offset_bodies[-1],3))
      body_velocities = np.zeros((2*len(bodies),3))
      for k, b in enumerate(bodies):
        istart = offset + 3*offset_bodies[k]+6*k
        body_densities[offset_bodies[k]:offset_bodies[k+1]] = np.reshape(x_all[istart : istart+3*b.Nblobs], (b.Nblobs,3))
        istart += 3*b.Nblobs
        body_velocities[2*k : 2*k+2] = np.reshape(x_all[istart : istart + 6], (2,3))

      fw = np.array([])
      if fibers:
        # Fibers' unknowns
        XT = x_all[offset+offset_bodies[-1]*3 + len(bodies)*6:]

        # Compute implicit fiber forces
        force_fibers = fibers_force_operator.dot(XT)
        # Reorder array
        fw = np.zeros((force_fibers.size // 3, 3))
        fw[:,0] = force_fibers[flat2mat[:,0]]
        fw[:,1] = force_fibers[flat2mat[:,1]]
        fw[:,2] = force_fibers[flat2mat[:,2]]

      # VELOCITY DUE TO BODIES
      if bodies:
        # Due to hydrodynamic density
        if self.useFMM and (trg_bdy_surf.size//3)*(trg_all.size//3) >= 500:
          vbdy2all = self.stresslet_kernel_source_target_stkfmm_partial(trg_bdy_surf, trg_all, normals_blobs, body_densities, eta = bodies[0].viscosity_scale*self.eta)
        else:
          vbdy2all = kernels.stresslet_kernel_source_target_numba(trg_bdy_surf, trg_all, normals_blobs, body_densities, eta = bodies[0].viscosity_scale*self.eta)

        if fibers:
          # Torque and force due to fiber-body link
          body_vel_xt = np.concatenate((body_velocities.flatten(),XT.flatten()), axis=0)
          y_BC = As_BC.dot(body_vel_xt)
          force_bodies, torque_bodies = np.zeros((len(bodies),3)), np.zeros((len(bodies),3))
          for k, b in enumerate(bodies):
            force_bodies[k,:] = y_BC[6*k : 6*k+3]
            torque_bodies[k,:] = y_BC[6*k+3 : 6*k+6]
          # Due to forces and torques
          if self.useFMM and (trg_bdy_cent.size//3)*(trg_all.size//3) >= 500:
            vbdy2all += self.oseen_kernel_source_target_stkfmm_partial(trg_bdy_cent, trg_all, force_bodies, eta = bodies[0].viscosity_scale*self.eta)
          else:
            vbdy2all += kernels.oseen_kernel_source_target_numba(trg_bdy_cent, trg_all, force_bodies, eta = bodies[0].viscosity_scale*self.eta)
          vbdy2all += kernels.rotlet_kernel_source_target_numba(trg_bdy_cent, trg_all, torque_bodies, eta = bodies[0].viscosity_scale*self.eta)

        vbdy2fib = vbdy2all[:3*offset_fibers[-1]]
        vbdy2bdy = vbdy2all[3*offset_fibers[-1]:3*offset_fibers[-1]+3*offset_bodies[-1]]
        vbdy2shell = np.array([])
        if shell is not None:
          vbdy2shell = vbdy2all[3*offset_fibers[-1]+3*offset_bodies[-1]:]
      else:
        vbdy2fib = np.zeros(offset_fibers[-1]*3)
        vbdy2shell = np.zeros(offset)
        vbdy2bdy = np.zeros(offset_bodies[-1]*3)
        y_BC = np.zeros(len(bodies)*6 + offset_fibers[-1]*4)

      # VELOCITY DUE TO FIBERS
      if fibers:
        vfib2all = tstep_utils.flow_fibers(fw, trg_fib, trg_all, self.fibers, offset_fibers, self.eta, integration = self.integration ,
          fib_mats = self.fib_mats, fib_mat_resolutions = self.fib_mat_resolutions, iupsample = self.iupsample, oseen_fmm = self.oseen_kernel_source_target_stkfmm_partial, fmm_max_pts = 500)
        vfib2fib = vfib2all[:3*offset_fibers[-1]]

        # Correct calculation by moving off fibers
        vfib2bdy = vfib2all[3*offset_fibers[-1]:3*offset_fibers[-1]+3*offset_bodies[-1]]

        vfib2shell = np.array([])
        if self.shell is not None:
          vfib2shell = vfib2all[3*offset_fibers[-1]+3*offset_bodies[-1]:]

        # subtract self-flow due to Stokeslet
        vfib2fib += tstep_utils.self_flow_fibers(fw, offset_fibers, self.fibers, Gfibers, self.eta, integration = self.integration,
                                       fib_mats = self.fib_mats, fib_mat_resolutions = self.fib_mat_resolutions, iupsample = self.iupsample)
      else:
        vfib2fib, vfib2shell, vfib2bdy = np.zeros(offset_fibers[-1]*3), np.zeros(offset), np.zeros(offset_bodies[-1]*3)

      # VELOCITY DUE TO SHELL
      if shell is not None:
        # Shell's self-interaction (includes singularity subtraction)
        res[:shell.Nblobs*3]  = np.dot(self.shell_stresslet, shell_density.flatten())
        # Complementary kernel
        res[:shell.Nblobs*3] += np.dot(self.shell_complementary, shell_density.flatten())

        # Shell velocity due to body density and fibers
        res[:shell.Nblobs*3] += vbdy2shell + vfib2shell

        # Shell to body and fiber
        if self.useFMM and (trg_shell_surf.size//3)*(trg_all.size//3-trg_shell_surf.size//3) > 500:
          vshell2all = self.stresslet_kernel_source_target_stkfmm_partial(trg_shell_surf, trg_all[:3*offset_fibers[-1]+3*offset_bodies[-1]], normals_shell, shell_density, eta = self.eta)
        else:
          vshell2all = kernels.stresslet_kernel_source_target_numba(trg_shell_surf, trg_all[:3*offset_fibers[-1]+3*offset_bodies[-1]], normals_shell, shell_density, eta = self.eta)

        vshell2fib = vshell2all[:3*offset_fibers[-1]]
        vshell2bdy = vshell2all[3*offset_fibers[-1]:]

      else:
        vshell2bdy, vshell2fib = np.zeros(3*offset_bodies[-1]), np.zeros(3*offset_fibers[-1])



      # STACK VELOCITIES
      if bodies:
        # Compute -K*U
        K_times_U = tstep_utils.K_matrix_vector_prod(bodies, body_velocities, offset_bodies, K_bodies = K_bodies)
        K_times_U = K_times_U.flatten()

        # Compute -K.T*w*mu
        K_T_times_lambda = tstep_utils.K_matrix_vector_prod(bodies, body_densities.flatten(), offset_bodies, K_bodies = K_bodies, Transpose = True)
        K_T_times_lambda = K_T_times_lambda.flatten()

      # Singularity subtraction and also add other velocities
      for k, b in enumerate(bodies):
        d = np.zeros(b.Nblobs)
        d[:] = body_densities[offset_bodies[k]:offset_bodies[k]+b.Nblobs,0]
        cx = ((d / b.quadrature_weights)[:,None] * b.ex).flatten()
        d[:] = body_densities[offset_bodies[k]:offset_bodies[k]+b.Nblobs,1]
        cy = ((d / b.quadrature_weights)[:,None] * b.ey).flatten()
        d[:] = body_densities[offset_bodies[k]:offset_bodies[k]+b.Nblobs,2]
        cz = ((d / b.quadrature_weights)[:,None] * b.ez).flatten()

        istart = offset + 3 * offset_bodies[k] + 6 * k
        res[istart : istart + 3*b.Nblobs] += -(cx + cy + cz) - K_times_U[3*offset_bodies[k]:3*offset_bodies[k+1]]
        res[istart : istart + 3*b.Nblobs] += vbdy2bdy[3*offset_bodies[k]:3*offset_bodies[k+1]] + vshell2bdy[3*offset_bodies[k]:3*offset_bodies[k+1]] + vfib2bdy[3*offset_bodies[k]:3*offset_bodies[k+1]]
        res[istart + 3*b.Nblobs : istart + 3*b.Nblobs + 6] = -K_T_times_lambda[6*k : 6*(k+1)] + body_velocities[2*k:2*k+2].flatten()
      # VELOCITIES ON FIBERS
      if fibers:
        # Add up all velocities on fibers
        v_on_fib =  vbdy2fib + vshell2fib + vfib2fib


        # Now copy it to right format
        v_on_fib = v_on_fib.reshape((v_on_fib.size // 3, 3))

        vT = np.zeros(offset_fibers[-1] * 4)
        vT_in = np.zeros(offset_fibers[-1] * 4)
        vT[flat2mat_vT[:,0]] = v_on_fib[:,0]
        vT[flat2mat_vT[:,1]] = v_on_fib[:,1]
        vT[flat2mat_vT[:,2]] = v_on_fib[:,2]

        # Tension part of exterior flow
        xs_vT_in = xsDs.dot(v_on_fib[:,0]) + ysDs.dot(v_on_fib[:,1]) + zsDs.dot(v_on_fib[:,2])
        vT[flat2mat_vT[:,3]] = xs_vT_in

        # Tension part of velocity boundary condition at the attachment
        xs_vT = np.zeros(offset_fibers[-1] * 4)
        if flat2mat_1st.any():
          xs_vT[flat2mat_tenBC] = v_on_fib[flat2mat_1st,0] * xfibers[flat2mat_1st,0] + v_on_fib[flat2mat_1st,1] * xfibers[flat2mat_1st,1] + v_on_fib[flat2mat_1st,2] * xfibers[flat2mat_1st,2]
        if flat2mat_last.any():
          xs_vT[flat2mat_lastTenBC] = v_on_fib[flat2mat_last,0] * xfibers[flat2mat_last,0] + v_on_fib[flat2mat_last,1] * xfibers[flat2mat_last,1] + v_on_fib[flat2mat_last,2] * xfibers[flat2mat_last,2]

        # Multiply with the interpolation matrix to get tension
        vT_in[flat2mat_vT[:,4]] = P_cheb_sprs.dot(vT[flat2mat_vT[:,5]])
        # Add it to fiber part of the residual and get self-interactions
        res[offset + offset_bodies[-1]*3+6*len(bodies):] = As_fibers.dot(XT) - vT_in + y_BC[6*len(bodies):] + xs_vT




      return res


    linear_operator_partial = partial(A_body_fiber_hyd,
                                      bodies = self.bodies,
                                      shell = self.shell,
                                      trg_bdy_surf = trg_bdy_surf,
                                      trg_bdy_cent = trg_bdy_cent,
                                      trg_shell_surf = self.trg_shell_surf,
                                      trg_all = trg_all,
                                      normals_blobs = normals_blobs,
                                      normals_shell = self.normals_shell,
                                      K_bodies = K_bodies,
                                      As_fibers = As_fibers,
                                      As_BC = As_BC,
                                      fibers = self.fibers,
                                      trg_fib = trg_fib,
                                      Gfibers = Gfibers,
                                      fibers_force_operator = fibers_force_operator,
                                      xfibers = xfibers)
    linear_operator_partial = scspla.LinearOperator((system_size, system_size), matvec = linear_operator_partial, dtype='float64')
    timer.timer('Build_A')

    # -----------------
    # 4. GMRES to SOLVE
    # -----------------


    timer.timer('GMRES')
    counter = gmres_counter(print_residual = False)

    (sol, info_precond) = gmres.gmres(linear_operator_partial,
                                      RHS_all,
                                      tol=self.tol_gmres,
                                      atol=0,
                                      M=P_inv_partial_LO,
                                      maxiter=200,
                                      restart=150,
                                      callback=counter)

    if info_precond != 0:
      self.write_message('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
      self.write_message('GMRES did not converge in ' + str(counter.niter) + ' iterations.')
      # sys.exit()
    else:
      self.write_message('GMRES converged in ' + str(counter.niter) + ' iterations.')
    timer.timer('GMRES')

    if len(self.fibers) > 0:
      body_velocities = np.zeros((2*len(self.bodies),3))
      for k, b in enumerate(self.bodies):
        istart = offset + 3*offset_bodies[k]+6*k + 3*b.Nblobs
        body_velocities[2*k : 2*k+2] = np.reshape(sol[istart : istart + 6], (2,3))
      XT = sol[offset+offset_bodies[-1]*3 + len(self.bodies)*6:]
      # Compute implicit fiber forces
      force_on_fibers = fibers_force_operator.dot(XT)
      fw_sol = np.zeros((force_on_fibers.size // 3, 3))
      fw_sol[:,0] = force_on_fibers[flat2mat[:,0]]
      fw_sol[:,1] = force_on_fibers[flat2mat[:,1]]
      fw_sol[:,2] = force_on_fibers[flat2mat[:,2]]
      body_vel_xt = np.concatenate((body_velocities.flatten(),XT.flatten()), axis=0)
      y_BC = As_BC.dot(body_vel_xt)
      self.force_on_mtoc = np.zeros((len(self.bodies),6))
      for k, b in enumerate(self.bodies):
        self.force_on_mtoc[k,:3] = y_BC[6*k : 6*k+3]
        self.force_on_mtoc[k,3:] = y_BC[6*k+3 : 6*k+6]


    # -------------------------------------------------
    # COMPUTE VELOCITY ON A GRID
    # -------------------------------------------------
    iComputeVelocity = self.iComputeVelocity
    if iComputeVelocity and np.remainder(self.step_now, self.ncompute_vel) == 0:
      # Generate the grid
      grid_points = generateGrid.generate_grid(self.velP,self.velMeasureRad)
      # Shift the grid points to the center of the body
      grid_points[:,0] += self.bodies[0].location[0]
      grid_points[:,1] += self.bodies[0].location[1]
      grid_points[:,2] += self.bodies[0].location[2]

      grid_points = grid_points.flatten()

      def compute_velocity(x_all, bodies, shell,
                        trg_bdy_surf, trg_bdy_cent, trg_shell_surf,
                        normals_blobs, normals_shell,
                        As_fibers, As_BC, Gfibers, fibers, trg_fib,
                        fibers_force_operator, xfibers, grid_points, K_bodies = None):


        # Extract shell density
        offset = 0
        if shell is not None:
          offset = 3*shell.Nblobs
          shell_density = np.reshape(x_all[:offset], (shell.Nblobs, 3))

        # Extract body density, velocity and fiber's position and tension from x_all
        body_densities = np.zeros((offset_bodies[-1],3))
        body_velocities = np.zeros((2*len(bodies),3))
        for k, b in enumerate(bodies):
          istart = offset + 3*offset_bodies[k]+6*k
          body_densities[offset_bodies[k]:offset_bodies[k+1]] = np.reshape(x_all[istart : istart+3*b.Nblobs], (b.Nblobs,3))
          istart += 3*b.Nblobs
          body_velocities[2*k : 2*k+2] = np.reshape(x_all[istart : istart + 6], (2,3))

        fw = np.array([])
        if fibers:
          # Fibers' unknowns
          XT = x_all[offset+offset_bodies[-1]*3 + len(bodies)*6:]

          # Compute implicit fiber forces
          force_fibers = fibers_force_operator.dot(XT)
          # Reorder array
          fw = np.zeros((force_fibers.size // 3, 3))
          fw[:,0] = force_fibers[flat2mat[:,0]]
          fw[:,1] = force_fibers[flat2mat[:,1]]
          fw[:,2] = force_fibers[flat2mat[:,2]]

        # VELOCITY DUE TO BODIES
        if bodies:
          # Due to hydrodynamic density
          vbdy2all = kernels.stresslet_kernel_source_target_numba(trg_bdy_surf, grid_points, normals_blobs, body_densities, eta = bodies[0].viscosity_scale*self.eta)

          if np.isnan(vbdy2all).any():
            print('body to grid has non')

          if fibers:
            # Torque and force due to fiber-body link
            body_vel_xt = np.concatenate((body_velocities.flatten(),XT.flatten()), axis=0)
            y_BC = As_BC.dot(body_vel_xt)
            force_bodies, torque_bodies = np.zeros((len(bodies),3)), np.zeros((len(bodies),3))
            for k, b in enumerate(bodies):
              force_bodies[k,:] = y_BC[6*k : 6*k+3]
              torque_bodies[k,:] = y_BC[6*k+3 : 6*k+6]
            # Due to forces and torques
            vbdy2all_force = kernels.oseen_kernel_source_target_numba(trg_bdy_cent, grid_points, force_bodies, eta = bodies[0].viscosity_scale*self.eta)

            if np.isnan(vbdy2all_force).any():
              print('body to grid (oseen) has nan')

            vbdy2all_force += kernels.rotlet_kernel_source_target_numba(trg_bdy_cent, grid_points, torque_bodies, eta = bodies[0].viscosity_scale*self.eta)
            if np.isnan(vbdy2all_force).any():
              print('body to grid (rotlet) has nan')
            vbdy2all += vbdy2all_force
        else:
          y_BC = np.zeros(len(bodies)*6 + offset_fibers[-1]*4)
          vbdy2all = np.zeros_like(grid_points)

        # VELOCITY DUE TO FIBERS
        if fibers:
          vfib2all = tstep_utils.flow_fibers(fw, trg_fib, grid_points, self.fibers, offset_fibers, self.eta, integration = self.integration ,
            fib_mats = self.fib_mats, fib_mat_resolutions = self.fib_mat_resolutions, iupsample = self.iupsample, oseen_fmm = None, fmm_max_pts = 500)
        else:
          vfib2all = np.zeros_like(grid_points)
        if np.isnan(vfib2all).any():
          print('fiber to grid has nan')

        # VELOCITY DUE TO SHELL
        if shell is not None:
          # Shell to body and fiber
          vshell2all = kernels.stresslet_kernel_source_target_numba(trg_shell_surf, grid_points, normals_shell, shell_density, eta = self.eta)
        else:
          vshell2all = np.zeros_like(grid_points)

        if np.isnan(vshell2all).any():
          print('shell to grid has nan')

        vgrid = vshell2all.reshape((vshell2all.size//3,3)) + vbdy2all.reshape((vbdy2all.size//3, 3)) + vfib2all.reshape((vfib2all.size//3, 3))

        return vgrid



    # ---------------------------------------
    # 5. UPDATE FIBER CONFIGURATION AND BODIES
    # ---------------------------------------

    # Shell
    if self.shell is not None: self.shell.density_new = sol[:offset]

    # Fibers
    for k, fib in enumerate(self.fibers):
      istart = offset + offset_bodies[-1]*3 + 6*len(self.bodies) + offset_fibers[k] * 4
      fib.tension_new = np.copy(sol[istart + 3 * fib.num_points : istart + 4 * fib.num_points])
      fib.x_new[:,0] = sol[istart + 0 * fib.num_points : istart + 1 * fib.num_points]
      fib.x_new[:,1] = sol[istart + 1 * fib.num_points : istart + 2 * fib.num_points]
      fib.x_new[:,2] = sol[istart + 2 * fib.num_points : istart + 3 * fib.num_points]
      fib.on_force = fw_sol[offset_fibers[k]:offset_fibers[k]+fib.num_points] + force_fibers[offset_fibers[k]:offset_fibers[k]+fib.num_points]

    # Bodies
    for k, b in enumerate(self.bodies):
      istart = offset + 3*offset_bodies[k]+6*k
      b.density_new = np.reshape(sol[istart : istart+3*b.Nblobs], (b.Nblobs,3))

      istart += 3*b.Nblobs
      b.location_new = b.location + sol[istart:istart+3] * self.dt
      quaternion_dt = quaternion.Quaternion.from_rotation(sol[istart+3:istart+6] * self.dt)
      b.orientation_new = quaternion_dt * b.orientation
      b.velocity_new = sol[istart:istart+3]
      b.angular_velocity_new = sol[istart+3:istart+6]
      b.on_force = self.force_on_mtoc[k,:] + force_bodies[k,:]
      self.write_message('Body velocity: ' + str(b.velocity_new))
      self.write_message('Angular velocity: ' + str(b.angular_velocity_new))

    # 3.7. Traction mat-vec
    timer.timer('Build_A')
    def A_traction(xin, body):
      # for each point, x, y, z components of the traction
      traction = np.reshape(xin,(body.Nblobs,3)) 
      
      centxyz = body.location
      
      weights = body.quadrature_weights
      normals = body.get_normals()
      surface_xyz = body.get_r_vectors_surface()

      # Extract body density, velocity and fiber's position and tension from x_all
      traction2 = np.zeros_like(traction) # residual

      for xn in range(body.Nblobs):
        for yn in range(body.Nblobs):
          if xn != yn:
            S = np.zeros((3,3,3)) # j, k, i
            drxyz = np.zeros(3)
            drxyz[0] = surface_xyz[yn,0] - surface_xyz[xn,0]
            drxyz[1] = surface_xyz[yn,1] - surface_xyz[xn,1]
            drxyz[2] = surface_xyz[yn,2] - surface_xyz[xn,2]
            dr = np.sqrt(drxyz[0]**2 + drxyz[1]**2 + drxyz[2]**2)
            # Stresslet contribution to the traction
            fr = 3.0 / (4.0 * np.pi * self.eta * np.sqrt(dr**2 + 25E-6)**5)
            rfy = np.dot(drxyz,traction[yn,:])
            rnx = np.dot(drxyz,normals[xn,:])
            rny = np.dot(drxyz,normals[yn,:])
            rfx = np.dot(drxyz,traction[xn,:])

            traction2[xn,:] += weights[yn]*fr*drxyz*(rfy*rnx+rny*rfx)
            
          # Stokeslet contribution to the traction
          drxyz_G = np.zeros(3)
          drxyz_G[0] = surface_xyz[yn,0]-centxyz[0]
          drxyz_G[1] = surface_xyz[yn,1]-centxyz[1]
          drxyz_G[2] = surface_xyz[yn,2]-centxyz[2]
          dr_G = np.sqrt(drxyz_G[0]**2 + drxyz_G[1]**2 + drxyz_G[2]**2)
          fr = 1.0 / (8.0 * np.pi * self.eta * np.sqrt(dr_G**2 + 25E-6))
          gr = 1.0 / (8.0 * np.pi * self.eta * np.sqrt(dr_G**2 + 25E-6)**3)
          rfy = np.dot(drxyz_G,traction[yn,:])
          traction2[xn,:] += (fr*traction[yn,:] + gr*drxyz_G*rfy)*weights[yn]

          # Rotlet contribution
          rot = np.zeros(3)
          drxyz_R = np.zeros(3)
          drxyz_R[0] = surface_xyz[xn,0]-centxyz[0]
          drxyz_R[1] = surface_xyz[xn,1]-centxyz[1]
          drxyz_R[2] = surface_xyz[xn,2]-centxyz[2]
          
          hr = -1.0 / (8.0 * np.pi * self.eta * np.sqrt(dr_G**2 + 25E-6)**3)

          rot[0] = (traction[yn,1]*drxyz_G[2]-traction[yn,2]*drxyz_G[1])
          rot[1] = (traction[yn,2]*drxyz_G[0]-traction[yn,0]*drxyz_G[2])
          rot[2] = (traction[yn,0]*drxyz_G[1]-traction[yn,1]*drxyz_G[0])
          
          traction2[xn,0] += hr*weights[yn]*(rot[1]*drxyz_R[2] - rot[2]*drxyz_R[1])
          traction2[xn,1] += hr*weights[yn]*(rot[2]*drxyz_R[0] - rot[0]*drxyz_R[2])
          traction2[xn,2] += hr*weights[yn]*(rot[0]*drxyz_R[1] - rot[1]*drxyz_R[0])
      return traction2.flatten()

    for k, b in enumerate(self.bodies):
      RHS_t = np.zeros((b.Nblobs,3))
      centxyz = b.location
      
      cent_velocity = b.velocity_new
      cent_ang_velocity = b.angular_velocity_new

      surface_xyz = b.get_r_vectors_surface()
      for xn in range(b.Nblobs):
        drxyz_R = np.zeros(3)
        drxyz_R[0] = surface_xyz[xn,0] - centxyz[0]
        drxyz_R[1] = surface_xyz[xn,1] - centxyz[1]
        drxyz_R[2] = surface_xyz[xn,2] - centxyz[2]
      
        angVel = np.zeros(3)
        angVel[0] = cent_ang_velocity[1]*drxyz_R[2] - cent_ang_velocity[2]*drxyz_R[1]
        angVel[1] = cent_ang_velocity[2]*drxyz_R[0] - cent_ang_velocity[0]*drxyz_R[2]
        angVel[2] = cent_ang_velocity[0]*drxyz_R[1] - cent_ang_velocity[1]*drxyz_R[0]

        RHS_t[xn,0] = cent_velocity[0] + angVel[0] 
        RHS_t[xn,1] = cent_velocity[1] + angVel[1] 
        RHS_t[xn,2] = cent_velocity[2] + angVel[2] 
      

      
      traction_operator_partial = partial(A_traction,body = b)
      linear_operator_partial = scspla.LinearOperator((b.Nblobs*3, b.Nblobs*3), matvec = traction_operator_partial, dtype='float64')
      
      counter2 = gmres_counter(print_residual = True)
      if np.remainder(self.step_now,self.n_save) == 0:
        (solTraj, info_precond) = gmres.gmres(linear_operator_partial,
                                        RHS_t.flatten(),
                                        tol=1E-8,
                                        atol=0,
                                        maxiter=200,
                                        restart=150,
                                        callback=counter2)

        if info_precond != 0:
          self.write_message('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
          self.write_message('GMRES did not converge in ' + str(counter2.niter) + ' iterations.')
          # sys.exit()
        else:
          self.write_message('GMRES converged in ' + str(counter2.niter) + ' iterations.')

        b.traction = -np.reshape(solTraj,(b.Nblobs,3)) # revert the direction

    if iComputeVelocity and np.remainder(self.step_now, self.ncompute_vel) == 0:
      vgrid = compute_velocity(sol, self.bodies, self.shell,
                      trg_bdy_surf, trg_bdy_cent, self.trg_shell_surf,
                      normals_blobs, self.normals_shell,
                      As_fibers, As_BC, Gfibers, self.fibers, trg_fib,
                      fibers_force_operator, xfibers, grid_points, K_bodies = K_bodies)

      if force_fibers.any():
        vfib2grid = tstep_utils.flow_fibers(force_fibers, trg_fib, grid_points, self.fibers, offset_fibers, self.eta, integration = self.integration ,
            fib_mats = self.fib_mats, fib_mat_resolutions = self.fib_mat_resolutions, iupsample = self.iupsample, oseen_fmm = None, fmm_max_pts = 500)
        vgrid += vfib2grid.reshape((vfib2grid.size//3,3))
        self.write_message('There is force_fiber causing vgrid')

      if force_bodies.any():
        vbdy2grid = kernels.oseen_kernel_source_target_numba(trg_bdy_cent, grid_points, force_bodies[:,:3], eta = self.bodies[0].viscosity_scale*self.eta)
        if np.isnan(vbdy2grid).any():
          print('body to grid (external force) has nan')
        vbdy2grid += kernels.rotlet_kernel_source_target_numba(trg_bdy_cent, grid_points, force_bodies[:,3:], eta = self.bodies[0].viscosity_scale*self.eta)
        if np.isnan(vbdy2grid).any():
          print('body to grid (external toruq) has nan')
        vgrid += vbdy2grid.reshape((vbdy2grid.size//3,3))

      grid_points = grid_points.reshape((grid_points.size//3,3))

      if self.bodies:
        loc = self.bodies[0].location
        ids = np.sqrt(((grid_points[:,0]-loc[0])/self.body_a)**2 + ((loc[1]-grid_points[:,1])/self.body_b)**2 + ((loc[2]-grid_points[:,2])/self.body_c)**2) <= 1
        vgrid[ids] = 0
        vgrid[ids] = self.bodies[0].velocity

        rgrid = grid_points[ids] - loc
        omega_b = self.bodies[0].angular_velocity
        vgrid[ids,0] += omega_b[1]*rgrid[:,2] - omega_b[2]*rgrid[:,1]
        vgrid[ids,1] += omega_b[2]*rgrid[:,0] - omega_b[0]*rgrid[:,2]
        vgrid[ids,2] += omega_b[0]*rgrid[:,1] - omega_b[1]*rgrid[:,0]

      # Save velocity and the grid points
      
      name = self.output_name + 'grid_at_step' + str(self.step_now) + '.txt'
      f_grid = open(name, 'w')
      np.savetxt(f_grid,grid_points)
      f_grid.close()

      name = self.output_name + '_velocity_at_step' + str(self.step_now) + '.txt'
      f_grid_velocity = open(name, 'w')
      np.savetxt(f_grid_velocity, vgrid)
      f_grid_velocity.close()

    return # time_step_hydro

  ##############################################################################################
  def time_step_dry(self, dt, *args, **kwargs):
    '''
    Time step without hydrodynamics
    System can have rigid bodies, fibers, (stationary) molecular motors
    '''
    # ------------------
    # 0. INITIALIZATION
    # ------------------
    # Update fibers' lengths
    timer.timer('update_length')
    if self.periphery_a is None and self.periphery_radius is not None:
      periphery_a = self.periphery_radius
      periphery_b = self.periphery_radius
      periphery_c = self.periphery_radius
    else:
      periphery_a = self.periphery_a
      periphery_b = self.periphery_b
      periphery_c = self.periphery_c
    # CORTICAL PUSHING
    if self.iNoSliding:
      self.fibers = tstep_utils.no_sliding_cortical_pushing(self.fibers, cortex_a = self.periphery_a, cortex_b = self.periphery_b,
        cortex_c = self.periphery_c, cortex_radius = self.periphery_radius)

    istep = self.step_now
    self.fibers, self.bodies = tstep_utils.dynamic_instability_v3(self.fibers,
                                                                  self.bodies,
                                                                  self.prams,
                                                                  self.site_idcs_nucleating[istep],
                                                                  self.site_idcs_dying[istep],
                                                                  self.site_idcs_hinged[istep],
                                                                  self.options,
                                                                  self.dt)


    # self.fibers, self.fibers_gone, self.bodies, self.num_nucleated_fibers, self.num_nucleated_imaginary, self.sites_gone_imaginary, self.terminal_time_gone_imaginary = tstep_utils.dynamic_instability_v2(self.fibers,
    #                                                                        self.bodies,
    #                                                                        self.prams,
    #                                                                        self.options,
    #                                                                        self.dt,
    #                                                                        periphery_a,
    #                                                                        periphery_b,
    #                                                                        periphery_c,
    #                                                                        self.time_now)

    timer.timer('update_length')

    timer.timer('initialization_step')
    # 1.1. Get number of particles and offsets
    num_bodies, num_fibers, offset_bodies, offset_fibers, system_size = tstep_utils.get_num_particles_and_offsets(self.bodies, self.fibers, self.shell, ihydro = False)

    # Get moving bodies with MMs info
    num_bodies_mm_attached = 0 # len(self.bodies_mm_attached)
    offset_bodies_mm_attached = num_bodies_mm_attached * 6

    # 1.2. Form initial guess for GMRES
    initGMRES, x0 = np.zeros(system_size + num_bodies_mm_attached*6), np.zeros((offset_fibers[-1],3))
    xEnds = np.zeros((len(self.fibers),3))

    # Fiber part of the solution, also compute derivatives of fiber configurations
    for k,fib in enumerate(self.fibers):
      istart = 6*len(self.bodies) + 4*offset_fibers[k] + num_bodies_mm_attached

      x0[offset_fibers[k] : offset_fibers[k+1]] = fib.x
      xEnds[k] = fib.x[-1]

      # Find the index for fib_mats
      indx = np.where(self.fib_mat_resolutions == fib.num_points)
      # Get the class that has the matrices
      fib_mat = self.fib_mats[indx[0][0]]

      # Call differentiation matrices
      D_1, D_2, D_3, D_4 = fib_mat.get_matrices(fib.length_previous, fib.num_points_up, 'Ds')

      # Compute derivatives along the fiber (scale diff. matrices with current length?)
      fib.xs = np.dot(D_1, fib.x)
      fib.xss = np.dot(D_2, fib.x)
      fib.xsss = np.dot(D_3, fib.x)
      fib.xssss = np.dot(D_4, fib.x)

    timer.timer('initialization_step')

    # NO CYTOPLASMIC MOTOR FORCE
    motor_force_fibers = np.zeros((offset_fibers[-1],3))
    force_fibers = np.zeros((offset_fibers[-1],3))
    # Compute external forces due to repulsion (ARTIFICIAL FORCES)
    if self.bodies_mm_attached:
      force_bodies_mm_attached, force_bodies, force_fibers = forces.compute_repulsion_force(self.bodies,
        self.fibers, self.bodies_mm_attached, offset_fibers, x0, self.len_nuc2fib, self.len_nuc2bdy, self.len_nuc2nuc,
        self.scale_nuc2fib, self.scale_nuc2bdy, self.scale_nuc2nuc)

    elif self.nucleus_radius:
      force_bodies, force_fibers = forces.compute_external_forces(self.bodies,
        self.fibers, x0, offset_fibers[-1], offset_fibers, offset_bodies,
        nucleus_radius = self.nucleus_radius, nucleus_position = self.nucleus_position)
      force_bodies_mm_attached = np.zeros((len(self.bodies_mm_attached),6))

    if periphery_a is not None:
      cort_force_fibers, cort_force_bodies, cort_force_bodies_mm_attached = forces.compute_cortex_forces(self.bodies,
        self.bodies_mm_attached, self.fibers, x0, offset_fibers[-1], periphery_a, periphery_b, periphery_c)
      force_fibers += cort_force_fibers
      force_bodies = cort_force_bodies
      force_bodies_mm_attached = cort_force_bodies_mm_attached

    if self.iFixObjects:
      force_bodies = np.zeros((len(self.bodies),6))
      force_bodies_mm_attached = np.zeros((len(self.bodies_mm_attached),6))

    self.offset_fibers = offset_fibers
    if self.isaveForces:
      # Save external forces
      self.fibers_repulsion_force = np.copy(force_fibers)
      self.bodies_repulsion_force = np.copy(force_bodies)
      self.bodies_mm_attached_repulsion_force = np.copy(force_bodies_mm_attached)


    # Molecular motors (Stationary)
    if self.molecular_motors and not self.iFixObjects:
      self.write_message('Motors on stationary surface(s) are in action')
      timer.timer('molecular_motor')
      # Compute fiber modes
      timer.timer('molecular_motor.modes')
      for fib in self.fibers: fib.compute_modes()
      timer.timer('molecular_motor.modes')

      # Compute x and xs
      timer.timer('molecular_motor.find_x_xs_and_length_MT')
      self.molecular_motors.find_x_xs_and_length_MT(self.fibers)
      timer.timer('molecular_motor.find_x_xs_and_length_MT')

      # Compute force
      timer.timer('molecular_motor.compute_force')
      self.molecular_motors.compute_force()
      timer.timer('molecular_motor.compute_force')

      # Spread force
      timer.timer('molecular_motor.spread_force')
      for fib in self.fibers: fib.force_motors[:,:] = 0.0
      self.molecular_motors.spread_force(self.fibers)
      force_fg = np.zeros_like(force_fibers)
      for k,fib in enumerate(self.fibers):
        force_fg[offset_fibers[k] : offset_fibers[k]+fib.num_points] = fib.force_motors
      timer.timer('molecular_motor.spread_force')

      # Walk and diffuse
      timer.timer('molecular_motor.walk')
      self.molecular_motors.walk(self.dt)
      timer.timer('molecular_motor.walk')
      timer.timer('molecular_motor.diffuse')
      self.molecular_motors.diffuse(self.dt)
      timer.timer('molecular_motor.diffuse')
      timer.timer('molecular_motor')

      # Update links
      timer.timer('molecular_motor.update_links')
      self.molecular_motors.update_links_numba(self.dt, self.fibers)
      timer.timer('molecular_motor.update_links')


      force_fibers += force_fg

    # Molecular motors (Moving with a body)
    if self.MM_on_moving_surf and not self.iFixObjects:
      self.write_message('Motors on moving surface(s) are in action')
      for mb, mm_on_body in enumerate(self.MM_on_moving_surf):
        timer.timer('molecular_motor')
        # Compute fiber modes
        timer.timer('molecular_motor.modes')
        for fib in self.fibers: fib.compute_modes()
        timer.timer('molecular_motor.modes')

        # Compute x and xs
        timer.timer('molecular_motor.find_x_xs_and_length_MT')
        mm_on_body.find_x_xs_and_length_MT(self.fibers)
        timer.timer('molecular_motor.find_x_xs_and_length_MT')

        # Compute force
        timer.timer('molecular_motor.compute_force')
        mm_on_body.compute_force()
        timer.timer('molecular_motor.compute_force')

        # Spread force
        timer.timer('molecular_motor.spread_force')
        for fib in self.fibers: fib.force_motors[:,:] = 0.0
        mm_on_body.spread_force(self.fibers)
        force_fg = np.zeros_like(force_fibers)
        for k,fib in enumerate(self.fibers):
          force_fg[offset_fibers[k] : offset_fibers[k]+fib.num_points] = fib.force_motors
        timer.timer('molecular_motor.spread_force')

        # Calculate force and torque on moving body
        timer.timer('molecular_motor.get_force_torque_body')
        force_body, torque_body = mm_on_body.spread_force_body(self.bodies_mm_attached[mb])
        force_bodies_mm_attached[mb,:3] += force_body.flatten()
        force_bodies_mm_attached[mb,3:] += torque_body.flatten()
        timer.timer('molecular_motor.get_force_torque_body')

        # Walk and diffuse
        timer.timer('molecular_motor.walk')
        mm_on_body.walk(self.dt)
        timer.timer('molecular_motor.walk')
        timer.timer('molecular_motor.diffuse')
        mm_on_body.diffuse(self.dt)
        timer.timer('molecular_motor.diffuse')
        timer.timer('molecular_motor')

        # Update links
        timer.timer('molecular_motor.update_links')
        mm_on_body.update_links_numba(self.dt, self.fibers)
        timer.timer('molecular_motor.update_links')


        force_fibers += force_fg

      if self.isaveForces:
        self.fibers_motor_force = force_fg


    # ---------------------------------------------------
    # 2. BLOCK DIAGONAL MATRICES AND RHSs FOR FIBERS
    # ---------------------------------------------------
    # 2.1. Build RHS (x,y,z) of point 1, then (x,y,z) of point 2. This is the order
    timer.timer('Build_linear_system_fibers')
    for k, fib in enumerate(self.fibers):
      # (-) end of fiber
      if fib.attached_to_body is not None:
        # Clamped
        BC_start_0, BC_start_1 = 'velocity', 'angular_velocity'
      else:
        # Force-torque free
        BC_start_0, BC_start_1 = 'force', 'torque'

      # (+) end of fiber

      if fib.iReachSurface:
        # HINGED
        BC_end_0, BC_end_1 = 'velocity', 'torque'
      else:
        # FORCE-TORQUE FREE
        BC_end_0, BC_end_1 = 'force','torque'

      fib.set_BC(BC_start_0 = BC_start_0,
                BC_start_1 = BC_start_1,
                BC_end_0 = BC_end_0,
                BC_end_1 = BC_end_1)




    As_fibers, A_fibers_blocks, RHS_all = tstep_utils.get_fibers_and_bodies_matrices(self.fibers, self.bodies, self.shell,
      system_size, offset_fibers, offset_bodies, force_fibers, motor_force_fibers, force_bodies, None, None, None, self.fib_mats, self.fib_mat_resolutions,
      inextensibility = self.inextensibility, ihydro = False)

    if self.bodies_mm_attached:
      RHS = np.zeros(len(self.bodies_mm_attached)*6)
      RHS[0:len(self.bodies_mm_attached)*6:6] = force_bodies_mm_attached[:,0]
      RHS[1:len(self.bodies_mm_attached)*6:6] = force_bodies_mm_attached[:,1]
      RHS[2:len(self.bodies_mm_attached)*6:6] = force_bodies_mm_attached[:,2]
      RHS[3:len(self.bodies_mm_attached)*6:6] = force_bodies_mm_attached[:,3]
      RHS[4:len(self.bodies_mm_attached)*6:6] = force_bodies_mm_attached[:,4]
      RHS[5:len(self.bodies_mm_attached)*6:6] = force_bodies_mm_attached[:,5]
      RHS_all = np.concatenate((RHS, RHS_all))
    timer.timer('Build_linear_system_fibers')

    # ---------------------------------------------------
    # 3. DEFINE LINEAR OPERATOR
    # ---------------------------------------------------
    # 3.1. QR factorization of fiber matrices
    iLU = True
    LU_fibers, P_fibers = [], []
    if self.fibers:
      timer.timer('Build_PC_fibers') # this can be avoided if factorization is done when building the matrix A_fibers_blocks
      LU_fibers, P_fibers = tstep_utils.build_block_diagonal_lu_preconditioner_fiber(self.fibers, A_fibers_blocks)
      timer.timer('Build_PC_fibers')

    # 3.2. Preconditioner for the system
    large_sys_size = system_size + num_bodies_mm_attached*6

    def P_inv(x, LU, P, offset_fibers, offset_bodies, offset_bodies_mm_attached):
      timer.timer('PC_apply_fibers')
      y = np.empty_like(x)
      # PC for bodies is the identity matrix
      y[0 : offset_bodies+offset_bodies_mm_attached] = x[0 : offset_bodies+offset_bodies_mm_attached]
      # For fibers is block diagonal
      for i in range(len(LU)):
        istart = offset_bodies_mm_attached + offset_bodies +offset_fibers[i]*4
        iend = offset_bodies_mm_attached + offset_bodies + offset_fibers[i+1]*4
        y[istart:iend] = scla.lu_solve((LU[i], P[i]),  x[istart:iend])
      timer.timer('PC_apply_fibers')
      return y
    P_inv_partial = partial(P_inv, LU = LU_fibers, P = P_fibers, offset_fibers = offset_fibers, offset_bodies = offset_bodies, offset_bodies_mm_attached = offset_bodies_mm_attached)
    P_inv_partial_LO = scspla.LinearOperator((large_sys_size, large_sys_size), matvec = P_inv_partial, dtype='float64')


    # 3.3. Build link matrix (fibers' boundary conditions)
    As_BC = None
    if self.fibers and self.bodies:
      timer.timer('Build_sparse_BC')
      As_BC = tstep_utils.build_link_matrix(4*offset_fibers[-1]+6*len(self.bodies), self.bodies,self.fibers, offset_fibers, 6*len(self.bodies),
        self.fib_mats, self.fib_mat_resolutions)
      timer.timer('Build_sparse_BC')

    # 3.4. Time Matrix-Vector Multiplication
    def A_body_fiber(x, As_fibers, As_BC, bodies, offset_bodies, offset_fibers, bodies_mm_attached, offset_bodies_mm_attached):
      # Create solution and body mobility
      timer.timer('Apply_A')
      y = np.empty_like(x)


      # a. Multiply by block diagonal matrices of bodies
      for k, b in enumerate(bodies_mm_attached):
        M = np.eye(6)
        radius = b.radius
        M[0:3, 0:3] = M[0:3, 0:3] * (6.0 * np.pi * self.eta * radius)
        M[3:6, 3:6] = M[3:6, 3:6] * (8.0 * np.pi * self.eta * radius**3)
        y[6*k : 6*(k+1)] = np.dot(M, x[6*k : 6*(k+1)])

      for k, b in enumerate(bodies):
        M = np.eye(6)
        radius = b.radius
        M[0:3, 0:3] = M[0:3, 0:3] * (6.0 * np.pi * self.eta * radius)
        M[3:6, 3:6] = M[3:6, 3:6] * (8.0 * np.pi * self.eta * radius**3)
        istart = offset_bodies_mm_attached + 6*k
        iend = offset_bodies_mm_attached + 6*k + 6
        y[istart : iend] = np.dot(M, x[istart : iend])

      # b. Multiply by block diagonal matrices of fibers
      if As_fibers is not None:
        y[offset_bodies+offset_bodies_mm_attached:] = As_fibers.dot(x[offset_bodies+offset_bodies_mm_attached:])

      # c. Add BC
      yBC = None
      if As_BC is not None:
        yBC = As_BC.dot(x[offset_bodies_mm_attached:])

      if yBC is not None:
        if not self.iFixObjects:
          yBC[:6*len(bodies)] = -yBC[:6*len(bodies)]
        else: # if fixing, then apply equal and opposite forces
          # ALLOWING ORIENTATION CHANGE
          #for k, b in enumerate(bodies):
          #  yBC[6*k+0 : 6*k+3] = 0
          #  yBC[6*k+3 : 6*k+6] = -yBC[6*k+3 : 6*k+6]
          # FIXED POSITION AND ORIENTATION
          yBC[:6*len(bodies)] = 0

        y[offset_bodies_mm_attached:] += yBC

      timer.timer('Apply_A')
      return y

    A_body_fiber_partial = partial(A_body_fiber,
                                   As_fibers = As_fibers,
                                   As_BC = As_BC,
                                   bodies = self.bodies,
                                   offset_bodies = offset_bodies,
                                   offset_fibers = offset_fibers,
                                   bodies_mm_attached = self.bodies_mm_attached,
                                   offset_bodies_mm_attached = offset_bodies_mm_attached)
    A_body_fiber_LO = scspla.LinearOperator((large_sys_size, large_sys_size), matvec = A_body_fiber_partial, dtype='float64')


    # ---------------------------------------------------
    # 3. CALL GMRES
    # ---------------------------------------------------
    timer.timer('GMRES')
    counter = gmres_counter(print_residual = False)
    (sol, info_precond) = gmres.gmres(A_body_fiber_LO,
                                      RHS_all,
                                      tol = self.tol_gmres,
                                      atol = 0,
                                      M = P_inv_partial_LO,
                                      maxiter = 200,
                                      restart = 150,
                                      callback=counter)

    if info_precond != 0:
      self.write_message('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
      self.write_message('GMRES did not converge in ' + str(counter.niter) + ' iterations.')
      # sys.exit()
    else:
      self.write_message('GMRES converged in ' + str(counter.niter) + ' iterations.')
    timer.timer('GMRES')

    # -----------------------------
    # 4. UPDATE FIBERS and BODIES CONFIGURATIONS
    # -----------------------------
    timer.timer('update_unknowns')
    # Fibers
    for k, fib in enumerate(self.fibers):
      istart = offset_bodies_mm_attached + offset_bodies + offset_fibers[k] * 4
      fib.tension_new = np.copy(sol[istart + 3*fib.num_points : istart + 4*fib.num_points])
      fib.x_new[:,0] = sol[istart + 0*fib.num_points : istart + 1*fib.num_points]
      fib.x_new[:,1] = sol[istart + 1*fib.num_points : istart + 2*fib.num_points]
      fib.x_new[:,2] = sol[istart + 2*fib.num_points : istart + 3*fib.num_points]


    # Bodies (mm attached)
    for k, b in enumerate(self.bodies_mm_attached):
      b.location_new = b.location + sol[k*6 : k*6 + 3] * self.dt
      quaternion_dt = quaternion.Quaternion.from_rotation(sol[k*6+3 : k*6 + 6] * self.dt)
      b.orientation_new = quaternion_dt * b.orientation
      b.velocity_new = sol[6*k : 6*k+3]
      b.angular_velocity_new = sol[6*k+3 : 6*k+6]

    # Bodies (not mm attached)
    for k, b in enumerate(self.bodies):
      istart = offset_bodies_mm_attached + 6*k
      b.location_new = b.location + sol[istart : istart + 3] * self.dt
      quaternion_dt = quaternion.Quaternion.from_rotation(sol[istart+3 : istart + 6] * self.dt)
      b.orientation_new = quaternion_dt * b.orientation
      b.velocity_new = sol[istart : istart+3]
      b.angular_velocity_new = sol[istart+3 : istart+6]
      self.write_message('Body velocity: ' + str(b.velocity_new))
      self.write_message('Angular velocity: ' + str(b.angular_velocity_new))

    timer.timer('update_unknowns')


    if self.isaveForces:
      # Compute and store forces and torques on bodies due to links
      y = np.zeros((6*len(self.bodies),1))
      if As_BC is not None:
        y = As_BC[:6*len(self.bodies),:].dot(sol[offset_bodies_mm_attached:])
      self.bodies_link_force_torque = y

    return # time_step_dry

  ##############################################################################################

  def check_error(self):
    '''
    Check error in fibers.x_new in terms of inextensibility,
    compare with tol_tstep and choose next time step size
    '''

    fiber_error = np.zeros(len(self.fibers))

    for k, fib in enumerate(self.fibers):

      # Find the index for fib_mats
      indx = np.where(self.fib_mat_resolutions == fib.num_points)
      indx = indx[0][0]

      # Get the class that has the matrices
      fib_mat = self.fib_mats[indx]

      D_1, D_2, D_3, D_4 = fib_mat.get_matrices(fib.length, fib.num_points_up, 'Ds')
      xs = np.dot(D_1, fib.x_new)
      fiber_error[k] = abs(max(np.sqrt(xs[:,0]**2 + xs[:,1]**2 + xs[:,2]**2) - 1.0, key=abs))

    # Maximum error in inextensibility
    max_error = max(fiber_error, key=abs)


    # Optimum time step size, avoid it going too large
    if self.adaptive_time:
      # If adaptive time stepping, then update time step size

      if max_error <= self.tol_tstep:
        accept = True
        if max_error <= self.tol_tstep * 0.9:
          dt_new = min(self.dt_max, self.dt * 1.2)
        elif max_error > self.tol_tstep * 0.9:
          dt_new = self.dt
      else:
        dt_new, accept = self.dt * 0.5, False

      if self.check_collision():
        self.write_message('There is a collision, reject solution and take a smaller step')
        dt_new = self.dt * 0.5
        accept = False

      # make sure that time step size does not shrink too much
      if dt_new < self.dt_min:
        self.write_message('Time step size becomes smaller than minimum allowed, so move on with minimum step size...')
        dt_new, accept = self.dt_min, True
    else:
      # if fixed time step size, then move on
      dt_new, accept = self.dt, True

    return accept, dt_new, max_error # check_error

  ##############################################################################################

  def check_collision(self):
    '''
    Check whether nucleus-nucleus, nucleus-body, nucleus-MT collisions occur
    '''
    bodies_mm_attached = self.bodies_mm_attached
    bodies = self.bodies
    fibers = self.fibers
    offset_fibers = self.offset_fibers

    icollision = False

    x0 = np.zeros(3*offset_fibers[-1])
    for k,fib in enumerate(fibers):
      x0[3*offset_fibers[k] + 0 : 3*offset_fibers[k] + 3*fib.num_points + 0 : 3] = fib.x[:,0]
      x0[3*offset_fibers[k] + 1 : 3*offset_fibers[k] + 3*fib.num_points + 1 : 3] = fib.x[:,1]
      x0[3*offset_fibers[k] + 2 : 3*offset_fibers[k] + 3*fib.num_points + 2 : 3] = fib.x[:,2]

    cortex_dim_a, cortex_dim_b, cortex_dim_c = [], [], []
    if self.cortex_radius is not None:
      cortex_dim_a, cortex_dim_b, cortex_dim_c = self.cortex_radius, self.cortex_radius, self.cortex_radius
    if self.periphery_radius is not None:
      cortex_dim_a, cortex_dim_b, cortex_dim_c = self.periphery_radius, self.periphery_radius, self.periphery_radius
    if self.periphery_a is not None:
      cortex_dim_a, cortex_dim_b, cortex_dim_c = self.periphery_a, self.periphery_b, self.periphery_c


    if cortex_dim_a:
      # cortex - nucleus
      #for i, b in enumerate(bodies_mm_attached):
      #  if np.linalg.norm(b.location) >= self.cortex_radius*0.98: icollision = True

      # cortex - body
      #for i, b in enumerate(bodies):
      #  if np.linalg.norm(b.location) >= self.cortex_radius*0.98: icollision = True

      # fiber - cortex collision
      xfib = x0[0::3]
      yfib = x0[1::3]
      zfib = x0[2::3]

      x = xfib/cortex_dim_a
      y = yfib/cortex_dim_b
      z = zfib/cortex_dim_c

      r_true = np.sqrt(xfib**2 + yfib**2 + zfib**2)
      r_fiber = np.sqrt(x**2 + y**2 + z**2)
      phi_fiber = np.arctan2(y,(x+1e-12))
      theta_fiber = np.arccos(z/(1e-12+r_fiber))

      x_cort = cortex_dim_a*np.sin(theta_fiber)*np.cos(phi_fiber)
      y_cort = cortex_dim_b*np.sin(theta_fiber)*np.sin(phi_fiber)
      z_cort = cortex_dim_c*np.cos(theta_fiber)

      d = np.sqrt((xfib-x_cort)**2 + (yfib-y_cort)**2 + (zfib-z_cort)**2)
      cortex_point_r = np.sqrt(x_cort**2 + y_cort**2 + z_cort**2)

      sel_out = r_true >= cortex_point_r

      if sel_out.any(): icollision = True


    for inuc, nucleus in enumerate(bodies_mm_attached):

      # 1. Nucleus-nucleus
      for jnuc in range(inuc):
        nucleus2 = bodies_mm_attached[jnuc]
        radius = nucleus.radius + nucleus2.radius
        distance = np.linalg.norm(nucleus2.location-nucleus.location)
        if distance - radius <= 0.001:
          icollision = True
          self.write_message('There is nucleus-nucleus collision.')



      # 2. Nucleus-bodies
      for i, b in enumerate(bodies):
        radius = nucleus.radius + b.radius
        distance = np.linalg.norm(b.location-nucleus.location)
        if distance - radius <= 0.001:
          icollision = True
          self.write_message('There is nucleus-body collision.')

      # 3. Nucleus-fibers
      if True:
        x = x0[0::3] - nucleus.location[0]
        y = x0[1::3] - nucleus.location[1]
        z = x0[2::3] - nucleus.location[2]
        distance = np.sqrt(x**2 + y**2 + z**2)
        sel_in = (distance-nucleus.radius) <= 0.001
        if sel_in.any():
          #icollision = True
          self.write_message('There is nucleus-fiber collision but going on.')


    return icollision # check_error

  ##############################################################################################

  def write_to_file(self,current_time,step,steps_rejected):
    '''
    Writing data to the output file
    '''
    body_names = self.body_names
    body_types = self.body_types
    fibers_names = self.fibers_names
    fibers_types = self.fibers_types
    f_bodies_ID = self.f_bodies_ID
    f_fibers_ID = self.f_fibers_ID
    f_molecular_motors_ID = self.f_molecular_motors_ID
    f_body_vels_ID = self.f_body_vels_ID
    f_body_forces_ID = self.f_body_forces_ID
    f_body_density_ID = self.f_body_density_ID
    f_traction_ID = self.f_traction_ID
    f_fibers_forces_ID = self.f_fibers_forces_ID



    step_accepted = step-steps_rejected
    if self.save_file is not None:
      save_folder = self.save_file + 'step' + str(step_accepted)
      os.mkdir(save_folder)
      save_folder = save_folder +'/run'

      with open(save_folder + '.random_state', 'wb') as f:
        cpickle.dump(np.random.get_state(), f)
    else:
      name = self.output_name + '.randomState'
      with open(name, 'wb') as f:
        cpickle.dump(np.random.get_state(), f)


    # save time step size, current time, number of accepted time steps and rejected time steps
    time_system = np.array([self.dt, current_time, step-steps_rejected, steps_rejected, self.n_save])
    if self.cortex_radius is not None:
      time_system = np.array([self.dt, current_time, step-steps_rejected, steps_rejected, self.cortex_radius, self.n_save])
    if self.periphery_radius is not None:
      time_system = np.array([self.dt, current_time, step-steps_rejected, steps_rejected, self.periphery_radius, self.n_save])
    if self.periphery_a is not None:
      time_system = np.array([self.dt, current_time, step-steps_rejected, steps_rejected, self.periphery_a, self.periphery_b, self.periphery_c, self.n_save])

    np.savetxt(self.f_time_system, time_system[None,:])

    bodies_offset = 0
    for i, ID in enumerate(body_names):

      if self.save_file is None:
        # save data for restart
        name = self.output_name + '_' + ID + '_resume.clones'
        f_resume = open(name, 'wb')
        f_resume.write(('%s \n' % body_types[i]).encode('utf-8'))

      else:
        # save data for restart
        name = save_folder + '_' + ID + '_resume.clones'
        f_resume = open(name, 'wb')
        f_resume.write(('%s \n' % body_types[i]).encode('utf-8'))

      # first save number of bodies
      if self.output_txt_files:
        np.savetxt(f_bodies_ID[i], np.ones((1,7), dtype=int)*body_types[i])
        np.savetxt(f_body_vels_ID[i], np.ones((1,6), dtype=int)*body_types[i])
        np.savetxt(f_body_density_ID[i], np.ones((1,3), dtype=int)*body_types[i])
        np.savetxt(f_traction_ID[i], np.ones((1,3), dtype=int)*body_types[i])
        np.savetxt(f_body_forces_ID[i], np.ones((1,6), dtype=int)*body_types[i])

      for j in range(body_types[i]):
        orientation = self.bodies[bodies_offset + j].orientation.entries
        if self.output_txt_files:
          out = np.ones((1,7))
          if self.body_shape is 'sphere':
            out[0][0] = self.body_r
          elif self.body_shape is 'ellipsoid':
            out[0][0] = self.body_a
            out[0][1] = self.body_b
            out[0][2] = self.body_c

          np.savetxt(f_bodies_ID[i], out)
          out_body = np.array([self.bodies[bodies_offset + j].location[0],
                               self.bodies[bodies_offset + j].location[1],
                               self.bodies[bodies_offset + j].location[2],
                               orientation[0],
                               orientation[1],
                               orientation[2],
                               orientation[3]])
          np.savetxt(f_bodies_ID[i], out_body[None,:])
        else:
          f_bodies_ID[i].write(('%s %s %s %s %s %s %s \n' % (self.bodies[bodies_offset + j].location[0],
                                                             self.bodies[bodies_offset + j].location[1],
                                                             self.bodies[bodies_offset + j].location[2],
                                                             orientation[0],
                                                             orientation[1],
                                                             orientation[2],
                                                             orientation[3])).encode('utf-8'))
        out_body =  np.array([self.bodies[bodies_offset + j].velocity[0],
                              self.bodies[bodies_offset + j].velocity[1],
                              self.bodies[bodies_offset + j].velocity[2],
                              self.bodies[bodies_offset + j].angular_velocity[0],
                              self.bodies[bodies_offset + j].angular_velocity[1],
                              self.bodies[bodies_offset + j].angular_velocity[2]])
        np.savetxt(f_body_vels_ID[i], out_body[None,:])
        out_body = np.array([self.bodies[bodies_offset + j].on_force[0],
                             self.bodies[bodies_offset + j].on_force[1],
                             self.bodies[bodies_offset + j].on_force[2],
                             self.bodies[bodies_offset + j].on_force[3],
                             self.bodies[bodies_offset + j].on_force[4],
                             self.bodies[bodies_offset + j].on_force[5]])
        np.savetxt(f_body_forces_ID[i], out_body[None,:])                             
        
        out = np.array([self.bodies[bodies_offset+j].Nblobs,
          self.bodies[bodies_offset+j].Nblobs,
          self.bodies[bodies_offset+j].Nblobs])

        np.savetxt(f_body_density_ID[i], out[None,:])
        np.savetxt(f_body_density_ID[i], self.bodies[bodies_offset+j].density)
        
        np.savetxt(f_traction_ID[i], out[None,:])
        np.savetxt(f_traction_ID[i],self.bodies[bodies_offset+j].traction)
      
        f_resume.write(('%s %s %s %s %s %s %s \n' % (self.bodies[bodies_offset + j].location[0],
                                                             self.bodies[bodies_offset + j].location[1],
                                                             self.bodies[bodies_offset + j].location[2],
                                                             orientation[0],
                                                             orientation[1],
                                                             orientation[2],
                                                             orientation[3])).encode('utf-8'))
      f_resume.close()
      bodies_offset += body_types[i]

    # Save fibers
    fiber_offset = 0
    for i, ID in enumerate(fibers_names):

      # save data for restart
      if self.save_file is None:
        name = self.output_name + '_' + ID + '_resume.fibers'
        f_resume = open(name, 'wb')
        f_resume.write(('%s \n' % fibers_types[i]).encode('utf-8'))
      else:
        name = save_folder + '_' + ID + '_resume.fibers'
        f_resume = open(name, 'wb')
        f_resume.write(('%s \n' % fibers_types[i]).encode('utf-8'))

      if self.output_txt_files:
        np.savetxt(f_fibers_ID[i], np.ones((1,4), dtype=int)*fibers_types[i])
      else:
        f_fibers_ID[i].write((str(fibers_types[i]) + '\n').encode('utf-8'))

      np.savetxt(self.f_fiber_tips, np.ones((1,6), dtype=int)*fibers_types[i])

      for j in range(int(fibers_types[i])):
        
        fiber_info = np.array([self.fibers[fiber_offset + j].num_points,
                              self.fibers[fiber_offset + j].E,
                              self.fibers[fiber_offset + j].length,
                              0])
        np.savetxt(f_fibers_ID[i], fiber_info[None,:])
        

        np.savetxt(f_fibers_ID[i], np.concatenate((self.fibers[fiber_offset + j].x, self.fibers[fiber_offset + j].tension[:,None]), axis=1))

        fiber_info = np.array([self.fibers[fiber_offset + j].num_points,
                              self.fibers[fiber_offset + j].E,
                              self.fibers[fiber_offset + j].length,
                              0])
        np.savetxt(f_fibers_ID[i], fiber_info[None,:])
        
        # save data for restart
        f_resume.write(('%s %s %s %s \n' % (self.fibers[fiber_offset + j].num_points, self.fibers[fiber_offset + j].E, self.fibers[fiber_offset + j].length, self.fibers[fiber_offset + j].growing)).encode('utf-8'))
        np.savetxt(f_resume, self.fibers[fiber_offset + j].x)

      f_resume.close()
      fiber_offset += int(fibers_types[i])



      





    return # write_to_file
  ##############################################################################################

  def write_message(self, message):
    '''
    Writing message to the log file
    '''
    if message is 'stars': message = '******************************************'
    f_log = open(self.output_name + '.logFile', 'a')
    f_log.write(message + '\n')
    f_log.close()
    print(message)

    return # write_message
  ##############################################################################################

class gmres_counter(object):
  '''
  Callback generator to count iterations
  '''
  def __init__(self, print_residual = False):
    self.print_residual = print_residual
    self.niter = 0

  def __call__(self, rk=None):
    self.niter += 1
    if self.print_residual is True:
      if self.niter == 1:
        print('gmres = 0 1')
      print('gmres = ', self.niter, rk)
