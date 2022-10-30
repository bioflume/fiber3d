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
import scipy.interpolate as scinter

# OUR CLASSES
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
from mpi4py import MPI

import _pickle as cpickle

try:
  from numba import njit, prange
  from numba.typed import List
except ImportError:
  print('Numba not found')

class postprocess_velocity(object):
  '''
  Time stepping algorithms: fixed or adaptive 1st order schemes
  Two routines:
  1. all structures (fiber, rigid body, periphery) with hydro (WET)
  2. all structures without hydro (DRY)
  '''


  ##############################################################################################
  def __init__(self, prams, options, time_steps, nsteps_all, time_all):
    self.options = options
    self.nsteps_all = nsteps_all
    self.time_all = time_all
    self.prams = prams
    self.nskip_steps = prams.nskip_steps
    self.time_steps = time_steps
    self.output_name = options.output_name # output_name: saving files to
    self.eta = prams.eta # suspending fluid viscosity
    self.iupsample = options.iupsample # upsampling for integration
    self.integration = options.integration # integration scheme 'trapz' or 'simpsons'
    self.repulsion = options.repulsion # flag for repulsion
    self.inextensibility = options.inextensibility
    self.dt = options.dt # if not adaptive time stepping
    self.tol_gmres = options.tol_gmres # gmres tolerance
    self.iCytoPulling = options.iCytoPulling # Cytoplasmic pulling flag
    self.iNoSliding = options.iNoSliding

    self.n_save = options.n_save # save data after this many time steps

    self.useFMM = options.useFMM # flag for using FMM
    self.fmm_order = options.fmm_order
    self.fmm_max_pts = options.fmm_max_pts
    self.oseen_kernel_source_target_stkfmm_partial = None
    self.stresslet_kernel_source_target_stkfmm_partial = None
    self.uprate = options.uprate

    if self.useFMM:
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

    # Get time stepping coefficients
    self.beta, self.Xcoeff, self.rhsCoeff = 1.0, [1.0], [1.0]
    # OPEN A FILE TO KEEP LOGS
    f_log = open(self.output_name + '_postprocessing.logFile', 'w+')

    self.motor_strength_gradient_function = prams.motor_strength_gradient_function
    self.motor_strength_gradient_along_axis = prams.motor_strength_gradient_along_axis
    self.motor_strength_gradient_regions = prams.motor_strength_gradient_regions
    self.motor_strength_gradient = prams.motor_strength_gradient
    self.xrange = prams.xrange
    self.yrange = prams.yrange
    self.zrange = prams.zrange

    # If Nblobs is given, then discretize rigid bodies with given Nblobs
    self.A_inv_bodies = []
    
    # INITIALIZE PERIPHERY IF EXISTS
    self.shell, self.M_inv_periphery, self.normals_shell, self.trg_shell_surf = None, [], [], []
    self.periphery_radius = prams.periphery_radius
    self.periphery_a = prams.periphery_a
    self.periphery_b = prams.periphery_b
    self.periphery_c = prams.periphery_c
    if prams.periphery is not None:
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

    # START TAKING TIME STEPS
    compute_steps = np.arange(self.nskip_steps, self.nsteps_all[-1]+self.nskip_steps, self.nskip_steps)
    for istep in compute_steps:
      # index in time_step container
      idx = np.where(self.nsteps_all == istep)
      idx = idx[0][0]
      self.write_message('stars')
      self.write_message('Step = ' + str(istep) + ', time = ' + str(self.time_all[idx]))

      self.fibers = self.time_steps[idx].fibers
      self.bodies = []
      
      self.fib_mats = self.time_steps[idx].fib_mats
      self.fib_mat_resolutions = self.time_steps[idx].fib_mat_resolutions

      for k, fib in enumerate(self.fibers): fib.dt = self.dt

      # STEP 1: Take a time step
      a_step = time.time()
      self.step_now = istep

      getattr(self, 'time_step_hydro')(self.dt, *args, **kwargs)
      self.write_message('Time step took ' + str(time.time() - a_step) + ' seconds.')
      self.write_to_file()


    return # take_time_steps

  ##############################################################################################
  def time_step_hydro(self, dt, *args, **kwargs):
    '''
    Time step including hydrodynamics
    System can have rigid bodies, fibers, (stationary) molecular motors, confinement
    '''


  
    # ------------------
    # 1. INITIALIZATION
    # ------------------


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
    
    trg_fib = np.zeros((offset_fibers[-1],3))
    xsDs_block, ysDs_block, zsDs_block = [], [], []
    for k,fib in enumerate(self.fibers):

      istart = offset + offset_bodies[-1]*3 + 6*len(self.bodies) + 4*offset_fibers[k]
      x0[3*offset_fibers[k] + 0 : 3*offset_fibers[k] + 3*fib.num_points + 0 : 3] = fib.x[:,0]
      x0[3*offset_fibers[k] + 1 : 3*offset_fibers[k] + 3*fib.num_points + 1 : 3] = fib.x[:,1]
      x0[3*offset_fibers[k] + 2 : 3*offset_fibers[k] + 3*fib.num_points + 2 : 3] = fib.x[:,2]
      
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
        strength = 0
        if self.motor_strength_gradient_along_axis is not None:
          if self.motor_strength_gradient_along_axis is 'x':
            check_loc = fib.x[0,0]
          elif self.motor_strength_gradient_along_axis is 'y':
            check_loc = fib.x[0,1]
          elif self.motor_strength_gradient_along_axis is 'z':
            check_loc = fib.x[0,2]
          if self.motor_strength_gradient_function is not None:
            strength = self.motor_strength_gradient_function(check_loc)
          else:
            for idx, region in enumerate(self.motor_strength_gradient_regions):
              if idx == 0:
                if check_loc >= region[0] and check_loc <= region[1]:
                  strength = self.motor_strength_gradient[idx]
              else:
                if check_loc > region[0] and check_loc <= region[1]:
                  strength = self.motor_strength_gradient[idx]
        else:
          strength = self.motor_strength_gradient[0]

        if strength == 0:
          print('Strength could not be found! Stop!')
          input()
        motor_force_fibers[offset_fibers[k]:offset_fibers[k+1]] = -strength * fib.xs


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
    GfibersNoFMM = tstep_utils.get_self_fibers_Stokeslet(self.fibers, self.eta,
                                              fib_mats = self.fib_mats,
                                              fib_mat_resolutions = self.fib_mat_resolutions,
                                              iupsample = True)

    Gfibers = []
    if self.fibers:
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


    # 2.2. Compute velocity due to fiber forces on fibers, bodies and shell
    # Concatenate target points
    trg_all = np.concatenate((trg_fib,trg_bdy_surf), axis = 0)
    if self.shell is not None: trg_all = np.concatenate((trg_all,self.trg_shell_surf.flatten()), axis = 0)

    if force_fibers.any():
      vfib2all = tstep_utils.flow_fibers(force_fibers, trg_fib, trg_all,
        self.fibers, offset_fibers, self.eta, integration = self.integration, fib_mats = self.fib_mats,
        fib_mat_resolutions = self.fib_mat_resolutions, iupsample = self.iupsample,
        oseen_fmm = self.oseen_kernel_source_target_stkfmm_partial, fmm_max_pts = 500)
      vfib2fib = vfib2all[:3*offset_fibers[-1]]

      vfib2bdy = np.array([])
      if self.bodies:
        vfib2bdy = np.copy(vfib2all[3*offset_fibers[-1]:3*offset_fibers[-1]+3*offset_bodies[-1]])

      vfib2shell = np.array([])
      if self.shell is not None:
        vfib2shell = vfib2all[3*offset_fibers[-1]+3*offset_bodies[-1]:]


      # Subtract self-interaction which is approximated by SBT
      vfib2fib += tstep_utils.self_flow_fibers(force_fibers, offset_fibers, self.fibers, Gfibers, self.eta,
        integration = self.integration, fib_mats = self.fib_mats, fib_mat_resolutions = self.fib_mat_resolutions, iupsample = self.iupsample)
    else:
      vfib2fib, vfib2bdy = np.zeros(offset_fibers[-1]*3), np.zeros(offset_bodies[-1]*3)
      vfib2shell = np.array([])
      if self.shell is not None: vfib2shell = np.zeros(self.shell.Nblobs*3)


    # 2.3. Compute velocity due to body forces on fibers, bodies and shell
    if force_bodies.any(): # non-zero body force
      if self.useFMM and (trg_bdy_cent.size//3) * (trg_all.size//3) >= 500:
        vbdy2all = self.oseen_kernel_source_target_stkfmm_partial(trg_bdy_cent, trg_all, force_bodies[:,:3], eta = self.bodies[0].viscosity_scale*self.eta)
      else:
        vbdy2all = kernels.oseen_kernel_source_target_numba(trg_bdy_cent, trg_all, force_bodies[:,:3], eta = self.bodies[0].viscosity_scale*self.eta)
      vbdy2all += kernels.rotlet_kernel_source_target_numba(trg_bdy_cent, trg_all, force_bodies[:,3:], eta = self.bodies[0].viscosity_scale*self.eta)

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

      # If there are fibers attached to a body, find the first points and the corresponding tension BC index

      # (-) end of fiber
      # CLAMPED
      BC_start_0, BC_start_1 = 'velocity', 'angular_velocity'

      # (+) end of fiber
      BC_end_0, BC_end_1 = 'force','torque'
      

      fib.set_BC(BC_start_0 = BC_start_0,
                BC_start_1 = BC_start_1,
                BC_end_0 = BC_end_0,
                BC_end_1 = BC_end_1)

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
    BC_start_vec_0 = np.zeros(3)
    BC_end_vec_0 = np.zeros(3)
    BC_start_0, BC_start_1 = 'velocity','angular_velocity'
    # 3.1. Build RHS (x,y,z) of point 1, then (x,y,z) of point 2. This is the order
    As_fibers, A_fibers_blocks, RHS_all = tstep_utils.get_fibers_and_bodies_matrices(self.fibers, self.bodies, self.shell,
      system_size, offset_fibers, offset_bodies, force_fibers, motor_force_fibers, force_bodies,
      vfib2fib+vbdy2fib, vbdy2bdy+vfib2bdy, vfib2shell+vbdy2shell, self.fib_mats, self.fib_mat_resolutions, inextensibility = self.inextensibility,
      BC_start_0 = BC_start_0, BC_start_vec_0 = BC_start_vec_0, BC_start_1 = BC_start_1 ,BC_end_0 = 'force', BC_end_vec_0 = BC_end_vec_0, ihydro = True)

    # 3.2. Fibers' force operator (sparse)
    fibers_force_operator = []
    if self.fibers:
      fibers_force_operator = tstep_utils.build_fibers_force_operator(self.fibers, self.fib_mats, self.fib_mat_resolutions)

    # 3.3. Build link matrix (fibers' boundary conditions)
    As_BC = []
    if self.fibers and self.bodies:
      As_BC = tstep_utils.build_link_matrix(4*offset_fibers[-1]+6*len(self.bodies), self.bodies,self.fibers, offset_fibers, 6*len(self.bodies),
        self.fib_mats, self.fib_mat_resolutions)

    # 3.4. Preconditioner for fibers
    LU_fibers, P_fibers = [], []
    if self.fibers:
      LU_fibers, P_fibers = tstep_utils.build_block_diagonal_lu_preconditioner_fiber(self.fibers, A_fibers_blocks)


    # 3.5. Preconditioner for bodies, if not precomputed
    LU_bodies, P_bodies = [], []

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
    def A_body_fiber_hyd(x_all, bodies, shell,
                        trg_bdy_surf, trg_bdy_cent, trg_shell_surf, trg_all,
                        normals_blobs, normals_shell,
                        As_fibers, As_BC, Gfibers, fibers, trg_fib,
                        fibers_force_operator, xfibers, dt, K_bodies = None):

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
                                      dt = self.dt,
                                      As_fibers = As_fibers,
                                      As_BC = As_BC,
                                      fibers = self.fibers,
                                      trg_fib = trg_fib,
                                      Gfibers = Gfibers,
                                      fibers_force_operator = fibers_force_operator,
                                      xfibers = xfibers)
    linear_operator_partial = scspla.LinearOperator((system_size, system_size), matvec = linear_operator_partial, dtype='float64')

    # -----------------
    # 4. GMRES to SOLVE
    # -----------------


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
      self.write_message('Body velocity: ' + str(b.velocity_new))
      self.write_message('Angular velocity: ' + str(b.angular_velocity_new))


    # -------------------------------------------------
    # COMPUTE VELOCITY ON A GRID
    # -------------------------------------------------

    if True:
      xin = self.xrange
      yin = self.yrange
      zin = self.zrange
      xin, yin, zin = xin.reshape((xin.size,1)), yin.reshape((yin.size,1)), zin.reshape((zin.size,1))
      grid_points = np.concatenate((xin, yin, zin), axis = 1)

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
          vbdy2all = kernels.stresslet_kernel_source_target_numba(trg_bdy_surf, grid_points, normals_blobs, body_densities, eta = self.eta)
          
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

          grid_off = 0
          grid_points_mat = grid_points.reshape((grid_points.size//3,3))
          for xyz in grid_points_mat:
            for k,fib in enumerate(self.fibers):
              dist = np.sqrt((xyz[0]-fib.x[:,0])**2 + (xyz[1]-fib.x[:,1])**2 + (xyz[2]-fib.x[:,2])**2)
              if False: #(dist<=2e-1).any() :
                indx = np.where(self.fib_mat_resolutions == fib.num_points)
                indx = indx[0][0]
                P_kerUp, P_kerDn, out3, out4 = self.fib_mats[indx].get_matrices(fib.length, fib.num_points_up, 'P_kernel')
                weights, weights_up, out3, out4 = self.fib_mats[indx].get_matrices(fib.length, fib.num_points_up, 'weights_all')
                fup = np.dot(P_kerUp, fw[offset_fibers[k]:offset_fibers[k+1]]) * weights_up[:,None]
                G = kernels.oseen_tensor_source_target(np.dot(P_kerUp,fib.x), fib.x, eta = self.eta)
                vself = G.dot(fup.flatten())
                vself = vself.reshape((vself.size//3,3))
                #vinter[0] = scinter.griddata(fib.x,vself[:,0],xyz.reshape((1,3)),method='linear')
                #vinter[1] = scinter.griddata(fib.x,vself[:,1],xyz.reshape((1,3)),method='linear')
                #vinter[2] = scinter.griddata(fib.x,vself[:,2],xyz.reshape((1,3)),method='linear')
                minIdx = np.argmin(dist)
                vinter = vself[minIdx]
                vfib2all[grid_off:grid_off+3] = vinter.flatten()
            grid_off += 3

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
        vbdy2all = vbdy2all.reshape((vbdy2all.size//3,3))
        vfib2all = vfib2all.reshape((vfib2all.size//3,3))

        return vgrid, vbdy2all, vfib2all


    if True:
      vgrid,vbdyAll2grid,vfibAll2grid = compute_velocity(sol, self.bodies, self.shell,
                      trg_bdy_surf, trg_bdy_cent, self.trg_shell_surf,
                      normals_blobs, self.normals_shell,
                      As_fibers, As_BC, Gfibers, self.fibers, trg_fib,
                      fibers_force_operator, xfibers, grid_points, K_bodies = K_bodies)
      if force_fibers.any():
        vfib2grid = tstep_utils.flow_fibers(force_fibers, trg_fib, grid_points, self.fibers, offset_fibers, self.eta, integration = self.integration ,
            fib_mats = self.fib_mats, fib_mat_resolutions = self.fib_mat_resolutions, iupsample = self.iupsample, oseen_fmm = None, fmm_max_pts = 500)
        vgrid += vfib2grid.reshape((vfib2grid.size//3,3))
        vfibAll2grid += vfib2grid.reshape((vfib2grid.size//3,3))
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
        loc = self.bodies[0].location_new
        ids = (grid_points[:,0]-loc[0])**2 + (loc[1]-grid_points[:,1])**2 + (loc[2]-grid_points[:,2])**2 <= self.bodies[0].radius**2
        vgrid[ids] = 0
        vgrid[ids] = self.bodies[0].velocity_new

        rgrid = grid_points[ids] - loc
        omega_b = self.bodies[0].angular_velocity_new
        vgrid[ids,0] += omega_b[1]*rgrid[:,2] - omega_b[2]*rgrid[:,1]
        vgrid[ids,1] += omega_b[2]*rgrid[:,0] - omega_b[0]*rgrid[:,2]
        vgrid[ids,2] += omega_b[0]*rgrid[:,1] - omega_b[1]*rgrid[:,0]

      # Save velocity and the grid points
      f_grid_points = open(self.output_name + '_grid_points.txt', 'w')
      np.savetxt(f_grid_points,grid_points)
      f_grid_points.close()

      name = self.output_name + '_velocity_at_step' + str(self.step_now) + '.txt'
      f_grid_velocity = open(name, 'w')
      np.savetxt(f_grid_velocity, vgrid)
      f_grid_velocity.close()

    
    return # time_step_hydro

  ##############################################################################################
  def write_to_file(self):
    '''
    Writing data to the output file
    '''

    # Save fibers
    name = self.output_name + '_fibers_step' + str(self.step_now) + '.txt'
    f_fibers = open(name,'w')
    np.savetxt(f_fibers, np.ones((1,4), dtype=int)*len(self.fibers))
    for k, fib in enumerate(self.fibers):
      fiber_info = np.array([fib.num_points,
                             fib.E,
                             fib.length,
                             fib.iReachSurface])
      np.savetxt(f_fibers, fiber_info[None,:])

      np.savetxt(f_fibers, np.concatenate((fib.x, fib.tension_new[:,None]), axis=1))
    f_fibers.close()



    return # write_to_file
  ##############################################################################################

  def write_message(self, message):
    '''
    Writing message to the log file
    '''
    if message is 'stars': message = '******************************************'
    f_log = open(self.output_name + '_postprocessing.logFile', 'a')
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
