from __future__ import print_function
import numpy as np
import sys
import argparse
import subprocess
import time
from functools import partial
import scipy.sparse as scsp
import scipy.linalg as scla
import scipy.sparse.linalg as scspla
try:
  import cPickle as cpickle
except:
  try:
    import cpickle
  except:
    import _pickle as cpickle
try:
  from mpi4py import MPI
except ImportError:
  print('It didn\'t find mpi4py')
try:
  from numba import njit, prange
  numba_found = True
except ImportError:
  print('Numba not found')

  
# Find project functions
found_functions = False
path_to_append = ''
while found_functions is False:
  try:
    from read_input import read_input
    from read_input import read_fibers_file
    from read_input import read_vertex_file
    from read_input import read_clones_file
    from read_input import read_links_file
    from utils import timer
    from utils import nonlinear 
    from utils import cheb
    from fiber import fiber
    #from integrators import integrators
    from kernels import kernels
    from body import body
    # from force_generator import force_generator as fg
    from molecular_motor import molecular_motor
    #from lib import periodic_fmm as fmm
    found_functions = True
  except ImportError:
    path_to_append += '../'
    print('searching functions in path ', path_to_append)
    sys.path.append(path_to_append)
    if len(path_to_append) > 21:
      print('\nProjected functions not found. Edit path in multi_bodies.py')
      sys.exit()



def flow_fibers(f, src, trg, fibers, offset_fibers, eta, iupsample = False):
  '''
  GK: inputs force and fibers, then upsamples here, if iupsample = True
  Flow created by fibers at target points.
  '''
  fw, src_up = [], []
  if iupsample:
    src = src.reshape(src.size // 3, 3)

  for k, fib in enumerate(fibers):
    if iupsample:
      if fib.tstep_order == 1:
        weights = fib.weights_up
      else:
        weights = fib.weights_up_ext
      fw.append(np.dot(fib.P_kerUp,f[offset_fibers[k]:offset_fibers[k+1]])*weights[:,None])
      src_up.append(np.dot(fib.P_kerUp,src[offset_fibers[k]:offset_fibers[k+1]])) 
    else:
      if fib.tstep_order == 1:
        weights = fib.weights
      else:
        weights = fib.weights_ext
      fw.append(f[offset_fibers[k]:offset_fibers[k+1]]*weights[:,None])
  
  fw = np.array(fw).flatten()

  if iupsample:
    src = np.array(src_up).flatten()

  return kernels.oseen_kernel_source_target_numba(src, trg, fw, eta = eta)


def self_flow_fibers_old(f, offset_fibers, fibers, eta, iupsample = False):
  '''
  Flow created by fibers at target points.
  Input: 
  force on fibers (num_points), then upsamples here
  Output:
  v = velocity, dimension (3*num_points)
  '''
  v = np.zeros(offset_fibers[-1] * 3)
  if iupsample:
    for k, fib in enumerate(fibers):
      # upsample force and multiply with weights
      if fib.tstep_order == 1:
        weights, x = fib.weights_up, fib.x
      else:
        weights, x = fib.weights_up_ext, fib.x_ext
      fw = np.dot(fib.P_kerUp, f[offset_fibers[k]:offset_fibers[k+1]]) * weights[:,None]
      v[3*offset_fibers[k] : 3*offset_fibers[k+1]] -= kernels.oseen_kernel_source_target_numba(np.dot(fib.P_kerUp,x),
                                                                                                x, fw, eta = eta)
  else:
    for k, fib in enumerate(fibers):
      
      if fib.tstep_order == 1:
        weights, x = fib.weights, fib.x
      else:
        weights, x = fib.weights_ext, fib.x_ext

      fw = f[offset_fibers[k] : offset_fibers[k+1]] * weights[:,None]
      v[3 * offset_fibers[k] : 3 * offset_fibers[k+1]] -= kernels.oseen_kernel_source_target_numba(x, 
                                                                                                 x, 
                                                                                                 fw, 
                                                                                                 eta = eta)
  return v 
  

def get_self_fibers_Stokeslet(offset_fibers, fibers, eta, iupsample = False):
  '''
  Get the block-diagonal matrix with each block being
  the Oseen kernel for one fiber.
  '''
  G_all = []

  if iupsample:
    for k, fib in enumerate(fibers):
      # Linear system
      G = kernels.oseen_tensor_source_target(np.dot(fib.P_kerUp,fib.x), fib.x, eta = eta)
      G_all.append(G)
  else:
    for k, fib in enumerate(fibers):
      # Linear system
      G = kernels.oseen_tensor(fib.x, eta = eta)
      G_all.append(G)
    
  G_block = scsp.block_diag(G_all) 
  G = scsp.csr_matrix(G_block)
  return G
  

def self_flow_fibers(f, offset_fibers, fibers, G, eta, iupsample = False):
  '''
  Flow created by fibers at target points.
  Input: 
  force on fibers (num_points), then upsamples here
  Output:
  v = velocity, dimension (3*num_points)
  '''
  fw = []
  if iupsample:
    for k, fib in enumerate(fibers):
      # upsample force and multiply with weights
      fw.append(np.dot(fib.P_kerUp, f[offset_fibers[k]:offset_fibers[k+1]]) * fib.weights_up[:,None])
  else:
    for k, fib in enumerate(fibers):
      fw.append(f[offset_fibers[k] : offset_fibers[k+1]] * fib.weights[:,None])

  fw = np.array(fw).flatten()
  v = -G.dot(fw)
  return v 


def flow_bodies(mu, F, tau, src, q, normals, trg, weights, eta):
  '''
  Flow created by bodies at target points.
  '''
  v = kernels.oseen_kernel_source_target_numba(q, trg, F, eta = eta)
  v += kernels.rotlet_kernel_source_target_numba(q, trg, tau, eta = eta)
  v += kernels.stresslet_kernel_source_target_numba(src, trg, normal, mu, eta = eta)
  return v


def build_fibers_force_operator(fibers):
  '''
  f = -E * X_ssss + (T*X_s)_s
  '''
  force_operator = []
  for fib in fibers:
    force_operator.append(fib.force_operator())
  force_operator_block = scsp.block_diag(force_operator) 
  
  return scsp.csr_matrix(force_operator_block)


def gather_fibers_weights(fibers, iupsample = True):
  '''
  GK: useless
  iupsample = flag showing whether we upsample for computing 
  Output:
  weights = array of dimension num_points * nfibers 
  '''

  weights = []
  for k, fib in enumerate(fibers):
    if iupsample:
      weights.append(fib.weights_up)
    else:
      weights.append(fib.weights)
    
  return np.copy(np.array(weights).flatten())


def gather_fibers_weights(fibers):
  '''

  '''
  weights = []
  for k, fib in enumerate(fibers):
    weights.append(fib.weights)
  return np.copy(np.array(weights).flatten())


def gather_target_points(bodies, fibers):
  '''
  Output: all target points in an array (x1,y1,z1; x2,y2,z2)
  '''
  x = []
  for b in bodies:
    x.append(b.get_r_vectors())
  for fib in fibers:
    x.append(fib.x)
  return np.array(x).flatten()


def gather_bodies_configuration_info(bodies):
  '''
  '''
  N = len(bodies)
  q = np.zeros((N, 3))
  xb = []
  normals = []
  for k, b in enumerate(bodies):
    q[k] = b.location
    xb.append(b.get_blobs_r_vectors)
    normals.append(b.get_normals)
  xb = np.array(xb).flatten()
  normals = np.array(normals).flatten()
  return q, xb, normals
  
def get_num_particles_and_offsets(bodies, fibers):
  '''
  '''
  num_bodies = len(bodies)
  num_fibers = len(fibers)
  offset_bodies = len(bodies) * 6
  offset_fibers = np.zeros(num_fibers + 1, dtype=int)

  for k, fib in enumerate(fibers):
    offset_fibers[k+1] = offset_fibers[k] + fib.num_points
  return num_bodies, num_fibers, offset_bodies, offset_fibers


def build_fibers_force_torque_link_operators(fibers):
  '''
  '''
  N = len(fibers)
  F_0 = []
  F_1 = []
  tau_0 = []
  tau_1 = []
  for k, fib in enumerate(fibers):
    Fi_0, Fi_1, taui_0, taui_1 = fib.force_torque_link_operators()
    F_0.append(Fi_0)
    F_1.append(Fi_1)
    tau_0.append(taui_0)
    tau_1.append(taui_0)
  F_0_block = scsp.block_diag(F_0)
  return scsp.csr_matrix(F_0_block)


def get_fibers_matrices(fibers, offset_fibers, external_flow, force_fibers):
  '''
  # GK: Upsampling when computing derivatives is implemented here
  '''
  external_flow = external_flow.reshape((external_flow.size // 3, 3))
  A_all = []
  RHS_all = np.zeros(offset_fibers[-1] * 4)
  
  for k, fib in enumerate(fibers):
    # Linear system
    A = fib.form_linear_operator()
    RHS = fib.compute_RHS(force_external = force_fibers[offset_fibers[k]:offset_fibers[k+1]], 
      flow = external_flow[offset_fibers[k] : offset_fibers[k+1]])

    # Apply BC
    A, RHS = fib.apply_BC_rectangular(A, RHS)

    # Save data
    A_all.append(A)
    RHS_all[4 * offset_fibers[k] : 4 * offset_fibers[k+1]] = RHS
    
  A_block = scsp.block_diag(A_all) 
  A = scsp.csr_matrix(A_block)
  return A, A_all, RHS_all
  



def build_link_matrix(system_size,bodies,fibers,offset_fibers,offset_bodies,fib_mats,fib_mat_resolutions):
  '''
  Building link matrix (fibers' boundary conditions)
  '''
  As_dok_BC = scsp.dok_matrix((system_size, system_size))
  for k, b in enumerate(bodies):
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
      offset_point = offset_fibers[offset_fiber] * 4 + offset_bodies
      fib = fibers[offset_fiber]

      # Find the index for fib_mats
      indx = np.where(fib_mat_resolutions == fib.num_points)
      indx = indx[0][0]

      # Get the class that has the matrices
      fib_mat = fib_mats[indx]

      if fib.filtering:
        P_up, P_down, out3, out4 = fib_mat.get_matrices(fib.length, fib.num_points_up, 'P_upsample')
        
        if fib.tstep_order == 1:
          out1, D_2, D_3, out4 = fib_mat.get_matrices(fib.length, fib.num_points_up, 'Ds')
          xs = fib.xs
        else:
          out1, D_2, D_3, out4 = fib_mat.get_matrices(fib.length, fib.num_points_up, 'Ds')
          xs = fib.xs_ext

        #D_2 = np.dot(P_down, np.dot(D_2_up, P_up))
        #D_3 = np.dot(P_down, np.dot(D_3_up, P_up))
          
      else:

        if fib.tstep_order == 1:
          xs = fib.xs
          out1, D_2, D_3, out4 = fib_mat.get_matrices(fib.length, fib.num_points_up, 'Ds')
        else:
          xs = fib.xs_ext
          out1, D_2, D_3, out4 = fib_mat.get_matrices(fib.length, fib.num_points_up, 'Ds')

      # Rectangular mathod, Driscoll and Hale
      # Matrix A_body_fiber, for position
      As_dok_BC[k*6 + 0, offset_point+fib.num_points*0 : offset_point+fib.num_points*1] = fib.E * D_3[0,:] 
      As_dok_BC[k*6 + 1, offset_point+fib.num_points*1 : offset_point+fib.num_points*2] = fib.E * D_3[0,:] 
      As_dok_BC[k*6 + 2, offset_point+fib.num_points*2 : offset_point+fib.num_points*3] = fib.E * D_3[0,:] 
      As_dok_BC[k*6 + 0, offset_point+fib.num_points*3] = -xs[0,0] 
      As_dok_BC[k*6 + 1, offset_point+fib.num_points*3] = -xs[0,1] 
      As_dok_BC[k*6 + 2, offset_point+fib.num_points*3] = -xs[0,2] 
      
      As_dok_BC[k*6 + 3, offset_point+fib.num_points*1 : offset_point+fib.num_points*2] = -fib.E * links_loc[i,2] * D_3[0,:] 
      As_dok_BC[k*6 + 3, offset_point+fib.num_points*2 : offset_point+fib.num_points*3] =  fib.E * links_loc[i,1] * D_3[0,:] 
      As_dok_BC[k*6 + 4, offset_point+fib.num_points*0 : offset_point+fib.num_points*1] =  fib.E * links_loc[i,2] * D_3[0,:] 
      As_dok_BC[k*6 + 4, offset_point+fib.num_points*2 : offset_point+fib.num_points*3] = -fib.E * links_loc[i,0] * D_3[0,:] 
      As_dok_BC[k*6 + 5, offset_point+fib.num_points*0 : offset_point+fib.num_points*1] = -fib.E * links_loc[i,1] * D_3[0,:] 
      As_dok_BC[k*6 + 5, offset_point+fib.num_points*1 : offset_point+fib.num_points*2] =  fib.E * links_loc[i,0] * D_3[0,:] 
      
      As_dok_BC[k*6 + 3, offset_point+fib.num_points*3] = (-links_loc[i,2]*xs[0,1]+links_loc[i,1]*xs[0,2]) 
      As_dok_BC[k*6 + 4, offset_point+fib.num_points*3] = ( links_loc[i,2]*xs[0,0]-links_loc[i,0]*xs[0,2]) 
      As_dok_BC[k*6 + 5, offset_point+fib.num_points*3] = (-links_loc[i,1]*xs[0,0]+links_loc[i,0]*xs[0,1]) 

      # Matrix A_fiber_body, for position 
      As_dok_BC[offset_point+fib.num_points*4-14, k*6 + 0] = -1.0 
      As_dok_BC[offset_point+fib.num_points*4-13, k*6 + 1] = -1.0 
      As_dok_BC[offset_point+fib.num_points*4-12, k*6 + 2] = -1.0 
      As_dok_BC[offset_point+fib.num_points*4-11, k*6 + 0] = -xs[0,0] 
      As_dok_BC[offset_point+fib.num_points*4-11, k*6 + 1] = -xs[0,1] 
      As_dok_BC[offset_point+fib.num_points*4-11, k*6 + 2] = -xs[0,2] 
      
      As_dok_BC[offset_point+fib.num_points*4-14, k*6 + 4] = -links_loc[i,2] 
      As_dok_BC[offset_point+fib.num_points*4-14, k*6 + 5] =  links_loc[i,1] 
      As_dok_BC[offset_point+fib.num_points*4-13, k*6 + 3] =  links_loc[i,2] 
      As_dok_BC[offset_point+fib.num_points*4-13, k*6 + 5] = -links_loc[i,0] 
      As_dok_BC[offset_point+fib.num_points*4-12, k*6 + 3] = -links_loc[i,1] 
      As_dok_BC[offset_point+fib.num_points*4-12, k*6 + 4] =  links_loc[i,0] 
      
      As_dok_BC[offset_point+fib.num_points*4-11, k*6 + 3] = ( xs[0,1]*links_loc[i,2] - xs[0,2]*links_loc[i,1]) 
      As_dok_BC[offset_point+fib.num_points*4-11, k*6 + 4] = (-xs[0,0]*links_loc[i,2] + xs[0,2]*links_loc[i,0]) 
      As_dok_BC[offset_point+fib.num_points*4-11, k*6 + 5] = ( xs[0,0]*links_loc[i,1] - xs[0,1]*links_loc[i,0]) 

      
      # Matrix A_body_fiber, for angle 
      As_dok_BC[k*6 + 3, offset_point+fib.num_points*1 : offset_point+fib.num_points*2] +=  fib.E * xs[0,2] * D_2[0,:] 
      As_dok_BC[k*6 + 3, offset_point+fib.num_points*2 : offset_point+fib.num_points*3] += -fib.E * xs[0,1] * D_2[0,:] 
      As_dok_BC[k*6 + 4, offset_point+fib.num_points*0 : offset_point+fib.num_points*1] += -fib.E * xs[0,2] * D_2[0,:] 
      As_dok_BC[k*6 + 4, offset_point+fib.num_points*2 : offset_point+fib.num_points*3] +=  fib.E * xs[0,0] * D_2[0,:] 
      As_dok_BC[k*6 + 5, offset_point+fib.num_points*0 : offset_point+fib.num_points*1] +=  fib.E * xs[0,1] * D_2[0,:] 
      As_dok_BC[k*6 + 5, offset_point+fib.num_points*1 : offset_point+fib.num_points*2] += -fib.E * xs[0,0] * D_2[0,:] 
      
      # Matrix A_fiber_body, for angle 
      As_dok_BC[offset_point+fib.num_points*4-10, k*6 + 4] = -links_loc[i,2] 
      As_dok_BC[offset_point+fib.num_points*4-10, k*6 + 5] =  links_loc[i,1] 
      As_dok_BC[offset_point+fib.num_points*4-9, k*6 + 3]  =  links_loc[i,2] 
      As_dok_BC[offset_point+fib.num_points*4-9, k*6 + 5]  = -links_loc[i,0] 
      As_dok_BC[offset_point+fib.num_points*4-8, k*6 + 3]  = -links_loc[i,1] 
      As_dok_BC[offset_point+fib.num_points*4-8, k*6 + 4]  =  links_loc[i,0] 

  return scsp.csr_matrix(As_dok_BC)



def get_fibers_and_bodies_matrices(fibers, bodies, system_size, 
  offset_fibers, offset_bodies, external_flow, force_bodies, force_fibers, 
  fib_mats, fib_mat_resolutions, BC_start_0=None, BC_start_1=None, BC_end_0=None):
  '''
  GK: this is similar to get_fibers_matrices but includes bodies as well
  So, is more general and will replace get_fibers_matrices
  '''

  if external_flow is not None:
    external_flow = external_flow.reshape((external_flow.size // 3, 3))

  A_all = []
  RHS_all = np.zeros(system_size)
  RHS_all[0:6*len(bodies):6] = force_bodies[:,0]
  RHS_all[1:6*len(bodies):6] = force_bodies[:,1]
  RHS_all[2:6*len(bodies):6] = force_bodies[:,2]
  for k, fib in enumerate(fibers):

    # Find the index for fib_mats
    indx = np.where(fib_mat_resolutions == fib.num_points)
    indx = indx[0][0]

    # Get the class that has the matrices
    fib_mat = fib_mats[indx]
    weights, out2, out3, out4 = fib_mat.get_matrices(fib.length, fib.num_points_up, 'weights_all')
    
    A = fib.form_linear_operator(fib_mat) 
   
    flow_on = None
    if external_flow is not None:
      flow_on = external_flow[offset_fibers[k]:offset_fibers[k+1]]
    
    
    RHS = fib.compute_RHS(fib_mat = fib_mat, 
      force_external = force_fibers[offset_fibers[k] : offset_fibers[k+1]], flow = flow_on)

    # Set BCs if they are given
    if BC_start_0 is not None:
      fib.set_BC(BC_start_0='velocity',
                BC_start_1='angular_velocity',
                BC_end_0='force',
                BC_end_vec_0=force_fibers[offset_fibers[k] + fib.num_points - 1]*weights[-1])

    # Apply BC
    A, RHS = fib.apply_BC_rectangular(A, RHS, fib_mat) # DONE

    # Save data
    A_all.append(A)
    istart = offset_bodies + 4*offset_fibers[k]
    RHS_all[istart : istart + 4*fib.num_points] = RHS

  # Transform to sparse matrix
  if len(A_all) > 0:
    As_fibers_block = scsp.block_diag(A_all)
    As_fibers = scsp.csr_matrix(As_fibers_block)
  else:
    As_fibers, As_fibers_block = None, None

  return As_fibers, A_all, RHS_all




def prepare_linear_operator(bodies, fibers):
  '''
  
  '''
  # Get number of particles and offsets
  num_bodies, num_fibers, offset_bodies, offset_fibers = get_num_particles_and_offsets(bodies, fibers)

  # Get fibers force operators
  fibers_force_operator = multi.build_fibers_force_operator(fibers)
  force_links_0, force_links_1, torque_links_0, torque_links_1 = build_fibers_force_torque_link_operators(fibers)

  # Get target points
  trg = gather_target_points(bodies, fibers)

  # Get fibers coordinates
  xf = gather_fibers_coordinates(fibers)

  # Get bodies locations, points and normals
  q, xb, normals = gather_bodies_configuration_info(bodies)

  # Get fibers A and RHS
  A_fibers, RHS_fibers, weights_fibers = get_fibers_matrices(fibers, offset_fibers)
  
  def linear_operator(y,
                      num_bodies,
                      num_fibers,
                      offset_bodies,
                      offset_fibers,
                      fibers_force_operator,
                      force_links_0,
                      torque_links_0,
                      trg,
                      xf,
                      q,
                      xb,
                      normals,
                      eta,
                      A_fibers):
    '''
    
    '''
    # Offsets to access certain variables
    offset_bodies_total = 6 * num_bodies + offset_bodies[num_bodies]
    
    # Get degrees of freedom (double layer potential,
    # bodies velocities and fibers coordinates and tensions).
    mu = y[0:offset_bodies[num_bodies]]
    U = y[offset_bodies[num_bodies]:offset_bodies_total] 
    XT = y[offset_bodies_total:] 
    
    # 1. Compute fibers density force
    force_fibers = fibers_force_operator.dot(XT)
    
    # 2. Compute fibers link force
    force_links = force_links_0.dot(XT)
    torque_links = torque_links_0.dot(XT)

    # 3. Compute total force torque on body
    force_body = np.sum(force_links, axis=1)
    torque_body = np.sum(torque_links, axis=1)
    
    # 4. Compute fluid velocity
    fw = force_fibers * fibers_weights[:, None]
    v  = kernels.oseen_kernel_source_target_numba(xf, trg, fw, eta = eta)
    v += kernels.oseen_kernel_source_target_numba(q, trg, force_body, eta = eta)
    v += kernels.rotlet_kernel_source_target_numba(q, trg, torque_body, eta = eta)
    v += kernels.stresslet_kernel_source_target_numba(xb, trg, normals, mu, eta = eta)

    # Rearrenge flow velocity at fibers
    vf = np.zeros(3 * offset_fibers[num_fibers])

    # 5. Multiply by fiber matrices
    AfXT = A_fibers.dot(XT)
    
  return v


def prepare_linear_system_fibers(fibers, external_forces, eta):
  '''
  Return linear operator, Preconditioner and RHS. 
  '''
  # Whether upsample or not when computing potentials
  iupsample = True

  # Get number of particles and offsets 
  num_bodies, num_fibers, offset_bodies, offset_fibers = get_num_particles_and_offsets([], fibers) 

  # Get the source and target points (they are the same) (num_points * nfibers * 3)
  trg = gather_target_points([],fibers)
  
  # Get fibers self-hydro Stokeslet
  G_self_fiber = get_self_fibers_Stokeslet(offset_fibers, fibers, eta, iupsample)

  # 1. Flow due to fiber forces on other fibers (includes self flow)
  v_external = flow_fibers(external_forces, trg, trg, fibers, offset_fibers, eta, iupsample)
  # 2. Subtract self-flow due to Stokeslet (instead use SBT)
  v_external += self_flow_fibers(external_forces, offset_fibers, fibers,  G_self_fiber, eta, iupsample)

  # Get fibers A and RHS (GK: Upsampling is done)
  A_fibers, A_fibers_blocks, RHS_fibers = get_fibers_matrices(fibers, offset_fibers, v_external) 
  
  # Get fibers force operators (GK: Upsampling is done)
  fibers_force_operator = build_fibers_force_operator(fibers)

  # prepare indexing to reshape outputs in GMRES
  flat2mat = np.zeros((3*offset_fibers[-1],3),dtype = bool)
  flat2mat_vT = np.zeros((4*offset_fibers[-1],5),dtype = bool)
  P_cheb_all = []
  for k, fib in enumerate(fibers):
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
  
  def linear_operator(y,
                      num_fibers,
                      offset_fibers,
                      trg,
                      eta,
                      A_fibers,
                      fibers_force_operator,
                      external_forces,
                      G_self_fiber):
    '''
    
    '''
    # Get degrees of freedom (fibers coordinates and tensions).
    XT = y # dimension: 4 * num_points
    
    # 1. Compute fibers density force (computed at high-res. then downsampled to num_points)
    force_fibers = fibers_force_operator.dot(XT)
    
    # 4. Compute fluid velocity due to fibers at trg
    # First, reorder forces
    fw = np.zeros((force_fibers.size // 3, 3))
    fw[:,0] = force_fibers[flat2mat[:,0]]
    fw[:,1] = force_fibers[flat2mat[:,1]]  
    fw[:,2] = force_fibers[flat2mat[:,2]]

    # Compute velocity due to force terms treated implicitly (bending and tension)
    v = flow_fibers(fw, trg, trg, fibers, offset_fibers, eta, iupsample)
    v += self_flow_fibers(fw, offset_fibers, fibers,  G_self_fiber, eta, iupsample)

    # Copy flow to right format
    v = v.reshape((v.size // 3, 3))
    vT = np.zeros(offset_fibers[-1] * 4)
    vT[flat2mat_vT[:,0]] = v[:,0]
    vT[flat2mat_vT[:,1]] = v[:,1]
    vT[flat2mat_vT[:,2]] = v[:,2]
    vT[flat2mat_vT[:,3]] = P_cheb_sprs.dot(vT[flat2mat_vT[:,4]])

    # 5. Multiply by fiber matrices
    AfXT = A_fibers.dot(XT)

    return AfXT - vT
    

  # Call partial
  system_size = offset_fibers[-1] * 4
  linear_operator_partial = partial(linear_operator,
                                    num_fibers=num_fibers,
                                    offset_fibers=offset_fibers,
                                    trg=trg,
                                    eta=eta,
                                    A_fibers=A_fibers,
                                    fibers_force_operator=fibers_force_operator,
                                    external_forces=external_forces,
                                    G_self_fiber=G_self_fiber)

  linear_operator_partial = scspla.LinearOperator((system_size, system_size), matvec = linear_operator_partial, dtype='float64')     

  # Build PC for fibers
  LU_all = [] 
  P_all = [] 
  for k, fib in enumerate(fibers): 
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

  return linear_operator_partial, RHS_fibers, P_inv_partial_LO


if __name__ == '__main__':
  print('Hola')
