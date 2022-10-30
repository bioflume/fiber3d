from __future__ import division, print_function
import numpy as np
import sys
import argparse
import subprocess
import time
from functools import partial
import scipy.sparse.linalg as scspla
from scipy.spatial import ConvexHull
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
  
# Find project functions
found_functions = False
path_to_append = ''
while found_functions is False:
  try:
    from mobility import mobility as mob
    from read_input import read_input
    # from read_input import read_fibers_file
    # from read_input import read_vertex_file
    from read_input import read_clones_file
    # from read_input import read_links_file
    # from utils import timer
    # from utils import nonlinear 
    # from utils import cheb
    from utils import gmres
    from quaternion import quaternion
    from utils import miscellaneous
    # from fiber import fiber
    # from integrators import integrators
    from kernels import kernels
    from body import body
    from shape_gallery import shape_gallery
    from quadratures import Smooth_Closed_Surface_Quadrature_RBF
    from periphery import periphery
    # from lib import periodic_fmm as fmm
    found_functions = True
  except ImportError:
    path_to_append += '../'
    print('searching functions in path ', path_to_append)
    sys.path.append(path_to_append)
    if len(path_to_append) > 21:
      print('\nProjected functions not found. Edit path in multi_bodies.py')
      sys.exit()


def set_mobility_vector_prod(implementation):
  '''
  Set the function to compute the matrix-vector
  product (M*F) with the mobility defined at the blob 
  level to the right implementation.


  ''' 
  return mob.single_layer_double_layer_numba


def get_blobs_r_vectors(bodies, Nblobs, periphery = None):
  '''
  Return coordinates of all the blobs with shape (Nblobs, 3).
  '''
  if periphery is None:
    Nperiphery = 0
    r_vectors = np.empty((Nblobs, 3))
  else:
    Nperiphery = periphery.Nblobs 
    r_vectors = np.empty((Nperiphery + Nblobs, 3))
    r_vectors[0 : Nperiphery] = periphery.get_r_vectors()
  offset = Nperiphery
  for b in bodies:
    num_blobs = b.Nblobs
    r_vectors[offset:(offset+num_blobs)] = b.get_r_vectors()
    offset += num_blobs
  return r_vectors


def get_blobs_normals(bodies, Nblobs, periphery = None):
  '''
  Return coordinates of all the blobs with shape (Nblobs, 3).
  '''
  if periphery is None:
    Nperiphery = 0
    normals = np.empty((Nblobs, 3))
  else:
    Nperiphery = periphery.Nblobs
    normals = np.empty((Nperiphery + Nblobs, 3))
    normals[0 : Nperiphery] = periphery.get_normals()
  offset = Nperiphery
  for b in bodies:
    num_blobs = b.Nblobs
    normals[offset:(offset+num_blobs)] = b.get_normals()
    offset += num_blobs
  return normals


def get_blobs_quadrature_weights(bodies, Nblobs, periphery = None):
  '''
  Return coordinates of all the blobs with shape (Nblobs, 3).
  '''
  if periphery is None:
    Nperiphery = 0
    weights = np.empty(Nblobs)
  else:
    Nperiphery = periphery.Nblobs 
    weights = np.empty(Nperiphery + Nblobs)
    weights[0 : Nperiphery] = periphery.quadrature_weights
  offset = Nperiphery
  for b in bodies:
    num_blobs = b.Nblobs
    weights[offset:(offset+num_blobs)] = b.quadrature_weights
    offset += num_blobs
  return weights


def K_matrix_vector_prod(bodies, vector, Nblobs, K_bodies = None, Transpose = False):
  '''
  Compute the matrix vector product K*vector where
  K is the geometrix matrix that transport the information from the 
  level of describtion of the body to the level of describtion of the blobs.
  ''' 
  # Prepare variables
  if Transpose:
    result = np.empty((len(bodies), 6))
    v = np.reshape(vector, (Nblobs * 3))
  else:
    result = np.empty((Nblobs, 3))
    v = np.reshape(vector, (len(bodies) * 6))

  # Loop over bodies
  offset = 0
  for k, b in enumerate(bodies):
    if K_bodies is None:
      K = b.calc_K_matrix()
    else:
      K = K_bodies[k]
    if Transpose:
      result[k : k+1] = np.dot(K.T, v[3*offset : 3*(offset+b.Nblobs)])
    else:
      result[offset : offset+b.Nblobs] = np.reshape(np.dot(K, v[6*k : 6*(k+1)]), (b.Nblobs, 3))
    offset += b.Nblobs    
  return result



@miscellaneous.static_var('initialized', [])
@miscellaneous.static_var('M_inv_periphery', [])
@miscellaneous.static_var('A_inv_bodies', [])
def build_block_diagonal_preconditioner_SK(bodies, eta, Nblobs, periphery = None, *args, **kwargs):
  '''

  '''
  initialized = build_block_diagonal_preconditioner_SK.initialized
  A_inv_bodies = []
  M_inv_periphery = None

  Nperiphery = 0
  if periphery is not None:
    Nperiphery = periphery.Nblobs

  if len(initialized) > 0:
    if periphery is not None:
      M_inv_periphery = build_block_diagonal_preconditioner_SK.M_inv_periphery[0]
    A_inv_bodies = build_block_diagonal_preconditioner_SK.A_inv_bodies
      
  else:
    build_block_diagonal_preconditioner_SK.initialized.append(1)
    if periphery is not None:
      N = periphery.Nblobs
      r_vectors = periphery.get_r_vectors()
      normals = periphery.get_normals()
      weights = periphery.quadrature_weights
      M = kernels.stresslet_kernel_times_normal_numba(r_vectors, normals, eta)
      if True:
        # Using singularity subtraction -q + D[q - q(x)](x)
        e = np.zeros((N, 3))
        e[:,0] = 1.0
        e *= weights[:,None]
        ex = kernels.stresslet_kernel_times_normal_times_density_numba(r_vectors, normals, e, eta)
        e[:,:] = 0.0
        e[:,1] = 1.0
        e *= weights[:,None]
        ey = kernels.stresslet_kernel_times_normal_times_density_numba(r_vectors, normals, e, eta)
        e[:,:] = 0.0
        e[:,2] = 1.0
        e *= weights[:,None]
        ez = kernels.stresslet_kernel_times_normal_times_density_numba(r_vectors, normals, e, eta)
        I = np.zeros((3*N, 3*N))
        for i in range(N):
          I[3*i:3*(i+1), 3*i+0] = ex[3*i:3*(i+1)] / weights[i]
          I[3*i:3*(i+1), 3*i+1] = ey[3*i:3*(i+1)] / weights[i]
          I[3*i:3*(i+1), 3*i+2] = ez[3*i:3*(i+1)] / weights[i]
        I_vec = np.ones(N*3)
        I_vec[0::3] /= (1.0 * weights)
        I_vec[1::3] /= (1.0 * weights)
        I_vec[2::3] /= (1.0 * weights)
        M += -np.diag(I_vec) - I
      else:
        # Using naive representation -0.5*q(x) + D[q](x)
        I_vec = np.ones(N*3)
        I_vec[0::3] /= (2.0 * weights)
        I_vec[1::3] /= (2.0 * weights)
        I_vec[2::3] /= (2.0 * weights)
        M -= np.diag(I_vec)
      M += kernels.complementary_kernel(r_vectors, normals)
      M_inv_periphery = np.linalg.pinv(M)
      build_block_diagonal_preconditioner_SK.M_inv_periphery.append(M_inv_periphery)

    for k, b in enumerate(bodies):
      # r_vectors = b.get_r_vectors()
      # normals = b.get_normals()
      r_vectors = b.reference_configuration
      normals = b.reference_normals
      weights = b.quadrature_weights
      M = kernels.stresslet_kernel_times_normal_numba(r_vectors, normals, eta)
      if True:
        # Using singularity subtraction D[q - q(x)](x)
        e = np.zeros((b.Nblobs, 3))
        e[:,0] = 1.0
        e *= weights[:,None]
        ex = kernels.stresslet_kernel_times_normal_times_density_numba(r_vectors, normals, e, eta)
        e[:,:] = 0.0
        e[:,1] = 1.0
        e *= weights[:,None]
        ey = kernels.stresslet_kernel_times_normal_times_density_numba(r_vectors, normals, e, eta)
        e[:,:] = 0.0
        e[:,2] = 1.0
        e *= weights[:,None]
        ez = kernels.stresslet_kernel_times_normal_times_density_numba(r_vectors, normals, e, eta)

        I = np.zeros((3*b.Nblobs, 3*b.Nblobs))
        for i in range(b.Nblobs):
          I[3*i:3*(i+1), 3*i+0] = ex[3*i:3*(i+1)] / weights[i]
          I[3*i:3*(i+1), 3*i+1] = ey[3*i:3*(i+1)] / weights[i]
          I[3*i:3*(i+1), 3*i+2] = ez[3*i:3*(i+1)] / weights[i]
        M -= I
      else:
        # Using naive representation -0.5*q(x) + D[q](x)
        I_vec = np.ones(b.Nblobs*3)
        I_vec[0::3] /= (2.0 * weights)
        I_vec[1::3] /= (2.0 * weights)
        I_vec[2::3] /= (2.0 * weights)
        M -= np.diag(I_vec)

      M_inv = np.linalg.pinv(M)
      A = np.zeros((3*b.Nblobs+6, 3*b.Nblobs+6))
      K = b.calc_K_matrix()
      A[0:3*b.Nblobs, 0:3*b.Nblobs] = M
      A[0:3*b.Nblobs, 3*b.Nblobs:3*b.Nblobs+6] = -K
      A[3*b.Nblobs:3*b.Nblobs+6, 0:3*b.Nblobs] = -K.T
      A[3*b.Nblobs:3*b.Nblobs+6, 3*b.Nblobs:3*b.Nblobs+6] = np.eye(6)
      A_inv_bodies.append(np.linalg.inv(A))
      build_block_diagonal_preconditioner_SK.A_inv_bodies.append(A_inv_bodies[-1])
 
  def block_diagonal_preconditioner(vector, A_inv_bodies = None, Nblobs = Nblobs, M_inv_periphery = None, Nperiphery = None):
    '''
    
    '''
    x = vector.flatten()
    y = np.empty_like(x)
    if Nperiphery > 0:
      y[0:3*Nperiphery] = np.dot(M_inv_periphery, x[0:3*Nperiphery])
    else:
      Nperiphery = 0

    offset = Nperiphery
    for k, b in enumerate(bodies):
      N = b.Nblobs
      # Get unknowns of body b
      xb = np.zeros(3*N+6)
      xb[0:3*N] = x[offset*3 : (offset+N)*3]
      xb[3*N:] = x[(Nperiphery+Nblobs)*3+k*6 : (Nperiphery+Nblobs)*3+(k+1)*6]

      # Rotate vectors to body frame
      rotation_matrix = b.orientation.rotation_matrix().T
      xb = xb.reshape((N+2, 3))
      xb_body = np.array([np.dot(rotation_matrix, vec) for vec in xb]).flatten()

      # Apply PC
      # yb = np.dot(A_inv_bodies[k], xb)
      yb_body = np.dot(A_inv_bodies[k], xb_body)

      # Rotate vectors to laboratory frame
      yb_body = yb_body.reshape((N+2, 3))
      yb = np.array([np.dot(rotation_matrix.T, vec) for vec in yb_body]).flatten()

      # Set unknowns of body b
      y[offset*3 : (offset+N)*3] = yb[0:3*N]
      y[(Nperiphery+Nblobs)*3+k*6 : (Nperiphery+Nblobs)*3+(k+1)*6] = yb[3*N:]
      offset += N

    # y[3*Nperiphery:] = x[3*Nperiphery:]
    return y

  block_diagonal_preconditioner_partial = partial(block_diagonal_preconditioner,
                                                  A_inv_bodies = A_inv_bodies,
                                                  Nblobs = Nblobs,
                                                  M_inv_periphery = M_inv_periphery, 
                                                  Nperiphery = Nperiphery)
  system_size = 3 * (Nperiphery+Nblobs) + 6 * len(bodies)
  PC = scspla.LinearOperator((system_size, system_size), matvec = block_diagonal_preconditioner_partial, dtype='float64')
  return PC


def linear_operator_rigid_SK(vector,
                             bodies,
                             r_vectors,
                             normals,
                             quadrature_weights,
                             eta, 
                             periphery = None,
                             K_bodies = None,
                             *args, **kwargs):
  '''
  | -0.5*I + D   -K||w*mu| = | s - G*F - R*L|
  |     -K^T      I|| U  |   |      0       |
  ''' 
  # Reserve memory for the solution and create some variables
  Nperiphery = 0
  #if ex_periphery is not None:
  #  Nperiphery = ex_periphery.size // 3
  if periphery is not None:
    Nperiphery = periphery.Nblobs

  Ncomp_blobs = r_vectors.size - Nperiphery*3
  Nblobs = r_vectors.size // 3 - Nperiphery
  Ncomp_bodies = 6 * len(bodies)
  res = np.zeros((Nperiphery*3 + Ncomp_blobs + Ncomp_bodies))
  v = np.reshape(vector, (vector.size // 3, 3))

  # Compute mobility part
  u_double = kernels.stresslet_kernel_times_normal_times_density_numba(r_vectors,
                                                                       normals,
                                                                       v[0:Nperiphery+Nblobs],
                                                                       eta)
  res[0:Nperiphery*3 + Ncomp_blobs] = u_double
  
  if periphery is not None:
    u_complementary = kernels.complementary_kernel_times_density_numba(r_vectors[0:Nperiphery],
                                                                       normals[0:Nperiphery],
                                                                       v[0:Nperiphery])
    res[0:Nperiphery*3] += u_complementary 

  if True:
    offset = Nperiphery
    for k, b in enumerate(bodies):
      d = np.zeros(b.Nblobs)
      d[:] = v[offset:offset+b.Nblobs, 0]
      cx = ((d / b.quadrature_weights)[:,None] * b.ex).flatten()
      d[:] = v[offset:offset+b.Nblobs, 1]
      cy = ((d / b.quadrature_weights)[:,None] * b.ey).flatten()
      d[:] = v[offset:offset+b.Nblobs, 2]
      cz = ((d / b.quadrature_weights)[:,None] * b.ez).flatten()
      
      res[offset*3:(offset+b.Nblobs)*3] += -(cx + cy + cz)
      # res[Nperiphery*3:Nperiphery*3+Ncomp_blobs] -= 0.5 * (v[Nperiphery:Nperiphery+Nblobs] / quadrature_weights[Nperiphery:Nperiphery+Nblobs, None]).flatten()  
      offset += b.Nblobs
    if periphery is not None:
      d = np.zeros(Nperiphery)
      d[:] = v[0:Nperiphery, 0] 
      cx = ((d / quadrature_weights[0:Nperiphery])[:,None] * periphery.ex).flatten()
      
      d = np.zeros(Nperiphery)
      d[:] = v[0:Nperiphery, 1] 
      cy = ((d / quadrature_weights[0:Nperiphery])[:,None] * periphery.ey).flatten()
      
      d = np.zeros(Nperiphery)
      d[:] = v[0:Nperiphery, 2] 
      cz = ((d / quadrature_weights[0:Nperiphery])[:,None] * periphery.ez).flatten()
    
      res[0:Nperiphery*3] -= (cx + cy + cz)
      res[0:Nperiphery*3] -= 1.0 * (v[0:Nperiphery] / quadrature_weights[0:Nperiphery, None]).flatten()
    
                
  # Compute -K*U
  K_times_U = K_matrix_vector_prod(bodies, v[Nperiphery+Nblobs : Nperiphery+Nblobs+2*len(bodies)], Nblobs, K_bodies=K_bodies) 
  res[Nperiphery*3:Nperiphery*3+Ncomp_blobs] -= K_times_U.flatten()

  # Compute -K.T*w*mu
  K_T_times_lambda = K_matrix_vector_prod(bodies, vector[Nperiphery*3:Nperiphery*3+Ncomp_blobs], Nblobs, K_bodies=K_bodies, Transpose=True)
  res[Nperiphery*3+Ncomp_blobs : Nperiphery*3+Ncomp_blobs+Ncomp_bodies] = -K_T_times_lambda.flatten()
  res[Nperiphery*3+Ncomp_blobs : Nperiphery*3+Ncomp_blobs+Ncomp_bodies] += v[Nperiphery+Nblobs:].flatten()

  return res







if __name__ == '__main__':
  # Get command line arguments
  parser = argparse.ArgumentParser(description='Run a multi-fiber simulation and save trajectory.')
  # parser.add_argument('--input-file', dest='input_file', type=str, default='data.main', help='name of the input file')
  parser.add_argument('--input-file', dest='input_file', type=str, default='data.main', help='name of the input file')
  parser.add_argument('--verbose', action='store_true', help='print convergence information')
  args=parser.parse_args()
  input_file = args.input_file
  
  # Read input file
  read = read_input.ReadInput(input_file)
  
  # Copy input file to output
  subprocess.call(["cp", input_file, read.output_name + '.inputfile'])

  # Set random generator state
  if read.random_state is not None:
    with open(read.random_state, 'rb') as f:
      np.random.set_state(cPickle.load(f))
  elif read.seed is not None:
    np.random.seed(int(read.seed))
  
  # Save random generator state
  with open(read.output_name + '.random_state', 'wb') as f:
    cpickle.dump(np.random.get_state(), f)




  # Create periphery
  Nresolution = 400
  if True:
    nodes_periphery, normals_periphery, h_periphery, gradh_periphery = shape_gallery.shape_gallery('sphere', Nresolution, radius=10.0)
    normals_periphery = -normals_periphery
    hull_periphery = ConvexHull(nodes_periphery)
    triangles_periphery = hull_periphery.simplices
    quadrature_weights_periphery = Smooth_Closed_Surface_Quadrature_RBF.Smooth_Closed_Surface_Quadrature_RBF(nodes_periphery, 
                                                                                                             triangles_periphery, 
                                                                                                             h_periphery,
                                                                                                             gradh_periphery)
    Nperiphery = quadrature_weights_periphery.size
    q = quaternion.Quaternion([1.0, 0.0, 0.0, 0.0])
    shell = periphery.Periphery(np.array([0., 0., 0.]), q, nodes_periphery, normals_periphery, quadrature_weights_periphery)

    # Update orientation
    # quaternion_dt = quaternion.Quaternion.from_rotation(np.array([1.0, 0., 0.]) * (2 * np.pi * np.random.rand()))
    # shell.orientation = quaternion_dt * shell.orientation

    # Compute correction for singularity subtractions
    r_vectors_blobs = shell.get_r_vectors()
    normals = shell.get_normals()
    quadrature_weights = shell.quadrature_weights
    e = np.zeros((Nperiphery, 3))
    e[:,0] = 1.0
    e *= quadrature_weights[:,None]
    ex = kernels.stresslet_kernel_times_normal_times_density_numba(r_vectors_blobs, normals, e, read.eta)
    e[:,:] = 0.0
    e[:,1] = 1.0
    e *= quadrature_weights[:,None]
    ey = kernels.stresslet_kernel_times_normal_times_density_numba(r_vectors_blobs, normals, e, read.eta)
    e[:,:] = 0.0
    e[:,2] = 1.0
    e *= quadrature_weights[:,None]
    ez = kernels.stresslet_kernel_times_normal_times_density_numba(r_vectors_blobs, normals, e, read.eta)
    shell.ex = np.copy(ex.reshape(Nperiphery, 3))
    shell.ey = np.copy(ey.reshape(Nperiphery, 3))
    shell.ez = np.copy(ez.reshape(Nperiphery, 3))

  else:
    Nperiphery = 0
    nodes_periphery = None
    normals_periphery = None
    quadrature_weights_periphery = None
    shell = None

    
  # Create rigid bodies
  bodies = []
  body_types = []
  body_names = []
  for ID, structure in enumerate(read.structures):
    print('Creating structures = ', structure[1])

    # TODO: read shape kind from imput file
    nodes, normals, h, gradh = shape_gallery.shape_gallery('sphere', Nresolution, radius=0.2)
    num_bodies_struct, struct_locations, struct_orientations = read_clones_file.read_clones_file(structure[1])
    # Read slip file if it exists
    slip = None
    if(len(structure) > 2):
      slip = read_slip_file.read_slip_file(structure[2])
    body_types.append(num_bodies_struct)
    body_names.append(read.structures_ID[ID])

    # Compute quadrature weights
    hull = ConvexHull(nodes)
    triangles = hull.simplices
    quadrature_weights = Smooth_Closed_Surface_Quadrature_RBF.Smooth_Closed_Surface_Quadrature_RBF(nodes, triangles, h ,gradh)

    # Creat each body of tyoe structure
    for i in range(num_bodies_struct):
      b = body.Body(struct_locations[i], struct_orientations[i], nodes, normals, quadrature_weights)
      # b.mobility_blobs = multi_bodies.set_mobility_blobs(read.mobility_blobs_implementation)
      b.ID = read.structures_ID[ID]

      # Update orientation
      # quaternion_dt = quaternion.Quaternion.from_rotation(np.random.rand(3) * (2 * np.pi * np.random.rand()))
      # b.orientation = quaternion_dt * b.orientation

      # multi_bodies_functions.set_slip_by_ID(b, slip, slip_options = read.slip_options)
      # Append bodies to total bodies list

      bodies.append(b)
  bodies = np.array(bodies)

  # Set some more variables
  num_of_body_types = len(body_types)
  num_bodies = bodies.size
  Nblobs = sum([b.Nblobs for b in bodies])
  # multi_bodies.mobility_vector_prod = multi_bodies.set_mobility_vector_prod(read.mobility_vector_prod_implementation)
  # multi_bodies.mobility_vector_prod = set_mobility_vector_prod(read.mobility_vector_prod_implementation)

  # Write bodies information
  with open(read.output_name + '.bodies_info', 'w') as f:
    f.write('num_of_body_types   ' + str(num_of_body_types) + '\n')
    f.write('body_names          ' + str(body_names) + '\n')
    f.write('body_types          ' + str(body_types) + '\n')
    f.write('num_bodies          ' + str(num_bodies) + '\n')
    f.write('num_blobs           ' + str(Nblobs) + '\n')
    f.write('num_blobs_periphery ' + str(Nperiphery) + '\n')


  start_time = time.time()
  for step in range(read.n_steps+1):
    print('========================================= ', step)

    # Save data if...
    if (step % read.n_save) == 0 and step >= 0:
      elapsed_time = time.time() - start_time
      print('Integrator = ', read.scheme, ', step = ', step, ', invalid configurations', 0, ', wallclock time = ', time.time() - start_time)
      # For each type of structure save locations and orientations to one file
      body_offset = 0
      if read.save_clones == 'one_file':
        structures_ID = ['spheres']
        for i, ID in enumerate(structures_ID):
          name = read.output_name + '.' + ID + '.config'
          if step == 0:
            status = 'w'
          else:
            status = 'a'
          with open(name, status) as f_ID:
            f_ID.write(str(body_types[i]) + '\n')
            for j in range(body_types[i]):
              orientation = bodies[body_offset + j].orientation.entries
              f_ID.write('%s %s %s %s %s %s %s\n' % (bodies[body_offset + j].location[0], 
                                                     bodies[body_offset + j].location[1], 
                                                     bodies[body_offset + j].location[2], 
                                                     orientation[0], 
                                                     orientation[1], 
                                                     orientation[2], 
                                                     orientation[3]))
            body_offset += body_types[i]


















    # Get blobs coordinates, normals and weights
    r_vectors_blobs = get_blobs_r_vectors(bodies, Nblobs, shell)
    normals_blobs = get_blobs_normals(bodies, Nblobs, shell)
    quadrature_weights_blobs = get_blobs_quadrature_weights(bodies, Nblobs, shell)

    # Set right hand side
    System_size = (Nperiphery + Nblobs) * 3 + num_bodies * 6
    RHS = np.zeros(System_size)
    # compute Oseen term
    r_bodies = np.array([b.location for b in bodies])
    force_torque = np.zeros(6*len(bodies))
    force_torque[0] = 0.
    forces = np.copy(force_torque[0::2])
    torques = np.copy(force_torque[1::2])
    u_stokeslet = kernels.oseen_kernel_source_target_numba(r_bodies,
                                                           r_vectors_blobs,
                                                           forces,
                                                           eta = read.eta)

    # compute rotlet term
    u_rotlet = kernels.rotlet_kernel_source_target_numba(r_bodies,
                                                         r_vectors_blobs,
                                                         torques,
                                                         eta = read.eta)

    if True:
      r_vec = bodies[0].get_r_vectors()
      x = r_vec[:,0] - bodies[0].location[0]
      y = r_vec[:,1] - bodies[0].location[1]
      z = r_vec[:,2] - bodies[0].location[2]
      r = np.sqrt(x**2 + y**2 + z**2)
      theta = np.zeros_like(r)
      sel = r > 0
      theta[sel] = np.arccos(z[sel] / r[sel])
      phi = np.arctan2(y, x)            
      slip_theta = np.sin(theta) * np.cos(theta) * 0
      v_slip = np.zeros_like(r_vec)
      v_slip[:,0] = np.cos(phi)*np.sin(theta)*0 - np.sin(theta)*0 + np.cos(phi)*np.cos(theta)*slip_theta
      v_slip[:,1] = np.sin(phi)*np.sin(theta)*0 + np.cos(phi)*0   + np.sin(phi)*np.cos(theta)*slip_theta
      v_slip[:,2] = np.cos(theta)*0                               - np.sin(theta)*slip_theta

      rotation_matrix = bodies[0].orientation.rotation_matrix()
      slip_rotated = np.empty_like(v_slip)
      for i in range(bodies[0].Nblobs):
        slip_rotated[i] = np.dot(rotation_matrix, v_slip[i])


    RHS[0:(Nperiphery+Nblobs)*3] = -u_stokeslet - u_rotlet 
    RHS[Nperiphery*3:(Nperiphery+Nblobs)*3] += slip_rotated.flatten()

    # Singularity subtraction vectors
    offset = 0
    for k, b in enumerate(bodies):
      b.calc_vectors_singularity_subtraction(read.eta, r_vectors_blobs[offset:offset+b.Nblobs], normals_blobs[offset:offset+b.Nblobs])
      # b.calc_vectors_singularity_subtraction(read.eta)
      offset += b.Nblobs
    
    # Set linear operators 
    linear_operator_partial = partial(linear_operator_rigid_SK,
                                      bodies=bodies,
                                      r_vectors=r_vectors_blobs,
                                      normals=normals_blobs,
                                      quadrature_weights=quadrature_weights_blobs,
                                      eta=read.eta,
                                      periphery = shell,
                                      K_bodies = None)
    A = scspla.LinearOperator((System_size, System_size), matvec = linear_operator_partial, dtype='float64')

    # Set preconditioner
    # PC = None
    PC = build_block_diagonal_preconditioner_SK(bodies, read.eta, Nblobs, periphery = shell)

    # Solve preconditioned linear system # callback=make_callback()
    counter = miscellaneous.gmres_counter(print_residual = args.verbose) 
    (sol_precond, info_precond) = gmres.gmres(A, RHS, tol=read.solver_tolerance, M=PC, maxiter=1000, restart=60, callback=counter) 
    
    # Extract velocities and constraint forces on blobs
    velocity = np.reshape(sol_precond[3*(Nperiphery+Nblobs): 3*(Nperiphery+Nblobs) + 6*num_bodies], (num_bodies, 6))
    lambda_blobs = np.reshape(sol_precond[0: 3*(Nperiphery+Nblobs)], (Nperiphery+Nblobs, 3))

    print('velocity = ', velocity)

    # Update
    for k, b in enumerate(bodies):
      quaternion_dt = quaternion.Quaternion.from_rotation(velocity[k, 6*k+3:6*k+6] * read.dt)
      b.orientation = quaternion_dt * b.orientation
      print(b.location.shape, velocity[k, 6*k:6*k+3].shape)
      b.location += velocity[k, 6*k:6*k+3] * read.dt




  # Save wallclock time 
  with open(read.output_name + '.time', 'w') as f:
    f.write(str(time.time() - start_time) + '\n')

  print('\n\n\n# End')
