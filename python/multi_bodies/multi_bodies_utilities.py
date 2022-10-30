'''
This modules solves the mobility or the resistance problem for one
configuration of a multibody supensions and it can save some data like
the velocities or forces on the bodies, the mobility of a body or
the mobility of the blobs.
'''
from __future__ import division, print_function
import argparse
import numpy as np
import scipy.linalg as sla
import scipy.sparse.linalg as spla
from scipy.spatial import ConvexHull
import subprocess
try:
  import cPickle as cpickle
except:
  try:
    import cpickle
  except:
    import _pickle as cpickle
from functools import partial, reduce
import sys
import time
from numba import njit, prange


# Find project functions
found_functions = False
path_to_append = ''
while found_functions is False:
  try:
    import multi_bodies
    from kernels import kernels
    from quaternion import quaternion
    from body import body 
    from periphery import periphery
    from read_input import read_input
    from read_input import read_vertex_file
    from read_input import read_clones_file
    from shape_gallery import shape_gallery
    from quadratures import Smooth_Closed_Surface_Quadrature_RBF
    from utils import miscellaneous
    from utils import gmres
    found_functions = True
  except ImportError:
    path_to_append += '../'
    print('searching functions in path ', path_to_append, sys.path)
    sys.path.append(path_to_append)
    if len(path_to_append) > 21:
      print('\nProjected functions not found. Edit path in multi_bodies_utilities.py')
      sys.exit()

print('searching functions in path ooo ', path_to_append, sys.path)

# Try to import the visit_writer (boost implementation)
# import visit.visit_writer as visit_writer
try:
  import visit.visit_writer as visit_writer
  # from visit import visit_writer
except ImportError:
  print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
  pass


# Callback generator
def make_callback():
  closure_variables = dict(counter=0, residuals=[]) 
  def callback(residuals):
    closure_variables["counter"] += 1
    closure_variables["residuals"].append(residuals)
    print(closure_variables["counter"], residuals)
  return callback


@njit(parallel=True)
def vzero_numba(r_vec, v, R1, R2):

  N = r_vec.size // 3
  r_vec = r_vec.reshape((N, 3))
  v = v.reshape((N,3))
  u = np.zeros_like(v)
  
  for xn in prange(N):
    x = r_vec[xn, 0]
    y = r_vec[xn, 1]
    z = r_vec[xn, 2]
    r = np.sqrt(x**2 + y**2 + z**2)

    if r < R1:
      u[xn,:] = 0
    elif r > R2:
      u[xn,:] = 0
    else:
      u[xn] = v[xn]

  return u


@njit(parallel=True)
def vth_2_numba(r_vec, R1, R2):
  '''
  From Kallemov paper
  '''
  v = np.zeros_like(r_vec)
  N = r_vec.size // 3
  r_vec = r_vec.reshape((N, 3))

  l = R1 / R2
  alpha = 1 - 9*l/4 + 5*l**3/2 - 9*l**5/4 + l**6
  K = (1 - l**5) / alpha
  A = -3.75 * (l**3 - l**5) / alpha / R1**2
  B = 1.5 * R1 * (1-l**5) / alpha
  C = 0.5 * (1 + 5*l**3/4 - 9*l**5/4) / alpha
  D = 0.25 * R1**3 * (1 - l**3) / alpha
  
  for xn in prange(N):
    x = r_vec[xn, 0]
    y = r_vec[xn, 1]
    z = r_vec[xn, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    if r > 0:
      theta = np.arccos(z / r)
    else:
      theta = 0
    phi = np.arctan2(y, x)

    if r > R1 and r < R2:
      ur = -np.cos(theta) * (A*r**2/5 - B/r + 2*C + 2*D/r**3)
      uphi = 0.
      utheta = np.sin(theta) * (2*A*r**2/5 - 0.5*B/r + 2*C - D/r**3)

      cos_theta = np.cos(theta)
      sin_theta = np.sin(theta)
      cos_phi = np.cos(phi)
      sin_phi = np.sin(phi)

      v[xn,0] = cos_phi*sin_theta*ur - sin_theta*uphi + cos_phi*cos_theta*utheta
      v[xn,1] = sin_phi*sin_theta*ur + cos_phi  *uphi + sin_phi*cos_theta*utheta
      v[xn,2] = cos_theta        *ur                  - sin_theta        *utheta + 1.0
  return v
   

@njit(parallel=True)
def vth_inf_numba(r_vec, R1, R2):
  v = np.zeros_like(r_vec)
  N = r_vec.size // 3
  r_vec = r_vec.reshape((N, 3))

  c = 3.0*R1 / 4.0
  d = R1**3 / 4.0
  
  for xn in prange(N):
    x = r_vec[xn, 0]
    y = r_vec[xn, 1]
    z = r_vec[xn, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    if r > 0:
      theta = np.arccos(z / r)
    else:
      theta = 0
    phi = np.arctan2(y, x)

    if r < R1:
      ur = 0
      utheta = 0
    elif r > R2:
      ur = 0
      utheta = 0
    else:
      ur = np.cos(theta) * (1.0 - 2*c/r + 2*d/r**3)
      uphi = 0.0
      utheta = -np.sin(theta) * (1.0 - c/r - d/r**3)

    if r > R1 and r < R2:
      cos_theta = np.cos(theta)
      sin_theta = np.sin(theta)
      cos_phi = np.cos(phi)
      sin_phi = np.sin(phi)

      v[xn,0] = cos_phi*sin_theta*ur - sin_theta*uphi + cos_phi*cos_theta*utheta
      v[xn,1] = sin_phi*sin_theta*ur + cos_phi*uphi   + sin_phi*cos_theta*utheta
      v[xn,2] = (cos_theta*ur                         - sin_theta*utheta) - 1.0

  return v



def plot_velocity_field(grid, r_vectors_blobs, lambda_blobs, eta, output, *args, **kwargs):
  '''
  This function plots the velocity field to a grid. 
  '''
  # Prepare grid values
  grid = np.reshape(grid, (3,3)).T
  grid_length = grid[1] - grid[0]
  grid_points = np.array(grid[2], dtype=np.int32)
  num_points = reduce(lambda x,y: x*y, grid_points)

  # Set grid coordinates
  dx_grid = grid_length / grid_points
  grid_x = np.array([grid[0,0] + dx_grid[0] * (x+0.5) for x in range(grid_points[0])])
  grid_y = np.array([grid[0,1] + dx_grid[1] * (x+0.5) for x in range(grid_points[1])])
  grid_z = np.array([grid[0,2] + dx_grid[2] * (x+0.5) for x in range(grid_points[2])])
  # Be aware, x is the fast axis.
  zz, yy, xx = np.meshgrid(grid_z, grid_y, grid_x, indexing = 'ij')
  grid_coor = np.zeros((num_points, 3))
  grid_coor[:,0] = np.reshape(xx, xx.size)
  grid_coor[:,1] = np.reshape(yy, yy.size)
  grid_coor[:,2] = np.reshape(zz, zz.size)

  # Compute velocity field 
  mobility_vector_prod_implementation = kwargs.get('mobility_vector_prod_implementation')
  if True:
    bodies = kwargs.get('bodies')
    force_torque = kwargs.get('force_torque')
    normal = kwargs.get('normal')
    density = kwargs.get('density')
    r_bodies = np.array([b.location for b in bodies])
    forces = np.copy(force_torque[0::2])
    torques = np.copy(force_torque[1::2])
      
    # compute Oseen term
    u_stokeslet = kernels.oseen_kernel_source_target_numba(r_bodies,
                                                           grid_coor,
                                                           forces,
                                                           eta)
    # compute rotlet term
    u_rotlet = kernels.rotlet_kernel_source_target_numba(r_bodies,
                                                         grid_coor,
                                                         torques,
                                                         eta)
    # Compute stresslet contribution
    u_stresslet = kernels.stresslet_kernel_source_target_numba(r_vectors_blobs,
                                                               grid_coor,
                                                               normal,
                                                               density,
                                                               eta)
    grid_velocity_full = u_stokeslet + u_rotlet + u_stresslet


  grid_velocity_th = vth_2_numba(grid_coor, 0.2, 2.0)
  grid_velocity    = vzero_numba(grid_coor, grid_velocity_full, 0.2, 2.0)
  grid_velocity_th = vzero_numba(grid_coor, grid_velocity_th, 0.2, 2.0)
  grid_velocity += grid_velocity_th 


  
  # Prepara data for VTK writer 
  variables = [np.reshape(grid_velocity, grid_velocity.size)] 
  dims = np.array([grid_points[0]+1, grid_points[1]+1, grid_points[2]+1], dtype=np.int32) 
  nvars = 1
  vardims = np.array([3])
  centering = np.array([0])
  varnames = ['velocity\0']
  name = output + '.velocity_field.vtk'
  grid_x = grid_x - dx_grid[0] * 0.5
  grid_y = grid_y - dx_grid[1] * 0.5
  grid_z = grid_z - dx_grid[2] * 0.5
  grid_x = np.concatenate([grid_x, [grid[1,0]]])
  grid_y = np.concatenate([grid_y, [grid[1,1]]])
  grid_z = np.concatenate([grid_z, [grid[1,2]]])

  print('dims = ', dims)
  print('\n\n\n')

  # Write velocity field
  visit_writer.boost_write_rectilinear_mesh(name,      # File's name
                                            0,         # 0=ASCII,  1=Binary
                                            dims,      # {mx, my, mz}
                                            grid_x,     # xmesh
                                            grid_y,     # ymesh
                                            grid_z,     # zmesh
                                            nvars,     # Number of variables
                                            vardims,   # Size of each variable, 1=scalar, velocity=3*scalars
                                            centering, # Write to cell centers of corners
                                            varnames,  # Variables' names
                                            variables) # Variables


  print('L2 norm 0%', np.linalg.norm(grid_velocity) / np.linalg.norm(grid_velocity_th))
  grid_velocity    = vzero_numba(grid_coor, grid_velocity, 0.21, 1.90)
  grid_velocity_th = vzero_numba(grid_coor, grid_velocity_th, 0.21, 1.90)
  print('L2 norm 5%', np.linalg.norm(grid_velocity) / np.linalg.norm(grid_velocity_th))
  grid_velocity    = vzero_numba(grid_coor, grid_velocity, 0.22, 1.80)
  grid_velocity_th = vzero_numba(grid_coor, grid_velocity_th, 0.22, 1.80)
  print('L2 norm 10%', np.linalg.norm(grid_velocity) / np.linalg.norm(grid_velocity_th))

  return


if __name__ ==  '__main__':
  # Get command line arguments
  parser = argparse.ArgumentParser(description='Solve the mobility or resistance problem'
                                   'for a multi-body suspension and save some data.')
  parser.add_argument('--input-file', dest='input_file', type=str, default='data.main', 
                      help='name of the input file')
  parser.add_argument('--verbose', action='store_true', help='print gmres and lanczos residuals')
  args=parser.parse_args()
  input_file = args.input_file

  # Read input file
  read = read_input.ReadInput(input_file)

  # Copy input file to output
  subprocess.call(["cp", input_file, read.output_name + '.inputfile'])
  name_kk = read.output_name + '.spheres.config'
  subprocess.call(["rm", name_kk])


  # Create periphery
  Nresolution = 800
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
    nodes, normals, h, gradh = shape_gallery.shape_gallery('sphere', Nresolution, radius=1.0)
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
      quaternion_dt = quaternion.Quaternion.from_rotation(np.random.rand(3) * (2 * np.pi * np.random.rand()))
      b.orientation = quaternion_dt * b.orientation

      # multi_bodies_functions.set_slip_by_ID(b, slip, slip_options = read.slip_options)
      # Append bodies to total bodies list

      bodies.append(b)
  bodies = np.array(bodies)

  # Set some more variables
  num_of_body_types = len(body_types)
  num_bodies = bodies.size
  Nblobs = sum([b.Nblobs for b in bodies])
  multi_bodies.mobility_vector_prod = multi_bodies.set_mobility_vector_prod(read.mobility_vector_prod_implementation)


  # Write bodies information
  with open(read.output_name + '.bodies_info', 'w') as f:
    f.write('num_of_body_types   ' + str(num_of_body_types) + '\n')
    f.write('body_names          ' + str(body_names) + '\n')
    f.write('body_types          ' + str(body_types) + '\n')
    f.write('num_bodies          ' + str(num_bodies) + '\n')
    f.write('num_blobs           ' + str(Nblobs) + '\n')
    f.write('num_blobs_periphery ' + str(Nperiphery) + '\n')

  # Calculate slip on blobs
  # if multi_bodies.calc_slip is not None:
  #   slip = multi_bodies.calc_slip(bodies, Nblobs)
  # else:
  #   slip = np.zeros((Nblobs, 3))
  slip = np.zeros((Nperiphery + Nblobs, 3))

  # Read forces file
  force_torque = np.zeros((num_bodies, 6))
  if read.force_file is not None:
    subprocess.call(["cp", read.force_file, read.output_name + '.force_file'])
    force_torque = np.loadtxt(read.force_file)
  force_torque = np.reshape(force_torque, (2*num_bodies, 3))
    
  # Read velocity file
  velocity = np.zeros((num_bodies, 6))
  if read.velocity_file is not None:
    subprocess.call(["cp", read.velocity_file, read.output_name + '.velocity_file'])
    velocity = np.loadtxt(read.velocity_file)
  velocity = np.reshape(velocity, (2*num_bodies, 3))


  # If scheme == mobility solve mobility problem
  if read.scheme == 'mobility':
    start_time = time.time()  
    distances = np.logspace(-3, 2, num=1000) + 2.0
    sample_max = 1
    step = 0
    for distance_i in distances:
      velocity_mean = np.zeros(12)
      velocity_error = np.zeros(12)
      # bodies[1].location[0] = distance_i
      for sample in range(sample_max):
        print('========================================= ', distance_i, sample)
        #for b in bodies:
        #  quaternion_dt = quaternion.Quaternion.from_rotation(np.random.rand(3) * (2 * np.pi * np.random.rand()))
        #  b.orientation = quaternion_dt * b.orientation

        # Get blobs coordinates, normals and weights
        r_vectors_blobs = multi_bodies.get_blobs_r_vectors(bodies, Nblobs, shell)
        normals_blobs = multi_bodies.get_blobs_normals(bodies, Nblobs, shell)
        quadrature_weights_blobs = multi_bodies.get_blobs_quadrature_weights(bodies, Nblobs, shell)

        # Set right hand side
        System_size = (Nperiphery + Nblobs) * 3 + num_bodies * 6
        RHS = np.zeros(System_size)
        # compute Oseen term
        r_bodies = np.array([b.location for b in bodies])
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
          y = r_vec[:,1] - bodies[0].location[0]
          z = r_vec[:,2] - bodies[0].location[0]
          r = np.sqrt(x**2 + y**2 + z**2)
          theta = np.zeros_like(r)
          sel = r > 0
          theta[sel] = np.arccos(z[sel] / r[sel])
          phi = np.arctan2(y, x)            
          slip_theta = np.cos(theta)*np.sin(theta)
          v_slip = np.zeros_like(r_vec)
          v_slip[:,0] = np.cos(phi)*np.sin(theta)*0 - np.sin(theta)*0 + np.cos(phi)*np.cos(theta)*slip_theta
          v_slip[:,1] = np.sin(phi)*np.sin(theta)*0 + np.cos(phi)*0   + np.sin(phi)*np.cos(theta)*slip_theta
          v_slip[:,2] = np.cos(theta)*0                               - np.sin(theta)*slip_theta


        
        RHS[0:(Nperiphery+Nblobs)*3] = -u_stokeslet - u_rotlet 
        RHS[Nperiphery*3:(Nperiphery+Nblobs)*3] = v_slip.flatten()

        # Singularity subtraction vectors
        offset = 0
        for k, b in enumerate(bodies):
          # b.calc_vectors_singularity_subtraction(read.eta, r_vectors_blobs[offset:offset+b.Nblobs], normals_blobs[offset:offset+b.Nblobs])
          b.calc_vectors_singularity_subtraction(read.eta)
          offset += b.Nblobs
    
        # Set linear operators 
        linear_operator_partial = partial(multi_bodies.linear_operator_rigid_SK,
                                          bodies=bodies,
                                          r_vectors=r_vectors_blobs,
                                          normals=normals_blobs,
                                          quadrature_weights=quadrature_weights_blobs,
                                          eta=read.eta,
                                          periphery = shell,
                                          K_bodies = None)
        A = spla.LinearOperator((System_size, System_size), matvec = linear_operator_partial, dtype='float64')

        # Set preconditioner
        # PC = None
        PC = multi_bodies.build_block_diagonal_preconditioner_SK(bodies, read.eta, Nblobs, periphery = shell)

        # Solve preconditioned linear system # callback=make_callback()
        counter = miscellaneous.gmres_counter(print_residual = args.verbose) 
        (sol_precond, info_precond) = gmres.gmres(A, RHS, tol=read.solver_tolerance, M=PC, maxiter=1000, restart=60, callback=counter) 
    
        # Extract velocities and constraint forces on blobs
        velocity = np.reshape(sol_precond[3*(Nperiphery+Nblobs): 3*(Nperiphery+Nblobs) + 6*num_bodies], (num_bodies, 6))
        lambda_blobs = np.reshape(sol_precond[0: 3*(Nperiphery+Nblobs)], (Nperiphery+Nblobs, 3))

        # Update
        for k, b in enumerate(bodies):
          quaternion_dt = quaternion.Quaternion.from_rotation(velocity[k, 6*k+3:6*k+6] * read.dt)
          b.orientation = quaternion_dt * b.orientation
          print(b.location.shape, velocity[k, 6*k:6*k+3].shape)
          b.location += velocity[k, 6*k:6*k+3] * read.dt

        with open(read.output_name + '.spheres.config', 'a') as f:
          orientation = bodies[0].orientation.entries
          f.write('1\n')
          f.write(str(bodies[0].location[0]) + ' ' + 
                  str(bodies[0].location[1]) + ' ' + 
                  str(bodies[0].location[2]) + ' ' + 
                  str(orientation[0]) + ' ' + 
                  str(orientation[1]) + ' ' + 
                  str(orientation[2]) + ' ' + 
                  str(orientation[3]) + '\n')



        step += 1
        # Plot velocity field
        if read.plot_velocity_field.size > 1: 
          print('plot_velocity_field')
          name = read.output_name + '.' + str(step) 
          plot_velocity_field(read.plot_velocity_field,
                              r_vectors_blobs, lambda_blobs,
                              read.eta,
                              name,
                              mobility_vector_prod_implementation = read.mobility_vector_prod_implementation,
                              bodies = bodies,
                              force_torque = force_torque,
                              normal = normals_blobs,
                              density = lambda_blobs)




        if False:
          velocity_error += sample * (velocity.flatten() - velocity_mean)**2 / (sample + 1)
          velocity_mean += (velocity.flatten() - velocity_mean) / (sample + 1)

      # Save velocities
      if False:
        velocity_error = np.sqrt(velocity_error) /  np.sqrt(sample_max * np.maximum(1.0,(sample_max-1.0)))
        result = np.zeros(25)
        result[0] = distance_i
        result[1:13] = velocity_mean
        result[13:] = velocity_error
        result = result.reshape((1,25))
        name = read.output_name + '.velocity_vs_distance.dat'
        status = 'a'
        if distance_i == distances[0]:
          status = 'w'
        with open(name, status) as f:
          np.savetxt(f, result, delimiter=' ')
        
    # Save velocity
    name = read.output_name + '.velocity.dat'
    np.savetxt(name, velocity, delimiter='  ')

    # Save lambda
    name = read.output_name + '.lambda.dat'
    np.savetxt(name, lambda_blobs, delimiter='  ')
    print('Time to solve mobility problem =', time.time() - start_time)
    

    
    # Plot velocity field
    if read.plot_velocity_field.size > 1: 
      print('plot_velocity_field')
      plot_velocity_field(read.plot_velocity_field,
                          r_vectors_blobs, lambda_blobs,
                          read.eta,
                          read.output_name,
                          mobility_vector_prod_implementation = read.mobility_vector_prod_implementation,
                          bodies = bodies,
                          force_torque = force_torque,
                          normal = normals_blobs,
                          density = lambda_blobs)
      
  # If scheme == resistance solve resistance problem 
  # elif read.scheme == 'resistance': 
  #   start_time = time.time() 
  #   # Get blobs coordinates 
  #   r_vectors_blobs = multi_bodies.get_blobs_r_vectors(bodies, Nblobs) 
    
  #   # Calculate block-diagonal matrix K
  #   K = multi_bodies.calc_K_matrix(bodies, Nblobs)

  #   # Set right hand side
  #   slip += multi_bodies.K_matrix_vector_prod(bodies, velocity, Nblobs) 
  #   RHS = np.reshape(slip, slip.size)
    
  #   # Calculate mobility (M) at the blob level
  #   mobility_blobs = multi_bodies.mobility_blobs(r_vectors_blobs, read.eta, read.blob_radius)

  #   # Compute constraint forces 
  #   force_blobs = np.linalg.solve(mobility_blobs, RHS)

  #   # Compute force-torques on bodies
  #   force = np.reshape(multi_bodies.K_matrix_T_vector_prod(bodies, force_blobs, Nblobs), (num_bodies, 6))
    
  #   # Save force
  #   name = read.output_name + '.force.dat'
  #   np.savetxt(name, force, delimiter='  ')
  #   print('Time to solve resistance problem =', time.time() - start_time)

  #   # Plot velocity field
  #   if read.plot_velocity_field.size > 1: 
  #     print('plot_velocity_field')
  #     lambda_blobs = np.reshape(force_blobs, (Nblobs, 3))
  #     plot_velocity_field(read.plot_velocity_field, r_vectors_blobs, lambda_blobs, read.blob_radius, read.eta, read.output_name, read.tracer_radius,
  #                         mobility_vector_prod_implementation = read.mobility_vector_prod_implementation)
  
  # elif read.scheme == 'body_mobility': 
  #   start_time = time.time()
  #   r_vectors_blobs = multi_bodies.get_blobs_r_vectors(bodies, Nblobs)
  #   mobility_blobs = multi_bodies.mobility_blobs(r_vectors_blobs, read.eta, read.blob_radius)
  #   resistance_blobs = np.linalg.inv(mobility_blobs)
  #   K = multi_bodies.calc_K_matrix(bodies, Nblobs)
  #   resistance_bodies = np.dot(K.T, np.dot(resistance_blobs, K))
  #   mobility_bodies = np.linalg.pinv(np.dot(K.T, np.dot(resistance_blobs, K)))
  #   name = read.output_name + '.body_mobility.dat'
  #   np.savetxt(name, mobility_bodies, delimiter='  ')
  #   print('Time to compute body mobility =', time.time() - start_time)


  print('\n\n\n# End')




