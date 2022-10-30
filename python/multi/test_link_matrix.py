import numpy as np
import sys
import time
sys.path.append('../')
from fiber import fiber
from body import body
from tstep import fiber_matrices
from quaternion import quaternion
import util_funcs

if __name__ == '__main__':

  num_fibers = 200 # num. fibers
  mean_fib_length = 10 # microns

  num_nuc_sites = 1000 # num. nucleating sites
  body_radius = 0.5
  body_center = np.array([0., 0., 0.])

  # Create body
  bodies = []
  orientation = quaternion.Quaternion([1., 0., 0., 0.])
  b = body.Body(body_center,orientation,np.array([0., 0., 0.]), np.array([0., 0., 0.]),np.ones(1))
  b.radius = 0.5
  # nucleating site file
  b.nuc_sites = util_funcs.create_nuc_sites_uniform(body_radius, num_nuc_sites)
  b.active_sites_idcs = []
  b.passive_sites_idcs = np.arange(b.nuc_sites.size//3).tolist()
  bodies.append(b)
  center, orientation = bodies[0].location, bodies[0].orientation
  rotation_matrix = orientation.rotation_matrix()

  # Create fibers randomly on the body
  fibers = []
  for ifib in range(num_fibers):
    # sample fiber length from exponential distribution
    fibL = np.random.exponential(mean_fib_length)
    while (fibL <= 0.5): fibL = np.random.exponential(mean_fib_length)
    # Given the length calculate the number of points to discretize fiber
    num_points = min(max(int(2**np.floor(np.log2(16 * np.sqrt(fibL/0.5)))),16), 196)

    # Pick a nucleating site
    idx_in_passive = np.random.randint(len(bodies[0].passive_sites_idcs))
    idx_in_all = bodies[0].passive_sites_idcs[idx_in_passive]
    ilink = bodies[0].nuc_sites[idx_in_all]
    site_location = np.dot(rotation_matrix, ilink)
    site_normal = ilink / np.linalg.norm(ilink)
    axis = np.dot(rotation_matrix, site_normal)
    s = np.linspace(0, 2, num_points)
    axis_s = np.empty((s.size, 3))


    # Draw fiber config
    r = np.random.rand()
    if r < 0.5:
      # then disturbed fiber
      axis_s[:,0] = (fibL/np.pi) * np.sin(s * np.pi / 2 * 0.2)
      axis_s[:,2] = (fibL/np.pi) * np.cos(s * np.pi / 2 * 0.2)
    else:
      axis_s[:,0] = axis[0] * s * fibL / 2
      axis_s[:,1] = axis[1] * s * fibL / 2
      axis_s[:,2] = axis[2] * s * fibL / 2

    fib = fiber.fiber(num_points = num_points,
                      E = 20,
                      length = fibL,
                      num_points_finite_diff = 4,
                      BC_start_0='velocity',
                      BC_start_1='angular_velocity')
    fib.x = axis_s + site_location + center
    fib.attached_to_body = 0
    fib.nuc_site_idx = idx_in_all
    fibers.append(fib)

  # Get fiber offsets and also build fiber_matrices
  fib_mats, fib_mat_resolutions = [], np.array([])
  offset_fibers = np.zeros(len(fibers)+1, dtype=int)
  for fib_idx, fib in enumerate(fibers):
    offset_fibers[fib_idx+1] = offset_fibers[fib_idx] + fib.num_points
    if fib.num_points not in fib_mat_resolutions:
      fib_mat = fiber_matrices.fiber_matrices(num_points = fib.num_points, num_points_finite_diff = fib.num_points_finite_diff)
      fib_mat.compute_matrices()
      fib_mats.append(fib_mat)
      fib_mat_resolutions = np.append(fib_mat_resolutions, fib.num_points)


  # Generate random unknowns for testing
  body_velocities = np.random.rand(2,3)
  fibers_xt = np.zeros(4*offset_fibers[-1])
  for fib_idx, fib in enumerate(fibers):
    Nfibs = fib.num_points
    istart = 4*offset_fibers[fib_idx]
    fibers_xt[istart + 0*Nfibs : istart + 1*Nfibs] = fib.x[:,0]
    fibers_xt[istart + 1*Nfibs : istart + 2*Nfibs] = fib.x[:,1]
    fibers_xt[istart + 2*Nfibs : istart + 3*Nfibs] = fib.x[:,2]
    fibers_xt[istart + 3*Nfibs : istart + 4*Nfibs] = np.random.rand(Nfibs)

  body_vel_xt = np.concatenate((body_velocities.flatten(),fibers_xt.flatten()), axis=0)
  # BUILD THE MATRIX FORM OF THE BCs and DO the MAT_VEC
  ti_matrix = time.time()
  As_BC = util_funcs.build_link_matrix(4*offset_fibers[-1] + 6, bodies,
                                       fibers, offset_fibers, 6, fib_mats,
                                       fib_mat_resolutions)
  mat_vec = As_BC.dot(body_vel_xt)
  tf_matrix = time.time()
  force_torque_on_body_matvec = mat_vec[:6]
  all_velocities_on_fiber_matvec = mat_vec[6:] # This includes all fiber points


  # CALCULATE VIA LOOP
  ti_loop = time.time()
  force_torque, velocities_on_fiber_loop = util_funcs.calculate_body_fiber_link_conditions(fibers,
                                            bodies, body_velocities, fibers_xt,
                                            offset_fibers, fib_mats, fib_mat_resolutions)


  tf_loop = time.time()
  force_torque_on_body_loop = np.zeros((1,6))
  # We need to sum up the contributions from each fiber,
  # now assuming that we compute force-torque on body indx 0
  for ft in force_torque:
    if ft[0] == 0: force_torque_on_body_loop += ft[1:]

  # For comparison, we need to do one more thing:
  # only boundary condition velocities need to be extracted from all_velocities_on_fiber_matvec
  velocities_on_fiber_matvec = np.zeros((num_fibers,7))
  for fib_idx, fib in enumerate(fibers):
    istart = 4*offset_fibers[fib_idx]+fib.num_points*4
    velocities_on_fiber_matvec[fib_idx,:] = all_velocities_on_fiber_matvec[istart-14:istart-7]      

  # Calculate the errors
  err_in_body = np.linalg.norm(force_torque_on_body_matvec-force_torque_on_body_loop)
  err_in_body *= 1/np.linalg.norm(force_torque_on_body_matvec)

  err_in_fiber = np.linalg.norm(velocities_on_fiber_matvec-velocities_on_fiber_loop)
  err_in_fiber *= 1/np.linalg.norm(velocities_on_fiber_matvec)
  
  #print('On body (matrix): ', force_torque_on_body_matvec)
  #print('On body (loop): ', force_torque_on_body_loop)
  print('Error in body part:', err_in_body)
  print('Error in fiber part:', err_in_fiber)
  print('Matrix-version took ', tf_matrix-ti_matrix, ' seconds')
  print('Loop-version took ', tf_loop-ti_loop, ' seconds')
