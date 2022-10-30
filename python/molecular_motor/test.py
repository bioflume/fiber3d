import numpy as np
import sys
sys.path.append('../')

import molecular_motor as mm
from fiber import fiber


if __name__ == '__main__':
  print('# Start')

  # Create molecular motors
  if True:
    r = np.random.rand(5,3)
    r[:,0] += 100
    r[0,:] = 0
    r[0,0] = 5.1
    radius = 0.5
    speed_0 = 2.0
    force_stall = 4.4
    spring_constant = 2.0
    rest_length = 0.5
    bind_frequency = 1000.0
    unbind_period_0 = 0.1
    kernel_sigma = 0.25
    mm_0 = mm.molecular_motor(r, radius, speed_0, force_stall, spring_constant, rest_length, bind_frequency, unbind_period_0, kernel_sigma)

  # Create fibers
  if True:
    fibers = []
    num_points = 32
    dt = 1e-03
    E = 10.0
    length = 10.0
    epsilon = 1e-03
    num_points_finite_diff = num_points
    fib = fiber.fiber(num_points = num_points, 
                      dt = dt, 
                      E=E, 
                      length = length, 
                      epsilon = epsilon, 
                      num_points_finite_diff = num_points_finite_diff)
    print('fiber.x = \n', fib.x)
    fibers.append(fib)


  # Test find_x_and_xs
  if True:
    print('=====================================================')
    print('Test find_x_and_xs')
    mm_0.attached_base[0:2] = -1
    mm_0.attached_head[1:3] = 0
    mm_0.s_head[:] = 5.0
    print('mm_0.attached_base = \n', mm_0.attached_base)
    print('mm_0.attached_head = \n', mm_0.attached_head)
    for fib in fibers:
      fib.compute_modes()
    mm_0.find_x_xs_and_length_MT(fibers)
    print('mm_0.x_base = \n', mm_0.x_base)
    print('mm_0.x_head = \n', mm_0.x_head)
    print('mm_0.xs_base = \n', mm_0.xs_base)
    print('mm_0.xs_head = \n', mm_0.xs_head)
    print('\n\n')

  # Test compute force
  if False:
    print('=====================================================')
    print('Test compute force')
    print('mm_0.force = \n', mm_0.force)
    mm_0.compute_force()
    print('mm_0.force = \n', mm_0.force)
    mm_0.force[1,:] = 0.0
    mm_0.force[1,0] = -2.0
    mm_0.force[1,1] = 2.0
    mm_0.force[1,2] = 3.0
    
  # Test walking
  if True:
    print('=====================================================')
    print('Test walking')
    dt = 0.01
    print('mm_0.s_base = \n', mm_0.s_base)
    print('mm_0.s_head = \n', mm_0.s_head)
    mm_0.walk(dt)
    print('mm_0.s_base = \n', mm_0.s_base)
    print('mm_0.s_head = \n', mm_0.s_head)
    print('\n\n')

  # Test diffusion
  if False:
    print('=====================================================')
    print('Test diffusion')
    print('x = \n', mm_0.x)
    mm_0.diffuse(dt)
    print('x = \n', mm_0.x)
    print('\n\n')
    
  # Test force spreading
  if False:
    print('=====================================================')
    print('Test force spreading')
    # print('fibers[0].force_motors = \n', fibers[0].force_motors)
    mm_0.spread_force(fibers)
    # print('fibers[0].force_motors = \n', fibers[0].force_motors)
    force = np.zeros((fibers[0].num_points, 4))
    force[:,0] = fibers[0].s
    force[:,1:] = fibers[0].force_motors
    np.savetxt('kk.dat', force)
    sum_f = np.sum(fibers[0].force_motors * fibers[0].weights[:,None], axis=0)
    print('sum_= ', sum_f)
    print('F_m = ', mm_0.force[1])


    print('\n\n')

  # Update links
  if True:
    print('=====================================================')
    print('Test update links')
    mm_0.attached_base[:] = -1
    print('force_motors = \n', np.linalg.norm(mm_0.force, axis=1))
    print('mm_0.attached_base = \n', mm_0.attached_base)
    print('mm_0.attached_head = \n', mm_0.attached_head)
    dt = 0.1
    mm_0.update_links_numba(dt, fibers)
    print('after call')
    print('mm_0.attached_base = \n', mm_0.attached_base)
    print('mm_0.attached_head = \n', mm_0.attached_head)
    print('x = \n', mm_0.x)
    print('mm_0.s_base = \n', mm_0.s_base)
    print('mm_0.s_head = \n', mm_0.s_head)

  print('# End')


