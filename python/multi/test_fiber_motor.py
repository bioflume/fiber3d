import numpy as np
import sys
import scipy.linalg as scla
sys.path.append('./../')
from fiber import fiber
from tstep import tstep
from tstep import initialize
from tstep import fiber_matrices
from molecular_motor import molecular_motor_simple as molecular_motor

if __name__ == '__main__':
  print('# Start')
  # Set parameters
  num_points = 32
  length = 10.0
  mm_radius = 0.5
  output_name = 'data/fiber_motor/run'
  dt = 0.001
  E = 10.0
  t_max = 10*dt
  n_save = 1
  num_points_finite_diff = 4
  
  

  # Build fiber matrices
  fib_mat = fiber_matrices.fiber_matrices(num_points = num_points, num_points_finite_diff = num_points_finite_diff)
  fib_mat.compute_matrices()
  alpha, s, output3, output4 = fib_mat.get_matrices(length,num_points_finite_diff,'alpha_and_s')
  D_1, D_2, D_3, D_4 = fib_mat.get_matrices(length,num_points_finite_diff,'Ds')
  
  # Create fibers
  fibers = []
  fib = fiber.fiber(num_points = num_points,
                    num_points_max = 256,
                    num_points_finite_diff = num_points_finite_diff,
                    dt = dt,
                    E = E,
                    length = length,
                    adaptive_num_points = False,
                    ireparam = False,
                    inonlocal = False,
                    growing = 0,
                    viscosity = 1)
  fib.x[:,0] = 0
  fib.x[:,1] = 0
  fib.x[:,2] = s
  fib.s = s
  fib.v_growth = 0
  fib.v_shrink = 0
  fib.rate_catastrophe = 0
  fib.rate_rescue = 0
  fib.rate_seed = 0
  fib.rate_catastrophe_stall = 0
  fib.force_stall = 0
  fib.length_min = 0.3
  fib.set_BC(BC_start_0 = 'velocity',BC_start_1 = 'angular_velocity',
    BC_end_0 = 'force', BC_end_1 = 'torque')
    
  fib.xs = np.dot(D_1, fib.x)
  fib.xss = np.dot(D_2, fib.x)
  fib.xsss = np.dot(D_3, fib.x)
  fib.xssss = np.dot(D_4, fib.x)

  fibers.append(fib)

  # Create a motor
  r_MM = np.array([[0, mm_radius, 3/4*fib.length],
                   [0, mm_radius/2, 1/2*fib.length],
                   [0, 3*mm_radius/2, 1/4*fib.length],
                   [mm_radius/2, mm_radius/2, 2/3*fib.length]])
  radius = mm_radius
  speed_0 = None
  force_stall = 4.0
  spring_constant = None
  rest_length = None
  bind_frequency = 1000.0
  unbind_frequency_0 = 0.1
  kernel_sigma = 0.25
  MM = molecular_motor.molecular_motor(r_MM, radius, speed_0, force_stall, spring_constant, rest_length, bind_frequency, unbind_frequency_0, kernel_sigma)
  attached_ends = np.zeros((MM.N, 2))
  attached_ends[:,0] = -1
  attached_ends[:,1] = -2
  MM.attached_base[:] = attached_ends[:,0]
  MM.attached_head[:] = attached_ends[:,1]

  # Initialize files
  name = output_name + '_fiber_x.txt'
  f_fiber = open(name, 'wb', buffering = 100)

  fiber_info = np.array([fibers[0].num_points,
    fibers[0].length,
    fibers[0].length_min])
  np.savetxt(f_fiber, fiber_info[None,:])

  name = output_name + '_fiber_force.txt'
  f_fiber_force = open(name, 'wb', buffering = 100)
  np.savetxt(f_fiber_force, fiber_info[None,:])

  MM_info = np.array([MM.N, MM.radius, 0])
  name = output_name + '_MM_xbase.txt'
  f_MM_xbase = open(name, 'wb', buffering = 100)
  np.savetxt(f_MM_xbase,MM_info[None,:])

  name = output_name + '_MM_xhead.txt'
  f_MM_xhead = open(name, 'wb', buffering = 100)
  np.savetxt(f_MM_xbase,MM_info[None,:])

  MM_info = np.array([MM.N, MM.radius])
  name = output_name + '_MM_attached_ends.txt'
  f_MM_attached = open(name, 'wb', buffering = 100)
  np.savetxt(f_MM_attached,MM_info[None,:])


  # TAKE TIME STEPS
  nsteps, current_time = 0, 0.0
  while current_time <= t_max:
    current_time += dt
    nsteps += 1


    # GET FIBER DERIVATIVES
    fibers[0].xs = np.dot(D_1, fibers[0].x)
    fibers[0].xss = np.dot(D_2, fibers[0].x)
    fibers[0].xsss = np.dot(D_3, fibers[0].x)
    fibers[0].xssss = np.dot(D_4, fibers[0].x)

    # MM steps
    fibers[0].compute_modes()
    MM.find_x_xs_and_length_MT(fibers)
    #print('After find_x_xs ..., s_head, x_head, attached_head', MM.s_head,
    #    MM.x_head, MM.attached_head)
    MM.compute_force()
    fibers[0].force_motors[:,:] = 0.0
    MM.spread_force(fibers)
    force_fibers = fibers[0].force_motors
    MM.walk(dt)
    #print('After walking, s_head, x_head, attached_head', MM.s_head, MM.x_head,
    #    MM.attached_head)
    MM.update_links_numba(dt, fibers)
    #print('After update_links, s_head, x_head, attached_head', MM.s_head,
    #     MM.x_head, MM.attached_head)
    MM.diffuse(dt)
    #print('After diffuse, s_head, x_head, attached_head', MM.s_head, MM.x_head,
    #    MM.attached_head)
    #input()

    # WRITE TO FILE
    np.savetxt(f_fiber,fibers[0].x)
    np.savetxt(f_fiber_force,force_fibers)
    np.savetxt(f_MM_xbase, MM.x_base)
    np.savetxt(f_MM_xhead, MM.x_head)
    attached_ends = np.hstack([[MM.attached_base, MM.attached_head]])
    np.savetxt(f_MM_attached, np.transpose(attached_ends))

    # BUILD MATRIX FOR MOBILITY
    flow_on = np.zeros((fib.num_points,3))
    force_on = np.zeros((fib.num_points,3))
    force_on += fibers[0].force_motors
    A = fibers[0].form_linear_operator(fib_mat, inextensibility = 'penalty')
    RHS = fibers[0].compute_RHS(fib_mat = fib_mat, force_external = force_fibers, 
      flow = flow_on, inextensibility = 'penalty')
    A, RHS = fibers[0].apply_BC_rectangular(A, RHS, fib_mat,flow_on,force_on)

    # FACTORIZE THE MATRIX
    LU, P = scla.lu_factor(A,check_finite=False)

    # SOLVE THE SYSTEM
    sol = scla.lu_solve((LU, P), RHS)

    fibers[0].x[:,0] = sol[0*fibers[0].num_points : 1*fibers[0].num_points]
    fibers[0].x[:,1] = sol[1*fibers[0].num_points : 2*fibers[0].num_points]
    fibers[0].x[:,2] = sol[2*fibers[0].num_points : 3*fibers[0].num_points]

    # Check error
    xs = np.dot(D_1, fibers[0].x)
    error = abs(max(np.sqrt(xs[:,0]**2 + xs[:,1]**2 + xs[:,2]**2) - 1.0, key=abs))
    print('**********************************')
    print('Time step: ', nsteps)
    print('attached_head: ', MM.attached_head)
    print('Motor x_head: ', MM.x_head)
    print('Motor s_head: ', MM.s_head)
    print('Error in inextensibility: ', error)
    input() 




