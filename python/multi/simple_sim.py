import numpy as np
import sys
sys.path.append('../')
from tstep import tstep_simple as tstep
from tstep import initialize_simple as initialize




if __name__ == '__main__':
  # MTs are clamped at their minus ends on to a periphery
  # There is a compressive force on MTs due to kinesins walking towards the plus end
  filename = 'output/simpleTest8/run'

  periphery = 'sphere' # shape of the periphery, could be 'sphere' or 'ellipsoid'
  periphery_radius = 3*1.04 # enter it if periphery is a sphere
  MT_length = 1 # it is the same for all MTs
  fiber_site = np.array([3, 0, 0])
  fiber_normal = np.array([-1, 0, 0])
  point_torque = None #np.array([0, 100, 0])
  point_torque_location = None #np.zeros((1,3))
  # MTs should be off the surface a bit, so enter geometric info about the MT surface
  mt_surface_radius = 3
  
  Nperiphery = 2000 # for dense 10000 #8000 Number of points to discretize periphery
  Nfiber = 32 # for dense 64 # Number of points discreting the MTs

  random_seed = 1 # initialization of rng
  dt = 1E-2 # for dense: 7.5E-3 # time step size
  ncompute_vel = 5000 # every ncompute_vel steps compute velocity field
  

  xrange = np.arange(-3, 3, 0.5)
  yrange = np.arange(-3, 3, 0.5)
  zrange = np.arange(-3, 3, 0.5)

  # if you define variable MOTOR density on MTs, give the axis along which the gradient is defined
  motor_strength_gradient_along_axis = None
  # if not None, then define the regions and the motor density
  # This can be done via giving regions discretely:
  motor_strength_gradient_regions = []
  motor_strength_gradient = []
  # In defining the regions: both points are included for the first entry,
  #     for the rest the first point is excluded (below has 0.5 for all.)
  #motor_strength_gradient_regions.append(np.array([-6.1,6.1]))
  motor_strength_gradient.append(0.4)
  # Or, you can give the motor strength as a function of the MTs nucleating site
  motor_strength_gradient_function = None #lambda x: 0.4*(5-x)

  

  #  time_step_scheme = 'fibers_and_bodies_and_periphery_hydro_implicit_time_step',
  # Following options can be set without the input file
  options = initialize.set_options(adaptive_num_points = False,
                                  num_points = Nfiber,
                                  adaptive_time = False,
                                  time_step_scheme = 'time_step_hydro',
                                  dt = dt,
                                  tol_tstep = 1e-2,
                                  tol_gmres = 1e-10,
                                  n_save = 10,
                                  output_name=filename,
                                  precompute_body_PC = True,
                                  useFMM = True,
                                  Nperiphery = Nperiphery,
                                  iCytoPulling = True,
                                  random_seed = random_seed)

  prams = initialize.set_parameters(eta = 1.0,
                                    Efib = 2.5E-3,
                                    final_time = 250000*dt,
                                    fiber_body_attached = False,
                                    periphery = periphery,
                                    periphery_radius = periphery_radius,
                                    mt_surface_radius = mt_surface_radius,
                                    MT_length = MT_length,
                                    fiber_site = fiber_site,
                                    fiber_normal = fiber_normal,
                                    point_torque = point_torque,
                                    point_torque_location = point_torque_location,
                                    motor_strength_gradient_function = motor_strength_gradient_function,
                                    motor_strength_gradient_along_axis = motor_strength_gradient_along_axis,
                                    motor_strength_gradient_regions = motor_strength_gradient_regions,
                                    motor_strength_gradient = motor_strength_gradient,
                                    ncompute_vel = ncompute_vel,
                                    xrange = xrange,
                                    yrange = yrange,
                                    zrange = zrange)

  # Create time stepping scheme
  tstep = tstep.tstep(prams,options, None,None,None)

  # Take time steps
  tstep.take_time_steps()


  print('\n\n\n# End')
