import numpy as np
import sys
sys.path.append('../')
from tstep import tstep_rotlet as tstep
from tstep import initialize_simple as initialize




if __name__ == '__main__':
  # MTs are clamped at their minus ends on to a periphery
  # There is a compressive force on MTs due to kinesins walking towards the plus end
  filename = 'output/' + sys.argv[1] + '/run'
  rotlet_off_time = 10
  rotlet_torque = np.array([0, float(sys.argv[6]), 0])
  periphery = 'ellipsoid' #'sphere' # shape of the periphery, could be 'sphere' or 'ellipsoid'
  periphery_radius = None # enter it if periphery is a sphere
  MT_length = 1 # it is the same for all MTs

  periphery_radius_x = 1.04*float(sys.argv[2]) #5.2 #6.24 #7.8 # Enter this and the following two, if periphery is ellipsoid (should be larger than attachment)
  periphery_radius_y = 1.04*float(sys.argv[3])
  periphery_radius_z = 1.04*float(sys.argv[4])

  # MTs should be off the surface a bit, so enter geometric info about the MT surface
  mt_surface_radius = None
  mt_surface_radius_a = float(sys.argv[2]) #5 #6 #7.5
  mt_surface_radius_b = float(sys.argv[3])
  mt_surface_radius_c = float(sys.argv[4])

  Nperiphery = 8000 #8000 Number of points to discretize periphery
  Nfiber = 32 # Number of points discreting the MTs

  random_seed = 1 # initialization of rng
  dt = 1E-2 # time step size
  ncompute_vel = 100 # every ncompute_vel steps compute velocity field
  if periphery is 'ellipsoid':
    radius_x, radius_y, radius_z = mt_surface_radius_a, mt_surface_radius_b, mt_surface_radius_c
  elif periphery is 'sphere':
    radius_x, radius_y, radius_z = mt_surface_radius, mt_surface_radius, mt_surface_radius
  dx, dy, dz = 0.25, 0.25, 0.25 # spacing between grid points in each direction
  xUpperLim, xlowerLim, stepInX = radius_x-0.75*MT_length+dx, -radius_x+0.75*MT_length, dx
  yUpperLim, ylowerLim, stepInY = radius_y-0.75*MT_length+dy, -radius_y+0.75*MT_length, dy
  zUpperLim, zlowerLim, stepInZ = radius_z-0.75*MT_length+dz, -radius_z+0.75*MT_length, dz

  xrange = np.arange(xlowerLim, xUpperLim, stepInX)
  yrange = np.arange(ylowerLim, yUpperLim, stepInY)
  zrange = np.arange(zlowerLim, zUpperLim, stepInZ)

  # if you define variable MOTOR density on MTs, give the axis along which the gradient is defined
  motor_strength_gradient_along_axis = 'uniform' 
  # if not None, then define the regions and the motor density
  # This can be done via giving regions discretely:
  motor_strength_gradient_regions = []
  motor_strength_gradient = []
  # In defining the regions: both points are included for the first entry,
  #     for the rest the first point is excluded (below has 0.5 for all.)
  #motor_strength_gradient_regions.append(np.array([-6.1,6.1]))
  motor_strength_gradient.append(0.05)
  # Or, you can give the motor strength as a function of the MTs nucleating site
  motor_strength_gradient_function = None #lambda x: 0.4*(5-x)

  # if you define the number of MTs varying in space, define the axis along which it changes
  Nmts_gradient_along_axis = 'x' # or None
  Nmts_regions = [] # define the regions (the first point included, the second excluded)
  Nmts_in_each_region = [] # enter the number of MTs in each region defined
  Nmts_regions.append(np.array([-6.1, 6.1]))
  #Nmts_regions.append(np.array([-5, 5]))
  Nmts_in_each_region.append(int(sys.argv[5]))
  #Nmts_regions.append(np.array([0, 5]))
  #Nmts_in_each_region.append(500)
  # the above example results in 1500 MTs in x = [-5,0), and 500 MTs in [0, 5]

  #  time_step_scheme = 'fibers_and_bodies_and_periphery_hydro_implicit_time_step',
  # Following options can be set without the input file
  options = initialize.set_options(adaptive_num_points = False,
                                  num_points = Nfiber,
                                  adaptive_time = False,
                                  time_step_scheme = 'time_step_hydro',
                                  dt = dt,
                                  tol_tstep = 1e-2,
                                  tol_gmres = 1e-10,
                                  n_save = 50,
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
                                    periphery_a = periphery_radius_x,
                                    periphery_b = periphery_radius_y,
                                    periphery_c = periphery_radius_z,
                                    mt_surface_radius = mt_surface_radius,
                                    mt_surface_radius_a = mt_surface_radius_a,
                                    mt_surface_radius_b = mt_surface_radius_b,
                                    mt_surface_radius_c = mt_surface_radius_c,
                                    MT_length = MT_length,
                                    Nmts_gradient_along_axis = Nmts_gradient_along_axis,
                                    Nmts_regions = Nmts_regions,
                                    Nmts_in_each_region = Nmts_in_each_region,
                                    motor_strength_gradient_function = motor_strength_gradient_function,
                                    motor_strength_gradient_along_axis = motor_strength_gradient_along_axis,
                                    motor_strength_gradient_regions = motor_strength_gradient_regions,
                                    motor_strength_gradient = motor_strength_gradient,
                                    ncompute_vel = ncompute_vel,
                                    xrange = xrange,
                                    yrange = yrange,
                                    zrange = zrange,
                                    rotlet_off_time = rotlet_off_time,
                                    rotlet_torque = rotlet_torque)

  # Create time stepping scheme
  tstep = tstep.tstep(prams,options, None,None,None)

  # Take time steps
  tstep.take_time_steps()


  print('\n\n\n# End')
