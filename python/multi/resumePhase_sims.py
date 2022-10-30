import numpy as np
import sys
sys.path.append('../')
from tstep import tstep_simple as tstep
from tstep import initialize_simple as initialize




if __name__ == '__main__':
  # MTs are clamped at their minus ends on to a periphery
  # There is a compressive force on MTs due to kinesins walking towards the plus end
  filename = '/mnt/ceph/users/gkabacaoglu/oocyteData/sphereRuns/' + sys.argv[1] + '_res3/run'
  fiber_resume = '/mnt/ceph/users/gkabacaoglu/oocyteData/sphereRuns/' + sys.argv[1] + '_res2/run_fibers_resume.fibers'

  periphery = 'sphere' #'sphere' # shape of the periphery, could be 'sphere' or 'ellipsoid'
  periphery_radius = 1.04*float(sys.argv[2]) # enter it if periphery is a sphere
  MT_length = 1 # it is the same for all MTs

  periphery_radius_x = None #5.2 #6.24 #7.8 # Enter this and the following two, if periphery is ellipsoid (should be larger than attachment)
  periphery_radius_y = None
  periphery_radius_z = None

  # MTs should be off the surface a bit, so enter geometric info about the MT surface
  mt_surface_radius = float(sys.argv[2])
  mt_surface_radius_a = None #5 #6 #7.5
  mt_surface_radius_b = None
  mt_surface_radius_c = None

  Nperiphery = 8000 # for dense 10000 #8000 Number of points to discretize periphery
  Nfiber = 32 # for dense 64 # Number of points discreting the MTs

  random_seed = 1 # initialization of rng
  dt = 1E-2 # for dense: 7.5E-3 # time step size
  ncompute_vel = 1000 # every ncompute_vel steps compute velocity field
  
  dx, dy, dz = 0.25, 0.25, 0.25 # spacing between grid points in each direction
  radius_xp, radius_xm = mt_surface_radius, -mt_surface_radius
  radius_yp, radius_ym = mt_surface_radius, -mt_surface_radius
  radius_zp, radius_zm = mt_surface_radius, -mt_surface_radius

  xUpperLim, xlowerLim, stepInX = radius_xp+dx, radius_xm, dx
  yUpperLim, ylowerLim, stepInY = radius_yp+dy, radius_ym, dy
  zUpperLim, zlowerLim, stepInZ = radius_zp+dz, radius_zm, dz

  xrange = np.arange(xlowerLim, xUpperLim, stepInX)
  yrange = np.arange(ylowerLim, yUpperLim, stepInY)
  zrange = np.arange(zlowerLim, zUpperLim, stepInZ)
  xx,yy,zz = np.meshgrid(xrange,yrange,zrange,sparse=False,indexing='ij')
  xrange = xx.flatten()
  yrange = yy.flatten()
  zrange = zz.flatten()

  # if you define variable MOTOR density on MTs, give the axis along which the gradient is defined
  motor_strength_gradient_along_axis = None 
  # if not None, then define the regions and the motor density
  # This can be done via giving regions discretely:
  motor_strength_gradient_regions = []
  motor_strength_gradient = []
  # In defining the regions: both points are included for the first entry,
  #     for the rest the first point is excluded (below has 0.5 for all.)
  #motor_strength_gradient_regions.append(np.array([-6.1,6.1]))
  motor_strength_gradient.append(float(sys.argv[3]))
  # Or, you can give the motor strength as a function of the MTs nucleating site
  motor_strength_gradient_function = None #lambda x: 0.4*(5-x)

  # if you define the number of MTs varying in space, define the axis along which it changes
  Nmts_gradient_along_axis = None
  Nmts_regions = [] # define the regions (the first point included, the second excluded)
  Nmts_in_each_region = [] # enter the number of MTs in each region defined
  # Nmts_regions.append(np.array([-6.1, 6.1]))
  #Nmts_regions.append(np.array([-5, 5]))
  # Nmts_in_each_region.append(int(sys.argv[4]))
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
                                    fiber_resume = fiber_resume)

  # Create time stepping scheme
  tstep = tstep.tstep(prams,options, None,None,None)

  # Take time steps
  tstep.take_time_steps()


  print('\n\n\n# End')
