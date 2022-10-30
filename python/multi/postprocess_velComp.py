import numpy as np
import sys
sys.path.append('../')
from tstep import time_step_container
from tstep import initialize_post as initialize
from tstep import postprocess_velocity


if __name__ == '__main__':
  
  runID = sys.argv[1] 
  nskip = int(sys.argv[2]) # multiplicative of 50
  numRes = int(sys.argv[3])

  output_file = '/mnt/ceph/users/gkabacaoglu/oocyteData/sphereRuns/run' + sys.argv[1] + '_velocity/run'

  time_step_file, fibers_file = [], []

  for ires in np.arange(numRes):
    if ires == 0:
      filename = '/mnt/ceph/users/gkabacaoglu/oocyteData/sphereRuns/run' + sys.argv[1] + '/run_fibers_fibers.txt'
      fibers_file.append(filename)
      filename = '/mnt/ceph/users/gkabacaoglu/oocyteData/sphereRuns/run' + sys.argv[1] + '/run_time_system_size.txt'
      time_step_file.append(filename)
    else:
      filename = '/mnt/ceph/users/gkabacaoglu/oocyteData/sphereRuns/run' + sys.argv[1] + '_res' +str(ires) + '/run_fibers_fibers.txt'
      fibers_file.append(filename)
      filename = '/mnt/ceph/users/gkabacaoglu/oocyteData/sphereRuns/run' + sys.argv[1] + '_res' +str(ires) + '/run_time_system_size.txt'
      time_step_file.append(filename)
    

  periphery = 'sphere' #'sphere' # shape of the periphery, could be 'sphere' or 'ellipsoid'
  periphery_radius = 1.04*5.0 # enter it if periphery is a sphere
  MT_length = 1 # it is the same for all MTs

  periphery_radius_x = None #5.2 #6.24 #7.8 # Enter this and the following two, if periphery is ellipsoid (should be larger than attachment)
  periphery_radius_y = None
  periphery_radius_z = None

  # MTs should be off the surface a bit, so enter geometric info about the MT surface
  mt_surface_radius = 5.0
  mt_surface_radius_a = None #5 #6 #7.5
  mt_surface_radius_b = None
  mt_surface_radius_c = None

  Nperiphery = 10000 # for dense 10000 #8000 Number of points to discretize periphery
  Nfiber = 64 # for dense 64 # Number of points discreting the MTs
  dt = 1E-2 # for dense: 7.5E-3 # time step size

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
  motor_strength_gradient.append(float(sys.argv[4]))
  motor_strength_gradient_function = None #lambda x: 0.4*(5-x)


  #  time_step_scheme = 'fibers_and_bodies_and_periphery_hydro_implicit_time_step',
  # Following options can be set without the input file
  options = initialize.set_options(adaptive_num_points = False,
                                  num_points = Nfiber,
                                  time_step_scheme = 'time_step_hydro',
                                  dt = dt,
                                  iupsample = True,
                                  tol_gmres = 1e-10,
                                  output_name=output_file,
                                  precompute_body_PC = True,
                                  useFMM = True,
                                  Nperiphery = Nperiphery,
                                  iCytoPulling = True,
                                  n_save = 50)

  prams = initialize.set_parameters(eta = 1.0,
                                    Efib = 2.5E-3,
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
                                    motor_strength_gradient_function = motor_strength_gradient_function,
                                    motor_strength_gradient_along_axis = motor_strength_gradient_along_axis,
                                    motor_strength_gradient_regions = motor_strength_gradient_regions,
                                    motor_strength_gradient = motor_strength_gradient,
                                    xrange = xrange,
                                    yrange = yrange,
                                    zrange = zrange,
                                    fibers_file = fibers_file,
                                    time_step_file = time_step_file,
                                    nskip_steps = nskip)


  # Initialize the files
  time_steps, time_all, nsteps_all = initialize.initialize_from_file(options, prams)

  # Create postprocessing object
  postprocess = postprocess_velocity.postprocess_velocity(prams, options, time_steps, nsteps_all, time_all)

  # Compute velocities
  postprocess.take_time_steps()
