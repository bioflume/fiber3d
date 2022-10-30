import numpy as np
import sys
sys.path.append('../')
from tstep import tstep_simple as tstep
from tstep import initialize_simple as initialize
import scipy.io as scio

from read_input import read_fibers_file

if __name__ == '__main__':
  # MTs are clamped at their minus ends on to a periphery
  # There is a compressive force on MTs due to kinesins walking towards the plus end
  filename = '/mnt/ceph/users/gkabacaoglu/oocyteData/sphereRuns/velocity/' + sys.argv[1] + '/run'
  fiber_resume = '/mnt/ceph/users/gkabacaoglu/oocyteData/sphereRuns/' + sys.argv[1] + '_res1/run_fibers_resume.fibers'

  periphery = 'sphere' #'sphere' # shape of the periphery, could be 'sphere' or 'ellipsoid'
  periphery_radius = 1.04*float(sys.argv[2])
  mt_surface_radius = float(sys.argv[2])

  MT_length = 1 # it is the same for all MTs

  Nperiphery = 8000
  Nfiber = 96 # Number of points discreting the MTs

  dt = 1E-3 # time step size
  ncompute_vel = 1# every ncompute_vel steps compute velocity field
  
  dx, dy, dz = 0.1, 0.1, 0.1 # spacing between grid points in each direction
  radius_xp, radius_xm = float(sys.argv[2]), -float(sys.argv[2])
  radius_yp, radius_ym = float(sys.argv[2]), -float(sys.argv[2])
  radius_zp, radius_zm = float(sys.argv[2]), -float(sys.argv[2])

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
  fibers_info, fibers_coor = read_fibers_file.read_fibers_file(fiber_resume)
  offset, idx = 0, 0
  for i in range(len(fibers_info)):
    if len(fibers_info[i]) > 0: num_points = fibers_info[i][0]
    if len(fibers_info[i]) > 2: length = fibers_info[i][2]

    fib_x = fibers_coor[offset + num_points-1]
    
    normx = fib_x / np.linalg.norm(fib_x)
    xq = fib_x - 0.3 * normx

    xrange = np.append(xrange,xq[0])
    yrange = np.append(yrange,xq[1])
    zrange = np.append(zrange,xq[2])

    offset += num_points
    
  # if you define variable MOTOR density on MTs, give the axis along which the gradient is defined
  motor_strength_gradient_along_axis = None
  # if not None, then define the regions and the motor density
  # This can be done via giving regions discretely:
  motor_strength_gradient_regions = []
  motor_strength_gradient = []
  # In defining the regions: both points are included for the first entry,
  #     for the rest the first point is excluded (below has 0.5 for all.)
  motor_strength_gradient.append(float(sys.argv[3]))
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
                                  n_save = 1,
                                  output_name=filename,
                                  precompute_body_PC = True,
                                  useFMM = True,
                                  Nperiphery = Nperiphery,
                                  iCytoPulling = True,
                                  random_seed = 1)

  prams = initialize.set_parameters(eta = 1.0,
                                    Efib = 2.5E-3,
                                    final_time = 2*dt,
                                    fiber_body_attached = False,
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
