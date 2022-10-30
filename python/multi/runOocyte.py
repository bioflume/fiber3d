import numpy as np
import sys
sys.path.append('../')
from tstep import tstep_oocyte as tstep
from tstep import initialize_oocyte as initialize
import scipy.io as scio



if __name__ == '__main__':
  # MTs are clamped at their minus ends on to a periphery
  # There is a compressive force on MTs due to kinesins walking towards the plus end
  filename = '/mnt/ceph/users/gkabacaoglu/oocyteData/mts_on_oocyte_normal/run'
  cortex_file = './normalMTs.mat'
  cortex_file2 = './eggShapeFine.mat'
  data = scio.loadmat(cortex_file)
  data2 = scio.loadmat(cortex_file2)
  fiber_sites = data['xyz_mt'] # or fiberSites2p5K or fiberSites3K
  fiber_norms = data['norms']
  cortex_xyz = data2['xyz_periphery']
  cortex_norms = data2['norms']
  Lx = np.max(cortex_xyz[:,0]) - np.min(cortex_xyz[:,0])
  xcut_mts = -2 
  #xyz2 = cortex_xyz/Lx * 7.5
  #cortex_xyz2L = cortex_xyz/Lx * 7.8

  # ooCyte fine has 2K fibers
  #idcs = np.random.choice(np.arange(xyz2.size//3),3000,replace=False)
  #fiber_sites = xyz2[idcs]
  #fiber_norms = cortex_norms[idcs]

  MT_length = 1 # it is the same for all MTs

  Nperiphery = cortex_xyz.size//3
  Nfiber = 64 # Number of points discreting the MTs

  dt = 5E-3 # time step size
  ncompute_vel = 5000 # every ncompute_vel steps compute velocity field
  
  dx, dy, dz = 0.25, 0.25, 0.25 # spacing between grid points in each direction
  radius_xp, radius_xm = np.max(cortex_xyz[:,0]), np.min(cortex_xyz[:,0])
  radius_yp, radius_ym = np.max(cortex_xyz[:,1]), np.min(cortex_xyz[:,1])
  radius_zp, radius_zm = np.max(cortex_xyz[:,2]), np.min(cortex_xyz[:,2])

  xUpperLim, xlowerLim, stepInX = radius_xp - 0.75*MT_length+dx, radius_xm + 0.75*MT_length, dx
  yUpperLim, ylowerLim, stepInY = radius_yp - 0.75*MT_length+dy, radius_ym + 0.75*MT_length, dy
  zUpperLim, zlowerLim, stepInZ = radius_zp - 0.75*MT_length+dz, radius_zm + 0.75*MT_length, dz

  xrange = np.arange(xlowerLim, xUpperLim, stepInX)
  yrange = np.arange(ylowerLim, yUpperLim, stepInY)
  zrange = np.arange(zlowerLim, zUpperLim, stepInZ)

  # if you define variable MOTOR density on MTs, give the axis along which the gradient is defined
  motor_strength_gradient_along_axis = None
  # if not None, then define the regions and the motor density
  # This can be done via giving regions discretely:
  motor_strength_gradient_regions = []
  motor_strength_gradient = []
  # In defining the regions: both points are included for the first entry,
  #     for the rest the first point is excluded (below has 0.5 for all.)
  motor_strength_gradient.append(0.05)
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
                                  n_save = 50,
                                  output_name=filename,
                                  precompute_body_PC = True,
                                  useFMM = True,
                                  Nperiphery = Nperiphery,
                                  iCytoPulling = True,
                                  random_seed = 1)

  prams = initialize.set_parameters(eta = 1.0,
                                    Efib = 2.5E-3,
                                    xcut_mts = xcut_mts,
                                    final_time = 250000*dt,
                                    fiber_body_attached = False,
                                    fiber_sites = fiber_sites,
                                    fiber_norms = fiber_norms,
                                    cortex_xyz = cortex_xyz,
                                    cortex_norms = cortex_norms,
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
