import numpy as np
import sys
import multi
sys.path.append('../')
from fiber import fiber
from body import body
from tstep import tstep
from tstep import initialize
from tstep import fiber_matrices
from quaternion import quaternion

if __name__ == '__main__':
  print('# Start')
  # Set parameters
  num_points = int(sys.argv[1])
  num_points_finite_diff = int(sys.argv[7])
  dt = float(sys.argv[2])
  Nf = int(sys.argv[3])
  output_name = 'data/fiber_hydro_fixDt/' + sys.argv[4] + '/run'

  d = float(sys.argv[5]) # diameter of the rim along which fibers placed
  iBody = int(sys.argv[6]) # flag for having rigid body or not

  #t_max = 2.0
  t_max = 500*dt
  n_save = 1
  
  E = 1.0
  length = 2.0
  force_density = -10.0 / length
  epsilon = 1e-3

  integration = 'trapz'
  iupsample = False
  tolerance = 1e-10
  
  # Create four fibers
  fibers = []
  xlocs = d/2 * np.cos(2*np.pi*np.arange(Nf)/Nf)
  ylocs = d/2 * np.sin(2*np.pi*np.arange(Nf)/Nf)
  for ifib in range(Nf):
    fib = fiber.fiber(num_points = num_points,
                      dt = dt,
                      E = E,
                      length = length,
                      epsilon = epsilon,
                      adaptive_num_points = False,
                      inonlocal = False,
                      num_points_finite_diff = num_points_finite_diff)
    fib.x[:,0] = xlocs[ifib]
    fib.x[:,1] = ylocs[ifib]
    fib.x[:,2] = fib.s # move from [0, 2] to [-1, 1]
    fibers.append(fib)

  bodies = None
  if iBody:
    bodies = []
    orientation = quaternion.Quaternion([1., 0., 0., 0.])
    b = body.Body(np.array([0., 0., 0.]),orientation,np.array([0., 0., 0.]), np.array([0., 0., 0.]),np.ones(1))
    b.radius = 0.5
    b.quadrature_radius = 0.5
    bodies.append(b)

  options = initialize.set_options(adaptive_num_points = False,
                                  num_points = num_points,
                                  ireparam = False,
                                  adaptive_time = False,
                                  time_step_scheme = 'time_step_hydro',
                                  order = 1,
                                  dt = dt,
                                  tol_gmres = tolerance,
                                  inonlocal = False,
                                  output_txt_files = True,
                                  n_save = n_save,
                                  isaveForces = False,
                                  output_name = output_name,
                                  precompute_body_PC = True,
                                  useFMM = False,
                                  Nblobs = 100,
                                  body_quadrature_radius = 0.5,
                                  iupsample = iupsample,
                                  integration = integration,
                                  inextensibility = 'penalty')

  prams = initialize.set_parameters(eta = 1.0, epsilon = epsilon, 
    fiber_body_attached = False, final_time = t_max)

  # Create time stepping scheme
  tstep = tstep.tstep(prams,options,None,fibers,bodies,None)

  # Take time steps
  tstep.take_time_steps()
  
