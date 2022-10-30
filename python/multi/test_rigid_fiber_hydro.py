import numpy as np
import sys
sys.path.append('../')
from bpm_utilities import gmres
from tstep import tstep as tstep
from tstep import initialize
from read_input import read_input
from read_input import read_fibers_file
from read_input import read_vertex_file
from read_input import read_clones_file
from read_input import read_links_file




 
if __name__ == '__main__':
  
  # INPUT FILE: includes fiber, molecular motor parameters and files to read fiber, body, mm configs
  input_file = './input_files/body_fiber_hydro_N50L5.inputfile'
  read = read_input.ReadInput(input_file)

  Nfiber = 8
  Nbody = 200
  Nperiphery = 800
  body_quad_radius = 4.0
  dt = 2E-4
  filename = 'data/body_fiber_hydro/N50L5/run'
  
  # SIMULATION'S OPTIONS AND PARAMETERS
  # adaptive_num_points = flag for using adaptive fiber resolution (needed when polymerization is on)
  # ireparam = flag for fiber reparameterization (needed when resolution is low)
  # filtering = flag for de-aliasing (should be on always)
  # adaptive_time = flag for adaptive time stepping
  # time_step_scheme = scheme to take time steps (fibers? bodies? hydro?)
  # order = time stepping order (only 1 and 2 exist, 2 is less stable)
  # dt = time step size (if fixed, otherwise initial step size)
  # dt_min = minimum time step size allowed in adaptive stepping
  # dt_max = maximum time step size allowed in adaptive stepping
  # tol_tstep = tolerance to accept solution when taking time steps (for inextensibility error)
  # tol_gmres = tolerance for GMRES
  # inonlocal = flag for including nonlocal part of the self-mobility of fibers
  # igrowing = flag for fibers whether they are growing or not (probably not used)
  # random_seed = random seeding, if 1, then always the same series
  # output_txt_files = flag for saving data in txt files to use in MATLAB
  # output_name = name to save outputs


  #  time_step_scheme = 'fibers_and_bodies_and_periphery_hydro_implicit_time_step',
  # Following options can be set without the input file
  options = initialize.set_options(adaptive_num_points = False,
                                  num_points = Nfiber,
                                  ireparam = False,
                                  filtering = True,
                                  adaptive_time = False,
                                  time_step_scheme = 'time_step_hydro',
                                  order = 1,
                                  dt = dt,
                                  tol_gmres = 1e-10,
                                  inonlocal = False,
                                  output_txt_files = True,
                                  n_save = 1,
                                  output_name=filename,
                                  precompute_body_PC = True,
                                  useFMM = True,
                                  Nperiphery = Nperiphery,
                                  Nblobs = Nbody,
                                  body_quadrature_radius = body_quad_radius)

  prams = initialize.set_parameters(eta = 1.0, epsilon = 1e-03, final_time = 10*dt,
          periphery = 'sphere', periphery_radius = 30.0)

  # Create time stepping scheme
  tstep = tstep.tstep(prams,options,input_file,None,None,None)

  # Take time steps
  tstep.take_time_steps()


  print('\n\n\n# End')    
