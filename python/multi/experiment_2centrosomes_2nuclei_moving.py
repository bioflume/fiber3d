import numpy as np
import sys
sys.path.append('../')
from tstep import tstep
from tstep import initialize_longMTsInRegion as initialize
from read_input import read_input
from read_input import read_fibers_file
from read_input import read_vertex_file
from read_input import read_clones_file
from read_input import read_links_file




 
if __name__ == '__main__':
  
  # INPUT FILE: includes fiber, molecular motor parameters and files to read fiber, body, mm configs
  # input_file = 'experiment_two_nuclei.inputfile'
  # input_file = './input_files/experiment_two_nuclei_movingMM.inputfile'
  #input_file = './input_files/two_cent_two_nuc_equil_moving_resume.inputfile'
  #input_file = './input_files/two_cent_two_far_nuc_moving_resume.inputfile'
  input_file = './input_files/Cents2Nucs2_FarMoving.inputfile'
  read = read_input.ReadInput(input_file)
  
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


  # Following options can be set without the input file
  options = initialize.set_options(adaptive_num_points = True,
                                  num_points = None,
                                  num_points_max = 192,
                                  ireparam = False,
                                  adaptive_time = True,
                                  time_step_scheme = 'time_step_dry',
                                  order = 1,
                                  dt = 1e-02,
                                  dt_min = 1e-04,
                                  dt_max = 1e-02,
                                  tol_tstep = 4e-02,
                                  tol_gmres = 1e-10,
                                  inonlocal = False,
                                  random_seed = 1,
                                  output_txt_files = True,
                                  n_save = 100,
                                  isaveForces = False,
                                  iupsample = False,
                                  integration = 'trapz',
                                  num_points_finite_diff = 4,
                                  inextensibility = 'penalty',
                                  iFixObjects = False,
                                  irelease = True,
                                  release_check = 'time',
                                  release_condition = 1000.0,
                                  output_name ='data/twoNuclei_twoCents/movingSimFarFewLongFibs/run',
                                  penalty_param = 50.0,
                                  iDynInstability = True)

  prams = initialize.set_parameters(eta = 1, 
                                    epsilon = 1e-03, 
                                    final_time = 200, 
                                    fiber_body_attached = True,
                                    len_nuc2nuc = 0.75,
                                    scale_nuc2nuc = 200.0,
                                    len_nuc2fib = 1e-1,
                                    scale_nuc2fib = 1.5,
                                    growing = 1)

  # Create time stepping scheme
  tstep = tstep.tstep(prams,options,input_file,None,None,None)

  # Take time steps
  tstep.take_time_steps()


  print('\n\n\n# End')    
