import numpy as np
import sys
sys.path.append('../')
from tstep import tstep
from tstep import initialize
from read_input import read_input
from read_input import read_fibers_file
from read_input import read_vertex_file
from read_input import read_clones_file
from read_input import read_links_file




 
if __name__ == '__main__':
  
  # INPUT FILE: includes fiber, molecular motor parameters and files to read fiber, body, mm configs
  # 2 centrosomes and 2 nuclei running from grown MTs (nuclei are moving with
  # motors attached)
  #input_file = './input_files/two_cent_two_nuc_equil_moving_resume.inputfile' 
  
  # 2 centrosomes and 2 nuclei with all MTs having 0.3 um length (nuclei are
  # moving with motors attached)
  #input_file = './input_files/two_cent_two_nuc_equil_moving.inputfile'
  
  # 1 centrosome and 1 nucleus with all MTs having 0.3 um length (nucleus is
  # moving with motors attached)
  #input_file = './input_files/one_cent_one_nuc_equil_moving.inputfile' # moving nucleus with motors attached
  
  # 1 centrosome and 1 FIXED nucleus (in this case enter nucleus radius and
  # position in PARAMETERS)
  input_file = './input_files/experiment_one_cent_one_nuc_fixed.inputfile'

  read = read_input.ReadInput(input_file)
  
  # SIMULATION'S OPTIONS AND PARAMETERS
  
  # OPTIONS:
  # adaptive_num_points = flag for using adaptive fiber resolution (needed when polymerization is on)
  # num_points = initial number of points per fiber (if you run from a checkpoint, then enter 'None')
  # num_points_max = # of points is variable, this is the maximum number
  # adaptive_time = flag for adaptive time stepping
  # time_step_scheme = scheme to take time steps (time_step_dry or time_step_hydro)
  # dt = time step size (if fixed, otherwise initial step size)
  # dt_min = minimum time step size allowed in adaptive stepping
  # dt_max = maximum time step size allowed in adaptive stepping
  # tol_tstep = tolerance to accept solution when taking time steps (for inextensibility error)
  # tol_gmres = tolerance for GMRES
  # n_save = the number of steps after which data is saved
  # isaveForces = flag to save forces on fibers and bodies
  # output_txt_files = flag for saving data in txt files to use in MATLAB
  # output_name = folder and file names to save outputs
  # num_points_finite_diff = (option name will be changed) is the order of the finite differencing scheme (2 or 4)
  # penalty_param = penalty parameter for inextensibility (affects stability)
  # iFixObjects = flag to fix the objects initially and then release at some point
  # release_check = how decide when to release ('time' or 'ave_length')
  # release_condition = if time then time after release or if average MT length, then length after release
  # irelease = flag to release a system or not, if this is true, then motor binding rate is slowly increased

  # PARAMETERS:
  # ALL LENGTHS ARE IN MICRONS
  # eta = fluid viscosity (for local drag in dry simulations), dimension = mg/um/s
  # final_time = time horizon
  # fiber_body_attached = True (if there are fibers attached to a body)
  # len_nuc2nuc = repulsion length scale for nucleus-nucleus interaction
  # scale_nuc2nuc = repulsion force strength (nucleus is body with motors attached to it)
  # Efib = Bending stiffness of MTs, [pNum^2]
  # cortex_radius = enter a value if there is one, if there is not, then write 'None'
  
  # nucleus_radius = radius of a FIXED nucleus (enter multiple, in an array if there are several)
  #               , if nucleus is moving, then it is given in input_file
  # nucleus_position = position of FIXED nucleus 
  # NOTE: this nucleus has motors on it and the motors' information is given in input_file

  # Following options can be set without the input file
  options = initialize.set_options(adaptive_num_points = True,
                                  num_points = 16,
                                  num_points_max = 256,
                                  adaptive_time = True,
                                  time_step_scheme = 'time_step_dry',
                                  dt = 1e-3,
                                  dt_min = 1e-04,
                                  dt_max = 1e-01,
                                  tol_tstep = 1e-01,
                                  tol_gmres = 1e-10,
                                  output_txt_files = True,
                                  n_save = 100,
                                  num_points_finite_diff = 4,
                                  penalty_param = 50.0,
                                  iFixObjects = False,
                                  isaveForces = False,
                                  irelease = False,
                                  release_check = 'time',
                                  release_condition = 30.0,
                                  output_name = 'data/oneCent/runTest/run',
                                  iDynInstability = True,
                                  iNoSliding = False)

  prams = initialize.set_parameters(eta = 1.0, 
                                    Efib = 10.0, 
                                    final_time = 2000, 
                                    cortex_radius = None, 
                                    len_nuc2nuc = 0.75,
                                    scale_nuc2nuc = 200.0,
                                    nucleus_radius = np.array([4.25]),
                                    nucleus_position = np.array([0]),
                                    growing = 1)

  # if there is a fixed nucleus then give: 
  # nucleus_radius = np.array([4.25])
  # nucleus_position = np.array([0]) in parameters (this is z coordinate)

  # Create time stepping scheme
  tstep = tstep.tstep(prams,options,input_file,None,None,None)

  # Take time steps
  tstep.take_time_steps()


  print('\n\n\n# End')    
