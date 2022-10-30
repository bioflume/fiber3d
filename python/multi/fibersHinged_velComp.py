import numpy as np
import sys
sys.path.append('../')
from tstep import tstep as tstep
from tstep import initialize
from read_input import read_input
from read_input import read_fibers_file
from read_input import read_vertex_file
from read_input import read_clones_file
from read_input import read_links_file





if __name__ == '__main__':

  # INPUT FILE: includes fiber, molecular motor parameters and files to read fiber, body, mm configs
  input_file = './input_files/fibers_cortex_dynInst_resume.inputfile'

  read = read_input.ReadInput(input_file)

  iComputeVelocity = True
  Nfiber = 48
  Nperiphery = 10000 # 8000, 10000
  max_length = 3
  dt = 5E-3
  filename = 'data/fibers_cortex_dynInst_velComp/run'
  active_sites_idcs_file = None
  passive_sites_idcs_file = None
  fib_attached_to_body_file = './Structures/fibers_cortex_dynInst/run_centrosome_at_x0_attached_to_body.fibers'
  fib_nuc_site_idx_file = './Structures/fibers_cortex_dynInst/run_centrosome_at_x0_nuc_site_idx.fibers'
  fib_length_before_die_file = './Structures/fibers_cortex_dynInst/run_centrosome_at_x0_length_before_die.fibers'

  # SIMULATION'S OPTIONS AND PARAMETERS

  #  time_step_scheme = 'fibers_and_bodies_and_periphery_hydro_implicit_time_step',
  # Following options can be set without the input file
  options = initialize.set_options(adaptive_num_points = False,
                                  num_points = Nfiber,
                                  adaptive_time = False,
                                  time_step_scheme = 'time_step_hydro',
                                  order = 1,
                                  dt = dt,
                                  dt_min = 5e-3,
                                  dt_max = 5e-2,
                                  tol_tstep = 1e-1,
                                  tol_gmres = 1e-10,
                                  inonlocal = False,
                                  output_txt_files = True,
                                  n_save = 1,
                                  output_name=filename,
                                  precompute_body_PC = True,
                                  useFMM = True,
                                  Nperiphery = Nperiphery,
                                  iupsample = False,
                                  integration = 'trapz',
                                  iFixObjects = False,
                                  iExternalForce = False,
                                  irelease = False,
                                  dynInstability = True,
                                  num_points_finite_diff = 4,
                                  inextensibility = 'penalty',
                                  iCytoPulling = False,
                                  iPeripheralNucleation = True,
                                  min_body_ds = 0.01,
                                  fiber_ds = 0.5/32,
                                  repulsion = False,
                                  iComputeVelocity = iComputeVelocity)

  prams = initialize.set_parameters(eta = 1.0,
      epsilon = 1e-03,
      Efib = 2.5E-2,
      final_time = 2*dt,
      fiber_body_attached = False,
      periphery = 'sphere',
      periphery_radius = 5.7,
      growing = 1,
      rate_catastrophe = 0.025,
      v_growth = 0.050,
      nucleation_rate = 50,
      max_length = max_length,
      active_sites_idcs_file = active_sites_idcs_file,
      passive_sites_idcs_file = passive_sites_idcs_file,
      fib_attached_to_body_file = fib_attached_to_body_file,
      fib_nuc_site_idx_file = fib_nuc_site_idx_file,
      fib_length_before_die_file = fib_length_before_die_file)

  # Create time stepping scheme
  tstep = tstep.tstep(prams,options,input_file,None,None,None)

  # Take time steps
  tstep.take_time_steps()


  print('\n\n\n# End')
