import numpy as np
import sys
sys.path.append('../')
from tstep import tstep_aster as tstep
from tstep import initialize_aster as initialize
from read_input import read_input
from read_input import read_fibers_file
from read_input import read_vertex_file
from read_input import read_clones_file
from read_input import read_links_file





if __name__ == '__main__':

  # INPUT FILE: includes fiber, molecular motor parameters and files to read fiber, body, mm configs
  cortex_sites_file = None
  cortex_fibers_file = None
  centrosome_position = np.array([0.0, 0.0, 3.5])
  # centrosome's fibers assuming at the center, then displace it given position
  centrosome_fibers_file = './Structures/300fibersOnR0p5_atCenter.fibers'
  centrosome_sites_file = './Structures/300sitesOnR0p5.links'

  cortexV_flow_normal = np.array([0., 1., 0.])
  cortexV_flow_direction = 'CCW'
  cortexV_strength = 1

  iComputeVelocity = True
  Nfiber = 32
  Nbody = 500
  Nperiphery = 8000 # 8000, 10000
  max_length = 3
  dt = 2E-2
  filename = './data/fibers_cortex_asterOnly/run'
  active_sites_idcs_file = None
  passive_sites_idcs_file = None
  fib_sites_lengths_file = None


  # SIMULATION'S OPTIONS AND PARAMETERS

  #  time_step_scheme = 'fibers_and_bodies_and_periphery_hydro_implicit_time_step',
  # Following options can be set without the input file
  options = initialize.set_options(adaptive_num_points = True,
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
                                  n_save = 25,
                                  output_name=filename,
                                  precompute_body_PC = True,
                                  useFMM = True,
                                  Nperiphery = Nperiphery,
                                  Nblobs = Nbody,
                                  body_quadrature_radius = 0.4,
                                  iupsample = False,
                                  integration = 'trapz',
                                  iFixObjects = False,
                                  iExternalForce = False,
                                  irelease = False,
                                  dynInstability = False,
                                  num_points_finite_diff = 4,
                                  inextensibility = 'penalty',
                                  iCytoPulling = False,
                                  iPeripheralNucleation = False,
                                  min_body_ds = 0.01,
                                  fiber_ds = 0.5/32,
                                  repulsion = False,
                                  iComputeVelocity = iComputeVelocity,
                                  ncompute_vel = 500,
                                  iNoSliding = False)

  prams = initialize.set_parameters(eta = 1.0,
      epsilon = 1e-03,
      Efib = 2.5E-2,
      final_time = 100.0,
      attached_to_cortex = False,
      fiber_body_attached = True,
      periphery = 'sphere',
      periphery_radius = 5.7,
      growing = 1,
      rate_catastrophe = 0.025,
      v_growth = 0.050,
      nucleation_rate = 50,
      scale_life_time = 1,
      scale_vg = 1,
      max_length = max_length,
      cortex_sites_file = cortex_sites_file,
      cortex_fibers_file = cortex_fibers_file,
      centrosome_fibers_file  = centrosome_fibers_file,
      centrosome_sites_file = centrosome_sites_file,
      active_sites_idcs_file = None,
      passive_sites_idcs_file = None,
      occupied_sites_idcs_file = None,
      fib_sites_lengths_file = None,
      site_idcs_nucleating_file = None,
      site_idcs_dying_file = None,
      site_idcs_hinged_file = None,
      cent_location = centrosome_position,
      body_viscosity_scale = 1,
      cortexV_flow_normal = cortexV_flow_normal,
      cortexV_flow_direction = cortexV_flow_direction,
      cortexV_strength = cortexV_strength)

  # Create time stepping scheme
  tstep = tstep.tstep(prams,options,None,None,None,None)

  # Take time steps
  tstep.take_time_steps()


  print('\n\n\n# End')
