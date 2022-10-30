import numpy as np
import sys
sys.path.append('../')
from tstep import tstep_simple as tstep
from tstep import initialize_simple as initialize


if __name__ == '__main__':

  # Manually generating structures
  centrosome_positions = None # or None
  #centrosome_positions.append(np.array([0.0, 0.0, 0.0]))
  centrosome_radii = None
  #centrosome_radii.append(0.5)
  centrosome_quad_radii = None #[]
  #centrosome_quad_radii.append(0.4)

  fibers_locations = None
  fibers_orientations = []
  fibers_lengths = []
  fibers_attachement = []
  fibers_BCs = []
  #fibers_locations.append([0., 0., 0.5])
  #fibers_orientations.append([0., 0., 1.])
  #fibers_lengths.append(3.25)
  #fibers_attachement.append(0)
  fibers_BCs.append(['clamped', 'free'])


  # Files for fibers and centrosomes, if exist
  cortex_sites_file = None
  cortex_fibers_file = None

  # centrosome's fibers assuming at the center, then displace it given position
  centrosome_fibers_file = []
  centrosome_fibers_file.append('./skellySim/Structures/1KfibersOnR0p5_L1.fibers')
  #centrosome_fibers_file.append('./skellySim/Structures/centZm2_500fibers_L1.fibers')
  centrosome_sites_file = []
  centrosome_sites_file.append('./skellySim/Structures/1KsitesOnR0p5.links')
  #centrosome_sites_file.append('./skellySim/Structures/cent500sites2.links')

  # Centrosome resume file
  centrosome_resume_files = []
  centrosome_resume_files.append('./skellySim/Structures/centrosome_atXm2.clones')
 # centrosome_resume_files.append('./skellySim/Structures/centrosome_atZm2.clones')
  active_sites_idcs_file = []
  active_sites_idcs_file.append(None)
  #active_sites_idcs_file.append(None)
  passive_sites_idcs_file = None
  fib_sites_lengths_file = []
  fib_sites_lengths_file.append(None)
  #fib_sites_lengths_file.append(None)


  iComputeVelocity = False
  iDynamicInstability = False
  iCytoPulling = False
  iNoSliding = False
  fiber_external_force = None
  body_external_force = [] # this includes torque too
  body_external_force.append(np.array([0.5, 0, 0, 0., 0., 0.]))
  #body_external_force.append(np.array([0., 0., 0.5, 0, 0, 0]))
  

  Nfiber = 16
  Nbody = 1600
  periphery = 'sphere' #None
  periphery_radius = 5 #None
  periphery_a = None
  periphery_b = None
  periphery_c = None
  Nperiphery = 5000 # 8000, 10000
  dt = 1E-2
  filename = './skellySim/run'


  # SIMULATION'S OPTIONS AND PARAMETERS

  #  time_step_scheme = 'fibers_and_bodies_and_periphery_hydro_implicit_time_step',
  # Following options can be set without the input file
  options = initialize.set_options(adaptive_num_points = True,
                                  num_points = Nfiber,
                                  num_points_max = 16,
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
                                  Nblobs = Nbody,
                                  iupsample = False,
                                  integration = 'trapz',
                                  irelease = False,
                                  num_points_finite_diff = 4,
                                  inextensibility = 'penalty',
                                  iCytoPulling = iCytoPulling,
                                  min_body_ds = 0.01,
                                  fiber_ds = 3.25/16,
                                  repulsion = False,
                                  iComputeVelocity = iComputeVelocity,
                                  ncompute_vel = 500,
                                  iNoSliding = iNoSliding)

  prams = initialize.set_parameters(eta = 1.0,
      epsilon = 1e-03,
      Efib = 1,
      final_time = 100.0,
      periphery = periphery,
      periphery_radius = periphery_radius,
      periphery_a = periphery_a,
      periphery_b = periphery_b,
      periphery_c = periphery_c,
      iDynamicInstability = iDynamicInstability,
      rate_catastrophe = 0.025,
      v_growth = 0.050,
      nucleation_rate = 50,
      scale_life_time = 1,
      scale_vg = 1,
      cortex_sites_file = cortex_sites_file,
      cortex_fibers_file = cortex_fibers_file,
      centrosome_fibers_file  = centrosome_fibers_file,
      centrosome_sites_file = centrosome_sites_file,
      active_sites_idcs_file = active_sites_idcs_file,
      passive_sites_idcs_file = None,
      occupied_sites_idcs_file = None,
      fib_sites_lengths_file = fib_sites_lengths_file,
      site_idcs_nucleating_file = None,
      site_idcs_dying_file = None,
      site_idcs_hinged_file = None,
      centrosome_positions = centrosome_positions,
      centrosome_radii = centrosome_radii,
      centrosome_quad_radii = centrosome_quad_radii,
      centrosome_resume_files = centrosome_resume_files,
      body_external_force = body_external_force,
      fiber_external_force = fiber_external_force,
      fibers_BCs = fibers_BCs,
      fibers_lengths = fibers_lengths,
      fibers_orientations = fibers_orientations,
      fibers_attachement = fibers_attachement,
      fibers_locations = fibers_locations,
      body_viscosity_scale = 1)

  # Create time stepping scheme
  tstep = tstep.tstep(prams,options,None,None,None,None)

  # Take time steps
  tstep.take_time_steps()


  print('\n\n\n# End')
