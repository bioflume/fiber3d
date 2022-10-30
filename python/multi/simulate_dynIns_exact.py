from __future__ import print_function
import numpy as np
import sys
sys.path.append('../')
from read_input import read_links_file
from tstep import initialize
from read_input import read_input
from tstep import tstep_utils_new as tstep_utils
from fiber import fiber

try:
  import cPickle as cpickle
except:
  try:
    import cpickle
  except:
    import _pickle as cpickle

if __name__ == '__main__':
  # RESUMING FILES
  resume_step = None
  radius = float(sys.argv[1])
  fileName = '/mnt/ceph/users/gkabacaoglu/time_series_set' +sys.argv[2] + '_dts/dt' + sys.argv[5] + '_radius' + sys.argv[1] + '/'
  grw_rate = 0.5
  cat_rate = 0.033
  nuc_rate = float(sys.argv[3]) # 54 for set25, 25 for set27, 50 for set 28
  
  # Scaling of catastrophe and growth rate on cortex
  scale_max_leng = 1/3 # 1/scale_max_leng is the scaling of catastrophe
  scale_grw_rate = 1

  if resume_step is None:
    fiber_sites_lengths_resume_file = None
    body_active_sites_resume_file = None
    body_passive_sites_resume_file = None
    body_occupied_sites_resume_file = None
    random_seed = 1
  else:
    fiber_sites_lengths_resume_file = './dynIns_Runs/radius' + sys.argv[1] + '/fiber_lengths.step' + str(resume_step)
    body_active_sites_resume_file = './dynIns_Runs/radius' + sys.argv[1] + '/active_nuc_site_idcs.step' + str(resume_step)
    body_passive_sites_resume_file = './dynIns_Runs/radius' + sys.argv[1] + '/passive_nuc_site_idcs.step' + str(resume_step)
    body_occupied_sites_resume_file = './dynIns_Runs/radius' + sys.argv[1] + '/occupied_site_idcs.step' + str(resume_step)
    random_seed = './dynIns_Runs/radius' + sys.argv[1] + '/randomState.step' + str(resume_step)


  # Time horizon
  #Th = 1E+4 # seconds, time horizon
  dt = 1E-2 # seconds, time step size
  dt = float(sys.argv[4])
  Th = 1E+4 / 1E-2 * dt

  time = np.arange(0,Th,dt)
  Nmts_hinged = np.zeros_like(time)
  name = fileName + 'nhinged_time.txt'
  f_nhinged_time = open(name, 'wb', buffering = 100)

  input_file = './jaySims/input_files/mtoc_at_center.inputfile'
  #input_file = './jaySims/input_files/largeMTOC_at_center.inputfile'

  options = initialize.set_options(adaptive_num_points = True,
                                  num_points = 32,
                                  num_points_max = 128,
                                  random_seed = random_seed,
                                  adaptive_time = False,
                                  time_step_scheme = 'time_step_dry',
                                  order = 1,
                                  dt = dt,
                                  dt_min = dt,
                                  dt_max = dt,
                                  tol_tstep = 1e-1,
                                  tol_gmres = 1e-10,
                                  n_save = 25,
                                  output_name=fileName,
                                  save_file = None,
                                  precompute_body_PC = False,
                                  useFMM = False,
                                  Nperiphery = 6000,
                                  Nblobs = 400,
                                  body_quadrature_radius = 0.4,
                                  iupsample = False,
                                  integration = 'trapz',
                                  iFixObjects = False,
                                  num_points_finite_diff = 4,
                                  inextensibility = 'penalty',
                                  iCytoPulling = False,
                                  dynInstability = True,
                                  iNoSliding = True,
                                  iExternalForce = False,
                                  min_body_ds = 0.01,
                                  fiber_ds = 0.5/32,
                                  repulsion = False)

  prams = initialize.set_parameters(eta = 1,
      scale_life_time = scale_max_leng,
      scale_vg = scale_grw_rate,
      epsilon = 1e-03,
      Efib = 20,
      minL = 0.5,
      final_time = 2400,
      fiber_body_attached = True,
      periphery = 'ellipsoid',
      periphery_a = radius,
      periphery_b = radius,
      periphery_c = 15, #np.floor(25/20 * radius),
      growing = 1,
      rate_catastrophe = cat_rate,
      v_growth = grw_rate,
      nucleation_rate = nuc_rate,
      max_nuc_sites = None,
      time_max = None,
      max_length = None)

  fibers, bodies, molecular_motors, fibers_names, body_names, fibers_types, body_types, f_fibers_ID, f_bodies_ID, f_molecular_motors_ID, f_time_system, f_fibers_forces_ID, f_bodies_forces_ID, MM_on_moving_surf, bodies_mm_attached, bodies_types_mm_attached, bodies_names_mm_attached, f_bodies_mm_attached_ID, f_bodies_mm_attached_forces_ID, f_mm_on_moving_surf_ID, f_body_vels_ID = initialize.initialize_from_file(input_file,options,prams)


  # If RESUMING, THEN BUILD FIBERS AND CHECK THEIR CORTICAL PUSHING STATUS
  s = np.linspace(0, 2, 64)
  if resume_step is not None:
    active_sites_idcs = np.loadtxt(body_active_sites_resume_file, dtype = np.int32)
    passive_sites_idcs = np.loadtxt(body_passive_sites_resume_file, dtype = np.int32)
    occupied_sites_idcs = np.loadtxt(body_occupied_sites_resume_file, dtype = np.int32)
    bodies[0].occupied_site_idcs = occupied_sites_idcs.tolist()
    bodies[0].active_sites_idcs = active_sites_idcs.tolist()
    bodies[0].passive_sites_idcs = passive_sites_idcs.tolist()

    fiber_sites_lengths = np.loadtxt(fiber_sites_lengths_resume_file, dtype = np.float64)
    for k in np.arange(len(fiber_sites_lengths)):
      nuc_site_idx = int(fiber_sites_lengths[k,0])
      site_location = bodies[0].nuc_sites[nuc_site_idx]
      site_normal = site_location / np.linalg.norm(site_location)
      axis_s = np.empty((s.size, 3))
      axis_s[:,0] = site_normal[0] * s
      axis_s[:,1] = site_normal[1] * s
      axis_s[:,2] = site_normal[2] * s
      fib_xyz = axis_s * (fiber_sites_lengths[k,1]/2.0) + site_location
      fib = fiber.fiber(
          num_points = 64,
          dt= options.dt,
          length=fiber_sites_lengths[k,1])
      fib.attached_to_body = 0
      fib.nuc_site_idx = nuc_site_idx
      fib.ID = bodies[0].ID
      fib.x = fib_xyz
      fib.length = fiber_sites_lengths[k,1]

      fibers.append(fib)

  # Loop in time
  site_idcs_nucleating, site_idcs_dying, site_idcs_hinged = np.array([],dtype=np.int32), np.array([],dtype=np.int32), np.array([],dtype=np.int32)
  for idx, tt in enumerate(time):
    print('Time step:', str(idx), ' in total number of steps: ', str(len(time)))

    # CHECK CORTICAL PUSHING STATUS
    site_idcs_hinged_t = []
    for k, fib in enumerate(fibers):
      if fib.iReachSurface is False:
        # Check if it is hinged now
        xfib = fib.x[:,0]
        yfib = fib.x[:,1]
        zfib = fib.x[:,2]

        x, y, z = xfib / radius, yfib / radius, zfib / 25

        r_true = np.sqrt(xfib[-1]**2 + yfib[-1]**2 + zfib[-1]**2)

        r_fiber = np.sqrt(x[-1]**2 + y[-1]**2 + z[-1]**2);
        phi_fiber = np.arctan2(y[-1], (x[-1] + 1e-12))
        theta_fiber = np.arccos(z[-1] / (1e-12 + r_fiber))

        x_cort = radius * np.sin(theta_fiber) * np.cos(phi_fiber)
        y_cort = radius * np.sin(theta_fiber) * np.sin(phi_fiber)
        z_cort = 25 * np.cos(theta_fiber)

        d2cort = np.sqrt((xfib[-1] - x_cort)**2 + (yfib[-1] - y_cort)**2 + (zfib[-1] - z_cort)**2)
        cortex_point_r = np.sqrt(x_cort**2 + y_cort**2 + z_cort**2)
        if r_true<= cortex_point_r:
          if d2cort <= 0.75:
            num_hinged += 1
            fib.iReachSurface = True
            site_idcs_hinged_t.append(fib.nuc_site_idx)
        else:
          fib.iReachSurface = True
          num_hinged += 1
          site_idcs_hinged_t.append(fib.nuc_site_idx)

    # DYNAMIC INSTABILITY
    fibers, fibers_gone, bodies, num_nucleated_fibers, num_nucleated_imaginary, sites_gone_imaginary, terminal_time_gone_imaginary, site_idcs_nucleating_t, site_idcs_dying_t = tstep_utils.dynamic_instability_v2(fibers,
                                                                           bodies,
                                                                           prams,
                                                                           options,
                                                                           dt,
                                                                           radius,
                                                                           radius,
                                                                           25,
                                                                           tt)




    # BASED ON FIBER LENGTH, UPDATE FIBER X, Y, Z
    b = bodies[0]

    fiber_sites_lengths = np.zeros((len(fibers),2))
    num_hinged = 0
    for k, fib in enumerate(fibers):
      site_location = b.nuc_sites[fib.nuc_site_idx]
      site_normal = site_location / np.linalg.norm(site_location)
      axis_s = np.empty((s.size, 3))
      axis_s[:,0] = site_normal[0] * s
      axis_s[:,1] = site_normal[1] * s
      axis_s[:,2] = site_normal[2] * s
      fib.x = axis_s * (fib.length/2.0) + site_location
      fiber_sites_lengths[k,0] = fib.nuc_site_idx
      fiber_sites_lengths[k,1] = fib.length
      if fib.iReachSurface: num_hinged += 1

    Nmts_hinged[idx] = num_hinged
    site_idcs_dying = np.concatenate((site_idcs_dying,np.array([len(site_idcs_dying_t)])))
    if len(site_idcs_dying_t) > 0:
      site_idcs_dying = np.concatenate((site_idcs_dying,np.array(site_idcs_dying_t)))
    site_idcs_nucleating = np.concatenate((site_idcs_nucleating, np.array([len(site_idcs_nucleating_t)])))
    if len(site_idcs_nucleating_t) > 0:
      site_idcs_nucleating = np.concatenate((site_idcs_nucleating, np.array(site_idcs_nucleating_t)))
    site_idcs_hinged = np.concatenate((site_idcs_hinged, np.array([len(site_idcs_hinged_t)])))
    if len(site_idcs_hinged_t) > 0:
      site_idcs_hinged = np.concatenate((site_idcs_hinged, np.array(site_idcs_hinged_t)))

    if np.remainder(idx, 100) == 0:
      print('There are ', str(Nmts_hinged[idx]) , ' hinged MTs')

      # Saving active and passive site information on body
      body_file_passive = open(fileName + 'passive_nuc_site_idcs.step' + str(idx),'wb')
      body_file_active = open(fileName + 'active_nuc_site_idcs.step' + str(idx),'wb')
      body_file_occupied = open(fileName + 'occupied_site_idcs.step' + str(idx),'wb')
      file_fiber_sites_lengths = open(fileName + 'fiber_lengths.step' + str(idx),'wb')

      np.savetxt(body_file_passive, b.passive_sites_idcs)
      body_file_passive.close()
      np.savetxt(body_file_active, b.active_sites_idcs)
      body_file_active.close()
      np.savetxt(body_file_occupied, b.occupied_site_idcs)
      body_file_occupied.close()
      np.savetxt(file_fiber_sites_lengths, fiber_sites_lengths)
      file_fiber_sites_lengths.close()

      # Saving hinged MT information and time
      nhinged_time = np.array([Nmts_hinged[idx], tt])
      np.savetxt(f_nhinged_time, nhinged_time[None,:])

  site_info_dying = open(fileName + 'site_info_dying.txt','wb')
  site_info_nucleating = open(fileName + 'site_info_nucleating.txt','wb')
  site_info_hinged = open(fileName + 'site_info_hinged.txt','wb')


  np.savetxt(site_info_dying, site_idcs_dying.flatten())
  site_info_dying.close()


  np.savetxt(site_info_nucleating, site_idcs_nucleating.flatten())
  site_info_nucleating.close()

  
  np.savetxt(site_info_hinged, site_idcs_hinged.flatten())
  site_info_hinged.close()
