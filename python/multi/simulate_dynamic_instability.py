from __future__ import print_function
import numpy as np
import sys
sys.path.append('../')
from read_input import read_links_file
try:
  import cPickle as cpickle
except:
  try:
    import cpickle
  except:
    import _pickle as cpickle

if __name__ == '__main__':
  # Set random seed to 1
  np.random.seed(1)
  #with open('./dynIns_Runs/radius12_lowMT/randomState.step300025', 'rb') as f:
    #np.random.set_state(cpickle.load(f))
  # Simulation parameters

  radius = 12
  grw_rate = 0.75
  cat_rate = 0.0625
  nuc_rate = 32

  fileName = './dynIns_Runs/radius10_lowMT/'

  # Scaling of catastrophe and growth rate on cortex
  scale_max_leng = 0.5
  scale_grw_rate = 0.8

  # Max nucleating sites
  max_nuc_sites = None
  min_ds = 0.01

  # Time horizon
  Th = 1E+4 # seconds, time horizon
  dt = 1E-2 # seconds, time step size
  time = np.arange(0,Th,dt)

  # Load nucleating sites
  spring_constants, spring_constants_angle, nuc_sites, axes = read_links_file.read_links_file('./Structures/centrosome.N1000.links')


  # INITIALIZE
  fibers_l = []
  fibers_hinged = []
  fibers_xyz = []
  fibers_site_idcs = []
  occupied_site_idcs = []
  passive_sites_idcs = np.arange(0,len(nuc_sites)).tolist()
  active_sites_idcs = []
  keep_occupied_sites = []
  when_empty_occupied_sites = []
  fibers_terminal_length = []
  empty_site_idcs = np.arange(0,len(nuc_sites)).tolist()
  Nmts_hinged = np.zeros_like(time)
  s = np.linspace(0, 2, 64)

  name = fileName + 'nhinged_time.txt'
  f_nhinged_time = open(name, 'wb', buffering = 100)

  # Loop in time
  for idx, tt in enumerate(time):
    print('Time step:', str(idx), ' in total number of steps: ', str(len(time)))

    # LOOP OVER IMAGINARY SITES
    temp_time, temp_idcs = [], []
    for jdx in range(len(occupied_site_idcs)):
      if tt >= when_empty_occupied_sites[jdx]:
        passive_sites_idcs.append(occupied_site_idcs[jdx])
      else:
        temp_time.append(when_empty_occupied_sites[jdx])
        temp_idcs.append(occupied_site_idcs[jdx])

    when_empty_occupied_sites = temp_time
    occupied_site_idcs = temp_idcs

    # LOOP OVER FIBERS
    num_hinged = 0
    temp_l, temp_hinged, temp_xyz, temp_site_idcs, temp_terminal_length = [], [], [], [], []
    for jdx, fibL in enumerate(fibers_l):
      max_length = fibers_terminal_length[jdx]
      if fibers_hinged[jdx]: max_length *= scale_max_leng

      if fibL >= max_length:
        passive_sites_idcs.append(fibers_site_idcs[jdx])
        index_in_active = active_sites_idcs.index(fibers_site_idcs[jdx])
        del active_sites_idcs[index_in_active]
      else:
        temp_site_idcs.append(fibers_site_idcs[jdx])
        temp_terminal_length.append(fibers_terminal_length[jdx])

        # If not going through catastrophe, then grow MT
        vlength = grw_rate
        if fibers_hinged[jdx]: vlength *= scale_grw_rate
        fibers_l[jdx] += vlength * dt

        site_location = nuc_sites[fibers_site_idcs[jdx]]
        site_normal = site_location / np.linalg.norm(site_location)

        axis_s = np.empty((s.size, 3))
        axis_s[:,0] = site_normal[0] * s
        axis_s[:,1] = site_normal[1] * s
        axis_s[:,2] = site_normal[2] * s
        fibers_xyz[jdx] = axis_s * (fibers_l[jdx]/2.0) + site_location

        # Check if it is hinged now
        xfib = fibers_xyz[jdx][:,0]
        yfib = fibers_xyz[jdx][:,1]
        zfib = fibers_xyz[jdx][:,2]

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
            fibers_hinged[jdx] = True
            num_hinged += 1
        else:
          fibers_hinged[jdx] = True
          num_hinged += 1
        temp_l.append(fibers_l[jdx])
        temp_xyz.append(fibers_xyz[jdx])
        temp_hinged.append(fibers_hinged[jdx])
    fibers_l, fibers_hinged, fibers_xyz, fibers_site_idcs, fibers_terminal_length = temp_l, temp_hinged, temp_xyz, temp_site_idcs, temp_terminal_length

    # NUCLEATE NEW MTs
    active_sites_arr = np.array([])
    if len(active_sites_idcs) > 0:
      active_sites_arr = np.reshape(nuc_sites[active_sites_idcs[0]],(1,3))
    for k in np.arange(1,len(active_sites_idcs)):
      loc = nuc_sites[active_sites_idcs[k]]
      active_sites_arr = np.concatenate((active_sites_arr,np.reshape(loc,(1,3))),axis = 0)

    num_to_nucleate = min(np.random.poisson(nuc_rate * dt), len(passive_sites_idcs))
    for k in np.arange(num_to_nucleate):
      cannot_place_fiber = True
      ntrial = 0
      while cannot_place_fiber and ntrial < 50:
        # choose a link
        idx_in_passive = np.random.randint(len(passive_sites_idcs))
        idx_in_all = passive_sites_idcs[idx_in_passive]
        site_location = nuc_sites[idx_in_all]
        site_normal = site_location / np.linalg.norm(site_location)

        # if there active links, then check interfilament spacing
        if active_sites_arr.size > 0:
          dummy_links = np.concatenate((active_sites_arr, np.reshape(site_location, (1,3))), axis=0)
          dx = dummy_links[:,0] - dummy_links[:,0,None]
          dy = dummy_links[:,1] - dummy_links[:,1,None]
          dz = dummy_links[:,2] - dummy_links[:,2,None]
          dr = np.sqrt(dx**2 + dy**2 + dz**2)
          dfilament = min(dr[0,1:])
          if dfilament > min_ds:
            cannot_place_fiber = False
          else:
            ntrial += 1
        else:
          # if there is no active site, then nucleate anywhere
          cannot_place_fiber = False

      if not cannot_place_fiber:
        zone_h = 0.2
        if abs(site_location[2]) > zone_h:
          occupied_site_idcs.append(idx_in_all)
          terminal_time = tt + np.random.exponential(1/cat_rate)
          when_empty_occupied_sites.append(terminal_time)
          del passive_sites_idcs[idx_in_passive]
          cannot_place_fiber = True

      if not cannot_place_fiber:
        # save occupied link location
        if active_sites_arr.size > 0:
          active_sites_arr = np.concatenate((active_sites_arr,np.reshape(site_location,(1,3))), axis=0)
        else:
          active_sites_arr = np.reshape(site_location,(1,3))
        # update active and remaining link lists:
        del passive_sites_idcs[idx_in_passive]
        active_sites_idcs.append(idx_in_all)

        # nucleate MT with initial length if minL
        axis_s = np.empty((s.size, 3))
        axis_s[:, 0] = site_normal[0] * s
        axis_s[:, 1] = site_normal[1] * s
        axis_s[:, 2] = site_normal[2] * s
        fibers_xyz.append(axis_s * (0.5 / 2.0) + site_location)
        fibers_l.append(0.5)
        fibers_hinged.append(False)
        fibers_site_idcs.append(idx_in_all)
        fibers_terminal_length.append(grw_rate * np.random.exponential(1/cat_rate))

    Nmts_hinged[idx] = num_hinged
    if np.remainder(idx, 75) == 0:
      print('There are ', str(Nmts_hinged[idx]) , ' hinged MTs')
      # Write the random state, Nmts_hinged and time and idx
      random_file = fileName + 'randomState.step' + str(idx)
      with open(random_file, 'wb') as f:
        cpickle.dump(np.random.get_state(), f)

      nhinged_time = np.array([Nmts_hinged[idx], tt])
      np.savetxt(f_nhinged_time, nhinged_time[None,:])
