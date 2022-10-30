import numpy as np
import sys
import ntpath
sys.path.append('../')
from tstep import tstep as tstep
from tstep import initialize
from body import body
from fiber import fiber
from tstep import tstep_utils
from read_input import read_input
from read_input import read_fibers_file
from read_input import read_vertex_file
from read_input import read_clones_file
from read_input import read_links_file
from quaternion import quaternion



if __name__ == '__main__':

  # INPUT FILE: includes fiber, molecular motor parameters and files to read fiber, body, mm configs
  nucleation_rate = 62.5
  rate_catastrophe = 0.015
  v_growth = 0.75
  minL = 0.3
  body_file = []
  body_file.append('./Structures/july15_centrosome_runs/centrosome_at_center.clones')
  body_file.append('./Structures/july15_centrosome_runs/centrosome_at_halfway.clones')
  nuc_sites_file = './Structures/july15_centrosome_runs/centrosome_Reff0p5_300nucs.links'

  fiber_ds = 0.5/16 # um per segment
  dt = 5e-3

  filename = 'data/dynInstability/run'

  options = initialize.set_options(adaptive_num_points = True,
                                  num_points_max = 96,
                                  adaptive_time = False,
                                  time_step_scheme = 'time_step_dry',
                                  order = 1,
                                  dt = dt,
                                  output_txt_files = True,
                                  n_save = 100,
                                  output_name=filename)

  prams = initialize.set_parameters(eta = 1.0,
                                   Efib = 10,
                                   epsilon = 1e-03,
                                   final_time = 20,
                                   growing = 1)

  # Create fibers
  fibers = []
  fibers_names = []
  f_fibers_ID = []

  # Create body
  bodies = []
  bodies_types = []
  bodies_names = []
  f_bodies_ID = []
  structures_body_fibers_ID = []
  num_bodies, body_locations, body_orientations = read_clones_file.read_clones_file(body_file[0])
  head, tail = ntpath.split(body_file[0])
  structures_body_fibers_ID = tail[:-7]

  bodies_types.append(num_bodies)
  bodies_names.append(structures_body_fibers_ID)
  ref_config = np.array([0,0,0])
  b = body.Body(body_locations[0], body_orientations[0], ref_config, ref_config, np.ones(ref_config.size//3))
  b.ID = structures_body_fibers_ID
  bodies.append(b)

  fibers_names.append(structures_body_fibers_ID)

  # Read nucleation nuc_sites_file
  spring_constants, spring_constants_angle, links_location, axes = read_links_file.read_links_file(nuc_sites_file)
  bodies[-1].nuc_sites = links_location
  bodies[-1].active_sites_idcs = []
  bodies[-1].passive_sites_idcs = np.arange(links_location.size//3).tolist()
  bodies[-1].radius = np.sqrt(links_location[0,0]**2 + links_location[0,1]**2 + links_location[0,2]**2)
  bodies[-1].min_ds = 0.10

  num_bodies, body_locations, body_orientations = read_clones_file.read_clones_file(body_file[1])
  head, tail = ntpath.split(body_file[1])
  structures_body_fibers_ID = tail[:-7]

  bodies_types.append(num_bodies)
  bodies_names.append(structures_body_fibers_ID)
  ref_config = np.array([0,0,0])
  b = body.Body(body_locations[0], body_orientations[0], ref_config, ref_config, np.ones(ref_config.size//3))
  b.ID = structures_body_fibers_ID
  bodies.append(b)

  fibers_names.append(structures_body_fibers_ID)

  # Read nucleation nuc_sites_file
  spring_constants, spring_constants_angle, links_location, axes = read_links_file.read_links_file(nuc_sites_file)
  bodies[-1].nuc_sites = links_location
  bodies[-1].active_sites_idcs = []
  bodies[-1].passive_sites_idcs = np.arange(links_location.size//3).tolist()
  bodies[-1].radius = np.sqrt(links_location[0,0]**2 + links_location[0,1]**2 + links_location[0,2]**2)
  bodies[-1].min_ds = 0.10



  # FILES
  name = options.output_name + '_' + bodies_names[0] + '_fibers.txt'
  fID = open(name, 'wb', buffering = 100)
  f_fibers_ID.append(fID)

  name = options.output_name + '_' + bodies_names[0] + '_clones.txt'
  fID = open(name,'wb', buffering = 100)
  f_bodies_ID.append(fID)

  name = options.output_name + '_' + bodies_names[1] + '_fibers.txt'
  fID = open(name, 'wb', buffering = 100)
  f_fibers_ID.append(fID)

  name = options.output_name + '_' + bodies_names[1] + '_clones.txt'
  fID = open(name,'wb', buffering = 100)
  f_bodies_ID.append(fID)
  name = options.output_name + '_time_system_size.txt'
  f_time_system = open(name, 'wb', buffering=100)

  f_log = open(options.output_name + '.logFile', 'w+')

  time = 0
  while time <= prams.final_time:
    time += dt
    f_log.write('Time = ' + str(time))
    fibers, fibers_gone, bodies, num_nucleated = tstep_utils.dynamic_instability(fibers,
                                                                                bodies,
                                                                                prams,
                                                                                options,
                                                                                nucleation_rate,
                                                                                rate_catastrophe,
                                                                                v_growth,
                                                                                options.dt,
                                                                                fiber_ds,
                                                                                minL,
                                                                                None,None,None)
    fiber_L = []
    fibers_types = np.zeros(len(bodies))
    for k, fib in enumerate(fibers):
      fiber_L.append(fib.length)
      fibers_types[fib.attached_to_body] += 1

    Larray = np.array(fiber_L)
    Lmean = np.mean(Larray)
    Lstd = np.std(Larray)
    message = 'There are ' + str(len(fibers)) + ' fibers.'
    print(message)
    f_log.write(message)
    message = 'Lmean = ' + str(Lmean)
    print(message)
    f_log.write(message)
    message = 'Lstd = ' + str(Lstd)
    print(message)
    f_log.write(message)

  # Write data to input file

  time_system = np.array([dt, time, int(time//dt), 0])
  np.savetxt(f_time_system, time_system[None, :])

  bodies_offset = 0
  for i, ID in enumerate(bodies_names):
    np.savetxt(f_bodies_ID[i], np.ones((1,7), dtype=int)*bodies_types[i])
    for j in range(bodies_types[i]):
      orientation = bodies[bodies_offset + j].orientation.entries
      np.savetxt(f_bodies_ID[i], np.ones((1,7), dtype=int)*bodies[bodies_offset+j].radius)
      out_body = np.array([bodies[bodies_offset + j].location[0],
                           bodies[bodies_offset + j].location[1],
                           bodies[bodies_offset + j].location[2],
                           orientation[0],
                           orientation[1],
                           orientation[2],
                           orientation[3]])
      np.savetxt(f_bodies_ID[i], out_body[None,:])

    bodies_offset += bodies_types[i]

  fiber_offset = 0
  for i, ID in enumerate(fibers_names):
    np.savetxt(f_fibers_ID[i], np.ones((1,4), dtype=int)*fibers_types[i])
    for j in range(int(fibers_types[i])):
      fiber_info = np.array([fibers[fiber_offset + j].num_points,
                             fibers[fiber_offset + j].E,
                             fibers[fiber_offset + j].length,
                             fibers[fiber_offset + j].growing])
      np.savetxt(f_fibers_ID[i], fiber_info[None,:])

      np.savetxt(f_fibers_ID[i], np.concatenate((fibers[fiber_offset + j].x, fibers[fiber_offset + j].tension[:,None]), axis=1))

    fiber_offset += int(fibers_types[i])

  print('\n\n\n# End')
