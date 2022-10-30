'''
Small module to read a vertex file of the rigid bodies.
'''
import numpy as np

def read_links_file(name_file):
  '''
  It reads a vertex file of the rigid bodies and return
  the coordinates as a numpy array with shape (Nblobs, 3).
  '''
  comment_symbols = ['#']   
  spring_constants = []
  spring_constants_angle = []
  coor = []
  axes = []
  with open(name_file, 'r') as f:
    i = 0
    for line in f:
      # Strip comments
      if comment_symbols[0] in line:
        line, comment = line.split(comment_symbols[0], 1)

      # Ignore blank lines
      line = line.strip()
      if line != '':
        if i == 0:
          Nlinks = int(float(line.split()[0]))
        else:
          data = line.split()
          spring_constant = [float(data[0])]
          spring_constant_angle = [float(data[1])]
          location = [float(data[2]), float(data[3]), float(data[4])]
          axis = [float(data[5]), float(data[6]), float(data[7])]
          axis_norm = np.linalg.norm(axis)
          axis = axis / axis_norm

          spring_constants.append(spring_constant)
          spring_constants_angle.append(spring_constant_angle)
          coor.append(location)
          axes.append(axis)
        i += 1

  spring_constants = np.array(spring_constants)
  spring_constants_angle = np.array(spring_constants_angle)
  coor = np.array(coor)
  axes = np.array(axes)
  return spring_constants, spring_constants_angle, coor, axes

