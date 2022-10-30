'''
Small module to read fibers configuration.

The format of the input file is:
number_of_fibers
number_of_points_fiber_0
x_0 y_0 z_0
.
.
.
number_of_points_fiber_1
x_0 y_0 z_0
.
.
.
'''
import numpy as np

def read_fibers_file(name_file):
  '''
  It reads the input file and returns the tuple
  (number_of_markers_per_fiber, configuration_fibers) where
  number_of_markers_per_fiber is a 1D array 
  configuration_fibers is 3D array.
  '''
  comment_symbols = ['#']   
  info = []
  coor = []
  counter = 0
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
          Nfibers = int(float(line.split()[0]))
        else:
          data = line.split()
          if counter == 0:
            # number of points
            info_f_local = [int(float(data[0]))]
            if len(data) > 1:
              # bending modulus
              info_f_local.append(float(data[1]))
            if len(data) > 2:
              # length
              info_f_local.append(float(data[2]))
            if len(data) > 3:
              # growing or shrinking
              if type(data[3]) == str:
                info_f_local.append(bool(data[3]))
              else:
                info_f_local.append(float(data[3]))
            info.append(info_f_local)
            counter = info[-1][0]
          else:
            location = [float(data[0]), float(data[1]), float(data[2])]
            coor.append(location)
            counter -= 1
        i += 1

  coor = np.array(coor)
  return info, coor



  return
