from __future__ import division, print_function
import numpy as np
import sys

if __name__ == '__main__':
  name = sys.argv[1]
  prefix = sys.argv[2]
  suffix = sys.argv[3]

  counter_frame = 0

  with open(name, 'r') as f:
    count_fibers = 0
    counter = 0

    for i, line in enumerate(f):
      if count_fibers == 0 and counter == 0:
        Nfibers = int(line.split()[0])
        count_fibers = Nfibers
        coor = []
        info_f = []
        continue


      data = line.split()
      if count_fibers > 0 and counter == 0:
        info_f_local = [int(data[0])]
        info_f.append(info_f_local)
        counter = int(data[0])
        count_fibers -= 1
      else:
        location = [float(data[0]), float(data[1]), float(data[2])]
        coor.append(location)
        counter -= 1

      if count_fibers == 0 and counter == 0:
        coor = np.array(coor)
        info_f = np.array(info_f).flatten()
        offset = 0       

        with open(prefix + str(counter_frame) + suffix, 'w') as f_out:
          for j in range(Nfibers):
            np.savetxt(f_out, coor[offset:offset + info_f[j]], delimiter=', ', header=str(j))
            offset += info_f[j]
        counter_frame += 1

