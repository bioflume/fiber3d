import numpy as np
import sys

if __name__ == '__main__':

  name = sys.argv[1]

  # counter = 0
  # Nfibers = 0
  # with open(name, 'r') as f:
  #   for line in f:
  #     if Nfibers == 0 and counter == 0:
  #       Nfibers = int(line.split()[0])
  #     else:
  #       if counter == 0:
  #         data = line.split()
  #         counter = int(data[0])
  #         Nfibers -= 1
  #         print counter, '\n#'
  #       else:
  #         print 'O', line,
  #         counter -= 1



  with open(name, 'r') as f:
    count_fibers = 0
    counter = 0
    for i, line in enumerate(f):
      if count_fibers == 0 and counter == 0:
        # print 'XXX ', line
        Nfibers = int(line.split()[0])
        count_fibers = Nfibers
        coor = []
        # counter = -1
        continue


      data = line.split()
      if count_fibers > 0 and counter == 0:
        # print 'AAA', line
        info_f_local = [int(data[0])]
        counter = int(data[0])
        count_fibers -= 1
      else:
        # print 'coor', line
        location = ['O', float(data[0]), float(data[1]), float(data[2])]
        coor.append(location)
        counter -= 1

      if count_fibers == 0 and counter == 0:
        print len(coor), '\n#'
        for e in coor:
          print '  '.join(str(x) for x in e)


