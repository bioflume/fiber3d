from __future__ import division, print_function
import numpy as np
import sys


if __name__ == '__main__':
  # Read parameters
  forgen_file_name = sys.argv[1] # 0 centered motors
  z_loc_top = float(sys.argv[2]) # z_location of top nuclue
  #z_loc_bot = float(sys.argv[3]) # z_location of bottom nucleus
  
  r_MM = np.loadtxt(forgen_file_name)

  # put the first ones
  name = 'kk.dat'
  with open(name, 'w') as f:
    for mm_loc in r_MM:
      f.write(str(mm_loc[0]) + '  ' + str(mm_loc[1]) + '  ' + str(mm_loc[2]+z_loc_top) + '\n')

    #for mm_loc in r_MM:
    #  f.write(str(mm_loc[0]) + '  ' + str(mm_loc[1]) + '  ' + str(mm_loc[2]+z_loc_bot) + '\n')

  
  
  
