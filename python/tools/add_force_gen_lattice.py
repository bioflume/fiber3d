from __future__ import division, print_function
import numpy as np
import sys


if __name__ == '__main__':
  # Read parameters
  forgen_file_name = sys.argv[1] # final file name
  forgen_set_1_file_name = sys.argv[2]
  forgen_set_2_file_name = sys.argv[3]

  r_MM_1 = np.loadtxt(forgen_set_1_file_name)
  r_MM_2 = np.loadtxt(forgen_set_2_file_name)

  # put the first ones

  with open(forgen_file_name, 'w') as f:
    for mm_loc in r_MM_1:
      f.write(str(mm_loc[0]) + '  ' + str(mm_loc[1]) + '  ' + str(mm_loc[2]) + '\n')

    for mm_loc in r_MM_2:
      f.write(str(mm_loc[0]) + '  ' + str(mm_loc[1]) + '  ' + str(mm_loc[2]) + '\n')

  
  
  
