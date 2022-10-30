import numpy as np
import sys


if __name__ == '__main__':
  name_1 = sys.argv[1]
  name_2 = sys.argv[2]

  r_1 = np.loadtxt(name_1)
  r_2 = np.loadtxt(name_2)

  diff = r_1[:,1:4] - r_2[:,1:4]
  distance = np.linalg.norm(diff, axis=1)
  time = r_1[:,0]
  result = np.empty((time.size, 5))
  result[:,0] = time
  result[:,1:4] = diff
  result[:,4] = distance
  
  np.savetxt(sys.stdout, result)

