'''
Extract Chebyshev modes from the files run.fibers.0.lines
'''
import numpy as np
import sys
sys.path.append('../')
sys.path.append('../../')

from utils import cheb

if __name__ == '__main__':

  name = sys.argv[1]
  num_points = int(sys.argv[2])
  num_fibers = int(sys.argv[3])
  r = np.zeros((num_points, 3))
  w = np.zeros((num_points, num_fibers))

  count_fiber = 0
  with open(name, 'r') as f:
    for k, line in enumerate(f):
      if k % (num_points+1) == 0:
        pass
      else:
        index = (k % (num_points+1)) - 1
        r[index] = np.fromstring(line, sep=', ')
      if k % (num_points+1) == num_points:
        wx = abs(cheb.cheb_calc_coef(r[:,0]))
        wy = abs(cheb.cheb_calc_coef(r[:,1]))
        wz = abs(cheb.cheb_calc_coef(r[:,2]))
        w[:,count_fiber] = np.sqrt(wx**2 + wy**2 + wz*2)
        count_fiber += 1
  np.savetxt('kk.dat', w, delimiter='  ')

