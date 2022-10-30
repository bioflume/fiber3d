from __future__ import division, print_function
import numpy as np
import sys
sys.path.append('../')
from read_input import read_fibers_file
from quaternion import quaternion

if __name__ == '__main__':
  
  name_links = sys.argv[2]
  name_fibers = sys.argv[1]
  fibers_info, fibers_coor = read_fibers_file.read_fibers_file(name_fibers)
  
  links = np.loadtxt(name_links, skiprows=1)
  N = links.size // 8

  zcent = float(sys.argv[3])
  angle = float(sys.argv[4]) # 0-90 degrees

  cent_file = sys.argv[5]
  


  # Find projections on PNC
  count = 0
  index_include = []
  for i, r in enumerate(links):
    

    xnorm = np.sqrt(r[2]**2 + r[3]**2 + r[4]**2)
    if zcent > 0: # then cut the ones below
      if r[4] < 0:
        cosAtPoint = -r[4] / xnorm
        if cosAtPoint <= np.cos(angle * np.pi /180):
          count += 1
          index_include.append(i)
      else:
        count+=1
        index_include.append(i)
    else:
      if r[4] > 0:
        cosAtPoint = r[4] / xnorm
        if cosAtPoint <= np.cos(angle * np.pi /180):
          count += 1
          index_include.append(i)
      else:
        count += 1
        index_include.append(i)




  
  fl1 = open(cent_file+ '.links','wb')
  
  ff1 = open(cent_file + '.fibers','wb')
  

  np.savetxt(fl1, np.array([count],dtype = int))
  np.savetxt(ff1, np.array([count],dtype = int))


  num_points = 32
  for i, idx in enumerate(index_include):
    
    link = links[idx] 
    np.savetxt(fl1,link[None,:])
    fib_info = np.array([fibers_info[idx][0],fibers_info[idx][1],fibers_info[idx][2]])
    np.savetxt(ff1,fib_info[None,:])
    
    fib_coor = fibers_coor[idx*num_points:(idx+1)*num_points]
    np.savetxt(ff1,fib_coor)
      
  
      
    
    
      

