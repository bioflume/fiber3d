from __future__ import division, print_function
import numpy as np
import sys
sys.path.append('../')
from read_input import read_fibers_file
from quaternion import quaternion

if __name__ == '__main__':
  
  name_links = sys.argv[1]
  name_fibers = sys.argv[2]
  fibers_info, fibers_coor = read_fibers_file.read_fibers_file(name_fibers)
  
  links = np.loadtxt(name_links, skiprows=1)
  N = links.size // 8

  R_cent = float(sys.argv[3])
  
  R_pnc = float(sys.argv[4])
  xcent = float(sys.argv[5])
  ycent = float(sys.argv[6])
  zcent = float(sys.argv[7])

  cent_file_1 = sys.argv[8]
  cent_file_2 = sys.argv[9]


  # Find projections on PNC
  rcents = np.zeros((N,3))
  norm_cents = np.zeros((N,3))
  ntop = 0
  nbot = 0
  for i, r in enumerate(links):
    if r[4] > 0:
      zc = zcent
      ntop += 1
    else:
      zc = -zcent
      nbot += 1

    normx = r[5]
    normy = r[6]
    normz = r[7]
    
    norm_cents[i,0] = normx
    norm_cents[i,1] = normy
    norm_cents[i,2] = normz

    b = -2*(normx*(r[2]-xcent)+normy*(r[3]-ycent)+normz*(r[4]-zc))
    c = -R_cent**2 + ((r[2]-xcent)**2+(r[3]-ycent)**2+(r[4]-zc)**2)
    coeff = [1, b, c]
    sol = np.roots(coeff)
    sol = sol[sol>0]
    if sol.size > 1:
      sol = np.min(sol)
    rcents[i,0] = r[2] - sol*normx
    rcents[i,1] = r[3] - sol*normy
    rcents[i,2] = r[4] - sol*normz

  spring_constant = 10.0
  spring_constant_angle = 1.0
  spring_info = np.array([spring_constant, spring_constant_angle],dtype=int)
  
  fl1 = open(cent_file_1 + '.links','wb')
  fl2 = open(cent_file_2 + '.links','wb')

  ff1 = open(cent_file_1 + '.fibers','wb')
  ff2 = open(cent_file_2 + '.fibers','wb')

  np.savetxt(fl1, np.array([ntop],dtype = int))
  np.savetxt(fl2, np.array([nbot],dtype = int))
  np.savetxt(ff1, np.array([ntop],dtype = int))
  np.savetxt(ff2, np.array([nbot],dtype = int))


  top_center = np.array([0, 0, 7.25])
  bot_center = np.array([0, 0, -7.25])
  top_orient = np.array([1, 0, 0, 0])

  #top_center = np.array([0, 0,      12.7997])
  #bot_center = np.array([0, 0.1406, -6.1997])
  #top_orient = np.array([0.99999316, -0.0036999,0,0])
  norm_orient = np.linalg.norm(top_orient)
  orientation = quaternion.Quaternion(top_orient/norm_orient)
  rotation_matrix = orientation.rotation_matrix()

  offset = 0
  s = np.linspace(0,2,32)

  for i, r in enumerate(rcents):
    
    coor = np.copy(r)
    num_points = fibers_info[i][0]
    Lfib = fibers_info[i][2]
    fib_info = np.array([fibers_info[i][0],fibers_info[i][1],fibers_info[i][2]])
    if r[2] > 0:
      coor[2] -= zcent
      location = np.dot(rotation_matrix,coor)
      allInfo = np.concatenate((spring_info,coor,norm_cents[i]),axis=0)
      np.savetxt(fl1,allInfo[None,:])
      np.savetxt(ff1,fib_info[None,:])
      
      axis = np.dot(rotation_matrix, norm_cents[i])
      axis_s = np.empty((s.size, 3))
      axis_s[:,0] = axis[0] * s
      axis_s[:,1] = axis[1] * s
      axis_s[:,2] = axis[2] * s
      axis_s = axis_s * (Lfib / 2.0) + location + top_center
      np.savetxt(ff1,axis_s)
      
    else:
      coor[2] += zcent
      location = np.dot(rotation_matrix,coor)
      allInfo = np.concatenate((spring_info,coor,norm_cents[i]),axis=0)
      np.savetxt(fl2,allInfo[None,:])
      np.savetxt(ff2,fib_info[None,:])
      axis = np.dot(rotation_matrix, norm_cents[i])
      axis_s = np.empty((s.size, 3))
      axis_s[:,0] = axis[0] * s
      axis_s[:,1] = axis[1] * s
      axis_s[:,2] = axis[2] * s
      axis_s = axis_s * (Lfib / 2.0) + location + bot_center
      np.savetxt(ff2,axis_s)
      
    offset += num_points
    
    
      

