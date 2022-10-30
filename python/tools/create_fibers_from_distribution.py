from __future__ import division, print_function
import numpy as np
import sys
sys.path.append('../')
sys.path.append('../../')

from fiber import fiber
from utils import cheb
from body import body
from quaternion import quaternion
from shape_gallery import shape_gallery
from quadratures import Smooth_Closed_Surface_Quadrature_RBF

if __name__ == '__main__':
  name_clones = sys.argv[1]
  name_links = sys.argv[2]
  body_quad_rad = float(sys.argv[3]) 
  
  # nucleation info
  Lmean = float(sys.argv[4])
  minNmt = int(sys.argv[5])
  
  # body info and cortex info
  Nbody = int(sys.argv[6])
  radius_a = float(sys.argv[7])
  radius_b = float(sys.argv[8])
  radius_c = float(sys.argv[9])

  active_link_file = sys.argv[11]
  fiber_file = sys.argv[10]

  # 1) FIRST DISCRETIZE BODY 
  # Read clones (assume one clone)
  x_clones = np.loadtxt(name_clones, skiprows=1)
  x_clones = x_clones.reshape(x_clones.size // 7, 7)
  center = x_clones[0, 0:3]
  orient = x_clones[0, 3:] 
  norm_orientation = np.linalg.norm(orient)
  orientation = quaternion.Quaternion(orient/norm_orientation)

  rigid_body = body.Body(center, orientation, np.array([0,0,0]), np.array([0,0,0]), np.ones(1))
  #rigid_body.discretize_body_surface(shape = 'sphere', Nblobs = Nbody, radius = body_quad_rad)
  #body_config = rigid_body.get_r_vectors_surface()
  #dx = body_config[:,0] - body_config[:,0,None]
  #dy = body_config[:,1] - body_config[:,1,None]
  #dz = body_config[:,2] - body_config[:,2,None]
  #dr = np.sqrt(dx**2 + dy**2 + dz**2)
  
  # find quadrature spacing
  #dquad = min(dr[0,1:])
  dquad = 0.16

  # to rotate link locations
  rotation_matrix = orientation.rotation_matrix()

  # Parameters
  num_points = 32
  E = 10.0
  
  # Get Chebyshev differential matrix
  s = np.linspace(0,2,num_points)
  
  # Load links
  links = np.loadtxt(name_links, skiprows=1)
  # N = links.shape[0]
  Nlinks = links.size // 8
  links_remain = np.copy(links)

  if Nlinks == 1:
    links = np.reshape(links, (1, 8))
  
  # March in time
  fiber_x = []
  fiber_L = []
  link_list = [] # keeps the reference link locations
  num_fibers = 0
  currentLmean = 0
  occupied_links = np.array([]) # keeps the original one
  while num_fibers < minNmt and links_remain.size//8>0:
    # choose a link
    ilink = np.random.randint(links_remain.size//8)
    location = np.dot(rotation_matrix, links_remain[ilink,2:5])
    link_norm = links_remain[ilink,5:]
    
    # check if this location leads to small interfilament spacing
    place_fiber = True
    if occupied_links.size > 0:
      dummy_links = np.concatenate((occupied_links,np.reshape(location,(1,3))),axis=0)
      dx = dummy_links[:,0] - dummy_links[:,0,None]
      dy = dummy_links[:,1] - dummy_links[:,1,None]
      dz = dummy_links[:,2] - dummy_links[:,2,None]
      dr = np.sqrt(dx**2 + dy**2 + dz**2)
      dfilament = min(dr[0,1:])
      if dfilament < dquad:
        place_fiber = False


    if place_fiber:
      iReachCortex = True
      ntrial = 0
      #while iReachCortex and ntrial < 10:
      if True:
        # now sample length given Lmean from exponential dist
        Lfib = 0
        while Lfib < 0.3:
          Lfib = np.random.exponential(Lmean)

        axis = np.dot(rotation_matrix, link_norm)
        axis_s = np.empty((s.size, 3))
        axis_s[:,0] = axis[0] * s
        axis_s[:,1] = axis[1] * s
        axis_s[:,2] = axis[2] * s
        axis_s = axis_s * (Lfib / 2.0) + location + center

        # check if this is close to the cortex
        xfib,yfib,zfib = axis_s[:,0],axis_s[:,1],axis_s[:,2]
        x = xfib / radius_a
        y = yfib / radius_b
        z = zfib / radius_c

        r_true = np.sqrt(xfib**2 + yfib**2 + zfib**2)

        r_fiber = np.sqrt(x**2 + y**2 + z**2)
        phi_fiber = np.arctan2(y,(x+1e-12))
        theta_fiber = np.arccos(z/(1e-12+r_fiber))

        x_cort = radius_a*np.sin(theta_fiber)*np.cos(phi_fiber)
        y_cort = radius_b*np.sin(theta_fiber)*np.sin(phi_fiber)
        z_cort = radius_c*np.cos(theta_fiber)

        d2cort = np.sqrt((xfib-x_cort)**2 + (yfib-y_cort)**2 + (zfib-z_cort)**2) 
        cortex_point_r = np.sqrt(x_cort**2 + y_cort**2 + z_cort**2)
        sel_out = r_true >= cortex_point_r
        sel_in = d2cort <= 0.1 * cortex_point_r
        if not sel_out.any() and not sel_in.any(): iReachCortex = False
        #if not sel_out.any(): iReachCortex = False
        #if not sel_in.any(): iReachCortex = False

        if not iReachCortex:
          # if does not reach cortex then place the fiber
          fiber_x.append(axis_s)
          fiber_L.append(Lfib)
          location = np.reshape(location,(1,3))
          if occupied_links.size > 0:
            occupied_links = np.concatenate((occupied_links,location),axis=0)
          else:
            occupied_links = np.copy(location)
          link_list.append(links_remain[ilink])
          links_remain = np.delete(links_remain,ilink,0)
          currentLmean = (currentLmean * num_fibers + Lfib)/(num_fibers+1)
          print(num_fibers+1, ' fibers placed, Lmean: ', currentLmean)
          num_fibers += 1
        else:
          print('Reaches cortex')
          ntrial += 1
    else:
      print('Do not place fiber because of small dfilament')
      #links_remain = np.delete(links_remain,ilink,0)
        
  print('Mean L: ' , currentLmean)
  # write fiber file and active link file
  flink = open(active_link_file,'wb')
  ffiber = open(fiber_file,'wb')
  np.savetxt(flink, np.array(([num_fibers]),dtype = int))
  np.savetxt(ffiber, np.array(([num_fibers]),dtype = int))
  
  for k, fib in enumerate(fiber_x):
    
    fiber_info = np.array([num_points, E, fiber_L[k]])
    np.savetxt(ffiber,fiber_info[None,:])
    np.savetxt(ffiber,fib)
    link = link_list[k]
    np.savetxt(flink,link[None,:])



