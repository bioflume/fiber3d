from __future__ import division, print_function
import numpy as np
import sys
#import statistics
from scipy import stats

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
  
  # nucleation info
  minNmt = int(sys.argv[3]) # minimum number of MTs
  maxNmt = int(sys.argv[4]) # max number of MTs 
  # body info and cortex info
  dquad = float(sys.argv[5]) # quadrature spacing allowed
  #dquad = 0.16
  radius_a = float(sys.argv[6])
  radius_b = float(sys.argv[7])
  radius_c = float(sys.argv[8])

  fiber_file = sys.argv[9]
  active_link_file = sys.argv[10]

  stat_tol = 5e-2
  
  if minNmt < 400:
    stat_coeff = 4
  else:
    stat_coeff = 10


  # Dynamic instability
  nucleation_rate = 62.5 #300
  delta_t = 1/300 #1/1500
  # nucleation rate is about 300, which makes my delta_t<1/300
  rate_catastrophe = 0.015 
  v_growth = 0.75 # 0.15 * 10 



  # 1) FIRST DISCRETIZE BODY 
  # Read clones (assume one clone)
  x_clones = np.loadtxt(name_clones, skiprows=1)
  x_clones = x_clones.reshape(x_clones.size // 7, 7)
  center = x_clones[0, 0:3]
  orient = x_clones[0, 3:] 
  norm_orientation = np.linalg.norm(orient)
  orientation = quaternion.Quaternion(orient/norm_orientation)

  # to rotate link locations
  rotation_matrix = orientation.rotation_matrix()

  # Parameters
  num_points = 32
  
  # Get Chebyshev differential matrix
  s = np.linspace(0,2,num_points)
  
  # Load links
  links = np.loadtxt(name_links, skiprows=1)
  Nlinks = links.size // 8
  links_remain = []
  for link in links:
    links_remain.append(link)
  


  # Go over all the nucleation sites to find the maximum L
  d2corts = []
  for j, jlink in enumerate(links):
    link_loc = np.dot(rotation_matrix, jlink[2:5]) + center
    xfib, yfib, zfib = link_loc[0], link_loc[1], link_loc[2]

    x = (xfib-center[0])/radius_a
    y = (yfib-center[1])/radius_b
    z = (zfib-center[2])/radius_c 

    r_fiber = np.sqrt(x**2 + y**2 + z**2)
    phi_fiber = np.arctan2(y,(x+1e-12))
    theta_fiber = np.arccos(z/(1e-12+r_fiber))

    x_cort = radius_a*np.sin(theta_fiber)*np.cos(phi_fiber)
    y_cort = radius_b*np.sin(theta_fiber)*np.sin(phi_fiber)
    z_cort = radius_c*np.cos(theta_fiber)

    d2cort = np.sqrt((x_cort-xfib)**2 + (y_cort-yfib)**2 + (z_cort-zfib)**2)

    d2corts.append(d2cort)
  d2corts = np.array(d2corts)
  Lmax = np.mean(d2corts)
  # THE DISTRIBUTION
  Ls = np.linspace(0.3,Lmax,100)
  psi = np.exp(-Ls * rate_catastrophe/v_growth) * nucleation_rate / v_growth
  psi = psi / np.trapz(psi, Ls)

  expMu = np.trapz(psi * Ls, Ls)
  expStd = np.sqrt(np.trapz(psi * (Ls-expMu)**2, Ls))
  expSkew = np.trapz(psi * (Ls-expMu)**3 / expStd**3, Ls)
  expKurt = np.trapz(psi * (Ls-expMu)**4 / expStd**4, Ls)
  #expMu = 6.9
  #expStd = 5.1


  # March in time
  fiber_x = []
  fiber_L = []
  fiber_growing = []

  link_list = [] # keeps the reference link locations
  LmeanList = []
  occupied_links = np.array([]) # keeps the original one
  
  iStatEquilReached = False
  it = 0
  while len(fiber_L) < minNmt or not iStatEquilReached:
    
    # 1) NUCLEATE MTs
    if len(links_remain) > 0 and len(fiber_L) < maxNmt:
      num_to_nucleate = np.random.poisson(nucleation_rate*delta_t)
      if num_to_nucleate > len(links_remain):
        num_to_nucleate = len(links_remain)
    else:
      num_to_nucleate = 0

    print(num_to_nucleate, ' number of MTs will be nucleated')
    # nucleate each fiber
    for k in np.arange(num_to_nucleate):

      # Find a nuc. site
      cannot_place_fiber = True
      ntrial = 0
      while cannot_place_fiber and ntrial < 10:
        print('Trying to find a nuc. site...')
        # choose a link
        idx = np.random.randint(len(links_remain))
        ilink = links_remain[idx]
        location = np.dot(rotation_matrix, ilink[2:5])
        link_norm = ilink[5:]
    
        # check if this location leads to small interfilament spacing
        
        if occupied_links.size > 0:
          dummy_links = np.concatenate((occupied_links,np.reshape(location,(1,3))),axis=0)
          dx = dummy_links[:,0] - dummy_links[:,0,None]
          dy = dummy_links[:,1] - dummy_links[:,1,None]
          dz = dummy_links[:,2] - dummy_links[:,2,None]
          dr = np.sqrt(dx**2 + dy**2 + dz**2)
          dfilament = min(dr[0,1:])
          if dfilament > dquad: 
            cannot_place_fiber = False
            print('Found one!')
          else:
            print('Failed to find one, dfilam = ', dfilament)
            ntrial += 1
        else:
          cannot_place_fiber = False
          print('Found one!')

      if not cannot_place_fiber:
        Lfib = 0.3
        axis = np.dot(rotation_matrix, link_norm)
        axis_s = np.empty((s.size, 3))
        axis_s[:,0] = axis[0] * s
        axis_s[:,1] = axis[1] * s
        axis_s[:,2] = axis[2] * s
        axis_s = axis_s * (Lfib / 2.0) + location + center

        #fiber_growing.append(True)
        fiber_x.append(axis_s)
        fiber_L.append(Lfib)
        # Also keep the position of the exact nuc. site for dfilament calculation
        location = np.reshape(location,(1,3))
        if occupied_links.size > 0:
          occupied_links = np.concatenate((occupied_links,location),axis=0)
        else:
          occupied_links = np.copy(location)

        # update active links and remaining links
        link_list.append(links_remain[idx])
        del links_remain[idx]

    # 2) LOOP OVER ACTIVE NUC. SITES AND DYNAMIC INSTABILITY:
    
    for k, fib in enumerate(fiber_x):
      # coordinates of fiber
      xfib, yfib, zfib = fib[:,0], fib[:,1], fib[:,2]
      # length of fiber
      Lfib = fiber_L[k]
      #growing = fiber_growing[k]
      r = np.random.rand(1)
      
      v_length = v_growth
      if r > np.exp(-delta_t*rate_catastrophe):
        #del fiber_growing[k]
        del fiber_x[k]
        del fiber_L[k]
        links_remain.append(link_list[k])
        del link_list[k]
        np.delete(occupied_links,k,0)
      else:
        Lfib += v_growth * delta_t

        #fiber_growing[k] = growing
        fiber_L[k] = Lfib
        ilink = link_list[k]
        location = np.dot(rotation_matrix, ilink[2:5])
        link_norm = ilink[5:]

        # Update the configuration, if it reaches the cortex
        # then remove the MT
        axis = np.dot(rotation_matrix, link_norm)
        axis_s = np.empty((s.size, 3))
        axis_s[:,0] = axis[0] * s
        axis_s[:,1] = axis[1] * s
        axis_s[:,2] = axis[2] * s
        axis_s = axis_s * (Lfib / 2.0) + location + center
        fiber_x[k] = axis_s

        iReachCortex = False
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
        sel_in = d2cort <= 0.05 * cortex_point_r
        if sel_out.any() or sel_in.any(): iReachCortex = True

        if iReachCortex:
          print('It reaches the cortex, removing the MT')
          #del fiber_growing[k]
          del fiber_x[k]
          del fiber_L[k]
          links_remain.append(link_list[k])
          del link_list[k]
          np.delete(occupied_links,k,0)
    
    Lsum, Lstd = 0, 0
    if len(fiber_L) > 0:
      Larray = np.array(fiber_L)
      # Calculate mean
      Lmean = np.mean(Larray)
      Lstd = np.std(Larray)
      Lskew = stats.skew(Larray)
      Lkurt = stats.kurtosis(Larray)
      LmeanList.append(Lsum)
    it += 1
    print('There are ', len(fiber_L), ' MTs, currently')
    
    if len(fiber_L) > minNmt:
      print('Expected mean, std, skewness, kurtosis: ', expMu, expStd, expSkew, expKurt)
      print('Current values: ', Lmean, Lstd, Lskew, Lkurt)
      errInMean = np.abs(expMu-Lmean) / np.abs(expMu+1e-12)
      errInStd = np.abs(expStd-Lstd) / np.abs(expStd+1e-12)
      errInSkew = np.abs(expSkew-Lskew) / np.abs(expSkew+1e-12)
      errInKurt = np.abs(expKurt-Lkurt) / np.abs(expKurt+1e-12)

      testPassed = True
      if errInMean > stat_tol: 
        testPassed = False
        print('Mean did not converge')
      if errInStd > stat_tol: 
        testPassed = False
        print('Std did not converge')
      if errInSkew > 10 * stat_tol: 
        testPassed = False
        print('Skewness did not converge')
      #if errInKurt > stat_tol: 
      #  testPassed = False
      #  print('Kurtosis did not converge')
      if testPassed: 
        iStatEquilReached = True
        print('STATISICAL EQUILIBRIUM IS REACHED')

    #if len(fiber_L) > minNmt:
    #  stat_test = stats.anderson(Larray,'expon')
    #  print('Stat test value: ', stat_test[0])
    #  print('Stat test critical values: ', stat_test[1])
    #  if (stat_test[0] <= stat_coeff*stat_test[1]).any():
    #    print('STATISICAL EQUILIBRIUM IS REACHED')
    #    iStatEquilReached = True
      
      #stdInLast100 = statistics.pstdev(LmeanList[-100:])
      #print('Std of Lmean in the last 100 steps: ', stdInLast100)
      #print('Std of length population:', Lstd)
      #meanStdDiff = np.abs(Lsum-Lstd)/Lsum
      #if meanStdDiff <= 0.01 and LmeanList[-1] > 4.5:
      #  print(LmeanList[-1])
      #  print('STATISTICAL EQUILIBRIUM IS REACHED!')
      #  iStatEquilReached = True


  # write fiber file and active link file
  flink = open(active_link_file,'wb')
  ffiber = open(fiber_file,'wb')
  for k, Lfib in enumerate(fiber_L):
    if Lfib <= 0.3:
      del fiber_x[k]
      del fiber_L[k]  
      del link_list[k]

  num_fibers = len(fiber_L)
  np.savetxt(flink, np.array(([num_fibers]),dtype = int))
  np.savetxt(ffiber, np.array(([num_fibers]),dtype = int))
  E = 10.0
  for k, fib in enumerate(fiber_x):
    
    fiber_info = np.array([num_points, E, fiber_L[k]])
    np.savetxt(ffiber,fiber_info[None,:])
    np.savetxt(ffiber,fib)
    link = link_list[k]
    np.savetxt(flink,link[None,:])



