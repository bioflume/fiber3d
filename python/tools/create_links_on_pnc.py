from __future__ import division, print_function
import numpy as np
import sys

def compute_energy(r):
  N = r.size // 3
  
  # Pair-pair contribution
  dx = r[:,0] - r[:,0, None]
  dy = r[:,1] - r[:,1, None]
  dz = r[:,2] - r[:,2, None]
  dr = np.sqrt(dx**2 + dy**2 + dz**2)
  sel = dr > 1e-12
  dr_inv = np.zeros_like(dr)
  dr_inv[sel] = 1.0 / dr[sel]
  energy = np.sum(dr_inv.flatten()) / N 

  # Self contribution
  #r_norm = np.linalg.norm(r, axis=1)
  #cosTheta = r[:,2] / r_norm 
  #sel = cosTheta > np.cos(135 * np.pi / 180)
  #energy += np.sum(sel) * 1000.0
  #print('# sel = ', np.sum(sel))

  return energy

def create_nucleation_sites(N, R_eff, kT, max_steps):
  # Create random initial configuration
  r = np.random.randn(N, 3)
  #r[:,2] += -10.0
  r_norm = np.linalg.norm(r, axis=1)
  r = (r / r_norm[:, None]) * R_eff
  energy = compute_energy(r)
  accepted = 0
  
  # Loop over steps to minimaze energy
  for step in range(max_steps):
    # Move configuration
    displacement = 1e-03 * R_eff * np.random.randn(N,3)
    r_guess = r + displacement
    r_guess_norm = np.linalg.norm(r_guess, axis=1)
    r_guess = (r_guess / r_guess_norm[:, None]) * R_eff

    # Compute new energy
    energy_guess = compute_energy(r_guess)
    
    if np.random.rand(1) < np.exp(-(energy_guess - energy) / kT):
      r = r_guess
      energy = energy_guess   
      accepted += 1

  return r

if __name__ == '__main__':
  
  N = int(sys.argv[1])
  R_cent = float(sys.argv[2])
  kT = float(sys.argv[3])
  max_steps = int(sys.argv[4])
  
  R_pnc = float(sys.argv[5])
  xcent = float(sys.argv[6])
  ycent = float(sys.argv[7])
  zcent = float(sys.argv[8])
  angle = float(sys.argv[9]) # should be 0-90 degrees


  # Create nucleation sites on centrosome
  r_links = create_nucleation_sites(N, R_cent, kT, max_steps)
  

  norm_links = []
  
  for i, r in enumerate(r_links):
    r_norm = np.linalg.norm(r)
    axis = r / r_norm
    norm_links.append(axis)
  norm_links = np.array(norm_links)
    
  # Find projections on PNC
  rnucleus = np.zeros_like(r_links)
  for i, r in enumerate(r_links):
    normx = norm_links[i,0]
    normy = norm_links[i,1]
    normz = norm_links[i,2]
    b = 2*(normx*xcent+normy*ycent+normz*zcent)
    c = -R_pnc**2 + xcent**2 + ycent**2 + zcent**2
    coeff = [1, b, c]
    sol = np.roots(coeff)
    sol = sol[sol>0]
    rnucleus[i,0] = xcent + sol*normx
    rnucleus[i,1] = ycent + sol*normy
    rnucleus[i,2] = zcent + sol*normz

  # Now include those within given angle (at the top)
  count = 0
  r_at_angle = []
  for i, r in enumerate(rnucleus):
    xnorm = np.sqrt(rnucleus[i,0]**2 + rnucleus[i,1]**2 + rnucleus[i,2]**2)
    cosAtPoint = rnucleus[i,2] / xnorm
    if rnucleus[i,2] > 0:
      if cosAtPoint >= np.cos(angle * np.pi / 180):
        count += 1
        r_at_angle.append(rnucleus[i])
        
  # Now write them
  r_at_angle = np.array(r_at_angle) 
  # Define center and spring constants
  center = np.array([0.0, 0.0, 0.0])
  spring_constant = 10.0
  spring_constant_angle = 1.0
  # Loop over locations and print link
  print(count*2)
  for i, r in enumerate(r_at_angle):
    
    site_norm = np.linalg.norm(r_at_angle[i] - np.array([xcent, ycent, zcent]))
    axis = r_at_angle[i] - np.array([xcent, ycent, zcent])
    axis = axis / site_norm
    
    print(spring_constant, ' ', spring_constant_angle, ' ', end='')
    np.savetxt(sys.stdout, r_at_angle[i], newline=' ') 
    np.savetxt(sys.stdout, axis, newline=' ')
    print('')
    
    rdown = r_at_angle[i]
    rdown[2] = -r_at_angle[i,2]
    site_norm = np.linalg.norm(rdown - np.array([xcent, ycent, -zcent]))
    axis = rdown - np.array([xcent, ycent, -zcent])
    axis = axis / site_norm
    print(spring_constant, ' ', spring_constant_angle, ' ', end='')
    np.savetxt(sys.stdout, rdown, newline=' ') 
    np.savetxt(sys.stdout, axis, newline=' ')
    print('')
      

