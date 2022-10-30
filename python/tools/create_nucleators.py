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
    
  


if __name__ == '__main__':
  # Read parameters
  N = int(sys.argv[1])
  R_eff = float(sys.argv[2])
  kT = float(sys.argv[3])
  max_steps = int(sys.argv[4])

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
    print(step, energy, accepted)


  # Save last configuration
  name = sys.argv[5]
  with open(name, 'w') as f:
    #f.write(str(N) + '\n# \n')
    for x in r:
      f.write(str(0) + '  ' + str(x[0]) + '  ' + str(x[1]) + '  ' + str(x[2]) + '\n')

