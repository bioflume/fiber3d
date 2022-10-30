from __future__ import division, print_function
import numpy as np
import sys


if __name__ == '__main__':
  # Read parameters
  N = int(sys.argv[1])
  R_eff = float(sys.argv[2])
  kT = float(sys.argv[3])
  max_steps = int(sys.argv[4])

  # Create random initial configuration
  r = np.random.randn(N, 3)
  r_norm = np.linalg.norm(r, axis=1)
  r = (r / r_norm[:, None]) * R_eff
  dx = r[:,0] - r[:,0, None]
  dy = r[:,1] - r[:,1, None]
  dz = r[:,2] - r[:,2, None]
  dr = np.sqrt(dx**2 + dy**2 + dz**2)
  sel = dr > 1e-12
  dr_inv = np.zeros_like(dr)
  dr_inv[sel] = 1.0 / dr[sel]
  energy = np.sum(dr_inv.flatten()) / N
  accepted = 0
  
  # Loop over steps to minimaze energy
  displacement_0 = 0.1
  for step in range(max_steps):
    # Move configuration
    if displacement_0 > 1e-03:
      displacement_0 -= 1e-04
    displacement = displacement_0 * R_eff * np.random.randn(N,3)
    r_guess = r + displacement
    r_guess_norm = np.linalg.norm(r_guess, axis=1)
    r_guess = (r_guess / r_guess_norm[:, None]) * R_eff

    # Compute new energy
    dx = r_guess[:,0] - r_guess[:,0, None]
    dy = r_guess[:,1] - r_guess[:,1, None]
    dz = r_guess[:,2] - r_guess[:,2, None]
    dr = np.sqrt(dx**2 + dy**2 + dz**2)
    sel = dr > 1e-12
    dr_inv = np.zeros_like(dr)
    dr_inv[sel] = 1.0 / dr[sel]
    energy_guess = np.sum(dr_inv.flatten()) / N
    
    if np.random.rand(1) < np.exp(-(energy_guess - energy) / kT):
      r = r_guess
      energy = energy_guess   
      accepted += 1
    print(step, energy, accepted)


  # Save last configuration
  name = 'kk.xyz'
  with open(name, 'w') as f:
    f.write(str(N) + '\n# \n')
    for x in r:
      f.write('O  ' + str(x[0]) + '  ' + str(x[1]) + '  ' + str(x[2]) + '\n')
  name = 'kk.dat'
  with open(name, 'w') as f:
    #f.write(str(N) + '\n')
    for x in r:
      f.write(str(x[0]) + '  ' + str(x[1]) + '  ' + str(x[2]) + '\n')
  
