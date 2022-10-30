'''
Small class to work with active force generators.
'''

from __future__ import print_function, division
import numpy as np
import imp

# If pycuda is installed import mobility_pycuda
try: 
  imp.find_module('pycuda')
  found_pycuda = True
except ImportError:
  found_pycuda = False
print('found_pycuda = ', found_pycuda)
if found_pycuda and False:
  try:
    import force_generator_pycuda
  except ImportError:
    from force_generator import force_generator_pycuda


class force_generator():
  '''
  Small class to handle force generators.
  '''

  def __init__(self, r, *args, **kwargs):
    # Store some parameters
    self.r = np.copy(r)
    self.force = np.empty_like(r)
    self.occupied = np.array([-1 for i in range(r.size // 3)], dtype=np.int32)
    self.radius = kwargs.get('radius', 0.0)
    self.spring_constant = kwargs.get('spring_constant', 0.0)
    self.active_force = kwargs.get('active_force', 0.0)
    self.unbind_period = kwargs.get('unbind_period', 0.0)
    self.alpha = kwargs.get('alpha', 1.0)

  def trap_particle(self, r_particles, num_particles,  num_points_particle, offset_particles, *args, **kwargs):
    '''
    If the active center is free, it captures one of the particles closer than
    the force_generator's radius. If there are several particles close, it selects one randomly.

    TODO: now the computational cost of this method is O(N**2),
    write a O(N) cost algorithm.
    '''
    
    # Loop over force generators
    for i in range(int(self.r.size / 3)):
      # Skip occupied force generators
      if self.occupied[i] > 0:
        continue

      # Compute distance between all particles and force_generator
      rxij = self.r[i,0] - r_particles[:,0,None]
      ryij = self.r[i,1] - r_particles[:,1,None]
      rzij = self.r[i,2] - r_particles[:,2,None]
      rij = np.sqrt(rxij**2 + ryij**2 + rzij**2)
      
      # Get possible pairs (there may be several points per particle)
      sel_ij = rij < (self.radius * (1 + self.alpha * 10))
      
      # Select possible particles
      sel_par = []
      for j in range(num_particles):
        if np.any(sel_ij[offset_particles[j] : offset_particles[j] +  num_points_particle[j]]):
          sel_par.append(j)       

      # Select one particle between the candidates
      if len(sel_par) > 0:
        self.occupied[i] = sel_par[np.random.random_integers(0, len(sel_par) - 1)]
    

  def release_particle(self, r_particles, num_points_particle, offset_particles, dt, *args, **kwargs):
    '''
    If the active center is occupied, it release its particle with 
    probability = release_period / dt.
    If the partilce is farther than the force_generator radius break the link.
    '''

    # Get occupied force generators
    sel_occupied = np.array([x > -1 for x in self.occupied])

    # Generate one random number between 0 and 1 for each
    # occupied force generator
    r = np.random.rand(sel_occupied.size)

    # unbind with probablity 1 - epx(-dt / tau)
    sel_unbind = r > np.exp(-dt / np.maximum(self.unbind_period, np.finfo(np.float).eps))
    self.occupied[sel_unbind] = -1

    # Loop over force generators
    for i in range(int(self.r.size / 3)):
      # Skip free force generators
      if self.occupied[i] < 0:
        continue

      first_point = offset_particles[self.occupied[i]]
      last_point = offset_particles[self.occupied[i]] + num_points_particle[self.occupied[i]]

      # Compute distance between force_generator and its particle
      rxij = self.r[i,0] - r_particles[first_point : last_point,0]
      ryij = self.r[i,1] - r_particles[first_point : last_point,1]
      rzij = self.r[i,2] - r_particles[first_point : last_point,2]
      rij = np.sqrt(rxij**2 + ryij**2 + rzij**2)
      
      # If the particle is far break the link
      sel_ij = rij > (self.radius * (1 + self.alpha * 10))
      if np.all(sel_ij):
        self.occupied[i] = -1

  def compute_force(self, r_particles, rs_particle, num_particles, num_points_particle, offset_particles, *args, **kwargs):
    '''
    If the active center it is occupied it computes the passive and
    active force generated on the particle.
    '''
    force = np.zeros_like(r_particles)
    self.force[:,:] = 0.0
    I = np.eye(3)
   
    # Loop over force generators
    for i in range(int(self.r.size / 3)):
      # Skip occupied force generators
      if self.occupied[i] < 0:
        continue        
      first_point = offset_particles[self.occupied[i]]
      last_point = offset_particles[self.occupied[i]] + num_points_particle[self.occupied[i]]

      # Loop over points in particle
      force_g = np.zeros(3)
      count = 0
      for j in range(first_point, last_point):
        rij = self.r[i] - r_particles[j]

        # Compute forces
        # if np.linalg.norm(rij) < self.radius:
        # Active force
        f = self.active_force * rs_particle[j]
         
        # Conservative force.
        f += self.spring_constant * np.dot(I - np.outer(rs_particle[j],rs_particle[j]), rij)

        # Smooth forces
        f = f * (1.0 - 1.0 / (1.0 + np.exp(-(np.linalg.norm(rij) - self.radius) / (self.alpha * self.radius))))

        # Add forces to particle and force generator
        force[j] += f
        force_g -= f
        count += 1

      # Save force acting on force generator
      self.force[i] += force_g / np.maximum(count, 1.0)
    return force


  def trap_particle_pycuda(self, r_particles, num_particles, num_points_particle, offset_particles, *args, **kwargs):
    '''
    If the active center is free, it captures one of the particles closer than
    the force_generator's radius. If there are several particles close, it selects one randomly.

    TODO: now the computational cost of this method is O(N**2),
    write a O(N) cost algorithm.
    '''
    force_generator_pycuda.trap_particle_pycuda(self.r, 
                                                self.occupied, 
                                                r_particles, 
                                                num_particles,  
                                                num_points_particle, 
                                                offset_particles, 
                                                self.radius, 
                                                self.alpha, 
                                                *args, **kwargs)
    


  

  def compute_force_pycuda(self, r_particles, rs_particle, num_particles, num_points_particle, offset_particles, *args, **kwargs):
    '''
    If the active center it is occupied it computes the passive and
    active force generated on the particle.
    '''

    return force_generator_pycuda.compute_force_pycuda(self.force, 
                                                       self.r, 
                                                       self.occupied, 
                                                       self.active_force,
                                                       self.spring_constant,
                                                       self.radius,
                                                       self.alpha,
                                                       r_particles, 
                                                       rs_particle, 
                                                       num_particles, 
                                                       num_points_particle, 
                                                       offset_particles, 
                                                       *args, **kwargs)

