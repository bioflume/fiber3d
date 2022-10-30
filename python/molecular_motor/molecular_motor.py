'''
Small class to work with active molecular motors.
'''

from __future__ import print_function, division
import numpy as np
import sys

try:
  from numba import njit, prange
  from numba.typed import List
except ImportError:
  print('Numba not found')

from bpm_utilities import find_near_points as find_np
from utils import cheb
from utils import timer


class molecular_motor():
  '''
  Small class for molecular motors. Some notation

  attached_base = -2, means the motor base is free
  attached_base = -1, means the motor head is attached but not to a microtubule
  attached_base = i > -1 means that the motor is attached to the microtubule i
  '''

  def __init__(self, x, radius, speed_0, force_stall, spring_constant, rest_length, bind_frequency, unbind_frequency_0, diffusion = 0, kernel_sigma = 0.5, *args, **kwargs):
    # Store some parameters
    self.x = np.copy(x).reshape((x.size // 3, 3))
    self.x_ref = np.copy(self.x)
    self.radius = radius
    self.speed_0 = speed_0
    self.force_stall = force_stall
    self.spring_constant = spring_constant
    self.rest_length = rest_length
    self.bind_frequency = bind_frequency
    self.unbind_frequency_0 = unbind_frequency_0
    self.diffusion = diffusion
    self.kernel_sigma = kernel_sigma

    # Create other variables
    self.N = self.x.size // 3
    self.force = np.zeros_like(x).reshape((x.size // 3, 3))
    self.attached = np.array([-2 for i in range(x.size // 3)], dtype=np.int32)
    self.attached_base = np.array([-2 for i in range(x.size // 3)], dtype=np.int32)
    self.attached_head = np.array([-2 for i in range(x.size // 3)], dtype=np.int32)
    self.x_base = np.copy(x).reshape((x.size // 3, 3))
    self.x_head = np.copy(x).reshape((x.size // 3, 3))
    self.speed_base_0 = speed_0
    self.speed_head_0 = speed_0
    self.s_base = np.zeros(self.N)
    self.s_head = np.zeros(self.N)
    self.xs_base = np.zeros((self.N, 3))
    self.xs_head = np.zeros((self.N, 3))
    self.length_MT_base = np.zeros(self.N)
    self.length_MT_head = np.zeros(self.N)

  
  def find_x_xs_and_length_MT(self, fibers):
    '''
    For attached motors find the location in space, the microtubule orientation
    at the attachment point and the microtubule length.    
    '''

    # Compute x and xs for motor base and motor head
    for i in range(self.N):
      if self.attached_base[i] > -1:
        index = self.attached_base[i]
        self.length_MT_base[i] = fibers[index].length
        alpha_array = np.array(self.s_base[i] * (2.0 / self.length_MT_base[i])) - 1
        alpha_array = np.clip(alpha_array, -1.0, 1.0)
        self.x_base[i, 0] = cheb.cheb_eval(alpha_array, fibers[index].x_modes[:, 0], order=1)
        self.x_base[i, 1] = cheb.cheb_eval(alpha_array, fibers[index].x_modes[:, 1], order=1)
        self.x_base[i, 2] = cheb.cheb_eval(alpha_array, fibers[index].x_modes[:, 2], order=1)
        self.xs_base[i, 0] = cheb.cheb_eval(alpha_array, fibers[index].xs_modes[:, 0], order=1)
        self.xs_base[i, 1] = cheb.cheb_eval(alpha_array, fibers[index].xs_modes[:, 1], order=1)
        self.xs_base[i, 2] = cheb.cheb_eval(alpha_array, fibers[index].xs_modes[:, 2], order=1)

      if self.attached_head[i] > -1:
        index = self.attached_head[i]
        self.length_MT_head[i] = fibers[index].length
        alpha_array = np.array(self.s_head[i] * (2.0 / self.length_MT_head[i])) - 1
        alpha_array = np.clip(alpha_array, -1.0, 1.0)
        self.x_head[i, 0] = cheb.cheb_eval(alpha_array, fibers[index].x_modes[:, 0], order=1)
        self.x_head[i, 1] = cheb.cheb_eval(alpha_array, fibers[index].x_modes[:, 1], order=1)
        self.x_head[i, 2] = cheb.cheb_eval(alpha_array, fibers[index].x_modes[:, 2], order=1)
        self.xs_head[i, 0] = cheb.cheb_eval(alpha_array, fibers[index].xs_modes[:, 0], order=1)
        self.xs_head[i, 1] = cheb.cheb_eval(alpha_array, fibers[index].xs_modes[:, 1], order=1)
        self.xs_head[i, 2] = cheb.cheb_eval(alpha_array, fibers[index].xs_modes[:, 2], order=1)

    # Select motors with attached base or head
    sel_base = self.attached_base > -2
    sel_head = self.attached_head > -2
    self.x[sel_base] = self.x_base[sel_base]
    self.x[sel_head] = self.x_head[sel_head]
     

  def compute_force(self):
    '''
    Compute harmonic force between base and head

    F = -k * (|r_head - r_base| - rest_length) * (r_head - r_base) / |r_head - r_base|

    for attached motors.
    '''
    # Select motors with attached base and head
    sel_base = self.attached_base > -2
    sel_head = self.attached_head > -2
    sel_double_link = np.logical_and(sel_base, sel_head)
    
    # Compute distance between base and head
    self.x_base.reshape((self.N, 3))
    self.x_head.reshape((self.N, 3))
    r = np.empty_like(self.x_base)
    r[sel_double_link] = self.x_head[sel_double_link] - self.x_base[sel_double_link]
    r_norm = np.zeros(r.size // 3)
    r_norm[sel_double_link] = np.linalg.norm(r[sel_double_link], axis=1)
    sel_distance = r_norm > 0.0
    sel = np.logical_and(sel_double_link, sel_distance)
    # Compute force between base and head
    self.force.reshape((self.N, 3))
    self.force[:,:] = 0
    self.force[sel] = -(self.spring_constant * (r_norm[sel,None] - self.rest_length) / r_norm[sel,None]) * (self.x_head[sel] - self.x_base[sel]) 
    

  def walk(self, dt):
    '''
    Walk along a microtubule. We use the force-velocity relationship,

    v = v_0 * max(0, min(1, 1 + F_{m,alpha} / F_m^{stall}))

    with

    F_{m,alpha} = F_m * xs
    '''
    # Select motors with attached base or head to a MT
    sel_base = self.attached_base > -1
    sel_head = self.attached_head > -1

    # Compute force parallel to microtubule at attachment point
    force_parallel_base = np.zeros(self.N)
    force_parallel_head = np.zeros(self.N)
    force_parallel_base[sel_base] = np.einsum('ij,ij->i', self.force[sel_base], self.xs_base[sel_base])
    force_parallel_head[sel_head] = np.einsum('ij,ij->i', self.force[sel_head], self.xs_head[sel_head])
    # Compute speed along microtubule
    speed_base = np.zeros(self.N)
    speed_head = np.zeros(self.N)
    speed_base[sel_base] = self.speed_base_0 * np.maximum(0, np.minimum(1.0, 1.0 + force_parallel_base[sel_base] / self.force_stall))
    speed_head[sel_head] = self.speed_head_0 * np.maximum(0, np.minimum(1.0, 1.0 + force_parallel_head[sel_head] / self.force_stall))

    # Update s
    self.s_base[sel_base] += speed_base[sel_base] * dt
    self.s_head[sel_head] += speed_head[sel_head] * dt
    # keep motors at microtubules ends
    sel = self.s_base > self.length_MT_base
    self.s_base[sel] = self.length_MT_base[sel]
    sel = self.s_base < 0
    self.s_base[sel] = 0
    sel = self.s_head > self.length_MT_head
    self.s_head[sel] = self.length_MT_head[sel]
    sel = self.s_head < 0
    self.s_head[sel] = 0

    # Keep track of x
    self.x[sel_head] = self.x_head[sel_head]
    self.x[sel_base] = self.x_head[sel_base]

    
  def diffuse(self, dt):
    '''
    Unlinked microtubules diffuse in 3D, the external flow is not
    taken into accound,

    self.x = self.x + sqrt(2*D*dt) * W
    '''
    # Select motors with free base and head
    sel_base = self.attached_base == -2
    sel_head = self.attached_head == -2
    sel = np.logical_and(sel_base, sel_head)
    
    #W = np.random.randn(self.N, 3)
    #self.x[sel] += np.sqrt(2.0 * self.diffusion * dt) * W[sel]
    #self.x_base[sel] = self.x[sel]
    #self.x_head[sel] = self.x[sel]
  
  def spread_force_body(self, body):
    '''
    Spread force onto body and calculate torque created
    Opposite sign to the force spread on fiber
    '''
    force_body = np.zeros((1,3))
    torque_body = np.zeros((1,3))

    for i in range(self.N):
      if self.attached_base[i] > -1: # base attached to fiber, head attached to body
        r = self.x_base[i]-body.location
        force2bdy = self.force[i]

        # Force at the center
        force_body += force2bdy

        # Torque at the center
        torque_body[0,0] += r[1] * force2bdy[2] - r[2] * force2bdy[1]
        torque_body[0,1] += r[2] * force2bdy[0] - r[0] * force2bdy[2]
        torque_body[0,2] += r[0] * force2bdy[1] - r[1] * force2bdy[0]

      if self.attached_head[i] > -1: # base attached to body, head attached to fiber
        r = self.x_base[i]-body.location
        force2bdy = -self.force[i]
  
        # Force at the center
        force_body += force2bdy

        # Torque at the center
        torque_body[0,0] += r[1] * force2bdy[2] - r[2] * force2bdy[1]
        torque_body[0,1] += r[2] * force2bdy[0] - r[0] * force2bdy[2]
        torque_body[0,2] += r[0] * force2bdy[1] - r[1] * force2bdy[0]

    return force_body, torque_body



  def spread_force(self, fibers):
    '''
    Spread force to the microtubule with the formula
    
    f(s) = S(s-s_end) * F_motor

    S(s) = exp(-s**2 / (2*sigma**2)) / sqrt(2*pi*sigma**2)

    with the width

    sigma = min(L/5, max(L/N, sigma_0))
    '''
    # Select motors with attached base and head
    sel_base = self.attached_base > -2
    sel_head = self.attached_head > -2

    # Spread force of both ends
    for i in range(self.N):
      if self.attached_base[i] > -1:
        index = self.attached_base[i]

        # Compute one-dimensional spreading function
        sigma2 = np.minimum(fibers[index].length / 5.0, np.maximum(fibers[index].length / fibers[index].num_points, self.kernel_sigma))**2
        s_imaginary_left  = -self.s_head[i]
        s_imaginary_right =  2.0 * self.length_MT_head[i] - self.s_head[i]
        
        alpha = np.linspace(-1, 1, fibers[index].num_points)
        fib_s = (1.0 + alpha)*(fibers[index].length / 2.0)

        s_diff       = (fib_s - self.s_base[i]) 
        s_diff_left  = (fib_s - s_imaginary_left) 
        s_diff_right = (fib_s - s_imaginary_right) 

        Spread = np.exp(-s_diff**2 / (2.0 * sigma2)) / np.sqrt(2.0 * np.pi * sigma2) + \
                 np.exp(-s_diff_left**2 / (2.0 * sigma2)) / np.sqrt(2.0 * np.pi * sigma2) + \
                 np.exp(-s_diff_right**2 / (2.0 * sigma2)) / np.sqrt(2.0 * np.pi * sigma2)

        # Spread force
        # Mind the minus sign, the force is defined on the motor head
        # so Force_base = -Force_head
        fibers[index].force_motors += -Spread[:,None] * self.force[i]

      if self.attached_head[i] > -1:
        index = self.attached_head[i]
        
        alpha = np.linspace(-1, 1, fibers[index].num_points)
        fib_s = (1.0 + alpha)*(fibers[index].length / 2.0)
        

        # Compute one-dimensional spreading function
        sigma2 = np.minimum(fibers[index].length / 5.0, np.maximum(fibers[index].length / fibers[index].num_points, self.kernel_sigma))**2
        s_imaginary_left  = -self.s_head[i]
        s_imaginary_right =  2.0 * self.length_MT_head[i] - self.s_head[i]
        
        s_diff       = (fib_s - self.s_head[i]) 
        s_diff_left  = (fib_s - s_imaginary_left) 
        s_diff_right = (fib_s - s_imaginary_right) 

        #s_diff       = (fibers[index].s - self.s_head[i]) 
        #s_diff_left  = (fibers[index].s - s_imaginary_left) 
        #s_diff_right = (fibers[index].s - s_imaginary_right)
        Spread = np.exp(-s_diff**2 / (2.0 * sigma2)) / np.sqrt(2.0 * np.pi * sigma2) + \
                 np.exp(-s_diff_left**2 / (2.0 * sigma2)) / np.sqrt(2.0 * np.pi * sigma2) + \
                 np.exp(-s_diff_right**2 / (2.0 * sigma2)) / np.sqrt(2.0 * np.pi * sigma2)
       
        # Spread force
        fibers[index].force_motors += Spread[:,None] * self.force[i]


  def update_links(self, dt):
    '''
    Update links. Bound motors can unbind and unbound motors can bind.
    '''
    # Select motors bound to a MT
    sel_base_bound = self.attached_base > -1
    sel_head_bound = self.attached_head > -1

    # Select motors with a free end
    sel_base_free = self.attached_base == -2
    sel_head_free = self.attached_head == -2


  def update_links_numba(self, dt, fibers):
    '''
    Update links. Bound motors can unbind and unbound motors can bind.
    '''
    # Select motors bound to a MT
    sel_base_bound = self.attached_base > -1
    sel_head_bound = self.attached_head > -1

    # Select motors with a free end
    sel_base_free = self.attached_base == -2
    sel_head_free = self.attached_head == -2

    offset = np.zeros(len(fibers) + 1)
    coor = List()
    lengths = np.zeros(len(fibers))
    for i, fib in enumerate(fibers):
      n = int(2 * fib.length / self.radius) + 2
      n += (n+1) % 2   
      alpha = np.linspace(-1, 1, num=n)
      r = np.zeros((n, 3))
      r[:, 0] = cheb.cheb_eval(alpha, fib.x_modes[:, 0], order=1)
      r[:, 1] = cheb.cheb_eval(alpha, fib.x_modes[:, 1], order=1)
      r[:, 2] = cheb.cheb_eval(alpha, fib.x_modes[:, 2], order=1)
      coor.append(r)
      offset[i+1] += offset[i] + n
      lengths[i] = fibers[i].length

    attached_base, attached_head, x, s_base, s_head = update_links_numba_implementation(dt, 
                                                                                        self.attached_base,
                                                                                        self.attached_head,
                                                                                        sel_base_bound, 
                                                                                        sel_head_bound, 
                                                                                        sel_base_free, 
                                                                                        sel_head_free,
                                                                                        self.N,
                                                                                        self.force,
                                                                                        self.force_stall,
                                                                                        self.unbind_frequency_0,
                                                                                        self.x,
                                                                                        self.x_base,
                                                                                        self.x_head,
                                                                                        offset,
                                                                                        coor,
                                                                                        lengths,
                                                                                        self.radius,
                                                                                        self.bind_frequency,
                                                                                        self.s_base,
                                                                                        self.s_head,
                                                                                        self.rest_length)

    self.attached_base = attached_base
    self.attached_head = attached_head 
    self.x = x
    self.s_base = s_base
    self.s_head = s_head

  
@njit(parallel=False, fastmath=False)
def update_links_numba_implementation(dt,
                                      attached_base,
                                      attached_head, 
                                      sel_base_bound, 
                                      sel_head_bound, 
                                      sel_base_free,
                                      sel_head_free,
                                      N,
                                      force,
                                      force_stall,
                                      unbind_frequency_0,
                                      x,
                                      x_base,
                                      x_head,
                                      offset,
                                      coor,
                                      lengths,
                                      radius,
                                      bind_frequency,
                                      s_base,
                                      s_head,
                                      rest_length):
  '''
  Update links. Bound motors can unbind and unbound motors can bind.
  '''
  targets = np.zeros(1000, dtype=np.int32)
        
  # Unbind motors base and head
  for i in prange(N):
    if sel_base_bound[i]:
      # Compute unbind frequency
      force_norm = np.linalg.norm(force[i])
      if force_norm < force_stall:
        fub = unbind_frequency_0 * np.exp(force_norm / (force_stall - force_norm))
      else:
        fub = 1e+20
        
      # Randomly unbind base
      if np.random.rand(1)[0] > np.exp(-dt * fub):
        attached_base[i] = -2
        if sel_head_bound[i]:
          x[i] = 0.5 * (x_base[i] + x_head[i])
        else:
          x[i] = x_base[i]

    if sel_head_bound[i]:
      # Compute unbind frequency
      force_norm = np.linalg.norm(force[i])
      if force_norm < force_stall:
        fub = unbind_frequency_0 * np.exp(force_norm / (force_stall - force_norm))
      else:
        fub = 1e+20
        
      # Randomly unbind base
      if np.random.rand(1)[0] > np.exp(-dt * fub):
        attached_head[i] = -2
        if sel_base_bound[i]:
          x[i] = 0.5 * (x_base[i] + x_head[i])
        else:
          x[i] = x_head[i]

  # Bind motors base and head
  for i in prange(N):
    if sel_base_free[i]:
      # Bind randomly
      if np.random.rand(1)[0] > np.exp(-dt * bind_frequency):
        num_targets = 0
        # Loop over fibers
        for j in range(len(coor)):
          r = coor[j]
          n = r.size // 3
          rx = r[n // 2, 0] - x[i, 0]
          ry = r[n // 2, 1] - x[i, 1]
          rz = r[n // 2, 2] - x[i, 2]
          d_middle = np.sqrt(rx**2 + ry**2 + rz**2)
          if d_middle <= (0.5 * lengths[j] + radius):
            for k in range(n):
              rx = r[k, 0] - x[i, 0]
              ry = r[k, 1] - x[i, 1]
              rz = r[k, 2] - x[i, 2]
              d = np.sqrt(rx**2 + ry**2 + rz**2)
              if d < radius:
                targets[num_targets] = j
                num_targets += 1
                break
        # if len(targets) > 1:
        if num_targets > 0:
          sel_target = targets[np.random.randint(0, num_targets)]
          attached_base[i] = sel_target
          k_min = 0
          d_min = 1e+20
          r = coor[attached_base[i]]
          n = r.size // 3
          for k in range(n):
            rx = r[k, 0] - x[i, 0]
            ry = r[k, 1] - x[i, 1]
            rz = r[k, 2] - x[i, 2]
            d = abs(np.sqrt(rx**2 + ry**2 + rz**2) - rest_length)
            if d < d_min:
              d_min = d
              k_min = k
          alpha = -1.0 + 2.0 * (k_min / (n-1))
          s_base[i] = (1.0 + alpha) * lengths[sel_target] * 0.5

    if sel_head_free[i]:
      # Bind randomly
      if np.random.rand(1)[0] > np.exp(-dt * bind_frequency):
        num_targets = 0
        # Loop over fibers
        for j in range(len(coor)):
          r = coor[j]
          n = r.size // 3
          rx = r[n // 2, 0] - x[i, 0]
          ry = r[n // 2, 1] - x[i, 1]
          rz = r[n // 2, 2] - x[i, 2]
          d_middle = np.sqrt(rx**2 + ry**2 + rz**2)
          if d_middle <= (0.5 * lengths[j] + radius):
            for k in range(n):
              rx = r[k, 0] - x[i, 0]
              ry = r[k, 1] - x[i, 1]
              rz = r[k, 2] - x[i, 2]
              d = abs(np.sqrt(rx**2 + ry**2 + rz**2) - rest_length)
              if d < radius:
                targets[num_targets] = j
                num_targets += 1
                break
        if num_targets > 0:
          sel_target = targets[np.random.randint(0, num_targets)]
          attached_head[i] = sel_target
          k_min = 0
          d_min = 1e+20
          r = coor[attached_head[i]]
          n = r.size // 3
          for k in range(n):
            rx = r[k, 0] - x[i, 0]
            ry = r[k, 1] - x[i, 1]
            rz = r[k, 2] - x[i, 2]
            d = abs(np.sqrt(rx**2 + ry**2 + rz**2) - rest_length)
            if d < d_min:
              d_min = d
              k_min = k
          alpha = -1.0 + 2.0 * (k_min / (n-1))
          s_head[i] = (1.0 + alpha) * lengths[sel_target] * 0.5

  return attached_base, attached_head, x, s_base, s_head
