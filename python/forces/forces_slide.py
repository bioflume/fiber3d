'''
 Module to compute external forces on bodies and fibers.

 This not include internal fiber forces (-E*X_sss + T*X_s) or
 body-force fibers in the joints.
'''
import sys
import numpy as np
import argparse
from body import body
from fiber import fiber


def compute_repulsion_force(bodies, fibers, bodies_mm_attached, offset_fibers, x0, len_nuc2fib,
  len_nuc2bdy, len_nuc2nuc, scale_nuc2fib, scale_nuc2bdy, scale_nuc2nuc):

  # Repulsion forces are between
  #   1. bodies_mm_attached and bodies
  #   2. bodies_mm_attached and fibers

  force_bodies = np.zeros((len(bodies), 6))
  force_fibers = np.zeros((offset_fibers[-1],3))
  force_bodies_mm_attached = np.zeros((len(bodies_mm_attached), 6))

  #f0 = 1.0 # 1
  #lamb_body = 1e-01 # 1e-01
  #lamb_fiber = 5e-02 # 5e-02
  #lamb_nuc = 0.3  # 2.0
  #fnuc = 250.0  # 100.0

  for inuc, nucleus in enumerate(bodies_mm_attached):

    # between nuclei
    for jnuc in range(inuc):
      nucleus2 = bodies_mm_attached[jnuc]
      radius = nucleus.radius + nucleus2.radius
      r = nucleus2.location - nucleus.location
      d = np.linalg.norm(r)
      if d - radius > 0:
        #fr = fnuc * np.exp(-(d-radius)/lamb_nuc) / d
        fr = scale_nuc2nuc * np.exp(-(d-radius)/len_nuc2nuc) / d
      else:
        #fr = (fnuc/lamb_nuc) / d
        fr = (scale_nuc2nuc / len_nuc2nuc) / d
      force_bodies_mm_attached[inuc,0:3] += -fr * r
      force_bodies_mm_attached[jnuc,0:3] += fr * r

    # mm_attached bodies to bodies
    for i, b in enumerate(bodies):
      radius = nucleus.radius + b.radius + 0.05 * nucleus.radius
      r = b.location - nucleus.location
      d = np.linalg.norm(r)
      if d - radius > 0:
        #fr = (f0 / lamb_body) * np.exp(-(d-radius) / lamb_body) / d
        fr = (scale_nuc2bdy / len_nuc2bdy) * np.exp(-(d-radius) / len_nuc2bdy) / d
      else:
        #fr  = (f0 / lamb_body) / d
        fr  = (scale_nuc2bdy / len_nuc2bdy) / d
      force_bodies[i,0:3] += fr * r
      force_bodies_mm_attached[inuc,0:3] += -fr * r

    # mm_attached bodies to fibers
    x = x0[0::3] - nucleus.location[0]
    y = x0[1::3] - nucleus.location[1]
    z = x0[2::3] - nucleus.location[2]
    d = np.sqrt(x**2 + y**2 + z**2)
    sel_out = (d-nucleus.radius) > 0
    sel_in = (d-nucleus.radius) <= 0
    fr = np.zeros_like(d)
    #fr[sel_out] = (f0 / lamb_fiber) * np.exp(-(d[sel_out]-nucleus.radius) / lamb_fiber) / d[sel_out]
    fr[sel_out] = (scale_nuc2fib / len_nuc2fib) * np.exp(-(d[sel_out]-nucleus.radius) / len_nuc2fib) / d[sel_out]
    #fr[sel_in]  = (f0 / lamb_fiber) / d[sel_in]
    fr[sel_in]  = (scale_nuc2fib / len_nuc2fib) / d[sel_in]
    force_fibers[:,0] += fr*x
    force_fibers[:,1] += fr*y
    force_fibers[:,2] += fr*z

    force_bodies_mm_attached[inuc,0] += -np.sum(fr*x)
    force_bodies_mm_attached[inuc,1] += -np.sum(fr*y)
    force_bodies_mm_attached[inuc,2] += -np.sum(fr*z)



  return force_bodies_mm_attached, force_bodies, force_fibers

def compute_hydro_repulsion_force(bodies, x0, offset_fibers, radius, periphery_a, periphery_b, periphery_c):

  # Repulsion forces are between
  #   1. shell and fibers
  #   2. shell and bodies

  force_bodies = np.zeros((len(bodies), 6))
  force_fibers = np.zeros((offset_fibers[-1],3))

  f0 = 50

  if radius is not None:
    periphery_a, periphery_b, periphery_c = radius, radius, radius

  xfib,yfib,zfib = x0[:,0], x0[:,1], x0[:,2]

  x = x0[:,0]/periphery_a
  y = x0[:,1]/periphery_b
  z = x0[:,2]/periphery_c

  r_true = np.sqrt(xfib**2 + yfib**2 + zfib**2)

  r_fiber = np.sqrt(x**2 + y**2 + z**2)
  phi_fiber = np.arctan2(y,(x+1e-12))
  theta_fiber = np.arccos(z/(1e-12+r_fiber))

  x_cort = periphery_a*np.sin(theta_fiber)*np.cos(phi_fiber)
  y_cort = periphery_b*np.sin(theta_fiber)*np.sin(phi_fiber)
  z_cort = periphery_c*np.cos(theta_fiber)

  d = np.sqrt((xfib-x_cort)**2 + (yfib-y_cort)**2 + (zfib-z_cort)**2)
  cortex_point_r = np.sqrt(x_cort**2 + y_cort**2 + z_cort**2)

  sel_out = r_true >= cortex_point_r
  sel_in = r_true < cortex_point_r

  fr = np.zeros_like(d)

  lamb = 0.5

  fr[sel_in] = f0 * np.exp(-d[sel_in] / lamb) / d[sel_in]
  fr[sel_out]  = f0 / d[sel_out]
  force_fibers[:,0] += fr * (xfib-x_cort)
  force_fibers[:,1] += fr * (yfib-y_cort)
  force_fibers[:,2] += fr * (zfib-z_cort)


  return force_bodies, force_fibers


def compute_external_forces(bodies, fibers, x0, Nfibers_markers, offset, offset_bodies, *args, **kwargs):
  N = len(bodies)
  force_bodies = np.zeros((N, 6))
  force_fibers = np.zeros((Nfibers_markers, 3))


  # 1. Compute one-body forces
  force_bodies += compute_one_body_forces(bodies, *args, **kwargs)

  # 2. Compute one-fiber forces
  force_fibers += compute_one_fiber_forces(bodies, fibers, x0, Nfibers_markers, offset, offset_bodies, *args, **kwargs)

  # 3. Compute body-body forces

  # 4. Compute fiber-fiber forces

  # 5. Compute fiber-body forces

  return force_bodies, force_fibers

def compute_one_body_forces(bodies, *args, **kwargs):
  nucleus_radius, nucleus_position = [], []
  for key, value in kwargs.items():
    if key == 'nucleus_radius':
      nucleus_radius = value
    if key == 'nucleus_position':
      nucleus_position = value

  force_bodies = np.zeros((len(bodies), 6))

  if True:
    # Spheres at (0,0,nucleus_position)
    f0 = 1.0
    lamb = 1e-01
    # Sphere mimics nucleus, its size is given
    if not nucleus_radius.all():
      nucleus_radius = np.array([5.1])
      nucleus_position = [np.array([0, 0, 0])]

    for inuc,nradius in enumerate(nucleus_radius):

      for i, b in enumerate(bodies):
        radius = nradius + b.radius + 0.05*nradius
        nuc_xyz = nucleus_position[inuc]
        r = b.location-np.array([nuc_xyz[0], nuc_xyz[1], nuc_xyz[2]])
        d = np.linalg.norm(r)
        if d - radius > 0:
          fr = (f0 / lamb) * np.exp(-(d-radius) / lamb) / d
        else:
          fr  = (f0 / lamb) / d
        force_bodies[i,0:3] += fr * r

  return force_bodies

def compute_one_fiber_forces(bodies, fibers, x0, Nfibers_markers, offset, offset_bodies, *args, **kwargs):
  # TODO: clean hard-coded forces and use input file

  nucleus_radius, nucleus_position = [], []
  for key, value in kwargs.items():
    if key == 'nucleus_radius':
      nucleus_radius = value
    if key == 'nucleus_position':
      nucleus_position = value

  f = np.zeros((Nfibers_markers, 3))

  if True:

    # Sphere at given position
    if not nucleus_radius.all():
      nucleus_radius = np.array([5.1])
      nucleus_position = [np.array([0, 0, 0])]

    f0 = 1.0
    lamb = 5e-02
    x,y,z = x0[:,0], x0[:,1], x0[:,2]

    for inuc,radius in enumerate(nucleus_radius):
      nuc_xyz = nucleus_position[inuc]
      d = np.sqrt((x-nuc_xyz[0])**2 + (y-nuc_xyz[1])**2 + (z-nuc_xyz[2])**2)
      sel_out = (d-radius) > 0
      sel_in = (d-radius) <= 0
      fr = np.zeros_like(d)
      fr[sel_out] = (f0 / lamb) * np.exp(-(d[sel_out]-radius) / lamb) / d[sel_out]
      fr[sel_in]  = (f0 / lamb) / d[sel_in]
      f[:,0] += fr * (x-nuc_xyz[0])
      f[:,1] += fr * (y-nuc_xyz[1])
      f[:,2] += fr * (z-nuc_xyz[2])

  return f

def compute_cortex_forces(bodies, bodies_mm_attached, fibers, x0, Nfibers_markers, periphery_a, periphery_b, periphery_c):
  force_bodies = np.zeros((len(bodies), 6))
  force_bodies_mm_attached = np.zeros((len(bodies_mm_attached), 6))
  force_fibers = np.zeros((Nfibers_markers, 3))

  # Spheres at (0,0,nucleus_position)
  f0 = 10.0 #10
  lamb = 2e-01 #2e-1
  # Sphere mimics cortex, its size is given

  for i, b in enumerate(bodies):
    # Find body location in spherical coordinates
    r_true = np.linalg.norm(b.location)
    x, y, z = b.location[0]/periphery_a, b.location[1]/periphery_b, b.location[2]/periphery_c
    r_body = np.sqrt(x**2 + y**2 + z**2)
    phi_body = np.arctan(y/(1e-12+x))
    theta_body = np.arccos(z/(1e-12+r_body))
    # Find projection of body on the cortex
    x_cort = periphery_a*np.sin(theta_body)*np.cos(phi_body)
    y_cort = periphery_b*np.sin(theta_body)*np.sin(phi_body)
    z_cort = periphery_c*np.cos(theta_body)

    r = b.location-np.array([x_cort, y_cort, z_cort])
    d = np.linalg.norm(r)
    cortex_point_r = np.sqrt(x_cort**2 + y_cort**2 + z_cort**2)
    if r_true < cortex_point_r:
      fr = (f0 / lamb) * np.exp(-(cortex_point_r-r_body) / lamb) / d
    else:
      fr  = (f0 / lamb) / d
    force_bodies[i,0:3] += fr * r

  for i, b in enumerate(bodies_mm_attached):
    # Find body location in spherical coordinates
    r_true = np.linalg.norm(b.location)
    x, y, z = b.location[0]/periphery_a, b.location[1]/periphery_b, b.location[2]/periphery_c
    r_body = np.sqrt(x**2 + y**2 + z**2)
    phi_body = np.arctan(y/(1e-12+x))
    theta_body = np.arccos(z/(1e-12+r_body))
    # Find projection of body on the cortex
    x_cort = periphery_a*np.sin(theta_body)*np.cos(phi_body)
    y_cort = periphery_b*np.sin(theta_body)*np.sin(phi_body)
    z_cort = periphery_c*np.cos(theta_body)

    r = b.location-np.array([x_cort, y_cort, z_cort])
    d = np.linalg.norm(r)
    cortex_point_r = np.sqrt(x_cort**2 + y_cort**2 + z_cort**2)
    if r_true < cortex_point_r:
      fr = (f0 / lamb) * np.exp(-(cortex_point_r-r_body) / lamb) / d
    else:
      fr  = (f0 / lamb) / d
    force_bodies_mm_attached[i,0:3] += fr * r

  # Fiber-Cortex interaction

  xfib = x0[:,0]
  yfib = x0[:,1]
  zfib = x0[:,2]
  x = np.copy(xfib)/periphery_a
  y = np.copy(yfib)/periphery_b
  z = np.copy(zfib)/periphery_c
  r_true = np.sqrt(xfib**2 + yfib**2 + zfib**2)
  r_fiber = np.sqrt(x**2 + y**2 + z**2)
  phi_fiber = np.arctan2(y, (x + 1e-12))
  theta_fiber = np.arccos(z / (1e-12 + r_fiber))

  x_cort = periphery_a * np.sin(theta_fiber) * np.cos(phi_fiber)
  y_cort = periphery_b * np.sin(theta_fiber) * np.sin(phi_fiber)
  z_cort = periphery_c * np.cos(theta_fiber)

  d2cort = np.sqrt((xfib - x_cort)**2 + (yfib - y_cort)**2 + (zfib - z_cort)**2)
  cortex_point_r = np.sqrt(x_cort**2 + y_cort**2 + z_cort**2)

  sel_out = r_true < cortex_point_r
  sel_in = r_true >= cortex_point_r
  fr = np.zeros_like(d2cort)
  fr[sel_out] = (f0 ) * np.exp(-(d2cort[sel_out]) / lamb) / d2cort[sel_out]
  fr[sel_in]  = (f0 ) / d2cort[sel_in]
  force_fibers[:,0] += fr * (xfib-x_cort)
  force_fibers[:,1] += fr * (yfib-y_cort)
  force_fibers[:,2] += fr * (zfib-z_cort)


  return force_fibers, force_bodies, force_bodies_mm_attached
