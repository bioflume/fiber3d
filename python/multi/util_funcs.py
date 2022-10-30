from __future__ import division, print_function
import numpy as np
import imp
import sys
import time
import argparse
import subprocess
import scipy.sparse as scsp
import scipy.linalg as scla
import scipy.sparse.linalg as scspla
sys.path.append('../')
sys.path.append('./')
from utils import barycentricMatrix as bary
from fiber import fiber
from body import body


##############################################################################################
def build_link_matrix(system_size,bodies,fibers,offset_fibers,offset_bodies,fib_mats,fib_mat_resolutions):
  '''
  Building link matrix (fibers' boundary conditions)
  '''
  As_dok_BC = scsp.dok_matrix((system_size, system_size))

  for offset_fiber, fib in enumerate(fibers):

    if fib.attached_to_body is not None:
      # Get body to which the fiber fib is attached
      k = fib.attached_to_body
      b = bodies[k]

      # Rotation matrix to get current config from reference config
      rotation_matrix = b.orientation.rotation_matrix()
      # Reference location of the nucleating site
      link_loc_ref = b.nuc_sites[fib.nuc_site_idx]

      # Location of link w.r.t. center of mass
      link = np.dot(rotation_matrix, link_loc_ref)

      # Find the location of the point in the matrix:
      offset_point = offset_fibers[offset_fiber] * 4 + offset_bodies

      # Rotation matrix to get current config from reference config
      rotation_matrix = b.orientation.rotation_matrix()
      # Reference location of the nucleating site
      link_loc_ref = b.nuc_sites[fib.nuc_site_idx]

      # Location of link w.r.t. center of mass
      link = np.dot(rotation_matrix, link_loc_ref)

      # Find the location of the point in the matrix:
      offset_point = offset_fibers[offset_fiber] * 4 + offset_bodies

      # Find the index for fib_mats
      indx = np.where(fib_mat_resolutions == fib.num_points)
      indx = indx[0][0]

      # Get the class that has the matrices
      fib_mat = fib_mats[indx]
      out1, D_2, D_3, out4 = fib_mat.get_matrices(fib.length, fib.num_points_up, 'Ds')
      xs = fib.xs


      # Rectangular mathod, Driscoll and Hale
      # Matrix A_body_fiber, for position
      # Force by fiber on body at s = 0, Fext = -F|s=0 = -(EXsss - TXs)
      # Bending term:
      As_dok_BC[k*6 + 0, offset_point+fib.num_points*0 : offset_point+fib.num_points*1] +=  -fib.E * D_3[0,:]
      As_dok_BC[k*6 + 1, offset_point+fib.num_points*1 : offset_point+fib.num_points*2] +=  -fib.E * D_3[0,:]
      As_dok_BC[k*6 + 2, offset_point+fib.num_points*2 : offset_point+fib.num_points*3] +=  -fib.E * D_3[0,:]
      # Tension term:
      As_dok_BC[k*6 + 0, offset_point+fib.num_points*3] += xs[0,0]
      As_dok_BC[k*6 + 1, offset_point+fib.num_points*3] += xs[0,1]
      As_dok_BC[k*6 + 2, offset_point+fib.num_points*3] += xs[0,2]

      # Torque by fiber on body at s = 0, Lext = (L + link_loc x F) = -(E(Xss x Xs) + link_loc x (EXsss - TXs))
      # Bending force term:
      # yz:
      As_dok_BC[k*6 + 3, offset_point+fib.num_points*2 : offset_point+fib.num_points*3] += -fib.E * link[1] * D_3[0,:]
      # zy:
      As_dok_BC[k*6 + 3, offset_point+fib.num_points*1 : offset_point+fib.num_points*2] +=  fib.E * link[2] * D_3[0,:]
      # zx:
      As_dok_BC[k*6 + 4, offset_point+fib.num_points*0 : offset_point+fib.num_points*1] += -fib.E * link[2] * D_3[0,:]
      # xz:
      As_dok_BC[k*6 + 4, offset_point+fib.num_points*2 : offset_point+fib.num_points*3] +=  fib.E * link[0] * D_3[0,:]
      # xy:
      As_dok_BC[k*6 + 5, offset_point+fib.num_points*1 : offset_point+fib.num_points*2] += -fib.E * link[0] * D_3[0,:]
      # yx:
      As_dok_BC[k*6 + 5, offset_point+fib.num_points*0 : offset_point+fib.num_points*1] +=  fib.E * link[1] * D_3[0,:]

      # Tension force term:
      # yz - zy:
      As_dok_BC[k*6 + 3, offset_point+fib.num_points*3] += (link[1]*xs[0,2] - link[2]*xs[0,1])
      # zx - xz:
      As_dok_BC[k*6 + 4, offset_point+fib.num_points*3] += (link[2]*xs[0,0] - link[0]*xs[0,2])
      # xy - yx:
      As_dok_BC[k*6 + 5, offset_point+fib.num_points*3] += (link[0]*xs[0,1] - link[1]*xs[0,0])

      # Fiber torque (L):
      # yz:
      As_dok_BC[k*6 + 3, offset_point+fib.num_points*1 : offset_point+fib.num_points*2] += -fib.E * xs[0,2] * D_2[0,:]
      # zy:
      As_dok_BC[k*6 + 3, offset_point+fib.num_points*2 : offset_point+fib.num_points*3] +=  fib.E * xs[0,1] * D_2[0,:]
      # zx:
      As_dok_BC[k*6 + 4, offset_point+fib.num_points*2 : offset_point+fib.num_points*3] += -fib.E * xs[0,0] * D_2[0,:]
      # xz:
      As_dok_BC[k*6 + 4, offset_point+fib.num_points*0 : offset_point+fib.num_points*1] +=  fib.E * xs[0,2] * D_2[0,:]
      # xy:
      As_dok_BC[k*6 + 5, offset_point+fib.num_points*0 : offset_point+fib.num_points*1] += -fib.E * xs[0,1] * D_2[0,:]
      # yx:
      As_dok_BC[k*6 + 5, offset_point+fib.num_points*1 : offset_point+fib.num_points*2] +=  fib.E * xs[0,0] * D_2[0,:]


      # Matrix A_fiber_body, for position
      # dx/dt = U + Omega x link_loc (move to LHS so -U -Omega x link_loc)
      # Linear velocity part (U)
      As_dok_BC[offset_point+fib.num_points*4-14, k*6 + 0] += -1.0
      As_dok_BC[offset_point+fib.num_points*4-13, k*6 + 1] += -1.0
      As_dok_BC[offset_point+fib.num_points*4-12, k*6 + 2] += -1.0
      # Angular velocity part (Omega)
      # yz:
      As_dok_BC[offset_point+fib.num_points*4-14, k*6 + 4] += -link[2]
      # zy:
      As_dok_BC[offset_point+fib.num_points*4-14, k*6 + 5] +=  link[1]
      # zx:
      As_dok_BC[offset_point+fib.num_points*4-13, k*6 + 5] += -link[0]
      # xz:
      As_dok_BC[offset_point+fib.num_points*4-13, k*6 + 3] +=  link[2]
      # xy:
      As_dok_BC[offset_point+fib.num_points*4-12, k*6 + 3] += -link[1]
      # yx:
      As_dok_BC[offset_point+fib.num_points*4-12, k*6 + 4] +=  link[0]

      # Tension equation, left hand side of it (U + Omega x link - \bar{u}_f).xs
      As_dok_BC[offset_point+fib.num_points*4-11, k*6 + 0] += -xs[0,0]
      As_dok_BC[offset_point+fib.num_points*4-11, k*6 + 1] += -xs[0,1]
      As_dok_BC[offset_point+fib.num_points*4-11, k*6 + 2] += -xs[0,2]

      As_dok_BC[offset_point+fib.num_points*4-11, k*6 + 3] += (xs[0,1]*link[2] - xs[0,2]*link[1])
      As_dok_BC[offset_point+fib.num_points*4-11, k*6 + 4] += (xs[0,2]*link[0] - xs[0,0]*link[2])
      As_dok_BC[offset_point+fib.num_points*4-11, k*6 + 5] += (xs[0,0]*link[1] - xs[0,1]*link[0])


      # Matrix A_fiber_body, for angle
      # Clamped boundary condition: dXs/dt = Omega x Xs or Omega x link_direction
      link_norm = np.sqrt(link[0]**2 + link[1]**2 + link[2]**2)
      link_dir = link / link_norm
      # yz:
      As_dok_BC[offset_point+fib.num_points*4-10, k*6 + 4] += -link_dir[2]
      # zy:
      As_dok_BC[offset_point+fib.num_points*4-10, k*6 + 5] +=  link_dir[1]
      # zx:
      As_dok_BC[offset_point+fib.num_points*4-9, k*6 + 5]  += -link_dir[0]
      # xz:
      As_dok_BC[offset_point+fib.num_points*4-9, k*6 + 3]  +=  link_dir[2]
      # xy:
      As_dok_BC[offset_point+fib.num_points*4-8, k*6 + 3]  += -link_dir[1]
      # yx:
      As_dok_BC[offset_point+fib.num_points*4-8, k*6 + 4]  +=  link_dir[0]


  return scsp.csr_matrix(As_dok_BC)
##############################################################################################
def calculate_body_fiber_link_conditions(fibers, bodies, body_velocities, fibers_xt, offset_fibers, fib_mats, fib_mat_resolutions):
  '''
  body_velocities: (len(bodies) x 6) matrix,
            translational velocity in x, y, z & rotational velocity in x, y, z comes

  fibers_xt: long array of fiber unknowns (comes from GMRES iteration)
    for each fiber (Nfib: num. points on fib), there are
    4*Nfib unknowns, 0*Nfib : 1*Nfib is x-coordinates of points
                     1*Nfib : 2*Nfib is y-coordinates
                     2*Nfib : 3*Nfib is z-coordinates
                     3*Nfib : 4*Nfib is tension
  '''
  # Fibers induce force and torque on the body they are attached to
  # Fibers' attachment points move with the body's translational velocity
  # Tangent along fiber at the attachement point rotates with the body's rotational velocity

  # I will gather force and torque contributions in this matrix
  # each fiber will result in an array - [body_indx, fx, fy, fz, Lx, Ly, Lz]
  force_torque_on_body = np.empty((len(fibers),7))
  # Velocity and angular velocity on a fiber will be saved in fiber or in a similar fashion
  # Also tension is calculated, so
  # vx, vy, vz, T, wx, wy, wz is the ordering in the matrix below
  velocities_on_fiber = np.empty((len(fibers),7))

  for fib_idx, fib in enumerate(fibers):

    # Check if the fiber fib is attached to any body
    if fib.attached_to_body is not None:
      # if so, find the body it is attached to
      body_idx = fib.attached_to_body
      body = bodies[body_idx]

      # Find the location of the nucleating site
      rotation_matrix = body.orientation.rotation_matrix() # matrix to rotate the reference to current config.
      ref_site_xyz = body.nuc_sites[fib.nuc_site_idx] # reference
      site_xyz = np.dot(rotation_matrix, ref_site_xyz) # current

      # Find the index for fib_mats given the resolution
      indx = np.where(fib_mat_resolutions == fib.num_points)
      indx = indx[0][0]
      # Get the object that has the differentiation matrices
      fib_mat = fib_mats[indx]
      out1, D_2, D_3, out4 = fib_mat.get_matrices(fib.length, fib.num_points_up, 'Ds')
      # Note: this is a implicit-explicit formulation in time, that is
      # calculations like (xs * xssss) discretized as (xs(t) * xssss(t + dt))
      # So, the higher order derivatives come from the next time step
      # Let's calculate those derivatives
      xs = fib.xs # at the current step, first derivative of x,y,z
      Nfib = fib.num_points
      # x, y, z coordinates of the points on the fiber and tension (at the next step)
      x_new = np.zeros((Nfib,3))
      x_new[:,0] = fibers_xt[4*offset_fibers[fib_idx] + 0*Nfib: 4*offset_fibers[fib_idx] + 1*Nfib]
      x_new[:,1] = fibers_xt[4*offset_fibers[fib_idx] + 1*Nfib: 4*offset_fibers[fib_idx] + 2*Nfib]
      x_new[:,2] = fibers_xt[4*offset_fibers[fib_idx] + 2*Nfib: 4*offset_fibers[fib_idx] + 3*Nfib]
      T_new = fibers_xt[4*offset_fibers[fib_idx] + 3*Nfib: 4*offset_fibers[fib_idx] + 4*Nfib]
      xss_new = np.dot(D_2, x_new)
      xsss_new = np.dot(D_3, x_new)

      # FIRST FROM FIBER ON-TO BODY
      # Force by fiber on body at s = 0, Fext = -F|s=0 = -(EXsss - TXs)
      # Bending term + Tension term:
      force_xyz  = -fib.E * xsss_new[0,:] + xs[0,:] * T_new[0]

      # Torque by fiber on body at s = 0, Lext = (L + link_loc x F) = -(E(Xss x Xs) + link_loc x (EXsss - TXs))
      torque_xyz = np.zeros(3)
      # bending contribution :
      torque_xyz[0] = fib.E * (-site_xyz[1] * xsss_new[0,2] + site_xyz[2] * xsss_new[0,1])
      torque_xyz[1] = fib.E * (-site_xyz[2] * xsss_new[0,0] + site_xyz[0] * xsss_new[0,2])
      torque_xyz[2] = fib.E * (-site_xyz[0] * xsss_new[0,1] + site_xyz[1] * xsss_new[0,0])

      # tension contribution :
      torque_xyz[0] += (site_xyz[1]*xs[0,2] - site_xyz[2]*xs[0,1]) * T_new[0]
      torque_xyz[1] += (site_xyz[2]*xs[0,0] - site_xyz[0]*xs[0,2]) * T_new[0]
      torque_xyz[2] += (site_xyz[0]*xs[0,1] - site_xyz[1]*xs[0,0]) * T_new[0]

      # fiber's torque L:
      torque_xyz[0] += fib.E * (-xs[0,2] * xss_new[0,1] + xs[0,1]*xss_new[0,2])
      torque_xyz[1] += fib.E * (-xs[0,0] * xss_new[0,2] + xs[0,2]*xss_new[0,0])
      torque_xyz[2] += fib.E * (-xs[0,1] * xss_new[0,0] + xs[0,0]*xss_new[0,1])

      # Store the contribution of each fiber in this array
      force_torque_on_body[fib_idx,0] = body_idx
      force_torque_on_body[fib_idx,1:4] = force_xyz
      force_torque_on_body[fib_idx,4:] = torque_xyz

      # SECOND FROM BODY ON-TO FIBER
      # Translational and angular velocities at the attacment point are calculated
      body_vxyz = body_velocities[2*body_idx,:]
      body_wxyz = body_velocities[2*body_idx+1,:]

      # dx/dt = U + Omega x link_loc (move to LHS so -U -Omega x link_loc)
      # Translational velocity part (U)
      fiber_vxyz = -1 * body_vxyz
      # Rotational velocity contribution to translation
      fiber_vxyz[0] += -site_xyz[2]*body_wxyz[1] + site_xyz[1]*body_wxyz[2]
      fiber_vxyz[1] += -site_xyz[0]*body_wxyz[2] + site_xyz[2]*body_wxyz[0]
      fiber_vxyz[2] += -site_xyz[1]*body_wxyz[0] + site_xyz[0]*body_wxyz[1]

      # tension condition = -(xs*vx + ys*vy + zs*wz)
      tension_condition = -np.dot(xs[0,:], body_vxyz)
      tension_condition += (xs[0,1]*site_xyz[2]-xs[0,2]*site_xyz[1]) * body_wxyz[0]
      tension_condition += (xs[0,2]*site_xyz[0]-xs[0,0]*site_xyz[2]) * body_wxyz[1]
      tension_condition += (xs[0,0]*site_xyz[1]-xs[0,1]*site_xyz[0]) * body_wxyz[2]

      # Rotational velocity condition on fiber
      site_dir = site_xyz / np.linalg.norm(site_xyz)
      fiber_wxyz = np.zeros(3)
      fiber_wxyz[0] = -site_dir[2] * body_wxyz[1] + site_dir[1] * body_wxyz[2]
      fiber_wxyz[1] = -site_dir[0] * body_wxyz[2] + site_dir[2] * body_wxyz[0]
      fiber_wxyz[2] = -site_dir[1] * body_wxyz[0] + site_dir[0] * body_wxyz[1]

      velocities_on_fiber[fib_idx,:3] = fiber_vxyz
      velocities_on_fiber[fib_idx,3] = tension_condition
      velocities_on_fiber[fib_idx,4:] = fiber_wxyz


  return force_torque_on_body, velocities_on_fiber

##############################################################################################
def create_nuc_sites_uniform(radius, Nsites):
  '''
  This creates uniformly distributed points on a sphere
  BUT DOES NOT CARE IF POINTS ARE WELL-SEPARATED
  '''

  theta = 2 * np.pi * np.random.rand(Nsites)
  phi = np.arccos(1-2*np.random.rand(Nsites))

  nuc_site_xyz = np.zeros((Nsites,3))
  nuc_site_xyz[:,0] = radius * np.sin(phi) * np.cos(theta)
  nuc_site_xyz[:,1] = radius * np.sin(phi) * np.sin(theta)
  nuc_site_xyz[:,2] = radius * np.cos(phi)

  return nuc_site_xyz

##############################################################################################
