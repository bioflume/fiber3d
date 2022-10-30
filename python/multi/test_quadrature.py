import numpy as np
import sys
sys.path.append('../')


import scipy.sparse as scsp
import scipy.linalg as scla
import scipy.sparse.linalg as scspla
from scipy.spatial import ConvexHull
from quaternion import quaternion
from shape_gallery import shape_gallery
from quadratures import Smooth_Closed_Surface_Quadrature_RBF
from periphery import periphery
from kernels import kernels


if __name__ == '__main__':

  Nblobs = np.array([400,800,1600,3200],dtype = int)
  Ntruth = int(8000)
  trg_points = np.array([7, 0, 0])
  
  nodes_periphery, normals_periphery, h_periphery, gradh_periphery = shape_gallery.shape_gallery('sphere', Ntruth, radius=3)
  normals_periphery = -normals_periphery
  hull_periphery = ConvexHull(nodes_periphery)
  triangles_periphery = hull_periphery.simplices

  quadrature_weights_periphery = Smooth_Closed_Surface_Quadrature_RBF.Smooth_Closed_Surface_Quadrature_RBF(nodes_periphery, 
                                                                                                           triangles_periphery, 
                                                                                                           h_periphery,
                                                                                                           gradh_periphery)
  shell = periphery.Periphery(np.array([0., 0., 0.]), quaternion.Quaternion([1.0, 0.0, 0.0, 0.0]), 
                nodes_periphery, normals_periphery, quadrature_weights_periphery)                                                                                                             
  
  trg_shell_surf = shell.get_r_vectors()
  density = np.zeros((Ntruth,3))
  density[:,0] = 100
  vel_truth = kernels.stresslet_kernel_source_target_numba(trg_shell_surf,
      trg_points, normals_periphery, density, eta = 1)
  errors = []
  vel_list = []
  for Nblob in Nblobs:
    nodes_periphery, normals_periphery, h_periphery, gradh_periphery = shape_gallery.shape_gallery('sphere', Nblob, radius=3)

    normals_periphery = -normals_periphery
    hull_periphery = ConvexHull(nodes_periphery)
    triangles_periphery = hull_periphery.simplices

    quadrature_weights_periphery = Smooth_Closed_Surface_Quadrature_RBF.Smooth_Closed_Surface_Quadrature_RBF(nodes_periphery, 
                                                                                                             triangles_periphery, 
                                                                                                             h_periphery,
                                                                                                             gradh_periphery)
    shell = periphery.Periphery(np.array([0., 0., 0.]), quaternion.Quaternion([1.0, 0.0, 0.0, 0.0]), 
                  nodes_periphery, normals_periphery, quadrature_weights_periphery)                                                                                                             
    
    trg_shell_surf = shell.get_r_vectors()
    density = np.zeros((Nblob,3))
    density[:,0] = 100
    vel = kernels.stresslet_kernel_source_target_numba(trg_shell_surf,
        trg_points, normals_periphery, density, eta = 1)
    error = np.linalg.norm(vel-vel_truth)/np.linalg.norm(vel_truth)
    print('N = ', Nblob, ', error = ', error)
    errors.append(error)
    vel_list.append(vel)
  print('errors = ', errors)
  vel_list.append(vel_truth)
  print('vel_list = ', vel_list)
