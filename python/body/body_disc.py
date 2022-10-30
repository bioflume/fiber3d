'''
Small class to handle a single body. The notation follows
loosely the paper Brownian dynamics of confined rigid
bodies, Steven Delong et al. The Journal of Chemical
Physics 143, 144107 (2015). doi: 10.1063/1.4932062
'''
import numpy as np
import copy
import sys
from kernels import kernels
from shape_gallery import shape_gallery
from quadratures import Smooth_Closed_Surface_Quadrature_RBF
from scipy.spatial import ConvexHull


class Body(object):
  '''
  Small class to handle a single body.
  '''  
  def __init__(self, location, orientation, reference_configuration, reference_normals, quadrature_weights):
    '''
    Constructor. Take arguments like ...
    '''
    # Location as np.array.shape = 3
    self.location = location
    self.location_new = np.copy(location)
    self.location_old = np.copy(location)
    # Orientation as Quaternion
    self.orientation = orientation
    self.orientation_new = copy.copy(orientation)
    self.orientation_old = copy.copy(orientation)
    # Number of blobs
    self.Nblobs = reference_configuration.size // 3
    self.quadrature_weights = quadrature_weights.flatten()
    self.reference_configuration = np.reshape(reference_configuration, (self.Nblobs, 3))
    self.reference_normals = np.reshape(reference_normals, (self.Nblobs, 3))
    # Blob masses
    self.blob_masses = np.ones(self.Nblobs)
    # Blob radius
    self.blob_radius = 1.0
    # Body length
    self.body_length = None
    # Name of body and type of body. A string or number
    self.name = None
    self.type = None
    self.mobility_blobs = None
    self.mobility_body = None
    # Geometrix matrix K (see paper Delong et al. 2015).
    self.K = None
    self.rotation_matrix = None
    # Some default functions
    self.prescribed_velocity = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    self.prescribed_kinematics = False
    self.mobility_blobs_cholesky = None
    self.ID = None
    self.force_torque = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    self.links_location = None
    # Vectors for singularity subtractions
    self.ex = None
    self.ey = None
    self.ez = None

    self.radius = None



    # Also keep body velocity and angular velocity
    self.velocity = np.zeros(3)
    self.angular_velocity = np.zeros(3)
    self.velocity_new = np.zeros(3)
    self.angular_velocity_new = np.zeros(3)


  def discretize_body_surface(self, shape = 'sphere', Nblobs = 80):
    '''
    To discretize a rigid body of given shape, if it is a confining wall, then change the 
    normal direction
    '''

    self.Nblobs = Nblobs

    nodes_body, normals_body, h_body, gradh_body = shape_gallery.shape_gallery(shape, self.Nblobs, radius = self.radius)

    self.reference_normals = normals_body

    self.reference_configuration = nodes_body # reference_configurat

    hull_body = ConvexHull(nodes_body)
    triangles_body = hull_body.simplices

    quadrature_weights = Smooth_Closed_Surface_Quadrature_RBF.Smooth_Closed_Surface_Quadrature_RBF(nodes_body,
                                                                                                        triangles_body, 
                                                                                                        h_body,
                                                                                                        gradh_body)
    self.quadrature_weights = quadrature_weights.flatten()

    # About discretization of rigid body
    self.density = np.zeros((self.Nblobs, 3))
    self.density_new = np.copy(self.density)
    self.density_old = np.copy(self.density)

    return
  

  def get_r_vectors(self, location = None, orientation = None):
    '''
    Return the coordinates of the blobs. Only the centers
    '''
    # Get location and orientation
    if location is None:
      location = self.location
    if orientation is None:
      orientation = self.orientation

    # Compute blobs coordinates
    r_vectors = location
    return r_vectors   

  def get_r_vectors_surface(self, location = None, orientation = None):
    '''
    Return the coordinates of the surface nodes. 
    '''
    # Get location and orientation
    if location is None:
      location = self.location
    if orientation is None:
      orientation = self.orientation

    # Compute blobs coordinates
    rotation_matrix = orientation.rotation_matrix()
    r_vectors = np.array([np.dot(rotation_matrix, vec) for vec in self.reference_configuration])
    r_vectors += location
    return r_vectors   


  def get_normals(self, orientation = None):
    '''
    Return the surface normals
    '''
    # Get orientation
    if orientation is None:
      orientation = self.orientation

    # Compute blobs coordinates
    rotation_matrix = orientation.rotation_matrix()
    normals = np.array([np.dot(rotation_matrix, vec) for vec in self.reference_normals])
    return normals


  def calc_rot_matrix(self, r_vectors = None, location = None, orientation = None, Nblobs = None):
    ''' 
    Calculate the matrix R, where the i-th 3x3 block of R gives
    (R_i x) = -1 (r_i cross x).
    R has shape (3*Nblobs, 3).
    '''
    if r_vectors is None:
      r_vectors = self.get_r_vectors_surface(location, orientation) - (self.location if location is None else location)    
    rot_matrix = np.array([[[0.0,    vec[2], -vec[1]],
                           [-vec[2], 0.0,    vec[0]],
                           [vec[1], -vec[0], 0.0]] for vec in r_vectors])
    return np.reshape(rot_matrix, (3*(self.Nblobs if Nblobs is None else Nblobs), 3))


  def calc_J_matrix(self, Nblobs = None):
    '''
    Returns a block matrix with dimensions (Nblobs, 1)
    with each block being a 3x3 identity matrix.
    '''  
    J = np.zeros((3*(self.Nblobs if Nblobs is None else Nblobs), 3))
    J[0::3,0] = 1.0
    J[1::3,1] = 1.0
    J[2::3,2] = 1.0
    return J


  def calc_K_matrix(self, r_vectors = None, location = None, orientation = None, Nblobs = None):
    '''
    Return geometric matrix K = [J, rot] with shape (3*Nblobs, 6)
    '''
    return np.concatenate([self.calc_J_matrix(Nblobs = Nblobs), 
                           self.calc_rot_matrix(location = location, orientation = orientation, r_vectors = r_vectors, Nblobs = Nblobs)], axis=1)


  def check_function(self, location = None, orientation = None, distance = None):
    ''' 
    Function to check that the body didn't cross the wall,
    i.e., all its blobs have z > distance. Default distance is 0.
    '''
    # Define distance
    if not distance:
      distance = 0.0
      
    # Get location and orientation
    if location is None:
      location = self.location
    if orientation is None:
      orientation = self.orientation      

    # Get current configuration
    r_vectors = self.get_r_vectors_surface(location, orientation)

    # Loop over blobs
    for vec in r_vectors:
      if vec[2] < distance:
        return False
    return True

  def calc_surface_integral_density(self):
    '''
    Surface integral of density, the output is of size (1,3)
    '''

    return np.sum(self.density*self.quadrature_weights[:,None], axis=0)

  def calc_surface_integral_density_cross_product(self, r_vectors = None, location = None, orientation = None, Nblobs = None):
    '''
    Surface integral of density's cross product with q
    '''
    rot_mat = self.calc_rot_matrix(location = location, orientation = orientation, r_vectors = r_vectors, Nblobs = Nblobs)

    out = np.zeros(((self.Nblobs if Nblobs is None else Nblobs), 3))
    for i in np.arange(self.Nblobs if Nblobs is None else Nblobs):
        out[i,:] = np.dot(rot_mat[3*i:3*i+3,:], self.density[i,:])

    return np.sum(out, axis=0)


  def calc_mobility_blobs(self, eta, a):
    '''
    Calculate blobs mobility. Shape (3*Nblobs, 3*Nblobs).
    '''
    r_vectors = self.get_r_vectors_surface()
    return self.mobility_blobs(r_vectors, eta, a)


  def calc_mobility_body(self, eta, a, M = None, M_inv = None):
    '''
    Calculate the 6x6 body mobility that maps
    forces and torques to velocities and angular
    velocites.
    '''
    K = self.calc_K_matrix()      
    if M_inv is not None:
      return np.linalg.pinv( np.dot(K.T, np.dot(M_inv, K)) )
    if M is None:
      M = self.calc_mobility_blobs(eta, a)
    return np.linalg.pinv( np.dot(K.T, np.dot(np.linalg.inv(M), K)) )


  def calc_mobility_body_scalar(self, eta, a, M = None, M_inv = None):
    '''
    Calculate the 6x6 body mobility that maps
    forces and torques to velocities and angular
    velocites.
    '''
    K = self.calc_K_matrix()      
    if M_inv is not None:
      return np.linalg.pinv(np.dot(K.T, K) * M_inv) 
    if M is None:
      M = self.calc_mobility_blobs(eta, a)
    return np.linalg.pinv(np.dot(K.T, K) / M) 


  def calc_mobility_blobs_cholesky(self, eta, a, M = None):
    '''
    Compute the Cholesky factorization L of the blobs mobility M=L*L.T.
    L is a lower triangular matrix with shape (3*Nblobs, 3*Nblobs).
    '''
    if M is None:
      M = self.calc_mobility_blobs(eta, a)
    return np.linalg.cholesky(M)


  def calc_body_length(self):
    '''
    It calculates, in one sense, the length of the body. Specifically, it
    returns the distance between the two furthest apart blobs in the body.
    '''
    max_distance = 0.
    for i in range(self.reference_configuration.size - 1):
      for blob in self.reference_configuration[i+1:]:
        blob_distance = np.linalg.norm(blob - self.reference_configuration[i]) 
        if blob_distance > max_distance:
          max_distance = blob_distance

    self.body_length = max_distance + 2*self.blob_radius
    return self.body_length

  def calc_vectors_singularity_subtraction(self, eta = 1.0, r_vectors = None, normals = None):
    if r_vectors is None:
      r_vectors = self.get_r_vectors_surface()
    if normals is None:
      normals = self.get_normals()      
    quadrature_weights = self.quadrature_weights

    e = np.zeros((self.Nblobs, 3))
    e[:,0] = 1.0
    e *= quadrature_weights[:,None]
    ex = kernels.stresslet_kernel_times_normal_times_density_numba(r_vectors, normals, e, eta)
    e[:,:] = 0.0
    e[:,1] = 1.0
    e *= quadrature_weights[:,None]
    ey = kernels.stresslet_kernel_times_normal_times_density_numba(r_vectors, normals, e, eta)
    e[:,:] = 0.0
    e[:,2] = 1.0
    e *= quadrature_weights[:,None]
    ez = kernels.stresslet_kernel_times_normal_times_density_numba(r_vectors, normals, e, eta)
    self.ex = np.copy(ex.reshape(self.Nblobs, 3))
    self.ey = np.copy(ey.reshape(self.Nblobs, 3))
    self.ez = np.copy(ez.reshape(self.Nblobs, 3))



  
    
