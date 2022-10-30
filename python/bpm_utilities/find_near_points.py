import numpy as np
import scipy as sp
import scipy.spatial

def find_near_points(px, qx, r, ptree=None, qtree=None, slow_trees=False, **kwargs):
  """
  Finds points in qx that are within a distance r of any point in px
  This function is basically a convenience wrapper to the function:
    scipy.spatial.cKDTree.query_ball_tree
  Parameters:
    px,      required, tuple(dim),   coordinates of px
    qx,      required, tuple(dim),   coordinates of qx
    r,       required, float,      radius of closeness
    ptree,     optional, cKDTree,    tree for px
    qtree,     optional, cKDTree,    tree for qx
    slow_trees,  optional, bool,       take more time to form trees
    Additionally, you may pass the keyword arguments:
      p   (minkowski norm)
      eps   (approximate search parameter)
    see the documentation of sp.spatialcKDTree.query_ball_tree for details
  The tuple px should have the same length as the dimension of the space
    the ith element of px should be the coordinate for the ith dimension
    and should be a numpy array of type float, with any shape
  The tuple qx should be the same as qx but for q; must have the same
    dimension but the underlying arrays need not have the same shape
  If slow_trees, will create balanced kd-trees with compact nodes. This takes
    longer but makes searches of the trees faster. If you intend to reuse
    the trees, this is probably worth setting to True.
  Outputs:
    Tuple of (result, pxtree, qxtree)
    result is a bool array with shape given by the shape of qx[0]
  """
  # get the dim
  dim = len(px)
  # get the shapes
  psh = px[0].shape
  qsh = qx[0].shape
  have_ptree = ptree is not None
  have_qtree = qtree is not None
  # if the trees don't exist, form them
  if not have_ptree:
    # put data into the correct format
    pf = np.zeros([np.prod(psh),dim], dtype=float)
    for i in range(dim):
      pf[:,i] = px[i].flatten()
    ptree = sp.spatial.cKDTree(pf, compact_nodes=slow_trees, balanced_tree=slow_trees, **kwargs)
  if not have_qtree:
    qf = np.zeros([np.prod(qsh),dim], dtype=float)
    for i in range(dim):
      qf[:,i] = qx[i].flatten()
    qtree = sp.spatial.cKDTree(qf, compact_nodes=slow_trees, balanced_tree=slow_trees, **kwargs)
  
  # determine which tree is bigger, run appropriate search
  p_is_bigger = ptree.data.shape[0] > qtree.data.shape[0]
  # find close points
  if p_is_bigger:
    groups = qtree.query_ball_tree(ptree, r)
  else:
    groups = ptree.query_ball_tree(qtree, r)
  # massage output as needed
  if p_is_bigger:
    close = [len(x)>0 for x in groups]
    out = np.array(close).reshape(qsh)
  else:
    out = np.zeros(np.prod(qsh), dtype=bool)
    for group in groups:
      out[np.array(group)] = True
    out = out.reshape(qsh)
  return out, ptree, qtree

def find_near_dists(px, qx, r, ptree=None, qtree=None, slow_trees=False, **kwargs):
  """
  Compute a sparse distance matrix between two sets of points
  This function is basically a convenience wrapper to the function:
    scipy.spatial.cKDTree.sparse_distance_matrix
  Parameters:
    px,      required, tuple(dim),   coordinates of px
    qx,      required, tuple(dim),   coordinates of qx
    r,       required, float,      radius of closeness
    ptree,     optional, cKDTree,    tree for px
    qtree,     optional, cKDTree,    tree for qx
    slow_trees,  optional, bool,       take more time to form trees
    Additionally, you may pass the keyword arguments:
      p       (minkowski norm)
    see the documentation of sp.spatialcKDTree.find_near_dists for details
  The tuple px should have the same length as the dimension of the space
    the ith element of px should be the coordinate for the ith dimension
    and should be a numpy array of type float, with any shape
  The tuple qx should be the same as qx but for q; must have the same
    dimension but the underlying arrays need not have the same shape
  If slow_trees, will create balanced kd-trees with compact nodes. This takes
    longer but makes searches of the trees faster. If you intend to reuse
    the trees, this is probably worth setting to True.
  Outputs:
    Tuple of (result, pxtree, qxtree)
    result is a tuple of (p_inds, q_inds, distances)
  """
  dim, psh, qsh, ptree, qtree = _prepare(px, qx, ptree, qtree, slow_trees)
  dists = ptree.sparse_distance_matrix(qtree, r, output_type='ndarray', **kwargs)
  p_inds = np.unravel_index(dists['i'], psh)
  q_inds = np.unravel_index(dists['j'], qsh)
  distances = dists['v']
  return (p_inds, q_inds, distances), ptree, qtree

def _prepare(px, qx, ptree=None, qtree=None, slow_trees=False):
  dim = len(px)
  # get the shapes
  psh = px[0].shape
  qsh = qx[0].shape
  have_ptree = ptree is not None
  have_qtree = qtree is not None
  # if the trees don't exist, form them
  if not have_ptree:
    # put data into the correct format
    pf = np.zeros([np.prod(psh),dim], dtype=float)
    for i in range(dim):
      pf[:,i] = px[i].flatten()
    ptree = sp.spatial.cKDTree(pf, compact_nodes=slow_trees, balanced_tree=slow_trees)
  if not have_qtree:
    qf = np.zeros([np.prod(qsh),dim], dtype=float)
    for i in range(dim):
      qf[:,i] = qx[i].flatten()
    qtree = sp.spatial.cKDTree(qf, compact_nodes=slow_trees, balanced_tree=slow_trees)
  return dim, psh, qsh, ptree, qtree


