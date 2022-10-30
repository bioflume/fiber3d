def get_vectors_frame_body(bodies, r_grid, frame_body):
  '''
  Get grid in the frame of reference of one body if frame_body >= 0.
  If frame_body < 0 use the lab frame of reference, i.e. do not translate or rotate anything.

  Inputs:
  bodies = list of bodies objects.
  r_grid = positions of the grid.
  frame_body = body to use as reference frame. If frame_body < 0 use laboratory reference frame.

  Outputs:
  r_grid_frame = grid positions in the body "frame_body" frame  of reference.
  '''

  # Prepare arrays
  r_grid_frame = np.empty((r_grid.size // 3, 3))

  if frame_body >= 0:
    # Get reference body rotation matrix and its location.
    # R0 rotates vectors to the body frame of reference
    R0 = bodies[frame_body].orientation.rotation_matrix().T
    location0 = bodies[frame_body].location

    # Translate and rotate grid postions
    for b in bodies:
      for i, ri in enumerate(r_grid):
        r_grid_frame[i] = np.dot(R0, (ri - location0))

  else:
    # If frame_body < 0 use the lab frame of reference; i.e. do not translate or rotate anything
    r_grid_frame = np.copy(r_grid)

  return r_grid_frame
