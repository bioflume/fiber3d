# List of files organized based on their functionality {#filelist}
======================================================

Prefix keys:
------------

- [FB]: Florencio Usabiaga Balboa
- [GK]: Gokberk Kabacaoglu
- [R] : Reviewed

Platform:
---------

Biophysics:
---------
- [FB] body/body.py
- [FB] fiber/fiber.py
- [FB] periphery/periphery.py ~ body.py
- [FB] force_generator/force_generator.py
- [FB] multi_bodies/multi_bodies.py
- [FB] multi/multi.py (the newest)


Mathematics:
-----------
- [FB] integrators/integrators.py: Building linear systems, preconditioners,
  iterative solvers. SBT only, SBT + HI
- [FB] kernels/kernels.py
- [FB] quaternion/quaternion.py
- [FB] quadratures/Smooth_Closed_Surface_Quadrature_RBF.py
 

Utilities:
----------
- [FB] bpm_utilities/find_near_points.py
- [FB] bpm_utilities/gmres.py: GMRES that permits right PC
- [FB] utils/cheb.py
- [FB] utils/barycentricMatrix.py
- [FB] utils/finite_diff.py
- [FB] utils/nonlinear.py: Newton-Krylov solver
- [FB] utils/timer.py


Initialization:
---------------
- [FB] Routines in tools/ create fiber, body, force generators and links
- [FB] read_input/read_input.py, read_fibers_file.py, read_vertex_file.py,
  read_clones_file.py, read_links_file.py
- [FB] shape_gallery/shape_gallery.py: 

Examples:
---------
- multi_fiber/many/multi_fiber.py: Many fibers with body, HI excluded


Tests:
------
- [FB] tests/test_aliasing.py, [GK] test aliasing when computing bending
- [FB] tests/test_kernels.py (Numba vs. Numpy)


Visualization:
--------------
- Use Visit. How to use visit?


Comments:
---------
