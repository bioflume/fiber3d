# QUESTIONS 
===========

1. Which multi_fiber.py is up-to-date? multi_fiber/many/multi_fiber.py calls
   time stepper from integrators, I guess that one is the most recent one.
  YES

2. Is "forces" complete, or useless? Does it need to involve body-body, body-fiber and
   fiber-fiber forces?
  REPULSION FORCE TO AVOID COLLISIONS, NOT COMPLETE

3. In fiber.py, when num_fiber_diff ~= 0, D_1_0_up is not initialized but later
   called. That part of the code has to be taken care of.
  RESOLVED but FINITE DIFFERENCE MIGHT BE USED

4. How is the inextensibility implemented? 
  TORNBERG & SHELLEY (2004) penalty method

5. Non-local contribution of self-mobility is not implemented in fiber.py?
  NEEDS TO BE IMPLEMENTED

6. O(N) algorithm for trap-particle in force_generator.py is needed?
  NOT URGENT, but WILL BE NEEDED  

7. Difference between nonlinear vs. linear in integrators? We solve a linear
   system using a direct vs. an iterative method, what does nonlinearity mean?
  NONLINEAR SOLVERS MIGHT WORK BETTER (REFER TO DAVID)

8. In linear, do we build a large matrix? I guess so. Don't we have a
   matrix-free solver which has loops over fibers and bodies?
  LARGE SPARSE MATRIX FOR SBT, but w/ HI IT WILL BE DENSE. MAT-FREE PRELIMINARY
  IN multi/multi.py prepare_linear_operator.  

9. Difference between linear_solver_new vs linear_solver? Additionally,
   linear_solver_fibers, linear_solver_fibers_and_bodies, what are they for?
  AGAIN GO BACK TO MULTI TO UNDERSTAND, 
  linear_solver_fiber_and_bodies, linear_solver_fiber are probably the newest.
  COMBINE THEM INTO ONE ROUTINE AND PASS # OF BODIES AND FIBERS

10. What are the bottlenecks in the code? What about building preconditioner?
  IN integrators.py/step 4-build link matrix, also used in preconditioner but
  maybe there is no need for that in PC

11. We should upsample whenever derivatives of fiber are computed and
  upsampling rate must be chosen based on bending term, not two always.
  THERE MIGHT BE A BUG IN UPSAMPLING (CHECK EXAMPLE, 1 FIBER, W/ and W/O
  UPSAMPLING). ANTI-ALIASING BASED ON BENDING NEEDED

12. The main thing needed is building linear system including hydrodynamics?
  THIS IS IN multi/multi.py (the most recent version)

13. Kernels, do we have singular quadratures for Oseen, Rotlet,...
  IN EHSSAN's PAPER. IN multi_bodies, linear_operator_rigid_SK is very
  preliminary. Miranda & Powers  

14. Where is near-singular integration scheme?
  NOT IMPLEMENTED, CHECK FROM EHSSAN's CODE, and TALK TO DAVID ABOUT NEW
  IMPLEMENTATION with ALEX.

15. Oseen_kernel_source_target is O(N^2), but can be vectorized and reduced to
    one for loop.
  YES.

16. There must be a threshold to switch between PVFMM and Direct evaluation.
    This must be decided based on experiments.
  YES.

17. StressletxNormal and Complementary kernels are for what?
  MIRANDA & POWERS, TO REMOVE THE NULL SPACE.

18. Are all the PyCuda files complete?
  FORGET ABOUT FOR NOW.

19. Periphery does not seem complete?
  NOT SURE, IF NEEDED.

20. In create_fibers, we get error message when we create a straight fiber?
    What does clone mean?
   FIX THE SECOND ELSE IF, elif
   CLONE HAS THE CONFIGURATION OF THE RIGID BODY. VERTEX HAS THE
   DISCRETIZATION.


21. Shapes in shape_gallery are for periphery? ALSO FOR RIGID BODIES.
   DISCRETIZATION HAS TO BE TAKEN CARE OF (FOR OUTER AND INNER). SBF
   QUADRATURE FOR DISCRETIZATION OF THE RIGID BODIES, but SPHERICAL HARMONICS CAN 
   BE IMPLEMENTED.

22. Which examples are working?
   NONE OF THEM.

23. Which subroutines are complete and useful?
  LOOK AT FILELIST.MD

24. Which subroutines have bugs or incomplete?
  EVERY STEP HAS TO BE TESTED

25. How to create fibers and links, molecular motors and periphery?
  CREATE NUCLEATORS CREATES UNIFORM DISTRIBUTION OF POINTS ON A SURFACE, CAN BE
  USED FOR PERIPHERY, RIGID PARTICLES, MOLECULAR MOTORS
  USE RUN5002.0.0.inputfile instead of data.main

  How to create force generators and centrosomes with fiber nucleators:
  a. create_nucleators_new.py: it generates points 
   equally distributed on a sphere. The points can be restricted
   to be closer to a pole than a given angle.

  b. create_links_from_body.py: it generates the links on a spherical     
   surface given the points on a surface (generated with create_nucleators_new.py).
   
  c. create_fibers_from_links.py: create fibers from the links.


26. Visualizing the output using Visit?
  .xyz for spheres and .lines for fibers can be input to Visit. Look at the
  script how to generate this from .txt files. scriptAnalysis

27. Inextensibility might be enforced without the penalty term, then we might
    do length correction. The attempt is in fiber.py but boundary conditions
    might cause a problem. BCs might be added as constraints to the
    optimization problem.


