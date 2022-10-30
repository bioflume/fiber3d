# TODOS 
=======

# PRIORITY:
------------------------

0. HIs with rigid bodies, fibers and links, molecular motors, cortex

1. PvFMM interface

2. Repulsion, collision detection, maybe near-singular integration

3. Profiling and improving stability, performance, memory usage

# MINOR
-----------------------
0. Reduce the number of for loops to 1 in direct evaluation of kernels.

1. Find a threshold to switch to PVFMM from direct evaluation of kernels.

2. Test all the subroutines.

3. Building preconditioner and also matrix for links is slow, speed that up. 

4. Currently we do not solve for nucleus, we solve only for centrosomes,
because we neglect nucleus-centrosome interaction and nucleus motion. Body
radius is set based on links file (MMs). Bodies can have different radius.
Radius also appears in forces/force.py and A_body in linear solver. Check if
bodies list include only one fiber, because we assign links to only bodies[-1]
not the other bodies in the list.

# DONE:
-------------------------

0. Adaptive time stepping of order 1 and 2.
   [10/25]

1. Input/Output [10/25]
 
2. Number of points per fiber is not changed from the input file, it should be
   upsampled or downsampled given the input regardless of the initialization.
   [10/25]

3. Modify output files so that we can visualize in MATLAB. 

  [(9/23) DONE]

4. Using the example (multi_fibers/many/multi_fiber.py), test aliasing issues
   and implement an anti-aliasing algorithm based on bending of fiber.
   Work with 1 fiber and 1 nucleus. Also Chebyshev differentiation via FFT is
   added, check if that is faster. 

  [(9/23) Anti-aliasing is implemented, and made as efficient as possible.
  However, upsampling matrices and differentiation maybe do not need to be computed again and
  again. If D4 = D2xD2 is the way to compute the matrix, then we should do it
  once for the fibers that have the same max. num. points.] 

5. Error computation based on fiber length is implemented on 9/23. Use it when
   taking time step size. Additionally, update length correction algorithm.
  
  [(9/24) Length correction algorithm is added but does not seem very useful, 
  it has several issues which need to be tested further. Maybe because of BCs.
  Maybe I should add BCs as constraints or should not touch the points attached.]

6. Implement nonlocal contribution of self-mobility in fiber.py.

  [(9/25) Although it seems there is no need to have the nonlocal term especially in dense
  suspensions, I have implemented it with regularization constant along
  arc-length.]

7. Implement singular quadratures for kernels (refer to Nazockdast et al. and
   Power & Miranda), (multi_bodies.py/linear_operator_rigid_SK)

  [(9/24) Singular quadratures are not used in our scheme, instead, a regularization
  term is added.]

8. Impelement near-singular integration scheme. Check if Ehssan's code has it,
   or talk to David.
   
  [(9/25) We do not have near-singular integration in the code, instead we use
  regularization. We can keep using regularization b/c I do not think special
  near-singular integration is needed in dense suspensions. But I can talk with
  David and see what we can implement.]

9. Implement linear solver with HIs (look at multi/multi.py).
  [(10/1) Floren has started doing that and e-mailed me the link. Look at that
  and improve]

10. Upsample when computing potentials.

11. Implement different number of points per fiber, offset is introduced to
   implement that idea.
   [(10/1) this is done and very useful. Now we allow different resolution for
   different fibers and maintain the maximum arc-length spacing the same.]

12. SOLVE STABILITY ISSUES EVEN WHEN THERE IS NO HIs. Remedies:
   - Length correction that respects BCs [(10/1) does not work]
   - Inextensibility correction 
   - Reparameterization along arc-length [(10/2) works]
   - Dealiasing (not 1/2)
   [(10/1) upsampling has to be implemented more carefully, i.e., whenever we
   compute derivatives, we should upsample, but this is not done in mobility
   problem]

  [(10/14) Upsampling is done rigorously]

13. Consistent ordering of variables in arrays (xcoords-ycoords-zcoords, then
    tension), not (x,y,z,tension,...,x,y,z,tension,...)

14. PVFMM needs to be linked to the fiber code.




