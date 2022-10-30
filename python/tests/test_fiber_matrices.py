'''
Here are the parameters I am thinking: Consider four fibers centered around a circle of radius d=0.2. 
The fibers have initial tangent vector X_s(t=0)=(0,0,1) (they are aligned in the z direction) and positions 
X(t=0)=(d,0,s-1), (0,d,s-1), (-d,0,s-1), and (0,-d,s-1), where 0 <= s <= L=2. The other parameters are 
\mu=E=1 and \epsilon=10^{-3}. The total gravitational force on the fibers is 10. Simulate from t=0 to 
t=0.25. The attached file shows the initial and final positions of the fibers. 
'''
import numpy as np
import sys
sys.path.append('../')
from fiber import fiber
from bpm_utilities import gmres
from tstep import tstep
from tstep import initialize
from tstep import fiber_matrices


if __name__ == '__main__':
  print('# Start')
  # Set parameters
  options = initialize.set_options(adaptive_num_points = False,
                                  ireparam = False,
                                  filtering = True,
                                  adaptive_time = True,
                                  time_step_scheme = 'fiber_hydro_implicit_time_step',
                                  order = 1,
                                  dt = 1e-03,
                                  dt_min = 1e-05,
                                  dt_max = 1,
                                  tol_tstep = 1e-04,
                                  tol_gmres = 1e-12,
                                  inonlocal = False,
                                  igrowing = False,
                                  output_name = 'data/tests/run4fibs')

  prams = initialize.set_parameters(eta = 1.0, epsilon = 1e-03, final_time = 1.0)

  # Create four fibers
  num_points = 16
  d = 0.2
  Nf = 4
  E = 0.1
  length = 2.0

  # Fiber 0
  fib = fiber.fiber(num_points = num_points, 
                      dt = options.dt, 
                      E = E, 
                      length = length, 
                      epsilon = prams.epsilon,
                      inonlocal = options.inonlocal,
                      ireparam = options.ireparam,
                      adaptive_num_points = options.adaptive_num_points,
                      tstep_order = options.order)
  fib.x[:,0] = d
  fib.x[:,1] = 0.0
  fib.x[:,2] = fib.s - 1.0
  

  

  fib_mat = fiber_matrices.fiber_matrices(num_points = num_points, filtering = options.filtering, uprate_poten = 2)
  fib_mat.compute_matrices()
  

  # Find upsampling rate and compute upsampling matrices
  fib.find_upsample_rate()

  # Compare matrices

  alpha, s, out3, out4 = fib_mat.get_matrices(length, fib.num_points_up, 'alpha_and_s')

  D_1, D_2, D_3, D_4 = fib_mat.get_matrices(length, fib.num_points_up, 'Ds')

  D_1_0, D_2_0, D_3_0, D_4_0 = fib_mat.get_matrices(length, fib.num_points_up, 'D0s')

  D_1_up, D_2_up, D_3_up, D_4_up = fib_mat.get_matrices(length, fib.num_points_up, 'D_ups')

  D_1_0_up, D_2_0_up, D_3_0_up, D_4_0_up = fib_mat.get_matrices(length, fib.num_points_up, 'D0_ups')  

  D_1_maxUp, D_2_maxUp, D_3_maxUp, D_4_maxUp = fib_mat.get_matrices(length, fib.num_points_up, 'D_maxUps')  

  alpha_maxUp, out2, out3, out4 = fib_mat.get_matrices(length, fib.num_points_up, 'alpha_maxUp')  

  alpha_roots, alpha_tension, out3, out4 = fib_mat.get_matrices(length, fib.num_points_up, 'alpha_roots_tension')

  weights, weights_up, out3, out4 = fib_mat.get_matrices(length, fib.num_points_up, 'weights_all')

  P_maxUp, P_maxDown, out3, out4 = fib_mat.get_matrices(length, fib.num_points_up, 'P_max')

  P_kerUp, P_kerDn, out3, out4 = fib_mat.get_matrices(length, fib.num_points_up, 'P_kernel')

  alpha_kerUp, s_kerUp, out3, out4 = fib_mat.get_matrices(length, fib.num_points_up, 'alpha_and_s_kernel')

  P_X, P_T, P_cheb_representations_all_dof, out4 = fib_mat.get_matrices(length, fib.num_points_up, 'P_X_and_P_T_and_P_cheb')

  alpha_up, s_up, out3, out4 = fib_mat.get_matrices(length, fib.num_points_up, 'alpha_and_s_up')

  P_up, P_down, out3, out4 = fib_mat.get_matrices(length, fib.num_points_up, 'P_upsample')

  P_X_filter, P_T_filter, P_cheb_representations_all_dof_filter, out4 = fib_mat.get_matrices(length, fib.num_points_up, 'PX_PT_Pcheb_filter')

  norm = np.linalg.norm(alpha-fib.alpha, 2)
  print('Alpha: ', norm)

  norm = np.linalg.norm(s-fib.s, 2)
  print('s: ', norm)

  norm = np.linalg.norm(D_1-fib.D_1, 'fro')
  print('D_1: ', norm)

  norm = np.linalg.norm(D_2-fib.D_2, 'fro')
  print('D_2: ', norm)

  norm = np.linalg.norm(D_3-fib.D_3, 'fro')
  print('D_3: ', norm)

  norm = np.linalg.norm(D_4-fib.D_4, 'fro')
  print('D_4: ', norm)

  norm = np.linalg.norm(D_1_0-fib.D_1_0, 'fro')
  print('D_1_0: ', norm)

  norm = np.linalg.norm(D_2_0-fib.D_2_0, 'fro')
  print('D_2_0: ', norm)

  norm = np.linalg.norm(D_3_0-fib.D_3_0, 'fro')
  print('D_3_0: ', norm)

  norm = np.linalg.norm(D_4_0-fib.D_4_0, 'fro')
  print('D_4_0: ', norm)

  norm = np.linalg.norm(D_1_up-fib.D_1_up, 'fro')
  print('D_1_up: ', norm)

  norm = np.linalg.norm(D_2_up-fib.D_2_up, 'fro')
  print('D_2_up: ', norm)

  norm = np.linalg.norm(D_3_up-fib.D_3_up, 'fro')
  print('D_3_up: ', norm)

  norm = np.linalg.norm(D_4_up-fib.D_4_up, 'fro')
  print('D_4_up: ', norm)  

  norm = np.linalg.norm(D_1_0_up-fib.D_1_0_up, 'fro')
  print('D_1_0_up: ', norm)

  norm = np.linalg.norm(D_2_0_up-fib.D_2_0_up, 'fro')
  print('D_2_0_up: ', norm)

  norm = np.linalg.norm(D_3_0_up-fib.D_3_0_up, 'fro')
  print('D_3_0_up: ', norm)

  norm = np.linalg.norm(D_4_0_up-fib.D_4_0_up, 'fro')
  print('D_4_0_up: ', norm)  

  norm = np.linalg.norm(D_1_maxUp-fib.D_1_maxUp, 'fro')
  print('D_1_maxUp: ', norm)

  norm = np.linalg.norm(D_2_maxUp-fib.D_2_maxUp, 'fro')
  print('D_2_maxUp: ', norm)

  norm = np.linalg.norm(D_3_maxUp-fib.D_3_maxUp, 'fro')
  print('D_3_maxUp: ', norm)

  norm = np.linalg.norm(D_4_maxUp-fib.D_4_maxUp, 'fro')
  print('D_4_maxUp: ', norm) 

  norm = np.linalg.norm(alpha_maxUp - fib.alpha_maxUp,2)
  print('alpha_maxUp: ', norm)

  norm = np.linalg.norm(alpha_roots - fib.alpha_roots,2)
  print('alpha_roots: ', norm)

  norm = np.linalg.norm(alpha_tension - fib.alpha_tension,2)
  print('alpha_tension: ', norm)

  norm = np.linalg.norm(weights - fib.weights,2)
  print('weights: ', norm)

  norm = np.linalg.norm(weights_up - fib.weights_up,2)
  print('weights_up: ', norm)
  
  norm = np.linalg.norm(P_maxUp-fib.P_maxUp, 'fro')
  print('P_maxUp: ', norm)   

  norm = np.linalg.norm(P_maxDown-fib.P_maxDown, 'fro')
  print('P_maxDown: ', norm)   

  norm = np.linalg.norm(P_kerUp-fib.P_kerUp, 'fro')
  print('P_kerUp: ', norm)   

  norm = np.linalg.norm(P_kerDn-fib.P_kerDn, 'fro')
  print('P_kerDn: ', norm)   

  norm = np.linalg.norm(alpha_kerUp - fib.alpha_kerUp,2)
  print('alpha_kerUp: ', norm)

  norm = np.linalg.norm(s_kerUp - fib.s_kerUp,2)
  print('s_kerUp: ', norm)

  norm = np.linalg.norm(P_X-fib.P_X, 'fro')
  print('P_X: ', norm)     

  norm = np.linalg.norm(P_T-fib.P_T, 'fro')
  print('P_T: ', norm)     

  norm = np.linalg.norm(P_cheb_representations_all_dof-fib.P_cheb_representations_all_dof, 'fro')
  print('P_cheb_representations_all_dof: ', norm)     

  norm = np.linalg.norm(P_X_filter-fib.P_X_filter, 'fro')
  print('P_X_filter: ', norm)     

  norm = np.linalg.norm(P_T_filter-fib.P_T_filter, 'fro')
  print('P_T_filter: ', norm)     

  norm = np.linalg.norm(P_cheb_representations_all_dof_filter-fib.P_cheb_representations_all_dof_filter, 'fro')
  print('P_cheb_representations_all_dof_filter: ', norm)     

  norm = np.linalg.norm(alpha_up - fib.alpha_up,2)
  print('alpha_up: ', norm)

  norm = np.linalg.norm(s_up - fib.s_up,2)
  print('s_up: ', norm)  

  norm = np.linalg.norm(P_up-fib.P_up, 'fro')
  print('P_up: ', norm)   

  norm = np.linalg.norm(P_down-fib.P_down, 'fro')
  print('P_down: ', norm)   


  
  

  print('\n\n\n# End')    
