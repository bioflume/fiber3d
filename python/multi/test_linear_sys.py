import numpy as np
import sys
sys.path.append('../')
from fiber import fiber
from bpm_utilities import gmres
from tstep import tstep
from tstep import initialize
from tstep import tstep_utils
from read_input import read_input
from read_input import read_fibers_file
from read_input import read_vertex_file
from read_input import read_clones_file
from read_input import read_links_file
import scipy.sparse as scsp
import scipy.linalg as scla
import scipy.sparse.linalg as scspla
import time
from utils import timer

from numba import njit, prange
from numba.typed import List


def sequential_preconditioner(A_fibers_blocks):
  Q_all = []
  R_all = []

  for k in range(len(A_fibers_blocks)):
    Q, R = scla.qr(A_fibers_blocks[k], check_finite=False)
    Q_all.append(Q)
    R_all.append(R)

  return Q_all, R_all

def lu_preconditioner(A_fibers_blocks):
  LU_all = []
  P_all = []
  for k in range(len(A_fibers_blocks)):
    LU, P = scla.lu(A_fibers_blocks[k], permute_l = True)
    LU_all.append(LU)
    P_all.append(P)

  return LU_all, P_all


def sequential_solve_lin(x,offset_fibers,Q_all,R_all):
  y = np.zeros_like(x)
  for k in range(len(Q_all)):
    istart = offset_fibers[k]*4
    iend = offset_fibers[k+1]*4
    y[istart:iend] = scla.solve_triangular(R_all[k], np.dot(Q_all[k].T, x[istart:iend]))

  return y

@njit(parallel=True)
def parallel_solve_lin(x,offset_fibers,A_fibers_blocks):
  y = np.zeros_like(x)
  for k in prange(len(A_fibers_blocks)):
    istart = offset_fibers[k]*4
    iend = offset_fibers[k+1]*4
    y[istart:iend] = np.linalg.solve(A_fibers_blocks[k], x[istart:iend])

  return y 



@njit(parallel=True)
def parallel_preconditioner(A_fibers_blocks):
  Q_all = [np.zeros((1,1)) for n in range(len(A_fibers_blocks))]
  R_all = [np.zeros((1,1)) for n in range(len(A_fibers_blocks))]
 
  for k in prange(len(A_fibers_blocks)):   
    Q, R = np.linalg.qr(A_fibers_blocks[k])
    Q_all[k] = Q.astype(np.float64)
    R_all[k] = np.triu(R)

  return Q_all, R_all


if __name__ == '__main__':

  input_file = './input_files/experiment_two_centrosomes_movingMM.inputfile'
  read = read_input.ReadInput(input_file)
  

  options = initialize.set_options(adaptive_num_points = True,
                                  num_points = 64,
                                  num_points_max = 96,
                                  ireparam = False,
                                  filtering = True,
                                  adaptive_time = True,
                                  time_step_scheme = 'time_step_dry',
                                  order = 1,
                                  dt = 1e-3,
                                  dt_min = 1e-04,
                                  dt_max = 1e-02,
                                  tol_tstep = 5e-02,
                                  tol_gmres = 1e-10,
                                  inonlocal = False,
                                  random_seed = 1,
                                  output_txt_files = True,
                                  n_save = 1,
                                  isaveForces = True,
                                  output_name = 'data/QR_test/run')

  prams = initialize.set_parameters(eta = 0.1, epsilon = 1e-03, final_time = 2000)

  # Create time stepping scheme
  tstep = tstep.tstep(prams,options,input_file,None,None,None)

  # Get fibers
  fibers = tstep.fibers

  num_bodies, num_fibers, offset_bodies, offset_fibers, system_size = tstep_utils.get_num_particles_and_offsets(tstep.bodies, tstep.fibers, tstep.shell, ihydro = False)
  
  force_bodies = np.zeros((len(tstep.bodies),6))
  force_fibers = np.zeros((offset_fibers[-1],3))

  timer.timer('build_linear_system')
  As_fibers, A_fibers_blocks, RHS_all = tstep_utils.get_fibers_and_bodies_matrices(tstep.fibers, tstep.bodies, tstep.shell,
      system_size, offset_fibers, offset_bodies, force_fibers, force_bodies, None, None, None, tstep.fib_mats, tstep.fib_mat_resolutions, 
      BC_start_0 = 'velocity', BC_start_1 = 'angular_velocity',BC_end_0 = 'force', ihydro = False)
  timer.timer('build_linear_system')

  # Compile the code
  Q_par, R_par = parallel_preconditioner(A_fibers_blocks)

  x = np.ones((offset_fibers[-1]*4,1))
  y_par = parallel_solve_lin(x,offset_fibers,A_fibers_blocks)
  
  timer.timer('sequential_preconditioner')
  Q_seq, R_seq = sequential_preconditioner(A_fibers_blocks)
  timer.timer('sequential_preconditioner')

  for i in range(20):
    #timer.timer('parallel_preconditioner')
    #Q_par, R_par = parallel_preconditioner(A_fibers_blocks)
    #timer.timer('parallel_preconditioner')

    timer.timer('parallel_solve')
    y_par = parallel_solve_lin(x,offset_fibers,A_fibers_blocks)
    timer.timer('parallel_solve')

    timer.timer('sequential_solve')
    y_seq = sequential_solve_lin(x,offset_fibers,Q_par,R_par)
    timer.timer('sequential_solve')

    #timer.timer('lu_preconditioner')
    #L_all, U_all = lu_preconditioner(A_fibers_blocks)
    #timer.timer('lu_preconditioner')

  timer.timer(' ', print_all = True)





