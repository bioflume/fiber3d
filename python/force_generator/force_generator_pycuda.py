# Standard imports
from __future__ import division, print_function 
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

# These lines set the precision of the cuda code
# to single or double. Set the precision
# in the following lines and edit the lines
# after   'mod = SourceModule("""'    accordingly
precision = 'single'
# precision = 'double'

mod = SourceModule("""
// Set real to single or double precision.
// This value has to agree witht the value
// for precision setted above.
typedef float real;
// typedef double real;

#include <stdio.h>
// #if __CUDA_ARCH__ < 600
//__device__ double atomicAdd_user(double* address, double val){
//    unsigned long long int* address_as_ull = (unsigned long long int*)address;
//    unsigned long long int old = *address_as_ull, assumed;
//    do{
//      assumed = old;
//      old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
//      // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
//    } while (assumed != old);
//    return __longlong_as_double(old);
//}
// #endif


/*

*/
__global__ void trap_particle(const real* x_fg, 
                              const int* occupied, 
                              int* sel_par,
                              int* sel_par_num,
                              const real* x_p, 
                              const int num_of_generators,
                              const int num_of_particles,  
                              const int* num_points_particle, 
                              const int* offset_particles, 
                              const real radius, 
                              const real alpha){

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i >= num_of_generators) return;   

  real rxij, ryij, rzij, rij;
  int count = 0;

  if(occupied[i] < 0){
    // Loop over particles
    for(int j=0; j<num_of_particles; j++){
      // Loop over point particles
      for(int k=0; k<num_points_particle[j]; k++){
        int offset = offset_particles[j] * 3;
        
        // Distance between point particle and force generator
        rxij = x_fg[i*3 + 0] - x_p[offset + k*3 + 0];
        ryij = x_fg[i*3 + 1] - x_p[offset + k*3 + 1];
        rzij = x_fg[i*3 + 2] - x_p[offset + k*3 + 2];
        rij = sqrt(rxij*rxij + ryij*ryij + rzij*rzij);
 
        // Get possible pairs -- there may be several points per particle
        if(rij < (radius * (1 + alpha * 5))){
          sel_par[i * 64 + count] = j;
          count++;
          break;
        }
      }
    }
  }
  sel_par_num[i] = count;

  // if(i == 19){
  //  printf("i = %i \\n", i);
  //  printf("count = %i \\n", count);
  //  printf("sel_par_num = %i \\n", sel_par_num[i]);
  //  printf("sel_par = %i \\n", sel_par[i * 64 + count - 1]);
  // }
}


/*

*/
__global__ void compute_force(const real* x_fg,
                              const real* x_p,
                              const real* rs_particle,
                              const int* occupied,
                              const int* offset_particles,
                              const int* num_points_particle,
                              real* force_g,
                              real* force_p,
                              const real active_force,
                              const real spring_constant,
                              const real radius,
                              const real alpha,
                              const int num_of_generators){

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i >= num_of_generators) return;   

  real rxij, ryij, rzij, rij;
  real fgx, fgy, fgz, fx, fy, fz;
  int count;
  
  // Skip occupied force generators
  if( occupied[i] > -1){
    // Get points of the particle interacting with the force generator
    int first_point = offset_particles[occupied[i]];
    int last_point = offset_particles[occupied[i]] + num_points_particle[occupied[i]];
   
    // Loop over points in particle
    count = 0;
    fgx = 0; fgy = 0; fgz = 0;
    for(int j=first_point; j<last_point; j++){
      rxij = x_fg[i*3 + 0] - x_p[j*3 + 0];
      ryij = x_fg[i*3 + 1] - x_p[j*3 + 1];
      rzij = x_fg[i*3 + 2] - x_p[j*3 + 2];
      rij = sqrt(rxij * rxij + ryij * ryij + rzij * rzij);

      // Compute forces
      // Active force
      fx = active_force * rs_particle[j*3 + 0];
      fy = active_force * rs_particle[j*3 + 1];
      fz = active_force * rs_particle[j*3 + 2];
        
      // Conservative force.
      fx += spring_constant * ((1.0 - rs_particle[j*3 + 0]*rs_particle[j*3 + 0]) * rxij +
                               (    - rs_particle[j*3 + 0]*rs_particle[j*3 + 1]) * ryij + 
                               (    - rs_particle[j*3 + 0]*rs_particle[j*3 + 2]) * rzij);
      fy += spring_constant * ((    - rs_particle[j*3 + 1]*rs_particle[j*3 + 0]) * rxij +
                               (1.0 - rs_particle[j*3 + 1]*rs_particle[j*3 + 1]) * ryij + 
                               (    - rs_particle[j*3 + 1]*rs_particle[j*3 + 2]) * rzij);
      fz += spring_constant * ((    - rs_particle[j*3 + 2]*rs_particle[j*3 + 0]) * rxij +
                               (    - rs_particle[j*3 + 2]*rs_particle[j*3 + 1]) * ryij + 
                               (1.0 - rs_particle[j*3 + 2]*rs_particle[j*3 + 2]) * rzij);

      // fx += spring_constant * rxij;
      // fy += spring_constant * ryij;
      // fz += spring_constant * rzij;

      // Smooth forces
      real smooth = (1.0 - 1.0 / (1.0 + exp(-(rij - radius) / (alpha * radius))));
      fx = fx * smooth;
      fy = fy * smooth;
      fz = fz * smooth;

      // Add forces to particle and force generator
      atomicAdd(&force_p[j*3 + 0], fx);
      atomicAdd(&force_p[j*3 + 1], fy);
      atomicAdd(&force_p[j*3 + 2], fz);
      fgx -= fx;
      fgy -= fy;
      fgz -= fz;
      count += 1;
    }
    // Save force acting on force generator
    force_g[i*3 + 0] += fgx / count;
    force_g[i*3 + 1] += fgy / count;
    force_g[i*3 + 2] += fgz / count;
  }
}
""")


def real(x):
  if precision == 'single':
    return np.float32(x)
  else:
    return np.float64(x)


def set_number_of_threads_and_blocks(num_elements):
  '''
  This functions uses a heuristic method to determine
  the number of blocks and threads per block to be
  used in CUDA kernels.
  '''
  threads_per_block=512
  if((num_elements//threads_per_block) < 512):
    threads_per_block = 256
  if((num_elements//threads_per_block) < 256):
    threads_per_block = 128
  if((num_elements//threads_per_block) < 128):
    threads_per_block = 64
  if((num_elements//threads_per_block) < 128):
    threads_per_block = 32
  num_blocks = (num_elements-1)//threads_per_block + 1

  return (threads_per_block, int(num_blocks))


def trap_particle_pycuda(r, occupied, r_particles, num_particles,  num_points_particle, offset_particles, radius, alpha, *args, **kwargs):
  '''
  If the active center is free, it captures one of the particles closer than
  the force_generator's radius. If there are several particles close, it selects one randomly.
  
  TODO: now the computational cost of this method is O(N**2),
  write a O(N) cost algorithm.
  '''

  # Determine number of threads and blocks for the GPU
  number_of_generators = np.int32(r.size // 3)
  number_of_points = np.int32(r_particles.size // 3)
  threads_per_block, num_blocks = set_number_of_threads_and_blocks(number_of_generators)

  # Reshape arrays
  x_fg = real(np.reshape(r, r.size))
  x_p = real(np.reshape(r_particles, r_particles.size))
  sel_par = np.full((r.size // 3) * 64, -1, dtype = np.int32)
  sel_par_num = np.zeros((r.size // 3), dtype = np.int32)

  # Allocate GPU memory
  x_fg_gpu = cuda.mem_alloc(x_fg.nbytes)
  x_p_gpu = cuda.mem_alloc(x_p.nbytes)
  occupied_gpu = cuda.mem_alloc(occupied.astype(np.int32).nbytes)
  num_points_particle_gpu = cuda.mem_alloc(num_points_particle.astype(np.int32).nbytes)
  offset_particles_gpu = cuda.mem_alloc(offset_particles.astype(np.int32).nbytes)
  sel_par_gpu = cuda.mem_alloc(sel_par.astype(np.int32).nbytes)
  sel_par_num_gpu = cuda.mem_alloc(sel_par_num.astype(np.int32).nbytes)

  # Copy data to the GPU (host to device)
  cuda.memcpy_htod(x_fg_gpu, x_fg)
  cuda.memcpy_htod(x_p_gpu, x_p)
  cuda.memcpy_htod(occupied_gpu, occupied.astype(np.int32))
  cuda.memcpy_htod(num_points_particle_gpu, num_points_particle.astype(np.int32))
  cuda.memcpy_htod(offset_particles_gpu, offset_particles.astype(np.int32))
  cuda.memcpy_htod(sel_par_gpu, sel_par.astype(np.int32))
  cuda.memcpy_htod(sel_par_gpu, sel_par_num.astype(np.int32))

  # Get GPU function
  trap_particle = mod.get_function("trap_particle")

  # Trap particles
  trap_particle(x_fg_gpu, 
                occupied_gpu, 
                sel_par_gpu, 
                sel_par_num_gpu, 
                x_p_gpu, 
                np.int32(r.size // 3), 
                np.int32(num_particles), 
                num_points_particle_gpu, 
                offset_particles_gpu, 
                real(radius), 
                real(alpha),
                block=(threads_per_block, 1, 1), grid=(num_blocks, 1))

  # Copy data from GPU to CPU (device to host)
  cuda.memcpy_dtoh(sel_par, sel_par_gpu)
  cuda.memcpy_dtoh(sel_par_num, sel_par_num_gpu)

  # Loop over force generators
  for i in range(int(r.size // 3)):
    # Skip occupied force generators
    if occupied[i] > -1:
      continue

    # Select one particle between the candidates
    if sel_par_num[i] > 0:
      occupied[i] = sel_par[i * 64 + np.random.random_integers(0, sel_par_num[i] - 1)]



def compute_force_pycuda(force_g, r, occupied, active_force, spring_constant, radius, alpha, 
                         r_particles, rs_particle, num_particles, num_points_particle, offset_particles, *args, **kwargs):
  '''
  If the active center it is occupied it computes the passive and
  active force generated on the particle.
  '''

  # Determine number of threads and blocks for the GPU
  number_of_generators = np.int32(r.size // 3)
  number_of_points = np.int32(r_particles.size // 3)
  threads_per_block, num_blocks = set_number_of_threads_and_blocks(number_of_generators)

  # Set forces to zero
  force_p = np.zeros_like(r_particles)
  force_g[:,:] = 0.0

  # Reshape arrays
  force_p = real(np.reshape(force_p, force_p.size))
  force_g = real(np.reshape(force_g, force_g.size))
  x_fg = real(np.reshape(r, r.size))
  x_p = real(np.reshape(r_particles, r_particles.size))
  rs_particle = real(np.reshape(rs_particle, r_particles.size))

  # Allocate GPU memory
  force_p_gpu = cuda.mem_alloc(force_p.nbytes)
  force_g_gpu = cuda.mem_alloc(force_g.nbytes)
  x_fg_gpu = cuda.mem_alloc(x_fg.nbytes)
  x_p_gpu = cuda.mem_alloc(x_p.nbytes)
  occupied_gpu = cuda.mem_alloc(occupied.astype(np.int32).nbytes)
  num_points_particle_gpu = cuda.mem_alloc(num_points_particle.astype(np.int32).nbytes)
  offset_particles_gpu = cuda.mem_alloc(offset_particles.astype(np.int32).nbytes)
  rs_particle_gpu = cuda.mem_alloc(rs_particle.nbytes)

  # Copy data to the GPU (host to device)
  cuda.memcpy_htod(force_g_gpu, force_g)
  cuda.memcpy_htod(force_p_gpu, force_p)
  cuda.memcpy_htod(x_fg_gpu, x_fg)
  cuda.memcpy_htod(x_p_gpu, x_p)
  cuda.memcpy_htod(occupied_gpu, occupied.astype(np.int32))
  cuda.memcpy_htod(num_points_particle_gpu, num_points_particle.astype(np.int32))
  cuda.memcpy_htod(offset_particles_gpu, offset_particles.astype(np.int32))
  cuda.memcpy_htod(rs_particle_gpu, rs_particle)

  # Get GPU function
  compute_force = mod.get_function("compute_force")
  
  compute_force(x_fg_gpu,
                x_p_gpu,
                rs_particle_gpu,
                occupied_gpu,
                offset_particles_gpu,
                num_points_particle_gpu,
                force_g_gpu,
                force_p_gpu,
                real(active_force),
                real(spring_constant),
                real(radius),
                real(alpha),
                np.int32(number_of_generators), 
                block=(threads_per_block, 1, 1), grid=(num_blocks, 1))
    
  # Copy data from GPU to CPU (device to host)
  cuda.memcpy_dtoh(force_p, force_p_gpu)
  cuda.memcpy_dtoh(force_g, force_g_gpu)

  force_g.reshape(force_g.size // 3, 3)
  return np.reshape(force_p, (force_p.size // 3, 3))
