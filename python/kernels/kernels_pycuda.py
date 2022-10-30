import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

# These lines set the precision of the cuda code
# to single or double. Set the precision
# in the following lines and edit the lines
# after   'mod = SourceModule("""'    accordingly
# precision = 'single'
precision = 'double'

mod = SourceModule("""
// Set real to single or double precision.
// This value has to agree witht the value
// for precision setted above.
// typedef float real;
typedef double real;

#include <stdio.h>

__global__ void oseen_kernel_source_target(real *r_source, 
                                           real *r_target, 
                                           real *density, 
                                           real *u, 
                                           int number_of_sources, 
                                           int number_of_targets, 
                                           real eta,
                                           real epsilon_distance){

  int xn = blockDim.x * blockIdx.x + threadIdx.x;
  if(xn >= number_of_targets) return;   

  real pi = real(4.0) * atan(real(1.0));
  real factor = 1.0 / (real(8.0) * pi * eta);
  real epsilon_distance_inv = real(1.0) / epsilon_distance;

  int offsetX, offsetY;
  real x, y, z;
  real r_norm_inv, fr, gr;
  real Mxx, Mxy, Mxz, Myy, Myz, Mzz;


  // Loop over targets
  offsetX = 3*xn;
  for(int yn=0; yn<number_of_sources; yn++){
    offsetY = 3*yn;
    x = r_target[offsetX+0] - r_source[offsetY+0];
    y = r_target[offsetX+1] - r_source[offsetY+1];
    z = r_target[offsetX+2] - r_source[offsetY+2];

    r_norm_inv = rsqrt(x*x + y*y + z*z);
    if(r_norm_inv > epsilon_distance_inv) continue;

    fr = factor * r_norm_inv;
    gr = factor * r_norm_inv * r_norm_inv * r_norm_inv;
    Mxx = fr + gr * x*x;
    Mxy =      gr * x*y;
    Mxz =      gr * x*z;
    Myy = fr + gr * y*y;
    Myz =      gr * y*z;
    Mzz = fr + gr * z*z;

    u[offsetX + 0] += Mxx * density[offsetY + 0] + Mxy * density[offsetY + 1] + Mxz * density[offsetY + 2];
    u[offsetX + 1] += Mxy * density[offsetY + 0] + Myy * density[offsetY + 1] + Myz * density[offsetY + 2];
    u[offsetX + 2] += Mxz * density[offsetY + 0] + Myz * density[offsetY + 1] + Mzz * density[offsetY + 2];   
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
  if((num_elements/threads_per_block) < 512):
    threads_per_block = 256
  if((num_elements/threads_per_block) < 256):
    threads_per_block = 128
  if((num_elements/threads_per_block) < 128):
    threads_per_block = 64
  if((num_elements/threads_per_block) < 128):
    threads_per_block = 32
  num_blocks = (num_elements-1)/threads_per_block + 1

  return (threads_per_block, int(num_blocks))


def oseen_kernel_source_target_pycuda(r_source, r_target, density, eta = 1.0, epsilon_distance = 1e-10):
  '''

  '''
  # Determine number of threads and blocks for the GPU
  number_of_sources = np.int32(r_source.size // 3)
  number_of_targets = np.int32(r_target.size // 3)
  threads_per_block, num_blocks = set_number_of_threads_and_blocks(number_of_targets)

  # Reshape and copy with real precision 
  r_source_real = real(np.reshape(r_source, r_source.size))
  r_target_real = real(np.reshape(r_target, r_target.size))
  density_real = real(np.reshape(density, density.size))
  
  # Allocate GPU memory
  r_source_gpu = cuda.mem_alloc(r_source_real.nbytes)
  r_target_gpu = cuda.mem_alloc(r_target_real.nbytes)
  density_gpu = cuda.mem_alloc(density_real.nbytes)
  u_gpu = cuda.mem_alloc(r_target_real.nbytes)
  number_of_sources_gpu = cuda.mem_alloc(number_of_sources.nbytes)
  number_of_targets_gpu = cuda.mem_alloc(number_of_targets.nbytes)

  # Copy data to the GPU (host to device)
  cuda.memcpy_htod(r_source_gpu, r_source_real)
  cuda.memcpy_htod(r_target_gpu, r_target_real)
  cuda.memcpy_htod(density_gpu, density_real)

  # Get GPU kernel
  K = mod.get_function("oseen_kernel_source_target")

  # Compute product
  K(r_source_gpu, r_target_gpu, density_gpu, u_gpu, number_of_sources, number_of_targets, real(eta), 
    real(epsilon_distance), block=(threads_per_block, 1, 1), grid=(num_blocks, 1)) 

  # Copy data from GPU to CPU (device to host)
  u = np.empty_like(r_target_real)
  cuda.memcpy_dtoh(u, u_gpu)
  return np.float64(u)



