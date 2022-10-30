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


__global__ void test(){
  printf("Hola from the GPU \\n");
}

""")



if __name__ == '__main__':
    
  print('# Start')
  test = mod.get_function("test")
  test(block=(2, 1, 1), grid=(1, 1)) 
  print('# End')



