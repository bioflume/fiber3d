// Functions to write VTK files from C++ 
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <vector>
#include "visit_writer.h"
#include "visit_writer.c"

namespace bp = boost::python;
namespace np = boost::python::numpy;

void visitWriterInterface(std::string name,
                            int format,
                            np::ndarray dims,
                            np::ndarray x,
                            np::ndarray y,
                            np::ndarray z,
                            int nvars,
                            np::ndarray vardims,
                            np::ndarray centering,
                            bp::list varnames,
                            bp::list variables){

  // Copy python variables to C++ variables
  // int dims_array[3];
  // for(int i=0; i<3; i++)
  //   dims_array[i] = bp::extract<int>(dims[i]);
  int *dims_array = reinterpret_cast<int *>(dims.get_data());

  // int vardims_array[1];
  // vardims_array[0] = bp::extract<int>(vardims[0]);
  int *vardims_array = reinterpret_cast<int *>(vardims.get_data());

  // int centering_array[1];
  // centering_array[0] = bp::extract<int>(centering[0]);
  int *centering_array = reinterpret_cast<int *>(centering.get_data());

  std::string varnames_array[1];
  varnames_array[0] = bp::extract<std::string>(varnames[0]);
  int sizeNameVelocity = varnames_array[0].size();
  char nameVelocity[sizeNameVelocity]; 
  varnames_array[0].copy(nameVelocity, sizeNameVelocity);
  char **varnames_char = new char* [1];
  varnames_char[0] = &nameVelocity[0];

  // // bp::numeric::array variables_velocity = bp::extract<bp::numeric::array>(variables[0]);
  // // int nCells = (dims_array[0]-1)*(dims_array[1]-1)*(dims_array[2]-1);
  // // double *velocity = new double [3*nCells];
  // // for(int i=0; i<nCells*3;i++){
  // //   velocity[i] = bp::extract<double>(variables_velocity[i]);
  // // }
  np::ndarray variables_velocity = bp::extract<np::ndarray>(variables[0]);
  double *velocity = reinterpret_cast<double *>(variables_velocity.get_data());

  double **vars;
  vars = new double* [1];
  vars[0] = velocity;
  
  double* xmesh = reinterpret_cast<double *>(x.get_data());
  double* ymesh = reinterpret_cast<double *>(y.get_data());
  double* zmesh = reinterpret_cast<double *>(z.get_data());
    

  // Print variables
  if(1){
    std::cout << std::endl << "visitWriterInterface " << std::endl;
    std::cout << "name: " << name << std::endl;
    std::cout << "format: " << format << std::endl;
    std::cout << "dims: " << dims_array[0] << "  " << dims_array[1] << "  " << dims_array[2] << std::endl;
    std::cout << "nvars: " << nvars << std::endl;
    std::cout << "vardims: " << vardims_array[0] << std::endl;
    std::cout << "centering: " << centering_array[0] << std::endl;
    std::cout << "varnames: " << varnames_array[0] << std::endl;
    // if(0)
	  // for(int i=0; i<nCells*3;i++){
	  //   std::cout << "velocity " << i << "  " << velocity[i] << std::endl;
	  //  }
    std::cout << std::endl;
  }

  // Call visit_writer
  /*Use visit_writer to write a regular mesh with data. */
  write_rectilinear_mesh(name.c_str(),    // Output file
                         format,          // 0=ASCII,  1=Binary
                         dims_array,      // {mx, my, mz}
                         xmesh,           
                         ymesh,
                         zmesh,
                         nvars,           // number of variables
                         vardims_array,   // Size of each variable, 1=scalar, velocity=3*scalars
                         centering_array,  
                         varnames_char,   
                         vars);

  // // Free memory in the heap
  // delete[] velocity;
  delete[] vars;
  delete[] varnames_char;
  // delete[] xmesh, ymesh, zmesh;
  // delete[] dims_array, vardims_array, centering_array;
}


BOOST_PYTHON_MODULE(visit_writer_interface)
{
  using namespace boost::python;

  // Initialize numpy
  Py_Initialize();
  np::initialize();

  // boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
  def("visit_writer_interface", visitWriterInterface);
}

