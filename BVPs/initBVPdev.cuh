#include <iostream>
#include <eigen3/Eigen/Core>
#include <BVPs/BVP.cuh>
#include "Domains/rectangle.cuh"
#include "EDPs/Monegros_Poisson.hpp"
__device__ bvpdev initBVPdev(void){
  bvpdev boundvalprob;
  boundvalprob.u = Equation_dev_u;
  boundvalprob.g = Equation_dev_g;
  boundvalprob.f = Equation_dev_f;
  boundvalprob.sigma = Equation_dev_sigma;
  boundvalprob.gradient = Equation_dev_grad;
  boundvalprob.distance = Rectangle2D_dev;
  boundvalprob.absorbing = Stopping_dev;
  return boundvalprob;
}
