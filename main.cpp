#include "Solvers/EMFKACSolver.hpp"
#include "BVPs/EDPs/Monegros_Poisson.hpp"
#include "BVPs/Domains/rectangle.hpp"
#include <boost/random/variate_generator.hpp>
int main(){
    //BoundaryValueProblem Definition
    bvp boundvalprob;
    boundvalprob.u = Equation_u;
    boundvalprob.g = Equation_g;
    boundvalprob.f = Equation_f;
    boundvalprob.sigma = Equation_sigma;
    boundvalprob.gradient = Equation_grad;
    boundvalprob.distance = Rectangle2D;
    boundvalprob.absorbing = Stopping;
    EMFKACSolver solver;
    Eigen::Vector2d position(3,4);
    double params[4] = {0,0,10,10};
    //solver.Solve_OMP(position,5000,0.08,0.16,boundvalprob,params);
    //solver.Update();
    //std::cout << boundvalprob.u(position,0.0) << "\t" << solver.phi_VR<< std::endl;
    printf("h var stat.err bias\n");
    for(double h = 0.02; h>0.0002;h = 0.5*h) solver.Solve_OMP(position,1.0,5000,h,0.2,boundvalprob,params);
}