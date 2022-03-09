#include "Solvers/EMFKACSolver.hpp"
#include "BVPs/EDPs/Monegros_Poisson.hpp"
#include "BVPs/Domains/rectangle.hpp"
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
    Eigen::Vector2d position(10,15);
    double params[4] = {0,0,20,20};
    solver.Solve_OMP(position,1000,0.08,0.16,boundvalprob,params);
    solver.Update();
    std::cout << boundvalprob.u(position,0.0) << "\t" << solver.phi<< std::endl;
    solver.Reset_sums();
}