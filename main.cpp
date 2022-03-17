#include "BVPs/EDPs/Monegros_2.hpp"
#include "BVPs/Domains/rectangle.hpp"
#include "PDDAlgorithms/PDDSparseGM.hpp"
int main(int argc, char *argv[]){
    //BoundaryValueProblem Definition
    bvp boundvalprob;
    boundvalprob.u = Equation_u;
    boundvalprob.g = Equation_g;
    boundvalprob.f = Equation_f;
    boundvalprob.c = Equation_c;
    boundvalprob.sigma = Equation_sigma;
    boundvalprob.gradient = Equation_grad;
    boundvalprob.distance = Rectangle2D;
    boundvalprob.absorbing = Stopping;
    std::string config("configuration.txt");
    PDDSparseGM PDDS(argc,argv,config);
    PDDS.Solve(boundvalprob);
    PDDS.Solve_Subdomains(boundvalprob);
}