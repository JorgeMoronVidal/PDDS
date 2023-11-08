#include "BVPs/EDPs/Monegros_Poisson_CPP.hpp"
#include "BVPs/Domains/rectangle.hpp"
#include "PDDAlgorithms/PDDSparseGM.hpp"

int main(int argc, char *argv[]){
    gsl_set_error_handler_off();
    //BoundaryValueProblem Definition
    bvp boundvalprob;
    boundvalprob.u = Equation_u;
    boundvalprob.g = Equation_g;
    boundvalprob.f = Equation_f;
    boundvalprob.gradient = Equation_grad;
    boundvalprob.sigma = Equation_sigma;
    boundvalprob.distance = Rectangle2D;
    boundvalprob.absorbing = Stopping;
    boundvalprob.c = Equation_c;
    std::string config("configuration.txt");
    PDDSparseGM PDDS(argc,argv,config);
    //PDDS.Print_Interface();
    PDDS.Solve(boundvalprob);
    MPI_Finalize();
}