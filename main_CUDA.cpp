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
    //boundvalprob.gradient = Equation_grad;
    std::string config("configuration.txt");
    PDDSparseGM PDDS(argc,argv,config);
    //First iteration
    PDDS.Solve_CUDA(boundvalprob);
    //PDDS.Solve_Subdomains_LinIt_First(boundvalprob);
    //Following iterations
    //for(int i = 0; i < 10; i++){
    //    PDDS.Solve_Iterative_numVR(i,nonlinboundprob);
    //   PDDS.Solve_Subdomains_LinIt(nonlinboundprob);
    //}
    MPI_Finalize();
}