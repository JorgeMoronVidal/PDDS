#include "BVPs/EDPs/Monegros_Poisson_CPP.hpp"
#include "BVPs/Domains/rectangle.hpp"
#include "PDDAlgorithms/PDDSparseGM.hpp"

int main(int argc, char *argv[]){
    gsl_set_error_handler_off();
    //BoundaryValueProblem Definition
    bvp boundvalprob;
    boundvalprob.u = Equation_u;
    boundvalprob.g = EquationLI_g;
    boundvalprob.f = EquationLI_f;
    boundvalprob.gradient = Equation_grad;
    boundvalprob.sigma = EquationLI_sigma;
    boundvalprob.distance = Rectangle2D;
    boundvalprob.absorbing = Stopping;
    boundvalprob.c = EquationLI_c;
    //boundvalprob.gradient = Equation_grad;
    std::string config("configuration.txt");
    PDDSparseGM PDDS(argc,argv,config);
    //First iteration
    PDDS.Solve(boundvalprob);
    //PDDS.Solve_Subdomains_LinIt_First(boundvalprob);
    //Following iterations
    //for(int i = 0; i < 10; i++){
    //    PDDS.Solve_Iterative_numVR(i,nonlinboundprob);
    //   PDDS.Solve_Subdomains_LinIt(nonlinboundprob);
    //}
    MPI_Finalize();
}