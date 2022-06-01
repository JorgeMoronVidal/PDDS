#include "BVPs/EDPs/Salou_Iterative.hpp"
#include "BVPs/Domains/rectangle.hpp"
#include "PDDAlgorithms/PDDSparseGM.hpp"

int main(int argc, char *argv[]){
    gsl_set_error_handler_off();
    //BoundaryValueProblem Definition
    bvp boundvalprob;
    boundvalprob.u = Equation_u;
    boundvalprob.g = EquationLI_g;
    boundvalprob.f = EquationLI_f;
    //boundvalprob.gradient = Equation_grad;
    boundvalprob.sigma = EquationLI_sigma;
    boundvalprob.distance = Rectangle2D;
    boundvalprob.absorbing = Stopping;
    boundvalprob.c = EquationLI_c;
    bvp nonlinboundprob;
    nonlinboundprob.u = Equation_u;
    nonlinboundprob.g = EquationLI_g;
    nonlinboundprob.num_f_2LUT = EquationLI_f_LUT;
    nonlinboundprob.num_c = EquationLI_c_LUT;
    nonlinboundprob.num_u = EquationLI_u_LUT;
    nonlinboundprob.num_gradient_LUT = EquationLI_grad_LUT;
    nonlinboundprob.sigma = EquationLI_sigma;
    nonlinboundprob.distance = Rectangle2D;
    nonlinboundprob.absorbing = Stopping;
    //boundvalprob.gradient = Equation_grad;
    std::string config("configuration.txt");
    PDDSparseGM PDDS(argc,argv,config);
    //First iteration
    PDDS.Solve(boundvalprob);
    PDDS.Solve_Subdomains_LinIt_First(boundvalprob);
    PDDS.Fullfill_Subdomains_Random(nonlinboundprob,0.1);
    //Following iterations
    for(int i = 0; i < 16; i++){
        PDDS.Solve_Iterative_numVR(i,nonlinboundprob);
        PDDS.Solve_Subdomains_LinIt(nonlinboundprob);
    }
    MPI_Finalize();
}