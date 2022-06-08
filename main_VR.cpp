#include "BVPs/EDPs/Semilinear_u3_VR.hpp"
#include "BVPs/Domains/rectangle.hpp"
#include "PDDAlgorithms/PDDSparseGM.hpp"

int main(int argc, char *argv[]){
    gsl_set_error_handler_off();
    //BoundaryValueProblem Definition
    bvp boundvalprob;
    boundvalprob.u = EquationSM_u;
    boundvalprob.g = EquationSM_g;
    boundvalprob.f = EquationSM_f;
    //boundvalprob.gradient = Equation_grad;
    boundvalprob.sigma = EquationSM_sigma;
    boundvalprob.distance = Rectangle2D;
    boundvalprob.absorbing = Stopping;
    boundvalprob.c = EquationSM_c;
    bvp nonlinboundprob;
    nonlinboundprob.u = EquationSM_u;
    nonlinboundprob.g = EquationSM_g;
    nonlinboundprob.num_f = EquationSM_f_LUT;
    nonlinboundprob.num_c = EquationSM_c_LUT;
    nonlinboundprob.num_u = EquationSM_u_LUT;
    nonlinboundprob.num_gradient_LUT = EquationSM_grad_LUT;
    nonlinboundprob.sigma = EquationSM_sigma;
    nonlinboundprob.distance = Rectangle2D;
    nonlinboundprob.absorbing = Stopping;
    //boundvalprob.gradient = Equation_grad;
    std::string config("configuration.txt");
    PDDSparseGM PDDS(argc,argv,config);
    //First iteration
    PDDS.Solve(boundvalprob);
    PDDS.Solve_Subdomains(boundvalprob);
    //PDDS.Fullfill_Subdomains_Random(nonlinboundprob,0.1);
    //Following iterations
    for(int i = 0; i < 10; i++){
        PDDS.Solve_SemiLin_numVR(i,nonlinboundprob);
        PDDS.Solve_Subdomains_SemiLin(2,nonlinboundprob);
    }
}