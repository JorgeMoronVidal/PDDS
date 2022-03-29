#include "BVPs/EDPs/Semilinear_u3.hpp"
#include "BVPs/Domains/rectangle.hpp"
#include "PDDAlgorithms/PDDSparseGM.hpp"

int main(int argc, char *argv[]){
    //BoundaryValueProblem Definition
    bvp boundvalprob;
    boundvalprob.u = EquationSM_u;
    boundvalprob.g = EquationSM_g;
    boundvalprob.f = EquationSM_f;
    //boundvalprob.gradient = Equation_grad;
    boundvalprob.sigma = EquationSM_sigma;
    boundvalprob.distance = Rectangle2D;
    boundvalprob.absorbing = Stopping;
    bvp nonlinboundprob;
    nonlinboundprob.u = EquationSM_u;
    nonlinboundprob.g = EquationSM_g;
    nonlinboundprob.num_f = EquationSM_Residual;
    nonlinboundprob.num_c = EquationSM_c;
    nonlinboundprob.sigma = EquationSM_sigma;
    nonlinboundprob.distance = Rectangle2D;
    nonlinboundprob.absorbing = Stopping;
    //boundvalprob.gradient = Equation_grad;
    std::string config("configuration.txt");
    PDDSparseGM PDDS(argc,argv,config);
    //First iteration
    //PDDS.Solve(boundvalprob);
    //PDDS.Solve_Subdomains(boundvalprob);
    PDDS.Fullfill_Subdomains_Random(boundvalprob,0.5);
    //Following iterations
    for(int i = 0; i < 5; i++){
        PDDS.Solve_SemiLin(i,nonlinboundprob);
        PDDS.Solve_Subdomains_SemiLin(i,nonlinboundprob);
    }
}