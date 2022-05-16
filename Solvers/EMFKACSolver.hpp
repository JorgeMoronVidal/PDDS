#include "../Integrators/FKACIntegrator.hpp"
#include "../Meshes/stencil.hpp"
#include "../Meshes/subdomain.hpp"
#include <omp.h>
#include <vector>
#include <iostream>
#ifndef EMFKACSOLVER
#define EMFKACSOLVER
enum sumindex
{   ScoreLinear = 0,
    ScoreSublinear = 1,
    ScoreLinearNum = 2,
    ScoreSublinearNum = 3,
    ScoreLinear2 = 4,
    ScoreSublinear2 = 5,
    ScoreLinearNum2 = 6,
    ScoreSublinearNum2 = 7,
    XiLinear = 8,
    XiSublinear = 9,
    XiLinearNum = 10,
    XiSublinearNum = 11,
    XiLinear2 = 8,
    XiSublinear2 = 9,
    XiLinearNum2 = 10,
    XiSublinearNum2 = 11,
    ScoreLinearVR = 12,
    ScoreSublinearVR = 13,
    ScoreLinearVRNum = 14,
    ScoreSublinearVRNum = 15,
    ScoreLinearVR2 = 16,
    ScoreSublinearVR2 = 17,
    ScoreLinearVRNum2 = 18,
    ScoreSublinearVRNum2 = 19,
    XiScoreLinear = 20,
    XiScoreSublinear = 21,
    XiScoreLinearNum = 22,
    XiScoreSublinearNum = 23,
    tauLinear = 24,
    tauSublinear= 25,
    RNGCalls = 26
};
class EMFKACSolver{
private:
    //Feynmann_Kac processes and final quantities
    std::vector<Eigen::Vector2d> X_tau_lin, X_tau_sublin;
    std::vector<double> Y_tau_lin, Y_tau_sublin, Z_tau_lin, Z_tau_sublin,
    tau_lin, tau_sublin,xi_lin,xi_sublin;
    std::vector<unsigned int> RNGcallsv;
    //RNG
    std::vector<boost::mt19937> RNG;
    std::vector<boost::normal_distribution<double>> normal_dist;
    std::vector<boost::exponential_distribution<double>> exp_dist;
    //sums of important quantities to control simulations
    double sums[30];
    //Random number generator
public:
    std::vector<int> threads;
    //Time discretization of the trajectories and initial time of the trajectory T
    double h,sqrth,T;
    //Number of trayectories
    unsigned int N;
    //Error objective
    double eps;
    //Important estimated quantities
    double phi, phiphi, phi_sublinear, phiphi_sublinear, phi_VR, phiphi_VR, phi_sublinearVR, phiphi_sublinearVR,
    xi, xixi, xi_sublinear, xixi_sublinear, xiphi, xiphi_sublinear,  phi_num, phiphi_num, phi_sublinear_num, 
    phiphi_sublinear_num, phi_VR_num, phiphi_VR_num, phi_sublinearVR_num, phiphi_sublinearVR_num, xi_num, xixi_num, 
    xi_sublinear_num, xixi_sublinear_num, xiphi_num, xiphi_sublinear_num,APL,RNGC;
    //Initializer of the class
    EMFKACSolver(void){
        N = 0;
        for(int i = 0; i < 30; i++) sums[i] = 0;
        #pragma omp parallel
        {
            #pragma omp single
            {
                RNG.resize(omp_get_num_threads());
                normal_dist.resize(omp_get_num_threads());
                exp_dist.resize(omp_get_num_threads());
            }
            int id = omp_get_thread_num();
            RNG[id].discard(id*1E+10);
        }
    }
    EMFKACSolver(unsigned int MPIrank){
        N = 0;
        for(int i = 0; i < 30; i++) sums[i] = 0;
        #pragma omp parallel
        {
            #pragma omp single
            {
                RNG.resize(omp_get_num_threads());
                normal_dist.resize(omp_get_num_threads());
                exp_dist.resize(omp_get_num_threads());
            }
            int id = omp_get_thread_num();
            RNG[id].discard(MPIrank*1E+12 + id*1E+10);
        }
    };
    void Simulate_OMP(Eigen::Vector2d X0, unsigned int N_tray, double time_discretization,
                   double rho, bvp BoundaryValueProblem, double *boundary_parameters)
    {
        //for(int i = 0; i < 30; i++) sums[i] = 0;
        N += N_tray;
        h = time_discretization;
        T = INFINITY;
        sqrth = sqrt(h);
        X_tau_lin.resize(N_tray);
        Y_tau_lin.resize(N_tray);
        Z_tau_lin.resize(N_tray);
        tau_lin.resize(N_tray);
        xi_lin.resize(N_tray);
        X_tau_sublin.resize(N_tray);
        Y_tau_sublin.resize(N_tray);
        Z_tau_sublin.resize(N_tray);
        tau_sublin.resize(N_tray);
        xi_sublin.resize(N_tray);
        RNGcallsv.resize(N_tray);
        threads.resize(N_tray);
        //Part of the algorithm that is going to happen inside a GPU
        #pragma omp parallel
        {
            int id = omp_get_thread_num();
            Eigen::Vector2d X,normal,normal_proyection,increment;
            double Y,Z,xi,t,ji_t,dist,dist_k;
            bool stoppingbc, Neumannbc;
            unsigned int RNGCalls_thread = 0;

            #pragma omp for
            for(unsigned int n = 0; n < N_tray; n++){
                //std::cout << n  << " " << id << std::endl;
                X = X0;
                Y = 1;
                Z = 0;
                xi = 0;
                ji_t = 0;
                t = INFINITY;
                RNGCalls_thread = 0;
                dist = BoundaryValueProblem.distance(boundary_parameters,X,
                normal_proyection,normal);
                stoppingbc = BoundaryValueProblem.absorbing(normal_proyection);
                Neumannbc = BoundaryValueProblem.Neumann(normal_proyection);
                threads[n] = id;

                do{
                    if(stoppingbc == true){

                        Step(X,normal,Y,Z,xi,t,ji_t,h,sqrth,RNG[id],normal_dist[id],
                        increment,BoundaryValueProblem.sigma,BoundaryValueProblem.f,
                        BoundaryValueProblem.b,BoundaryValueProblem.c,BoundaryValueProblem.psi,
                        BoundaryValueProblem.varphi, BoundaryValueProblem.gradient);
                        RNGCalls_thread += 2;
                    }else{
                        if(Neumannbc == true){

                            Step(X,normal,normal_proyection,Y,Z,xi,t,ji_t,rho,h,increment,
                            BoundaryValueProblem.sigma,BoundaryValueProblem.f,
                            BoundaryValueProblem.b,BoundaryValueProblem.c,
                            BoundaryValueProblem.psi,BoundaryValueProblem.varphi,
                            BoundaryValueProblem.gradient, BoundaryValueProblem.distance,
                            boundary_parameters,dist_k,RNG[id],normal_dist[id],exp_dist[id],
                            sqrth,RNGCalls_thread);
                             
                        }else{

                            Step(X,normal,Y,Z,xi,t,ji_t,h,sqrth,RNG[id],normal_dist[id],
                            increment,BoundaryValueProblem.sigma,BoundaryValueProblem.f,
                            BoundaryValueProblem.b,BoundaryValueProblem.c,BoundaryValueProblem.psi,
                            BoundaryValueProblem.varphi, BoundaryValueProblem.gradient);
                            RNGCalls_thread += 2;
                        }
                    }
                    dist_k = dist;
                    stoppingbc = BoundaryValueProblem.absorbing(normal_proyection);
                    Neumannbc = BoundaryValueProblem.Neumann(normal_proyection);
                }while(Inside(dist,stoppingbc,Neumannbc,X,t,ji_t,T,
                sqrth,normal_proyection, normal,boundary_parameters,
                -0.5826,BoundaryValueProblem.distance,
                BoundaryValueProblem.absorbing,BoundaryValueProblem.Neumann,
                BoundaryValueProblem.sigma) != outcome::stop);
                X_tau_lin[n] = normal_proyection;
                Y_tau_lin[n] = Y;
                Z_tau_lin[n] = Z;
                tau_lin[n] = t;
                xi_lin[n] = xi;
                do{
                        Step(X,normal,Y,Z,xi,t,ji_t,h,sqrth,RNG[id],normal_dist[id],
                        increment,BoundaryValueProblem.sigma,BoundaryValueProblem.f,
                        BoundaryValueProblem.b,BoundaryValueProblem.c,BoundaryValueProblem.psi,
                        BoundaryValueProblem.varphi, BoundaryValueProblem.gradient);
                        RNGCalls_thread += 2;

                }while(Inside(dist,stoppingbc,Neumannbc,X,t,ji_t,T,
                sqrth,normal_proyection, normal,boundary_parameters,
                0.0,BoundaryValueProblem.distance,
                BoundaryValueProblem.absorbing,BoundaryValueProblem.Neumann,
                BoundaryValueProblem.sigma)!=outcome::stop);
                //End of sublinear step
                X_tau_sublin[n] = normal_proyection;
                Y_tau_sublin[n] = Y;
                Z_tau_sublin[n] = Z;
                tau_sublin[n] = t;
                xi_sublin[n] = xi;
                RNGcallsv[n] = RNGCalls_thread;
            }
            #pragma omp barrier
        }
    }
    void Simulate_OMP(Eigen::Vector2d X0, unsigned int N_tray, double time_discretization,
                   double rho, bvp BoundaryValueProblem, double *boundary_parameters, gsl_spline2d *LUT_u,
                   gsl_interp_accel *xacc_u, gsl_interp_accel *yacc_u, gsl_spline2d *LUT_v,
                   gsl_interp_accel *xacc_v, gsl_interp_accel *yacc_v)
    {
        //for(int i = 0; i < 30; i++) sums[i] = 0;
        N += N_tray;
        h = time_discretization;
        T = INFINITY;
        sqrth = sqrt(h);
        X_tau_lin.resize(N_tray);
        Y_tau_lin.resize(N_tray);
        Z_tau_lin.resize(N_tray);
        tau_lin.resize(N_tray);
        xi_lin.resize(N_tray);
        X_tau_sublin.resize(N_tray);
        Y_tau_sublin.resize(N_tray);
        Z_tau_sublin.resize(N_tray);
        tau_sublin.resize(N_tray);
        xi_sublin.resize(N_tray);
        RNGcallsv.resize(N_tray);
        threads.resize(N_tray);
        //Part of the algorithm that is going to happen inside a GPU
        #pragma omp parallel
        {
            int id = omp_get_thread_num();
            Eigen::Vector2d X,normal,normal_proyection,increment;
            double Y,Z,xi,t,ji_t,dist,dist_k;
            bool stoppingbc, Neumannbc;
            unsigned int RNGCalls_thread = 0;

            #pragma omp for
            for(unsigned int n = 0; n < N_tray; n++){
                //std::cout << n  << " " << id << std::endl;
                X = X0;
                Y = 1;
                Z = 0;
                xi = 0;
                ji_t = 0;
                t = INFINITY;
                RNGCalls_thread = 0;
                dist = BoundaryValueProblem.distance(boundary_parameters,X,
                normal_proyection,normal);
                stoppingbc = BoundaryValueProblem.absorbing(normal_proyection);
                Neumannbc = BoundaryValueProblem.Neumann(normal_proyection);
                threads[n] = id;

                do{
                    if(stoppingbc == true){

                        Step(X,normal,Y,Z,xi,t,ji_t,h,sqrth,RNG[id],normal_dist[id],
                        increment,BoundaryValueProblem.sigma,BoundaryValueProblem.num_f_2LUT,
                        BoundaryValueProblem.b,BoundaryValueProblem.num_c,BoundaryValueProblem.psi,
                        BoundaryValueProblem.varphi, BoundaryValueProblem.gradient,LUT_u,xacc_u,yacc_u,
                        LUT_v,xacc_v,yacc_v);
                        RNGCalls_thread += 2;
                    }else{
                        if(Neumannbc == true){

                            Step(X,normal,normal_proyection,Y,Z,xi,t,ji_t,rho,h,increment,
                            BoundaryValueProblem.sigma,BoundaryValueProblem.f,
                            BoundaryValueProblem.b,BoundaryValueProblem.c,
                            BoundaryValueProblem.psi,BoundaryValueProblem.varphi,
                            BoundaryValueProblem.gradient, BoundaryValueProblem.distance,
                            boundary_parameters,dist_k,RNG[id],normal_dist[id],exp_dist[id],
                            sqrth,RNGCalls_thread);
                             
                        }else{

                            Step(X,normal,Y,Z,xi,t,ji_t,h,sqrth,RNG[id],normal_dist[id],
                            increment,BoundaryValueProblem.sigma,BoundaryValueProblem.f,
                            BoundaryValueProblem.b,BoundaryValueProblem.c,BoundaryValueProblem.psi,
                            BoundaryValueProblem.varphi, BoundaryValueProblem.gradient);
                            RNGCalls_thread += 2;
                        }
                    }
                    dist_k = dist;
                    stoppingbc = BoundaryValueProblem.absorbing(normal_proyection);
                    Neumannbc = BoundaryValueProblem.Neumann(normal_proyection);
                }while(Inside(dist,stoppingbc,Neumannbc,X,t,ji_t,T,
                sqrth,normal_proyection, normal,boundary_parameters,
                -0.5826,BoundaryValueProblem.distance,
                BoundaryValueProblem.absorbing,BoundaryValueProblem.Neumann,
                BoundaryValueProblem.sigma) != outcome::stop);
                X_tau_lin[n] = normal_proyection;
                Y_tau_lin[n] = Y;
                Z_tau_lin[n] = Z;
                tau_lin[n] = t;
                xi_lin[n] = xi;
                do{
                        Step(X,normal,Y,Z,xi,t,ji_t,h,sqrth,RNG[id],normal_dist[id],
                        increment,BoundaryValueProblem.sigma,BoundaryValueProblem.f,
                        BoundaryValueProblem.b,BoundaryValueProblem.c,BoundaryValueProblem.psi,
                        BoundaryValueProblem.varphi, BoundaryValueProblem.gradient);
                        RNGCalls_thread += 2;

                }while(Inside(dist,stoppingbc,Neumannbc,X,t,ji_t,T,
                sqrth,normal_proyection, normal,boundary_parameters,
                0.0,BoundaryValueProblem.distance,
                BoundaryValueProblem.absorbing,BoundaryValueProblem.Neumann,
                BoundaryValueProblem.sigma)!=outcome::stop);
                //End of sublinear step
                X_tau_sublin[n] = normal_proyection;
                Y_tau_sublin[n] = Y;
                Z_tau_sublin[n] = Z;
                tau_sublin[n] = t;
                xi_sublin[n] = xi;
                RNGcallsv[n] = RNGCalls_thread;
            }
            #pragma omp barrier
        }
    }
        void Simulate_OMP_VR_Loop(Eigen::Vector2d X0, unsigned int N_tray, double time_discretization,
                   double rho, bvp BoundaryValueProblem, double *boundary_parameters, gsl_spline2d *LUT_u,
                   gsl_interp_accel *xacc_u, gsl_interp_accel *yacc_u)
    {
        //for(int i = 0; i < 30; i++) sums[i] = 0;
        N += N_tray;
        h = time_discretization;
        T = INFINITY;
        sqrth = sqrt(h);
        X_tau_lin.resize(N_tray);
        Y_tau_lin.resize(N_tray);
        Z_tau_lin.resize(N_tray);
        tau_lin.resize(N_tray);
        xi_lin.resize(N_tray);
        X_tau_sublin.resize(N_tray);
        Y_tau_sublin.resize(N_tray);
        Z_tau_sublin.resize(N_tray);
        tau_sublin.resize(N_tray);
        xi_sublin.resize(N_tray);
        RNGcallsv.resize(N_tray);
        threads.resize(N_tray);
        //Part of the algorithm that is going to happen inside a GPU
        #pragma omp parallel
        {
            int id = omp_get_thread_num();
            Eigen::Vector2d X,normal,normal_proyection,increment;
            double Y,Z,xi,t,ji_t,dist,dist_k;
            bool stoppingbc, Neumannbc;
            unsigned int RNGCalls_thread = 0;

            #pragma omp for
            for(unsigned int n = 0; n < N_tray; n++){
                //std::cout << n  << " " << id << std::endl;
                X = X0;
                Y = 1;
                Z = 0;
                xi = 0;
                ji_t = 0;
                t = INFINITY;
                RNGCalls_thread = 0;
                dist = BoundaryValueProblem.distance(boundary_parameters,X,
                normal_proyection,normal);
                stoppingbc = BoundaryValueProblem.absorbing(normal_proyection);
                Neumannbc = BoundaryValueProblem.Neumann(normal_proyection);
                threads[n] = id;

                do{
                    if(stoppingbc == true){
                        Step(X,normal,Y,Z,xi,t,ji_t,h,sqrth,RNG[id],normal_dist[id],
                        increment,BoundaryValueProblem.sigma,BoundaryValueProblem.num_f,
                        BoundaryValueProblem.b,BoundaryValueProblem.num_c,BoundaryValueProblem.psi,
                        BoundaryValueProblem.varphi, BoundaryValueProblem.num_gradient_LUT,LUT_u,xacc_u,yacc_u);
                        RNGCalls_thread += 2;
                    }else{
                        if(Neumannbc == true){

                            Step(X,normal,normal_proyection,Y,Z,xi,t,ji_t,rho,h,increment,
                            BoundaryValueProblem.sigma,BoundaryValueProblem.f,
                            BoundaryValueProblem.b,BoundaryValueProblem.c,
                            BoundaryValueProblem.psi,BoundaryValueProblem.varphi,
                            BoundaryValueProblem.gradient, BoundaryValueProblem.distance,
                            boundary_parameters,dist_k,RNG[id],normal_dist[id],exp_dist[id],
                            sqrth,RNGCalls_thread);
                             
                        }else{

                            Step(X,normal,Y,Z,xi,t,ji_t,h,sqrth,RNG[id],normal_dist[id],
                            increment,BoundaryValueProblem.sigma,BoundaryValueProblem.f,
                            BoundaryValueProblem.b,BoundaryValueProblem.c,BoundaryValueProblem.psi,
                            BoundaryValueProblem.varphi, BoundaryValueProblem.gradient);
                            RNGCalls_thread += 2;
                        }
                    }
                    dist_k = dist;
                    stoppingbc = BoundaryValueProblem.absorbing(normal_proyection);
                    Neumannbc = BoundaryValueProblem.Neumann(normal_proyection);
                }while(Inside(dist,stoppingbc,Neumannbc,X,t,ji_t,T,
                sqrth,normal_proyection, normal,boundary_parameters,
                -0.5826,BoundaryValueProblem.distance,
                BoundaryValueProblem.absorbing,BoundaryValueProblem.Neumann,
                BoundaryValueProblem.sigma) != outcome::stop);
                X_tau_lin[n] = normal_proyection;
                Y_tau_lin[n] = Y;
                Z_tau_lin[n] = Z;
                tau_lin[n] = t;
                xi_lin[n] = xi;
                do{
                        Step(X,normal,Y,Z,xi,t,ji_t,h,sqrth,RNG[id],normal_dist[id],
                        increment,BoundaryValueProblem.sigma,BoundaryValueProblem.f,
                        BoundaryValueProblem.b,BoundaryValueProblem.c,BoundaryValueProblem.psi,
                        BoundaryValueProblem.varphi, BoundaryValueProblem.gradient);
                        RNGCalls_thread += 2;

                }while(Inside(dist,stoppingbc,Neumannbc,X,t,ji_t,T,
                sqrth,normal_proyection, normal,boundary_parameters,
                0.0,BoundaryValueProblem.distance,
                BoundaryValueProblem.absorbing,BoundaryValueProblem.Neumann,
                BoundaryValueProblem.sigma)!=outcome::stop);
                //End of sublinear step
                X_tau_sublin[n] = normal_proyection;
                Y_tau_sublin[n] = Y;
                Z_tau_sublin[n] = Z;
                tau_sublin[n] = t;
                xi_sublin[n] = xi;
                RNGcallsv[n] = RNGCalls_thread;
            }
            #pragma omp barrier
        }
    }
    void Reduce_Analytic(bvp BoundaryValueProblem, unsigned int N_tray){
        #pragma omp parallel
            #pragma omp master
            {
                double score_linear_vr_thread,score_sublinear_vr_thread,
                score_linear_nvr_thread, score_sublinear_nvr_thread,
                score_linear_num_vr_thread,score_sublinear_num_vr_thread,
                score_linear_num_nvr_thread, score_sublinear_num_nvr_thread;
                for(unsigned int n = 0; n<N_tray; n++){
                    score_linear_nvr_thread = Z_tau_lin[n] + Y_tau_lin[n]*BoundaryValueProblem.g(X_tau_lin[n],tau_lin[n]);
                    score_sublinear_nvr_thread = Z_tau_sublin[n] + Y_tau_sublin[n]*BoundaryValueProblem.g(X_tau_sublin[n],tau_sublin[n]);
                    score_linear_vr_thread = score_linear_nvr_thread + xi_lin[n];
                    score_sublinear_vr_thread = score_sublinear_nvr_thread + xi_sublin[n];
                    sums[ScoreLinear] += score_linear_nvr_thread;
                    sums[ScoreSublinear] += score_sublinear_nvr_thread;
                    sums[ScoreLinear2] += score_linear_nvr_thread* score_linear_nvr_thread;
                    sums[ScoreSublinear2] += score_sublinear_nvr_thread*score_sublinear_nvr_thread;
                    sums[XiLinear] += xi_lin[n];
                    sums[XiSublinear] += xi_sublin[n];
                    sums[XiLinear2] += xi_lin[n]*xi_lin[n];
                    sums[XiSublinear2] += xi_sublin[n]*xi_sublin[n];
                    sums[ScoreLinearVR] += score_linear_nvr_thread + xi_lin[n];
                    sums[ScoreSublinearVR] += score_sublinear_nvr_thread + xi_sublin[n];
                    sums[ScoreLinearVR2] += pow(score_linear_nvr_thread + xi_lin[n],2);
                    sums[ScoreSublinearVR2] += pow(score_sublinear_nvr_thread + xi_sublin[n],2);
                    sums[XiScoreLinear] += xi_lin[n]*(score_linear_nvr_thread);
                    sums[XiScoreSublinear] += xi_sublin[n]*(score_sublinear_nvr_thread);
                    sums[RNGCalls] += RNGcallsv[n];
                    sums[tauLinear] += tau_lin[n];
                    sums[tauSublinear] += tau_sublin[n];
            }
        }
    }

    void Reduce_Analytic(bvp BoundaryValueProblem, unsigned int N_tray, Stencil & stencil_knot,
        double c2, std::vector<double>& G, std::vector<double>& G_var, std::vector<int> &G_j,
        double & B, double & BB){
        //#pragma omp parallel
        //{
            //#pragma omp master
            //{
                double score_linear_vr_thread,score_sublinear_vr_thread,
                score_linear_nvr_thread, score_sublinear_nvr_thread,
                score_linear_num_vr_thread,score_sublinear_num_vr_thread,
                score_linear_num_nvr_thread, score_sublinear_num_nvr_thread, 
                b = 0, bb = 0;
                double stencil_parameters[4] = {stencil_knot.stencil_parameters[0],stencil_knot.stencil_parameters[1],
                stencil_knot.stencil_parameters[2],stencil_knot.stencil_parameters[3]},
                boundary_parameters[4] = {stencil_knot.global_parameters[0],stencil_knot.global_parameters[1],
                stencil_knot.global_parameters[2],stencil_knot.global_parameters[3]};
                //std::cout << stencil_parameters[0] << " " << stencil_parameters[1] << " " <<
                //stencil_parameters[2] << " " << stencil_parameters[3] << "\n";
                Eigen::Vector2d aux_np, aux_n;
                stencil_knot.Reset();
                for(unsigned int n = 0; n<N_tray; n++){
                    //std::cout << n << std::endl;
                    score_linear_nvr_thread = Z_tau_lin[n] + Y_tau_lin[n]*BoundaryValueProblem.g(X_tau_lin[n],tau_lin[n]);
                    score_sublinear_nvr_thread = Z_tau_sublin[n] + Y_tau_sublin[n]*BoundaryValueProblem.g(X_tau_sublin[n],tau_sublin[n]);
                    score_linear_vr_thread = score_linear_nvr_thread + xi_lin[n];
                    score_sublinear_vr_thread = score_sublinear_nvr_thread + xi_sublin[n];
                    sums[ScoreLinear] += score_linear_nvr_thread;
                    sums[ScoreSublinear] += score_sublinear_nvr_thread;
                    sums[ScoreLinear2] += score_linear_nvr_thread* score_linear_nvr_thread;
                    sums[ScoreSublinear2] += score_sublinear_nvr_thread*score_sublinear_nvr_thread;
                    sums[XiLinear] += xi_lin[n];
                    sums[XiSublinear] += xi_sublin[n];
                    sums[XiLinear2] += xi_lin[n]*xi_lin[n];
                    sums[XiSublinear2] += xi_sublin[n]*xi_sublin[n];
                    sums[ScoreLinearVR] += score_linear_nvr_thread + xi_lin[n];
                    sums[ScoreSublinearVR] += score_sublinear_nvr_thread + xi_sublin[n];
                    sums[ScoreLinearVR2] += pow(score_linear_nvr_thread + xi_lin[n],2);
                    sums[ScoreSublinearVR2] += pow(score_sublinear_nvr_thread + xi_sublin[n],2);
                    sums[XiScoreLinear] += xi_lin[n]*(score_linear_nvr_thread);
                    sums[XiScoreSublinear] += xi_sublin[n]*(score_sublinear_nvr_thread);
                    sums[RNGCalls] += RNGcallsv[n];
                    sums[tauLinear] += tau_lin[n];
                    sums[tauSublinear] += tau_sublin[n];
                    if(BoundaryValueProblem.distance(boundary_parameters,X_tau_lin[n],aux_np,aux_n)>=-1E-08){
                        b +=score_linear_vr_thread;
                        bb +=score_linear_vr_thread*score_linear_vr_thread;
                    }else{
                        stencil_knot.G_update(X_tau_lin[n],Y_tau_lin[n],BoundaryValueProblem,c2);
                        b += Z_tau_lin[n] + xi_lin[n];
                        bb += pow(Z_tau_lin[n] + xi_lin[n],2);
                    }
                }
                
                stencil_knot.G_return_withrep(G_j, G, G_var,N_tray);
                B = b/N_tray;
                BB = bb/N_tray;
            //}
        //}
        
    }
        void Reduce_Numeric(bvp BoundaryValueProblem, unsigned int N_tray, Stencil & stencil_knot,
        double c2, std::vector<double>& G, std::vector<double>& G_var, std::vector<int> &G_j,
        double & B, double & BB, gsl_spline2d *LUT_u, gsl_interp_accel *xacc_u, gsl_interp_accel *yacc_u){
        //#pragma omp parallel
        //{
            //#pragma omp master
            //{
                double score_linear_vr_thread,score_sublinear_vr_thread,
                score_linear_nvr_thread, score_sublinear_nvr_thread,
                score_linear_num_vr_thread,score_sublinear_num_vr_thread,
                score_linear_num_nvr_thread, score_sublinear_num_nvr_thread, 
                b = 0, bb = 0;
                double stencil_parameters[4] = {stencil_knot.stencil_parameters[0],stencil_knot.stencil_parameters[1],
                stencil_knot.stencil_parameters[2],stencil_knot.stencil_parameters[3]},
                boundary_parameters[4] = {stencil_knot.global_parameters[0],stencil_knot.global_parameters[1],
                stencil_knot.global_parameters[2],stencil_knot.global_parameters[3]};
                //std::cout << boundary_parameters[0] << " " << boundary_parameters[1] << " " <<
                //boundary_parameters[2] << " " << boundary_parameters[3] << "\n";
                //std::cout << stencil_parameters[0] << " " << stencil_parameters[1] << " " <<
                //stencil_parameters[2] << " " << stencil_parameters[3] << "\n";
                Eigen::Vector2d aux_np, aux_n;
                stencil_knot.Reset();
                for(unsigned int n = 0; n<N_tray; n++){
                    //std::cout << n << std::endl;
                    score_linear_nvr_thread = Z_tau_lin[n] + Y_tau_lin[n]*BoundaryValueProblem.u(X_tau_lin[n],tau_lin[n]);
                    score_sublinear_nvr_thread = Z_tau_sublin[n] + Y_tau_sublin[n]*BoundaryValueProblem.u(X_tau_sublin[n],tau_sublin[n]);
                    score_linear_vr_thread = score_linear_nvr_thread + xi_lin[n];
                    score_sublinear_vr_thread = score_sublinear_nvr_thread + xi_sublin[n];
                    score_linear_num_nvr_thread = Z_tau_lin[n] + Y_tau_lin[n]*BoundaryValueProblem.u(X_tau_lin[n],tau_lin[n]);//*BoundaryValueProblem.num_u(X_tau_lin[n],tau_lin[n],LUT_u,xacc_u,yacc_u);
                    score_sublinear_num_nvr_thread = Z_tau_sublin[n] + Y_tau_sublin[n]*BoundaryValueProblem.u(X_tau_lin[n],tau_lin[n]);//*BoundaryValueProblem.num_u(X_tau_sublin[n],tau_sublin[n],LUT_u,xacc_u,yacc_u);
                    score_linear_num_vr_thread = score_linear_nvr_thread + xi_lin[n];
                    score_sublinear_num_vr_thread = score_sublinear_nvr_thread + xi_sublin[n];
                    sums[ScoreLinear] += score_linear_nvr_thread;
                    sums[ScoreSublinear] += score_sublinear_nvr_thread;
                    sums[ScoreLinear2] += score_linear_nvr_thread* score_linear_nvr_thread;
                    sums[ScoreSublinear2] += score_sublinear_nvr_thread*score_sublinear_nvr_thread;
                    sums[ScoreLinearNum] += score_linear_num_nvr_thread;
                    sums[ScoreSublinearNum] += score_sublinear_num_nvr_thread;
                    sums[ScoreLinearNum2] += score_linear_num_nvr_thread* score_linear_num_nvr_thread;
                    sums[ScoreSublinearNum2] += score_sublinear_num_nvr_thread*score_sublinear_num_nvr_thread;
                    sums[XiLinear] += xi_lin[n];
                    sums[XiSublinear] += xi_sublin[n];
                    sums[XiLinear2] += xi_lin[n]*xi_lin[n];
                    sums[XiSublinear2] += xi_sublin[n]*xi_sublin[n];
                    sums[XiLinearNum] += xi_lin[n];
                    sums[XiSublinearNum] += xi_sublin[n];
                    sums[XiLinearNum2] += xi_lin[n]*xi_lin[n];
                    sums[XiSublinearNum2] += xi_sublin[n]*xi_sublin[n];
                    sums[ScoreLinearVR] += score_linear_nvr_thread + xi_lin[n];
                    sums[ScoreSublinearVR] += score_sublinear_nvr_thread + xi_sublin[n];
                    sums[ScoreLinearVR2] += pow(score_linear_nvr_thread + xi_lin[n],2);
                    sums[ScoreSublinearVR2] += pow(score_sublinear_nvr_thread + xi_sublin[n],2);
                    sums[XiScoreLinear] += xi_lin[n]*(score_linear_nvr_thread);
                    sums[XiScoreSublinear] += xi_sublin[n]*(score_sublinear_nvr_thread);
                    sums[RNGCalls] += RNGcallsv[n];
                    sums[tauLinear] += tau_lin[n];
                    sums[tauSublinear] += tau_sublin[n];
                    if(BoundaryValueProblem.distance(boundary_parameters,X_tau_lin[n],aux_np,aux_n)>=-1E-08){
                        b +=score_linear_vr_thread;
                        bb +=score_linear_vr_thread*score_linear_vr_thread;
                    }else{
                        stencil_knot.G_update(X_tau_lin[n],Y_tau_lin[n],BoundaryValueProblem,c2);
                        b += Z_tau_lin[n] + xi_lin[n];
                        bb += pow(Z_tau_lin[n] + xi_lin[n],2);
                    }
                }
                //std::cout << "Xi " << sums[XiLinear]/N_tray<< "\n";
                stencil_knot.G_return_withrep(G_j, G, G_var,N_tray);
                B = b/N_tray;
                BB = bb/N_tray;
            //}
        //}
        
    }
    void Reset_sums(void){
        #pragma omp parallel for
        for(int i = 0; i < 30; i++) sums[i] = 0.0;
    }
    void Update(void){
        phi = sums[ScoreLinear]/N;
        phiphi = sums[ScoreLinear2]/N;
        phi_sublinear = sums[ScoreSublinear]/N;
        phiphi_sublinear = sums[ScoreSublinear2]/N;
        phi_VR = sums[ScoreLinearVR]/N;
        phiphi_VR = sums[ScoreLinearVR2]/N;
        xi = sums[XiLinear]/N;
        xixi = sums[XiLinear2]/N;
        xi_sublinear = sums[XiSublinear]/N;
        xixi_sublinear = sums[XiSublinear2]/N;
        xiphi = sums[XiScoreLinear]/N;
        xiphi_sublinear = sums[XiScoreSublinear]/N;
        phi_num = sums[ScoreLinearNum]/N;
        phiphi_num = sums[ScoreLinearNum2]/N;
        phi_sublinear_num = sums[ScoreSublinearNum]/N;
        phiphi_sublinear_num = sums[ScoreSublinearNum2]/N;
        phi_VR_num = sums[ScoreLinearVRNum]/N;
        phiphi_VR_num = sums[ScoreLinearVRNum2]/N;
        phi_sublinearVR_num = sums[ScoreSublinearVRNum]/N;
        phiphi_sublinearVR_num = sums[ScoreSublinearVRNum2]/N;
        xi_num = sums[XiLinearNum]/N;
        xixi_num = sums[XiLinearNum2]/N;
        xi_sublinear_num = sums[XiSublinearNum]/N;
        xixi_sublinear_num = sums[XiSublinearNum2]/N;
        xiphi_num = sums[XiScoreLinearNum]/N;
        xiphi_sublinear_num = sums[XiScoreSublinearNum]/N;
        RNGC = sums[RNGCalls];
        APL = sums[tauLinear]/N;
    }
    void Solve_OMP_Analytic(Eigen::Vector2d X0, unsigned int N_tray, double time_discretization,
                   double rho, bvp BoundaryValueProblem, double *boundary_parameters){
        Simulate_OMP(X0,N_tray,time_discretization,rho,BoundaryValueProblem,boundary_parameters);
        Reduce_Analytic(BoundaryValueProblem, N_tray);
        Update();
    }
    void Solve_OMP_Analytic(Eigen::Vector2d X0, unsigned int N_tray, double time_discretization,
                   double rho, bvp BoundaryValueProblem, Stencil & stencil_knot, double c2, 
                   std::vector<double>& G, std::vector<double>& G_var, std::vector<int> &G_j,
                   double & B, double & BB){
        double stencil_parameters[4] = {stencil_knot.stencil_parameters[0],stencil_knot.stencil_parameters[1],
               stencil_knot.stencil_parameters[2],stencil_knot.stencil_parameters[3]},
               boundary_parameters[4] = {stencil_knot.global_parameters[0],stencil_knot.global_parameters[1],
               stencil_knot.global_parameters[2],stencil_knot.global_parameters[3]};
        Eigen::Vector2d aux1,aux2;
        if(fabs(BoundaryValueProblem.distance(boundary_parameters,X0,aux1,aux2))>1E-08){
            Simulate_OMP(X0,N_tray,time_discretization,rho,BoundaryValueProblem,stencil_parameters);
            Reduce_Analytic(BoundaryValueProblem, N_tray, stencil_knot,c2,G,G_var,G_j,B, BB);
            Update(); 
        }else{
            B = BoundaryValueProblem.g(X0,0.0);
            BB = B*B;
        }
                
    }
    void Solve_OMP_Semilinear(Eigen::Vector2d X0, unsigned int N_tray, double time_discretization,
                   double rho, bvp BoundaryValueProblem, Stencil & stencil_knot, double c2, 
                   std::vector<double>& G, std::vector<double>& G_var, std::vector<int> &G_j,
                   double & B, double & BB, gsl_spline2d *LUT_u, gsl_interp_accel *xacc_u,
                   gsl_interp_accel *yacc_u, gsl_spline2d *LUT_v, gsl_interp_accel *xacc_v, gsl_interp_accel *yacc_v){
        double stencil_parameters[4] = {stencil_knot.stencil_parameters[0],stencil_knot.stencil_parameters[1],
               stencil_knot.stencil_parameters[2],stencil_knot.stencil_parameters[3]},
               boundary_parameters[4] = {stencil_knot.global_parameters[0],stencil_knot.global_parameters[1],
               stencil_knot.global_parameters[2],stencil_knot.global_parameters[3]};
        Eigen::Vector2d aux1,aux2;
        if(fabs(BoundaryValueProblem.distance(boundary_parameters,X0,aux1,aux2))>1E-08){
            Simulate_OMP(X0,N_tray,time_discretization,rho,BoundaryValueProblem,stencil_parameters
                        ,LUT_u, xacc_u, yacc_u, LUT_v, xacc_v, yacc_v);
            Reduce_Analytic(BoundaryValueProblem, N_tray, stencil_knot,c2,G,G_var,G_j,B, BB);
            Update(); 
        }else{
            B = BoundaryValueProblem.g(X0,0.0);
            BB = B*B;
        }
                
    }
    void Test_Solve_OMP_Analytic(Eigen::Vector2d X0, unsigned int eps, unsigned int N_tray, 
                   double time_discretization, double rho, bvp BoundaryValueProblem,
                   double *boundary_parameters){
        Reset_sums();
        N = 0;
        do{
            Solve_OMP_Analytic(X0,N_tray,time_discretization,rho,BoundaryValueProblem,boundary_parameters);
            //std::cout << sqrt((phiphi_VR-phi_VR*phi_VR)/N) << "\t" << eps*fabs(BoundaryValueProblem.u(X0,INFINITY)-phi_VR) << std::endl;
        }while(2*sqrt((phiphi_VR-phi_VR*phi_VR)/N)>eps*fabs(BoundaryValueProblem.u(X0,INFINITY)-phi_VR));
        printf("%e %e %e %e\n",time_discretization,phiphi_VR-phi_VR*phi_VR,2*sqrt((phiphi_VR-phi_VR*phi_VR)/N),
        BoundaryValueProblem.u(X0,INFINITY)-phi_VR);
    }
};
#endif