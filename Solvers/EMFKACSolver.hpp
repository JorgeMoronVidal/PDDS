#include "../BVPs/BVP.hpp"
#include "../BVPs/LUT.hpp"
#include "../Integrators/FKACIntegrator.hpp"
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
    xi, xixi, xi_sublinear, xixi_sublinear, phi_num, phiphi_num, phi_sublinear_num, phiphi_sublinear_num, phi_VR_num,
    phiphi_VR_num, phi_sublinearVR_num, phiphi_sublinearVR_num, xi_num, xixi_num, xi_sublinear_num, xixi_sublinear_num,
    APL,RNGC;
    //Initializer of the class
    EMFKACSolver(void){
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
    void Solve_OMP(Eigen::Vector2d X0, unsigned int N_tray, double time_discretization,
                   double rho, bvp BoundaryValueProblem, double *boundary_parameters)
    {
        //for(int i = 0; i < 30; i++) sums[i] = 0;
        N = N_tray;
        h = time_discretization;
        T = INFINITY;
        sqrth = sqrt(h);
        X_tau_lin.resize(N);
        Y_tau_lin.resize(N);
        Z_tau_lin.resize(N);
        tau_lin.resize(N);
        xi_lin.resize(N);
        X_tau_sublin.resize(N);
        Y_tau_sublin.resize(N);
        Z_tau_sublin.resize(N);
        tau_sublin.resize(N);
        xi_sublin.resize(N);
        RNGcallsv.resize(N);
        threads.resize(N);
        //Part of the algorithm that is going to happen inside a GPU
        #pragma omp parallel
        {
            int id = omp_get_thread_num();
            Eigen::Vector2d X,normal,normal_proyection,increment;
            double Y,Z,xi,t,ji_t,dist,dist_k;
            bool stoppingbc, Neumannbc;
            unsigned int RNGCalls_thread;

            #pragma omp for
            for(unsigned int n = 0; n < N; n++){

                X = X0;
                Y = 1;
                Z = 0;
                t = INFINITY;
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
                BoundaryValueProblem.sigma)!=outcome::stop);
                //End of linear step
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

        }
        //Ends part that is going to take place in a GPU
        #pragma omp parallel
        {   double score_linear_vr_thread,score_sublinear_vr_thread,
            score_linear_nvr_thread, score_sublinear_nvr_thread,
            score_linear_num_vr_thread,score_sublinear_num_vr_thread,
            score_linear_num_nvr_thread, score_sublinear_num_nvr_thread;
            #pragma omp for ordered
            for(unsigned int n = 0; n< N; n++){
                score_linear_nvr_thread = Z_tau_lin[n] + Y_tau_lin[n]*BoundaryValueProblem.g(X_tau_lin[n],tau_lin[n]);
                score_sublinear_nvr_thread = Z_tau_sublin[n] + Y_tau_sublin[n]*BoundaryValueProblem.g(X_tau_sublin[n],tau_lin[n]);
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
                #pragma omp ordered
                std::cout << BoundaryValueProblem.u(X0,0.0) << "\t" << Y_tau_sublin[n] << "\t" << score_linear_vr_thread << std::endl;
            }
        }
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
        phiphi_VR = sums[ScoreLinearVR]/N;
        xi = sums[XiLinear]/N;
        xixi = sums[XiLinear2]/N;
        xi_sublinear = sums[XiSublinear]/N;
        xixi_sublinear = sums[XiSublinear2]/N;
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
        RNGC = sums[RNGCalls];
        APL = sums[tauLinear]/N;
    }
};
#endif