# include "../BVPs/initBVPdev.cuh"
//#include "../BVPs/LUT.hpp"
//#include "../BVPs/EDPs/Monegros_Poisson.hpp"
#include "../Integrators/FKACIntegrator.cuh"
#include <omp.h>
#include <math.h>
#include <eigen3/Eigen/Core>
#include <vector>
#include <iostream>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>
#include <curand_mtgp32_host.h>
#include <curand_mtgp32dc_p_11213.h>
#define NUMBLOCKS 128
#define BLOCKSIZE 256

template <typename Typef,typename Typec>
__global__ void SolveLoop(double X01,double X02, double discr,double T, double rho,
                          double sqrth,double bp0,double bp1,double bp2,double bp3,curandStateMtgp32* state,
                          double* d_sums, bool* seAcabo, bool* continua,bool* cuenta,double*d_X0, double*d_X1, double*d_Y,
                          double*d_Z, double*d_xi, double*d_ji_t,int N, int N_tray,int Nx, int Ny,
                          Typef f, Typec c, pfscalar g, bool EulerMaruyama, bool VARC){
  int id=blockIdx.x * blockDim.x + threadIdx.x;
  int bid=blockIdx.x;
  int stride = blockDim.x * gridDim.x;
  double boundary_parameters[4]={bp0,bp1,bp2,bp3};
  double h=discr;
  Eigen::Vector2d X0;
  X0 << X01, X02;
  Eigen::Vector2d X,normal,normal_proyection,increment;
  double Y,Z,xi,t,ji_t,dist,dist_k;
  bool stoppingbc, Neumannbc;
  //Feynmann_Kac processes and final quantities
  Eigen::Vector2d X_tau_lin;
  Eigen::Vector2d X_tau_sublin= X_tau_lin ;
  double Y_tau_lin, Y_tau_sublin, Z_tau_lin, Z_tau_sublin,
  tau_lin, tau_sublin,xi_lin,xi_sublin;
  unsigned int RNGcallsv;
  //Random number generator
  unsigned int RNGCalls_thread = 0;
  int threads;
  bvpdev boundvalprob;
  boundvalprob=initBVPdev();
  double score_linear_vr_thread,score_sublinear_vr_thread,
  score_linear_nvr_thread, score_sublinear_nvr_thread,
  score_linear_num_vr_thread,score_sublinear_num_vr_thread,
  score_linear_num_nvr_thread, score_sublinear_num_nvr_thread;
  int seAcaboAntes=0;
  int seAcaboAntes2=0;
    X = X0;
    Y = 1;
    Z = 0;
    xi = 0;
    ji_t = 0;
    t = INFINITY;
    if(continua[id]==true){
      X<<d_X0[id],d_X1[id];
      Y=d_Y[id];
      Z=d_Z[id];
      xi=d_xi[id];
      ji_t=d_ji_t[id];
      if(seAcabo[id]==true){
        seAcaboAntes=1;
        h=0;
        if(seAcabo[id+NUMBLOCKS*BLOCKSIZE]==true){
          seAcaboAntes2=1;}
      }
    }
    if(continua[id]==false&&N>=N_tray){
      cuenta[id]=false;
    }
    double Gobet_Constant1=0.5826;
    double Gobet_Constant2=0.0;
    RNGCalls_thread = 0;
    dist = boundvalprob.distance(boundary_parameters,X,
    normal_proyection,normal);
    stoppingbc = boundvalprob.absorbing(normal_proyection);
    Neumannbc = boundvalprob.Neumann(normal_proyection);
    threads = id;
    Eigen::Matrix2d sigma_aux=boundvalprob.sigma(X,t);
    double Gobet_Condition=boundvalprob.distance(boundary_parameters,X,
        normal_proyection,normal)+Gobet_Constant1*(normal.transpose()*sigma_aux).norm()*sqrth;
    double Gobet_Condition2=boundvalprob.distance(boundary_parameters,X,
      normal_proyection,normal)+Gobet_Constant2*(normal.transpose()*sigma_aux).norm()*sqrth;
    int acabo=seAcabo[id];
    int acabo2=seAcabo[id+NUMBLOCKS*BLOCKSIZE];
    X_tau_lin = normal_proyection;
    Y_tau_lin = Y;
    Z_tau_lin = Z;
    tau_lin = t;
    xi_lin = xi;
    X_tau_sublin = normal_proyection;
    Y_tau_sublin = Y;
    Z_tau_sublin = Z;
    tau_sublin = t;
    xi_sublin = xi;
	     if(VARC==false){
         for(int algo=0;algo<1000;algo++){
              if(stoppingbc == true){
                  StepCUDA(X,normal,Y,Z,t,ji_t,h,sqrth,state,
                  increment,boundvalprob.sigma,f,
                  boundvalprob.b,c,boundvalprob.psi,
                  boundvalprob.varphi, boundary_parameters, Nx, Ny);
                  RNGCalls_thread += 2;
              }
              dist_k = dist;
              stoppingbc = boundvalprob.absorbing(normal_proyection);
              Neumannbc = boundvalprob.Neumann(normal_proyection);
             sigma_aux=boundvalprob.sigma(X,t);
              Gobet_Condition=boundvalprob.distance(boundary_parameters,X,
              normal_proyection,normal)+Gobet_Constant1*(normal.transpose()*sigma_aux).norm()*sqrth;
             if(acabo==0&&Gobet_Condition>0){
                 if(Neumannbc){
                   X-=dist_k*normal_proyection;
                 }else{
                 acabo=1;
                 h=0;
                 seAcabo[id]=true;
                 X_tau_lin = normal_proyection;
                 Y_tau_lin = Y;
                 Z_tau_lin = Z;
                 tau_lin = t;
                 xi_lin = xi;
                 score_linear_nvr_thread = Z_tau_lin + Y_tau_lin*boundvalprob.g(X_tau_lin,tau_lin);
                 }
               }
             }
         }else{
              for(int algo=0;algo<1000;algo++){
                    if(stoppingbc == true){
                        StepCUDA(X,normal,Y,Z,xi,t,ji_t,h,sqrth,state,
                          increment,boundvalprob.sigma,f,
                          boundvalprob.b,c,boundvalprob.psi,
                          boundvalprob.varphi, boundvalprob.gradient, boundary_parameters, Nx, Ny);
                          RNGCalls_thread += 2;
                        }else{
                          if(Neumannbc == true){
                            StepCUDA(X,normal,normal_proyection,Y,Z,xi,t,ji_t,rho,h,increment,
                              boundvalprob.sigma,f,
                              boundvalprob.b,c,
                              boundvalprob.psi,boundvalprob.varphi,
                              boundvalprob.gradient, boundvalprob.distance,
                              boundary_parameters,dist_k,state,
                              sqrth,RNGCalls_thread, Nx, Ny);
                            }else{
                              StepCUDA(X,normal,Y,Z,xi,t,ji_t,h,sqrth,state,
                                increment,boundvalprob.sigma,f,
                                boundvalprob.b,c,boundvalprob.psi,
                                boundvalprob.varphi, boundvalprob.gradient, boundary_parameters, Nx, Ny);
                                RNGCalls_thread += 2;
                              }
                            }
                            dist_k = dist;
                            stoppingbc = boundvalprob.absorbing(normal_proyection);
                            Neumannbc = boundvalprob.Neumann(normal_proyection);
                            sigma_aux=boundvalprob.sigma(X,t);
                            Gobet_Condition=boundvalprob.distance(boundary_parameters,X,
                              normal_proyection,normal)+Gobet_Constant1*(normal.transpose()*sigma_aux).norm()*sqrth;
                              if(acabo==0&&Gobet_Condition>0){
                                acabo=1;
                                h=0;
                                seAcabo[id]=true;
                                X_tau_lin = normal_proyection;
                                Y_tau_lin = Y;
                                Z_tau_lin = Z;
                                tau_lin = t;
                                xi_lin = xi;
                                score_linear_nvr_thread = Z_tau_lin + Y_tau_lin*boundvalprob.g(X_tau_lin,tau_lin);
                              }
                            }
       }
    d_X0[id]=X(0);
    d_X1[id]=X(1);
    d_Y[id]=Y;
    d_Z[id]=Z;
    d_xi[id]=xi;
    d_ji_t[id]=ji_t;
    if(EulerMaruyama){
      h=discr;
       if(seAcaboAntes==1){
         X<<d_X0[id+NUMBLOCKS*BLOCKSIZE],d_X1[id+NUMBLOCKS*BLOCKSIZE];
         Y=d_Y[id+NUMBLOCKS*BLOCKSIZE];
         Z=d_Z[id+NUMBLOCKS*BLOCKSIZE];
         xi=d_xi[id+NUMBLOCKS*BLOCKSIZE];
         ji_t=d_ji_t[id+NUMBLOCKS*BLOCKSIZE];
       }
       for(int algo=0;algo<10000;algo++){
   //        do{
                   StepCUDA(X,normal,Y,Z,xi,t,ji_t,h,sqrth,state,
                   increment,boundvalprob.sigma,f,
                   boundvalprob.b,c,boundvalprob.psi,
                   boundvalprob.varphi, boundvalprob.gradient, boundary_parameters, Nx, Ny);
                   RNGCalls_thread += 2;
                   sigma_aux=boundvalprob.sigma(X,t);
                   Gobet_Condition2=boundvalprob.distance(boundary_parameters,X,
                     normal_proyection,normal)+Gobet_Constant2*(normal.transpose()*sigma_aux).norm()*sqrth;
                   //End of sublinear step
                   if(acabo2==0&&Gobet_Condition2>0){
                     acabo2=1;
                     seAcabo[id+BLOCKSIZE*NUMBLOCKS]=true;
                     X_tau_sublin = normal_proyection;
                     Y_tau_sublin = Y;
                     Z_tau_sublin = Z;
                     tau_sublin = t;
                     xi_sublin = xi;
                   }
                 }
           d_X0[id+NUMBLOCKS*BLOCKSIZE]=X(0);
           d_X1[id+NUMBLOCKS*BLOCKSIZE]=X(1);
           d_Y[id+NUMBLOCKS*BLOCKSIZE]=Y;
           d_Z[id+NUMBLOCKS*BLOCKSIZE]=Z;
           d_xi[id+NUMBLOCKS*BLOCKSIZE]=xi;
           d_ji_t[id+NUMBLOCKS*BLOCKSIZE]=ji_t;
}
	      int seSuma=cuenta[id]*acabo*(1-seAcaboAntes);
        int seSuma2=cuenta[id]*acabo*acabo2*(1-seAcaboAntes2);
        RNGcallsv = RNGCalls_thread;
        score_linear_nvr_thread = Z_tau_lin + Y_tau_lin*boundvalprob.g(X_tau_lin,tau_lin);
        score_sublinear_nvr_thread = Z_tau_sublin + Y_tau_sublin*boundvalprob.g(X_tau_sublin,tau_sublin);
        score_linear_vr_thread = score_linear_nvr_thread + xi_lin;
        score_sublinear_vr_thread = score_sublinear_nvr_thread + xi_sublin;
        d_sums[23*id+0] += seSuma*score_linear_nvr_thread;
        d_sums[23*id+1] += seSuma2*score_sublinear_nvr_thread;
        d_sums[23*id+2] += seSuma*score_linear_nvr_thread* score_linear_nvr_thread;
        d_sums[23*id+3] += seSuma2*score_sublinear_nvr_thread*score_sublinear_nvr_thread;
        d_sums[23*id+4] += seSuma*xi_lin;
        d_sums[23*id+5] += seSuma2*xi_sublin;
        d_sums[23*id+6] += seSuma*xi_lin*xi_lin;
        d_sums[23*id+7] += seSuma2*xi_sublin*xi_sublin;
        d_sums[23*id+8] += seSuma*(score_linear_nvr_thread + xi_lin);
        d_sums[23*id+9] += seSuma2*(score_sublinear_nvr_thread + xi_sublin);
        d_sums[23*id+10]+= seSuma*pow(score_linear_nvr_thread + xi_lin,2);
        d_sums[23*id+11] += seSuma2*pow(score_sublinear_nvr_thread + xi_sublin,2);
        d_sums[23*id+12] += seSuma*xi_lin*(score_linear_nvr_thread);
        d_sums[23*id+13] += seSuma2*xi_sublin*(score_sublinear_nvr_thread);
        d_sums[23*id+14] += RNGcallsv;
        d_sums[23*id+15] += (1-acabo)*tau_lin;
        d_sums[23*id+16] += (1-acabo*acabo2)*tau_sublin;
        d_sums[23*id+17] += seSuma*score_linear_nvr_thread* score_linear_nvr_thread*score_linear_nvr_thread;
        d_sums[23*id+18] += seSuma*score_linear_nvr_thread* score_linear_nvr_thread*score_linear_nvr_thread*score_linear_nvr_thread;
        d_sums[23*id+19] += (1-acabo)*tau_lin*tau_lin;
        d_sums[23*id+20] += seSuma2*score_sublinear_nvr_thread*score_sublinear_nvr_thread*score_sublinear_nvr_thread;
        d_sums[23*id+21] += seSuma2*score_sublinear_nvr_thread*score_sublinear_nvr_thread*score_sublinear_nvr_thread*score_sublinear_nvr_thread;
        d_sums[23*id+22] += (1-acabo*acabo2)*tau_sublin*tau_sublin;
        if(EulerMaruyama){
          if((seSuma2==1)){
            continua[id]=false;
          //ALTERNATIVA DE CONTAR
            seAcabo[id]=false;
            seAcabo[id+NUMBLOCKS*BLOCKSIZE]=false;
           }else{continua[id]=true;}
        }else{ if((seSuma==1)){
          continua[id]=false;
          //ALTERNATIVA DE CONTAR
          seAcabo[id]=false;
          seAcabo[id+NUMBLOCKS*BLOCKSIZE]=false;
        }else{continua[id]=true;}}
    }

    template <typename Typef,typename Typec, typename Typeu>
    __global__ void SolveLoop(double X01,double X02, double discr,double T, double rho,
                              double sqrth,double bp0,double bp1,double bp2,double bp3,curandStateMtgp32* state,
                              double* d_sums, bool* seAcabo, bool* continua,bool* cuenta,double*d_X0, double*d_X1, double*d_Y,
                              double*d_Z, double*d_xi, double*d_ji_t,int N, int N_tray,int Nx, int Ny,
                              Typef f, Typec c, pfscalar g, bool EulerMaruyama, bool VARC, Typeu ux, Typeu uy){
      int id=blockIdx.x * blockDim.x + threadIdx.x;
      int bid=blockIdx.x;
      int stride = blockDim.x * gridDim.x;
      double boundary_parameters[4]={bp0,bp1,bp2,bp3};
      double h=discr;
      Eigen::Vector2d X0;
      X0 << X01, X02;
      Eigen::Vector2d X,normal,normal_proyection,increment;
      double Y,Z,xi,t,ji_t,dist,dist_k;
      bool stoppingbc, Neumannbc;
      //Feynmann_Kac processes and final quantities
      Eigen::Vector2d X_tau_lin;
      Eigen::Vector2d X_tau_sublin= X_tau_lin ;
      double Y_tau_lin, Y_tau_sublin, Z_tau_lin, Z_tau_sublin,
      tau_lin, tau_sublin,xi_lin,xi_sublin;
      unsigned int RNGcallsv;
      //Random number generator
      unsigned int RNGCalls_thread = 0;
      int threads;
      bvpdev boundvalprob;
      boundvalprob=initBVPdev();
      double score_linear_vr_thread,score_sublinear_vr_thread,
      score_linear_nvr_thread, score_sublinear_nvr_thread,
      score_linear_num_vr_thread,score_sublinear_num_vr_thread,
      score_linear_num_nvr_thread, score_sublinear_num_nvr_thread;
      int seAcaboAntes=0;
      int seAcaboAntes2=0;
        X = X0;
        Y = 1;
        Z = 0;
        xi = 0;
        ji_t = 0;
        t = INFINITY;
        if(continua[id]==true){
          X<<d_X0[id],d_X1[id];
          Y=d_Y[id];
          Z=d_Z[id];
          xi=d_xi[id];
          ji_t=d_ji_t[id];
          if(seAcabo[id]==true){
            h=0;
            seAcaboAntes=1;
            if(seAcabo[id+NUMBLOCKS*BLOCKSIZE]==true){
              seAcaboAntes2=1;}
          }
        }
        if(continua[id]==false&&N>=N_tray){
          cuenta[id]=false;
        }
        double Gobet_Constant1=0.5826;
        double Gobet_Constant2=0.0;
        RNGCalls_thread = 0;
        dist = boundvalprob.distance(boundary_parameters,X,
        normal_proyection,normal);
        stoppingbc = boundvalprob.absorbing(normal_proyection);
        Neumannbc = boundvalprob.Neumann(normal_proyection);
        threads = id;
        Eigen::Matrix2d sigma_aux=boundvalprob.sigma(X,t);
        double Gobet_Condition=boundvalprob.distance(boundary_parameters,X,
            normal_proyection,normal)+Gobet_Constant1*(normal.transpose()*sigma_aux).norm()*sqrth;
        double Gobet_Condition2=boundvalprob.distance(boundary_parameters,X,
          normal_proyection,normal)+Gobet_Constant2*(normal.transpose()*sigma_aux).norm()*sqrth;
        int acabo=seAcabo[id];
        int acabo2=seAcabo[id+NUMBLOCKS*BLOCKSIZE];
        X_tau_lin = normal_proyection;
        Y_tau_lin = Y;
        Z_tau_lin = Z;
        tau_lin = t;
        xi_lin = xi;
        X_tau_sublin = normal_proyection;
        Y_tau_sublin = Y;
        Z_tau_sublin = Z;
        tau_sublin = t;
        xi_sublin = xi;
    	     for(int algo=0;algo<1000;algo++){
                if(stoppingbc == true){//
                    StepCUDA(X,normal,Y,Z,xi,t,ji_t,h,sqrth,state,
                    increment,boundvalprob.sigma,f,
                    boundvalprob.b,c,boundvalprob.psi,
                    boundvalprob.varphi, ux,uy, boundary_parameters, Nx, Ny);
                    RNGCalls_thread += 2;
                }else{
                    if(Neumannbc == true){//
                        StepCUDA(X,normal,normal_proyection,Y,Z,xi,t,ji_t,rho,h,increment,
                        boundvalprob.sigma,f,
                        boundvalprob.b,c,
                        boundvalprob.psi,boundvalprob.varphi,
                        ux,uy, boundvalprob.distance,
                        boundary_parameters,dist_k,state,
                        sqrth,RNGCalls_thread, Nx, Ny);//
                    }else{//
                        StepCUDA(X,normal,Y,Z,xi,t,ji_t,h,sqrth,state,
                        increment,boundvalprob.sigma,f,
                        boundvalprob.b,c,boundvalprob.psi,
                        boundvalprob.varphi, ux,uy, boundary_parameters, Nx, Ny);
                        RNGCalls_thread += 2;
                    }
                }
                dist_k = dist;
                stoppingbc = boundvalprob.absorbing(normal_proyection);
                Neumannbc = boundvalprob.Neumann(normal_proyection);
    	          sigma_aux=boundvalprob.sigma(X,t);
                Gobet_Condition=boundvalprob.distance(boundary_parameters,X,
                normal_proyection,normal)+Gobet_Constant1*(normal.transpose()*sigma_aux).norm()*sqrth;
    	          if(acabo==0&&Gobet_Condition>0){
                   acabo=1;
                   h=0;
                   seAcabo[id]=true;
                   X_tau_lin = normal_proyection;
                   Y_tau_lin = Y;
                   Z_tau_lin = Z;
                   tau_lin = t;
                   xi_lin = xi;
                   score_linear_nvr_thread = Z_tau_lin + Y_tau_lin*boundvalprob.g(X_tau_lin,tau_lin);
                 }
               }

        d_X0[id]=X(0);
        d_X1[id]=X(1);
        d_Y[id]=Y;
        d_Z[id]=Z;
        d_xi[id]=xi;
        d_ji_t[id]=ji_t;
        if(EulerMaruyama){
          h=discr;
           if(seAcaboAntes==1){
             X<<d_X0[id+NUMBLOCKS*BLOCKSIZE],d_X1[id+NUMBLOCKS*BLOCKSIZE];
             Y=d_Y[id+NUMBLOCKS*BLOCKSIZE];
             Z=d_Z[id+NUMBLOCKS*BLOCKSIZE];
             xi=d_xi[id+NUMBLOCKS*BLOCKSIZE];
             ji_t=d_ji_t[id+NUMBLOCKS*BLOCKSIZE];
           }
           for(int algo=0;algo<10000;algo++){
       //        do{
                       StepCUDA(X,normal,Y,Z,xi,t,ji_t,h,sqrth,state,
                       increment,boundvalprob.sigma,f,
                       boundvalprob.b,c,boundvalprob.psi,
                       boundvalprob.varphi, ux,uy, boundary_parameters, Nx, Ny);
                       RNGCalls_thread += 2;
                       sigma_aux=boundvalprob.sigma(X,t);
                       Gobet_Condition2=boundvalprob.distance(boundary_parameters,X,
                         normal_proyection,normal)+Gobet_Constant2*(normal.transpose()*sigma_aux).norm()*sqrth;
                       //End of sublinear step
                       if(acabo2==0&&Gobet_Condition2>0){
                         acabo2=1;
                         seAcabo[id+BLOCKSIZE*NUMBLOCKS]=true;
                         X_tau_sublin = normal_proyection;
                         Y_tau_sublin = Y;
                         Z_tau_sublin = Z;
                         tau_sublin = t;
                         xi_sublin = xi;
                       }
                     }
               d_X0[id+NUMBLOCKS*BLOCKSIZE]=X(0);
               d_X1[id+NUMBLOCKS*BLOCKSIZE]=X(1);
               d_Y[id+NUMBLOCKS*BLOCKSIZE]=Y;
               d_Z[id+NUMBLOCKS*BLOCKSIZE]=Z;
               d_xi[id+NUMBLOCKS*BLOCKSIZE]=xi;
               d_ji_t[id+NUMBLOCKS*BLOCKSIZE]=ji_t;
    }
    	      int seSuma=cuenta[id]*acabo*(1-seAcaboAntes);
            int seSuma2=cuenta[id]*acabo*acabo2*(1-seAcaboAntes2);
            RNGcallsv = RNGCalls_thread;
            score_linear_nvr_thread = Z_tau_lin + Y_tau_lin*boundvalprob.g(X_tau_lin,tau_lin);
            score_sublinear_nvr_thread = Z_tau_sublin + Y_tau_sublin*boundvalprob.g(X_tau_sublin,tau_sublin);
            score_linear_vr_thread = score_linear_nvr_thread + xi_lin;
            score_sublinear_vr_thread = score_sublinear_nvr_thread + xi_sublin;
            d_sums[23*id+0] += seSuma*score_linear_nvr_thread;
            d_sums[23*id+1] += seSuma2*score_sublinear_nvr_thread;
            d_sums[23*id+2] += seSuma*score_linear_nvr_thread* score_linear_nvr_thread;
            d_sums[23*id+3] += seSuma2*score_sublinear_nvr_thread*score_sublinear_nvr_thread;
            d_sums[23*id+4] += seSuma*xi_lin;
            d_sums[23*id+5] += seSuma2*xi_sublin;
            d_sums[23*id+6] += seSuma*xi_lin*xi_lin;
            d_sums[23*id+7] += seSuma2*xi_sublin*xi_sublin;
            d_sums[23*id+8] += seSuma*(score_linear_nvr_thread + xi_lin);
            d_sums[23*id+9] += seSuma2*(score_sublinear_nvr_thread + xi_sublin);
            d_sums[23*id+10]+= seSuma*pow(score_linear_nvr_thread + xi_lin,2);
            d_sums[23*id+11] += seSuma2*pow(score_sublinear_nvr_thread + xi_sublin,2);
            d_sums[23*id+12] += seSuma*xi_lin*(score_linear_nvr_thread);
            d_sums[23*id+13] += seSuma2*xi_sublin*(score_sublinear_nvr_thread);
            d_sums[23*id+14] += RNGcallsv;
            d_sums[23*id+15] += (1-acabo)*tau_lin;
            d_sums[23*id+16] += (1-acabo*acabo2)*tau_sublin;
            d_sums[23*id+17] += seSuma*score_linear_nvr_thread* score_linear_nvr_thread*score_linear_nvr_thread;
            d_sums[23*id+18] += seSuma*score_linear_nvr_thread* score_linear_nvr_thread*score_linear_nvr_thread*score_linear_nvr_thread;
            d_sums[23*id+19] += (1-acabo)*tau_lin*tau_lin;
            d_sums[23*id+20] += seSuma2*score_sublinear_nvr_thread*score_sublinear_nvr_thread*score_sublinear_nvr_thread;
            d_sums[23*id+21] += seSuma2*score_sublinear_nvr_thread*score_sublinear_nvr_thread*score_sublinear_nvr_thread*score_sublinear_nvr_thread;
            d_sums[23*id+22] += (1-acabo*acabo2)*tau_sublin*tau_sublin;
            if(EulerMaruyama){
              if((seSuma2==1)){
                continua[id]=false;
              //ALTERNATIVA DE CONTAR
                seAcabo[id]=false;
                seAcabo[id+NUMBLOCKS*BLOCKSIZE]=false;
               }else{continua[id]=true;}
            }else{ if((seSuma==1)){
              continua[id]=false;
              //ALTERNATIVA DE CONTAR
              seAcabo[id]=false;
              seAcabo[id+NUMBLOCKS*BLOCKSIZE]=false;
            }else{continua[id]=true;}}
        }//
        template <typename Typef,typename Typec>
        __global__ void SolveLoop(double X01,double X02, double discr,double T, double rho,
                                  double sqrth,double bp0,double bp1,double bp2,double bp3,curandStateMtgp32* state,
                                  double* d_sums, bool* seAcabo, bool* continua,bool* cuenta,double*d_X0, double*d_X1, double*d_Y,
                                  double*d_Z, double*d_xi, double*d_ji_t,int N, int N_tray,int Nx, int Ny,
                                  Typef f, Typec c, bool EulerMaruyama, bool VARC ){
          int id=blockIdx.x * blockDim.x + threadIdx.x;
          int bid=blockIdx.x;
          int stride = blockDim.x * gridDim.x;
          double boundary_parameters[4]={bp0,bp1,bp2,bp3};
          double h=discr;
          Eigen::Vector2d X0;
          X0 << X01, X02;
          Eigen::Vector2d X,normal,normal_proyection,increment;
          double Y,Z,xi,t,ji_t,dist,dist_k;
          bool stoppingbc, Neumannbc;
          //Feynmann_Kac processes and final quantities
          Eigen::Vector2d X_tau_lin;
          Eigen::Vector2d X_tau_sublin= X_tau_lin ;
          double Y_tau_lin, Y_tau_sublin, Z_tau_lin, Z_tau_sublin,
          tau_lin, tau_sublin,xi_lin,xi_sublin;
          unsigned int RNGcallsv;
          //Random number generator
          unsigned int RNGCalls_thread = 0;
          int threads;
          bvpdev boundvalprob;
          boundvalprob=initBVPdev();
          int seAcaboAntes=0;
          int seAcaboAntes2=0;
            X = X0;
            Y = 1;
            Z = 0;
            xi = 0;
            ji_t = 0;
            t = INFINITY;
            if(continua[id]==true){
              X<<d_X0[id],d_X1[id];
              Y=d_Y[id];
              Z=d_Z[id];
              xi=d_xi[id];
              ji_t=d_ji_t[id];
              if(seAcabo[id]==true){
                h=0;
                seAcaboAntes=1;
                if(seAcabo[id+NUMBLOCKS*BLOCKSIZE]==true){
                  seAcaboAntes2=1;}
              }
            }
            if(continua[id]==false&&N>=N_tray){
              cuenta[id]=false;
            }
            double Gobet_Constant1=0.5826;
            double Gobet_Constant2=0.0;
            RNGCalls_thread = 0;
            dist = boundvalprob.distance(boundary_parameters,X,
            normal_proyection,normal);
            stoppingbc = boundvalprob.absorbing(normal_proyection);
            Neumannbc = boundvalprob.Neumann(normal_proyection);
            threads = id;
            Eigen::Matrix2d sigma_aux=boundvalprob.sigma(X,t);
            double Gobet_Condition=boundvalprob.distance(boundary_parameters,X,
                normal_proyection,normal)+Gobet_Constant1*(normal.transpose()*sigma_aux).norm()*sqrth;
            double Gobet_Condition2=boundvalprob.distance(boundary_parameters,X,
              normal_proyection,normal)+Gobet_Constant2*(normal.transpose()*sigma_aux).norm()*sqrth;
            int acabo=seAcabo[id];
            int acabo2=seAcabo[id+NUMBLOCKS*BLOCKSIZE];
            X_tau_lin = normal_proyection;
            Y_tau_lin = Y;
            Z_tau_lin = Z;
            tau_lin = t;
            xi_lin = xi;
            X_tau_sublin = normal_proyection;
            Y_tau_sublin = Y;
            Z_tau_sublin = Z;
            tau_sublin = t;
            xi_sublin = xi;
            if(VARC==false){

              for(int algo=0;algo<1000;algo++){
                   if(stoppingbc == true){//
                       StepCUDA(X,normal,Y,Z,t,ji_t,h,sqrth,state,
                       increment,boundvalprob.sigma,f,
                       boundvalprob.b,c,boundvalprob.psi,
                       boundvalprob.varphi, boundary_parameters, Nx, Ny);
                       RNGCalls_thread += 2;
                   }
                   dist_k = dist;
                   stoppingbc = boundvalprob.absorbing(normal_proyection);
                   Neumannbc = boundvalprob.Neumann(normal_proyection);
                  sigma_aux=boundvalprob.sigma(X,t);
                   Gobet_Condition=boundvalprob.distance(boundary_parameters,X,
                   normal_proyection,normal)+Gobet_Constant1*(normal.transpose()*sigma_aux).norm()*sqrth;
                  if(acabo==0&&Gobet_Condition>0){
                      if(Neumannbc){
                        X-=dist_k*normal_proyection;
                      }else{
                      acabo=1;
                      h=0;
                      seAcabo[id]=true;
                      X_tau_lin = normal_proyection;
                      Y_tau_lin = Y;
                      Z_tau_lin = Z;
                      tau_lin = t;
                      xi_lin = xi;
                      //score_linear_nvr_thread = Z_tau_lin + Y_tau_lin*boundvalprob.g(X_tau_lin,tau_lin);
                      }
                    }
                  }
              }else{
        	     for(int algo=0;algo<1000;algo++){
                    if(stoppingbc == true){//
                        StepCUDA(X,normal,Y,Z,xi,t,ji_t,h,sqrth,state,
                        increment,boundvalprob.sigma,f,
                        boundvalprob.b,c,boundvalprob.psi,
                        boundvalprob.varphi, boundvalprob.gradient, boundary_parameters, Nx, Ny);
                        RNGCalls_thread += 2;
                    }else{
                        if(Neumannbc == true){//
                            StepCUDA(X,normal,normal_proyection,Y,Z,xi,t,ji_t,rho,h,increment,
                            boundvalprob.sigma,f,
                            boundvalprob.b,c,
                            boundvalprob.psi,boundvalprob.varphi,
                            boundvalprob.gradient, boundvalprob.distance,
                            boundary_parameters,dist_k,state,
                            sqrth,RNGCalls_thread, Nx, Ny);//
                        }else{//
                            StepCUDA(X,normal,Y,Z,xi,t,ji_t,h,sqrth,state,
                            increment,boundvalprob.sigma,f,
                            boundvalprob.b,c,boundvalprob.psi,
                            boundvalprob.varphi, boundvalprob.gradient, boundary_parameters, Nx, Ny);
                            RNGCalls_thread += 2;
                        }
                    }
                    dist_k = dist;
                    stoppingbc = boundvalprob.absorbing(normal_proyection);
                    Neumannbc = boundvalprob.Neumann(normal_proyection);
        	          sigma_aux=boundvalprob.sigma(X,t);
                    Gobet_Condition=boundvalprob.distance(boundary_parameters,X,
                    normal_proyection,normal)+Gobet_Constant1*(normal.transpose()*sigma_aux).norm()*sqrth;
        	          if(acabo==0&&Gobet_Condition>0){
                       h=0;
                       acabo=1;
                       seAcabo[id]=true;
                       X_tau_lin = normal_proyection;
                       Y_tau_lin = Y;
                       Z_tau_lin = Z;
                       tau_lin = t;
                       xi_lin = xi;
                     }
                   }
                 }
            d_X0[id]=X(0);
            d_X1[id]=X(1);
            d_Y[id]=Y;
            d_Z[id]=Z;
            d_xi[id]=xi;
            d_ji_t[id]=ji_t;
            if(EulerMaruyama){
              h=discr;
               if(seAcaboAntes==1){
                 X<<d_X0[id+NUMBLOCKS*BLOCKSIZE],d_X1[id+NUMBLOCKS*BLOCKSIZE];
                 Y=d_Y[id+NUMBLOCKS*BLOCKSIZE];
                 Z=d_Z[id+NUMBLOCKS*BLOCKSIZE];
                 xi=d_xi[id+NUMBLOCKS*BLOCKSIZE];
                 ji_t=d_ji_t[id+NUMBLOCKS*BLOCKSIZE];
               }
               for(int algo=0;algo<10000;algo++){
           //        do{
                           StepCUDA(X,normal,Y,Z,xi,t,ji_t,h,sqrth,state,
                           increment,boundvalprob.sigma,f,
                           boundvalprob.b,c,boundvalprob.psi,
                           boundvalprob.varphi, boundvalprob.gradient, boundary_parameters, Nx, Ny);
                           RNGCalls_thread += 2;
                           sigma_aux=boundvalprob.sigma(X,t);
                           Gobet_Condition2=boundvalprob.distance(boundary_parameters,X,
                             normal_proyection,normal)+Gobet_Constant2*(normal.transpose()*sigma_aux).norm()*sqrth;
                           //End of sublinear step
                           if(acabo2==0&&Gobet_Condition2>0){
                             acabo2=1;
                             seAcabo[id+BLOCKSIZE*NUMBLOCKS]=true;
                             X_tau_sublin = normal_proyection;
                             Y_tau_sublin = Y;
                             Z_tau_sublin = Z;
                             tau_sublin = t;
                             xi_sublin = xi;
                           }
                         }
                   d_X0[id+NUMBLOCKS*BLOCKSIZE]=X(0);
                   d_X1[id+NUMBLOCKS*BLOCKSIZE]=X(1);
                   d_Y[id+NUMBLOCKS*BLOCKSIZE]=Y;
                   d_Z[id+NUMBLOCKS*BLOCKSIZE]=Z;
                   d_xi[id+NUMBLOCKS*BLOCKSIZE]=xi;
                   d_ji_t[id+NUMBLOCKS*BLOCKSIZE]=ji_t;
        }
        	      int seSuma=cuenta[id]*acabo*(1-seAcaboAntes);
                int seSuma2=cuenta[id]*acabo*acabo2*(1-seAcaboAntes2);
                RNGcallsv = RNGCalls_thread;
                d_sums[13*id+0] += RNGcallsv;
                d_sums[13*id+1] += (1-acabo)*tau_lin;
                d_sums[13*id+2] += (1-acabo*acabo2)*tau_sublin;
                d_sums[13*id+3] += (1-acabo)*tau_lin*tau_lin;
                d_sums[13*id+4] += (1-acabo*acabo2)*tau_sublin*tau_sublin;
                d_sums[13*id+5] = d_X0[id];
                d_sums[13*id+6] = d_X1[id];
                d_sums[13*id+7] = d_Y[id];
                d_sums[13*id+8] = d_Z[id];
                d_sums[13*id+9] = d_X0[id+NUMBLOCKS*BLOCKSIZE];
                d_sums[13*id+10] = d_X1[id+NUMBLOCKS*BLOCKSIZE];
                d_sums[13*id+11] = d_Y[id+NUMBLOCKS*BLOCKSIZE];
                d_sums[13*id+12] = d_Z[id+NUMBLOCKS*BLOCKSIZE];
                if(EulerMaruyama){
                  if((seSuma2==1)){
                    continua[id]=false;
                  //ALTERNATIVA DE CONTAR
                    seAcabo[id]=false;
                    seAcabo[id+NUMBLOCKS*BLOCKSIZE]=false;
                   }else{continua[id]=true;}
                }else{ if((seSuma==1)){
                  continua[id]=false;
                  //ALTERNATIVA DE CONTAR
                  seAcabo[id]=false;
                  seAcabo[id+NUMBLOCKS*BLOCKSIZE]=false;
                }else{continua[id]=true;}}
            }//
            template <typename Typef,typename Typec, typename Typeu>
            __global__ void SolveLoop(double X01,double X02, double discr,double T, double rho,
                                      double sqrth,double bp0,double bp1,double bp2,double bp3,curandStateMtgp32* state,
                                      double* d_sums, bool* seAcabo, bool* continua,bool* cuenta,double*d_X0, double*d_X1, double*d_Y,
                                      double*d_Z, double*d_xi, double*d_ji_t,int N, int N_tray,int Nx, int Ny,
                                      Typef f, Typec c, bool EulerMaruyama, bool VARC, Typeu ux, Typeu uy){
              int id=blockIdx.x * blockDim.x + threadIdx.x;
              int bid=blockIdx.x;
              int stride = blockDim.x * gridDim.x;
              double boundary_parameters[4]={bp0,bp1,bp2,bp3};
              double h=discr;
              Eigen::Vector2d X0;
              X0 << X01, X02;
              Eigen::Vector2d X,normal,normal_proyection,increment;
              double Y,Z,xi,t,ji_t,dist,dist_k;
              bool stoppingbc, Neumannbc;
              //Feynmann_Kac processes and final quantities
              Eigen::Vector2d X_tau_lin;
              Eigen::Vector2d X_tau_sublin= X_tau_lin ;
              double Y_tau_lin, Y_tau_sublin, Z_tau_lin, Z_tau_sublin,
              tau_lin, tau_sublin,xi_lin,xi_sublin;
              unsigned int RNGcallsv;
              //Random number generator
              unsigned int RNGCalls_thread = 0;
              int threads;
              bvpdev boundvalprob;
              boundvalprob=initBVPdev();
              int seAcaboAntes=0;
              int seAcaboAntes2=0;
                X = X0;
                Y = 1;
                Z = 0;
                xi = 0;
                ji_t = 0;
                t = INFINITY;
                if(continua[id]==true){
                  X<<d_X0[id],d_X1[id];
                  Y=d_Y[id];
                  Z=d_Z[id];
                  xi=d_xi[id];
                  ji_t=d_ji_t[id];
                  if(seAcabo[id]==true){
                    seAcaboAntes=1;
                    h=0;
                    if(seAcabo[id+NUMBLOCKS*BLOCKSIZE]==true){
                      seAcaboAntes2=1;}
                  }
                }
                if(continua[id]==false&&N>=N_tray){
                  cuenta[id]=false;
                }
                double Gobet_Constant1=0.5826;
                double Gobet_Constant2=0.0;
                RNGCalls_thread = 0;
                dist = boundvalprob.distance(boundary_parameters,X,
                normal_proyection,normal);
                stoppingbc = boundvalprob.absorbing(normal_proyection);
                Neumannbc = boundvalprob.Neumann(normal_proyection);
                threads = id;
                Eigen::Matrix2d sigma_aux=boundvalprob.sigma(X,t);
                double Gobet_Condition=boundvalprob.distance(boundary_parameters,X,
                    normal_proyection,normal)+Gobet_Constant1*(normal.transpose()*sigma_aux).norm()*sqrth;
                double Gobet_Condition2=boundvalprob.distance(boundary_parameters,X,
                  normal_proyection,normal)+Gobet_Constant2*(normal.transpose()*sigma_aux).norm()*sqrth;
                int acabo=seAcabo[id];
                int acabo2=seAcabo[id+NUMBLOCKS*BLOCKSIZE];
                X_tau_lin = normal_proyection;
                Y_tau_lin = Y;
                Z_tau_lin = Z;
                tau_lin = t;
                xi_lin = xi;
                X_tau_sublin = normal_proyection;
                Y_tau_sublin = Y;
                Z_tau_sublin = Z;
                tau_sublin = t;
                xi_sublin = xi;
                for(int algo=0;algo<1000;algo++){
                        if(stoppingbc == true){//
                            StepCUDA(X,normal,Y,Z,xi,t,ji_t,h,sqrth,state,
                            increment,boundvalprob.sigma,f,
                            boundvalprob.b,c,boundvalprob.psi,
                            boundvalprob.varphi, ux,uy, boundary_parameters, Nx, Ny);
                            RNGCalls_thread += 2;
                        }else{
                            if(Neumannbc == true){//
                                StepCUDA(X,normal,normal_proyection,Y,Z,xi,t,ji_t,rho,h,increment,
                                boundvalprob.sigma,f,
                                boundvalprob.b,c,
                                boundvalprob.psi,boundvalprob.varphi,
                                ux,uy, boundvalprob.distance,
                                boundary_parameters,dist_k,state,
                                sqrth,RNGCalls_thread, Nx, Ny);//
                            }else{//
                                StepCUDA(X,normal,Y,Z,xi,t,ji_t,h,sqrth,state,
                                increment,boundvalprob.sigma,f,
                                boundvalprob.b,c,boundvalprob.psi,
                                boundvalprob.varphi, ux,uy, boundary_parameters, Nx, Ny);
                                RNGCalls_thread += 2;
                            }
                        }
                        dist_k = dist;
                        stoppingbc = boundvalprob.absorbing(normal_proyection);
                        Neumannbc = boundvalprob.Neumann(normal_proyection);
            	          sigma_aux=boundvalprob.sigma(X,t);
                        Gobet_Condition=boundvalprob.distance(boundary_parameters,X,
                        normal_proyection,normal)+Gobet_Constant1*(normal.transpose()*sigma_aux).norm()*sqrth;
            	          if(acabo==0&&Gobet_Condition>0){
                           acabo=1;
                           h=0;
                           seAcabo[id]=true;
                           X_tau_lin = normal_proyection;
                           Y_tau_lin = Y;
                           Z_tau_lin = Z;
                           tau_lin = t;
                           xi_lin = xi;
                         }
                       }
                     
                d_X0[id]=X(0);
                d_X1[id]=X(1);
                d_Y[id]=Y;
                d_Z[id]=Z;
                d_xi[id]=xi;
                d_ji_t[id]=ji_t;
                if(EulerMaruyama){
                  h=discr;
                   if(seAcaboAntes==1){
                     X<<d_X0[id+NUMBLOCKS*BLOCKSIZE],d_X1[id+NUMBLOCKS*BLOCKSIZE];
                     Y=d_Y[id+NUMBLOCKS*BLOCKSIZE];
                     Z=d_Z[id+NUMBLOCKS*BLOCKSIZE];
                     xi=d_xi[id+NUMBLOCKS*BLOCKSIZE];
                     ji_t=d_ji_t[id+NUMBLOCKS*BLOCKSIZE];
                   }
                   for(int algo=0;algo<10000;algo++){
               //        do{
                               StepCUDA(X,normal,Y,Z,xi,t,ji_t,h,sqrth,state,
                               increment,boundvalprob.sigma,f,
                               boundvalprob.b,c,boundvalprob.psi,
                               boundvalprob.varphi, ux,uy, boundary_parameters, Nx, Ny);
                               RNGCalls_thread += 2;
                               sigma_aux=boundvalprob.sigma(X,t);
                               Gobet_Condition2=boundvalprob.distance(boundary_parameters,X,
                                 normal_proyection,normal)+Gobet_Constant2*(normal.transpose()*sigma_aux).norm()*sqrth;
                               //End of sublinear step
                               if(acabo2==0&&Gobet_Condition2>0){
                                 acabo2=1;
                                 seAcabo[id+BLOCKSIZE*NUMBLOCKS]=true;
                                 X_tau_sublin = normal_proyection;
                                 Y_tau_sublin = Y;
                                 Z_tau_sublin = Z;
                                 tau_sublin = t;
                                 xi_sublin = xi;
                               }
                             }
                       d_X0[id+NUMBLOCKS*BLOCKSIZE]=X(0);
                       d_X1[id+NUMBLOCKS*BLOCKSIZE]=X(1);
                       d_Y[id+NUMBLOCKS*BLOCKSIZE]=Y;
                       d_Z[id+NUMBLOCKS*BLOCKSIZE]=Z;
                       d_xi[id+NUMBLOCKS*BLOCKSIZE]=xi;
                       d_ji_t[id+NUMBLOCKS*BLOCKSIZE]=ji_t;
            }
            	      int seSuma=cuenta[id]*acabo*(1-seAcaboAntes);
                    int seSuma2=cuenta[id]*acabo*acabo2*(1-seAcaboAntes2);
                    RNGcallsv = RNGCalls_thread;
                    d_sums[13*id+0] += RNGcallsv;
                    d_sums[13*id+1] += (1-acabo)*tau_lin;
                    d_sums[13*id+2] += (1-acabo*acabo2)*tau_sublin;
                    d_sums[13*id+3] += (1-acabo)*tau_lin*tau_lin;
                    d_sums[13*id+4] += (1-acabo*acabo2)*tau_sublin*tau_sublin;
                    d_sums[13*id+5] = d_X0[id];
                    d_sums[13*id+6] = d_X1[id];
                    d_sums[13*id+7] = d_Y[id];
                    d_sums[13*id+8] = d_Z[id];
                    d_sums[13*id+9] = d_X0[id+NUMBLOCKS*BLOCKSIZE];
                    d_sums[13*id+10] = d_X1[id+NUMBLOCKS*BLOCKSIZE];
                    d_sums[13*id+11] = d_Y[id+NUMBLOCKS*BLOCKSIZE];
                    d_sums[13*id+12] = d_Z[id+NUMBLOCKS*BLOCKSIZE];
                    if(EulerMaruyama){
                      if((seSuma2==1)){
                        continua[id]=false;
                      //ALTERNATIVA DE CONTAR
                        seAcabo[id]=false;
                        seAcabo[id+NUMBLOCKS*BLOCKSIZE]=false;
                       }else{continua[id]=true;}
                    }else{ if((seSuma==1)){
                      continua[id]=false;
                      //ALTERNATIVA DE CONTAR
                      seAcabo[id]=false;
                      seAcabo[id+NUMBLOCKS*BLOCKSIZE]=false;
                    }else{continua[id]=true;}}
                }
