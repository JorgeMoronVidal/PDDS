#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <boost/random.hpp>
#include <eigen3/Eigen/Core>
#include "../BVPs/BVP.cuh"
#include <curand.h>
#include <curand_kernel.h>
#include <curand_mtgp32_host.h>
#include <curand_mtgp32dc_p_11213.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "cubicTex2D.cu"

/*Updates increment vector*/
__device__ inline void Increment_UpdateCUDA(Eigen::Vector2d & increment, curandStateMtgp32* state, double sqrth){
  int bid=blockIdx.x;
  double x=curand_normal_double(&state[bid]);
  increment(0) = sqrth*x;
  x=curand_normal_double(&state[bid]);
  increment(1) = sqrth*x;
};
/////
/*One step of plain Euler Maruyama*/
__device__ inline void StepCUDA(Eigen::Vector2d & X, Eigen::Vector2d & N, double & Y, double & Z, double & t,
double ji_t, double h, double sqrth, curandStateMtgp32* state,
Eigen::Vector2d & increment, pfmatrix sigma, cudaTextureObject_t tex_f, pfvector b, cudaTextureObject_t tex_c, pfscalarN psi,
pfscalarN varphi, double* params, int Nx, int Ny){
  Increment_UpdateCUDA(increment, state, sqrth);
  float x=Nx*(X(0)-params[0])/(params[2]-params[0]);
  float y=Ny*(X(1)-params[1])/(params[3]-params[1]);
  Z += cubicTex2DSimple(tex_f,x,y)*Y*h + psi(X,N,t)*Y*ji_t;
  Y += cubicTex2DSimple(tex_c,x,y)*Y*h + varphi(X,N,t) * Y * ji_t;
  X += b(X,t)*h + sigma(X,t)*increment;
  t -= h;
}
__device__ inline void StepCUDA(Eigen::Vector2d & X, Eigen::Vector2d & N, double & Y, double & Z, double & t,
double ji_t, double h, double sqrth, curandStateMtgp32* state,
Eigen::Vector2d & increment, pfmatrix sigma, pfscalar f, pfvector b, cudaTextureObject_t tex_c, pfscalarN psi,
pfscalarN varphi, double* params, int Nx, int Ny){
  Increment_UpdateCUDA(increment, state, sqrth);
  float x=Nx*(X(0)-params[0])/(params[2]-params[0]);
  float y=Ny*(X(1)-params[1])/(params[3]-params[1]);
  Z += f(X,t)*Y*h + psi(X,N,t)*Y*ji_t;
  Y += cubicTex2DSimple(tex_c,x,y)*Y*h + varphi(X,N,t) * Y * ji_t;
  X += b(X,t)*h + sigma(X,t)*increment;
  t -= h;
}
__device__ inline void StepCUDA(Eigen::Vector2d & X, Eigen::Vector2d & N, double & Y, double & Z, double & t,
double ji_t, double h, double sqrth, curandStateMtgp32* state,
Eigen::Vector2d & increment, pfmatrix sigma, cudaTextureObject_t tex_f, pfvector b, pfscalar c, pfscalarN psi,
pfscalarN varphi, double* params, int Nx, int Ny){
  Increment_UpdateCUDA(increment, state, sqrth);
  float x=Nx*(X(0)-params[0])/(params[2]-params[0]);
  float y=Ny*(X(1)-params[1])/(params[3]-params[1]);
  Z += cubicTex2DSimple(tex_f,x,y)*Y*h + psi(X,N,t)*Y*ji_t;
  Y += c(X,t)*Y*h + varphi(X,N,t) * Y * ji_t;
  X += b(X,t)*h + sigma(X,t)*increment;
  t -= h;
}
__device__ inline void StepCUDA(Eigen::Vector2d & X, Eigen::Vector2d & N, double & Y, double & Z, double & t,
double ji_t, double h, double sqrth, curandStateMtgp32* state,
Eigen::Vector2d & increment, pfmatrix sigma, pfscalar f, pfvector b, pfscalar c, pfscalarN psi,
pfscalarN varphi, double* params, int Nx, int Ny){
  Increment_UpdateCUDA(increment, state, sqrth);
  Z += f(X,t)*Y*h + psi(X,N,t)*Y*ji_t;
  Y += c(X,t)*Y*h + varphi(X,N,t) * Y * ji_t;
  X += b(X,t)*h + sigma(X,t)*increment;
  t -= h;
}
/////
/*One step of the plain Euler's discretization with Variance Reduction*/
__device__ inline void StepCUDA(Eigen::Vector2d & X, Eigen::Vector2d & N, double & Y, double & Z, double & t,
double ji_t, double h, double sqrth, curandStateMtgp32* state,
Eigen::Vector2d & increment, pfmatrix sigma, pfscalar f,  pfvector b, pfscalar c, pfscalarN psi, pfscalarN varphi,
pfvector gradient, pfscalar u, double* params, int Nx, int Ny){
  Eigen::Matrix2d sigma_aux = sigma(X,t);
  Increment_UpdateCUDA(increment, state, sqrth);
  Z += f(X,t)*Y*h + psi(X,N,t)*Y*ji_t;
  Y += c(X,t)*Y*h +
  Y *(-sigma_aux.transpose()*gradient(X,t)/u(X,t)).transpose().dot(increment) +
  varphi(X,N,t) * Y * ji_t;
  X += (b(X,t)-sigma_aux*(-sigma_aux.transpose()*gradient(X,t)/u(X,t)))*h + sigma_aux*increment;
  t -= h;
}
__device__ inline void StepCUDA(Eigen::Vector2d & X, Eigen::Vector2d & N, double & Y, double & Z, double & t,
double ji_t, double h, double sqrth, curandStateMtgp32* state,
Eigen::Vector2d & increment, pfmatrix sigma,  pfscalar f, pfvector b, cudaTextureObject_t tex_c,
pfscalarN psi, pfscalarN varphi, pfvector gradient, pfscalar u,double* params, int Nx, int Ny){

  Eigen::Matrix2d sigma_aux = sigma(X,t);
  Increment_UpdateCUDA(increment, state, sqrth);
  float x=Nx*(X(0)-params[0])/(params[2]-params[0]);
  float y=Ny*(X(1)-params[1])/(params[3]-params[1]);
  Z += f(X,t)*Y*h + psi(X,N,t)*Y*ji_t;
  Y += cubicTex2DSimple(tex_c,x,y)*Y*h +
  Y *(-sigma_aux.transpose()*gradient(X,t)/u(X,t)).transpose().dot(increment) +
  varphi(X,N,t) * Y * ji_t;
  X += (b(X,t)-sigma_aux*(-sigma_aux.transpose()*gradient(X,t)/u(X,t)))*h + sigma_aux*increment;
  t -= h;
}
__device__ inline void StepCUDA(Eigen::Vector2d & X, Eigen::Vector2d & N, double & Y, double & Z, double & t,
double ji_t, double h, double sqrth, curandStateMtgp32* state,
Eigen::Vector2d & increment, pfmatrix sigma, cudaTextureObject_t tex_f, pfvector b, pfscalar c,
pfscalarN psi, pfscalarN varphi, pfvector gradient, pfscalar u,double* params, int Nx, int Ny){

  Eigen::Matrix2d sigma_aux = sigma(X,t);
  Increment_UpdateCUDA(increment, state, sqrth);
  float x=Nx*(X(0)-params[0])/(params[2]-params[0]);
  float y=Ny*(X(1)-params[1])/(params[3]-params[1]);
  Z += cubicTex2DSimple(tex_f,x,y)*Y*h + psi(X,N,t)*Y*ji_t;
  Y += c(X,t)*Y*h +
  Y *(-sigma_aux.transpose()*gradient(X,t)/u(X,t)).transpose().dot(increment) +
  varphi(X,N,t) * Y * ji_t;
  X += (b(X,t)-sigma_aux*(-sigma_aux.transpose()*gradient(X,t)/u(X,t)))*h + sigma_aux*increment;
  t -= h;
}
__device__ inline void StepCUDA(Eigen::Vector2d & X, Eigen::Vector2d & N, double & Y, double & Z, double & t,
double ji_t, double h, double sqrth, curandStateMtgp32* state,
Eigen::Vector2d & increment, pfmatrix sigma, cudaTextureObject_t tex_f, pfvector b, cudaTextureObject_t tex_c,
pfscalarN psi, pfscalarN varphi, pfvector gradient, pfscalar u,double* params, int Nx, int Ny){

  Eigen::Matrix2d sigma_aux = sigma(X,t);
  Increment_UpdateCUDA(increment, state, sqrth);
  float x=Nx*(X(0)-params[0])/(params[2]-params[0]);
  float y=Ny*(X(1)-params[1])/(params[3]-params[1]);
  Z += cubicTex2DSimple(tex_f,x,y)*Y*h + psi(X,N,t)*Y*ji_t;
  Y += cubicTex2DSimple(tex_c,x,y)*Y*h +
  Y *(-sigma_aux.transpose()*gradient(X,t)/u(X,t)).transpose().dot(increment) +
  varphi(X,N,t) * Y * ji_t;
  X += (b(X,t)-sigma_aux*(-sigma_aux.transpose()*gradient(X,t)/u(X,t)))*h + sigma_aux*increment;
  t -= h;
}
__device__ inline void StepCUDA(Eigen::Vector2d & X, Eigen::Vector2d & N, double & Y, double & Z, double & t,
double ji_t, double h, double sqrth, curandStateMtgp32* state,
Eigen::Vector2d & increment, pfmatrix sigma, cudaTextureObject_t tex_f, pfvector b, cudaTextureObject_t tex_c,
pfscalarN psi, pfscalarN varphi, cudaTextureObject_t tex_ux,cudaTextureObject_t tex_uy, pfscalar u,
double* params, int Nx, int Ny){
  Eigen::Matrix2d sigma_aux = sigma(X,t);
  Increment_UpdateCUDA(increment, state, sqrth);
  float x=Nx*(X(0)-params[0])/(params[2]-params[0]);
  float y=Ny*(X(1)-params[1])/(params[3]-params[1]);
  Z += cubicTex2DSimple(tex_f,x,y)*Y*h + psi(X,N,t)*Y*ji_t;
  Eigen::Vector2d gradient;
  gradient<< cubicTex2DSimple(tex_ux,x,y),cubicTex2DSimple(tex_uy,x,y);
  Y += cubicTex2DSimple(tex_c,x,y)*Y*h +
  Y *(-sigma_aux.transpose()*gradient/u(X,t)).transpose().dot(increment) +
  varphi(X,N,t) * Y * ji_t;
  X += (b(X,t)-sigma_aux*(-sigma_aux.transpose()*gradient/u(X,t)))*h + sigma_aux*increment;
  t -= h;
}
////////
/*One step of the plain Euler's discretization with Control Variates*/
__device__ inline void StepCUDA(Eigen::Vector2d & X, Eigen::Vector2d & N, double & Y, double & Z , double & xi,
double & t, double ji_t, double h, double sqrth, curandStateMtgp32* state,
Eigen::Vector2d & increment, pfmatrix sigma, cudaTextureObject_t tex_f, pfvector b, cudaTextureObject_t tex_c,
pfscalarN psi, pfscalarN varphi,  cudaTextureObject_t tex_ux,cudaTextureObject_t tex_uy,double* params, int Nx, int Ny){
  Eigen::Matrix2d sigma_aux = sigma(X,t);
  Increment_UpdateCUDA(increment, state, sqrth);
  float x=Nx*(X(0)-params[0])/(params[2]-params[0]);
  float y=Ny*(X(1)-params[1])/(params[3]-params[1]);
  Z += cubicTex2DSimple(tex_f,x,y)*Y*h + psi(X,N,t)*Y*ji_t;
  Eigen::Vector2d gradient;
  gradient<< cubicTex2DSimple(tex_ux,x,y),cubicTex2DSimple(tex_uy,x,y);
  xi +=  Y *(-sigma_aux.transpose()*gradient).dot(increment);
  Y += cubicTex2DSimple(tex_c,x,y)*Y*h + varphi(X,N,t) * Y * ji_t;
  X += b(X,t)*h + sigma_aux*increment;
  t -= h;
}
__device__ inline void StepCUDA(Eigen::Vector2d & X, Eigen::Vector2d & N, double & Y, double & Z , double & xi,
double & t, double ji_t, double h, double sqrth, curandStateMtgp32* state,
Eigen::Vector2d & increment, pfmatrix sigma, cudaTextureObject_t tex_f, pfvector b, cudaTextureObject_t tex_c,
pfscalarN psi, pfscalarN varphi,  pfvector gradient,double* params, int Nx, int Ny){
  Eigen::Matrix2d sigma_aux = sigma(X,t);
  Increment_UpdateCUDA(increment, state, sqrth);
  float x=Nx*(X(0)-params[0])/(params[2]-params[0]);
  float y=Ny*(X(1)-params[1])/(params[3]-params[1]);
  Z += cubicTex2DSimple(tex_f,x,y)*Y*h + psi(X,N,t)*Y*ji_t;
  xi +=  Y *(-sigma_aux.transpose()*gradient(X,t)).dot(increment);
  Y += cubicTex2DSimple(tex_c,x,y)*Y*h + varphi(X,N,t) * Y * ji_t;
  X += b(X,t)*h + sigma_aux*increment;
  t -= h;
}
__device__ inline void StepCUDA(Eigen::Vector2d & X, Eigen::Vector2d & N, double & Y, double & Z , double & xi,
double & t, double ji_t, double h, double sqrth, curandStateMtgp32* state,
Eigen::Vector2d & increment, pfmatrix sigma,  pfscalar f, pfvector b, cudaTextureObject_t tex_c,
pfscalarN psi, pfscalarN varphi,  pfvector gradient,double* params, int Nx, int Ny){
  Eigen::Matrix2d sigma_aux = sigma(X,t);
  Increment_UpdateCUDA(increment, state, sqrth);
  float x=Nx*(X(0)-params[0])/(params[2]-params[0]);
  float y=Ny*(X(1)-params[1])/(params[3]-params[1]);
  Z += f(X,t)*Y*h + psi(X,N,t)*Y*ji_t;
  xi +=  Y *(-sigma_aux.transpose()*gradient(X,t)).dot(increment);
  Y += cubicTex2DSimple(tex_c,x,y)*Y*h + varphi(X,N,t) * Y * ji_t;
  X += b(X,t)*h + sigma_aux*increment;
  t -= h;
}
__device__ inline void StepCUDA(Eigen::Vector2d & X, Eigen::Vector2d & N, double & Y, double & Z , double & xi,
double & t, double ji_t, double h, double sqrth, curandStateMtgp32* state,
Eigen::Vector2d & increment, pfmatrix sigma, cudaTextureObject_t tex_f, pfvector b, pfscalar c,
pfscalarN psi, pfscalarN varphi,  pfvector gradient,double* params, int Nx, int Ny){
  Eigen::Matrix2d sigma_aux = sigma(X,t);
  Increment_UpdateCUDA(increment, state, sqrth);
  float x=Nx*(X(0)-params[0])/(params[2]-params[0]);
  float y=Ny*(X(1)-params[1])/(params[3]-params[1]);
  Z += cubicTex2DSimple(tex_f,x,y)*Y*h + psi(X,N,t)*Y*ji_t;
  xi +=  Y *(-sigma_aux.transpose()*gradient(X,t)).dot(increment);
  Y += c(X,t)*Y*h + varphi(X,N,t) * Y * ji_t;
  X += b(X,t)*h + sigma_aux*increment;
  t -= h;
}
__device__ inline void StepCUDA(Eigen::Vector2d & X, Eigen::Vector2d & N, double & Y, double & Z , double & xi,
double & t, double ji_t, double h, double sqrth, curandStateMtgp32* state,
Eigen::Vector2d & increment, pfmatrix sigma, pfscalar f, pfvector b, pfscalar c,
pfscalarN psi, pfscalarN varphi,  pfvector gradient,  double* params, int Nx, int Ny){
  Eigen::Matrix2d sigma_aux = sigma(X,t);
  Increment_UpdateCUDA(increment, state, sqrth);
  Z += f(X,t)*Y*h + psi(X,N,t)*Y*ji_t;
  xi +=  Y *(-sigma_aux.transpose()*gradient(X,t)).dot(increment);
  Y += c(X,t)*Y*h + varphi(X,N,t) * Y * ji_t;
  X += b(X,t)*h + sigma_aux*increment;
  t -= h;
}
///
/*One step of the plain Euler's discretization with Variance Reduction and Control variates*/
__device__ inline void StepCUDA(Eigen::Vector2d & X, Eigen::Vector2d & N, double & Y, double & Z, double & xi,
double & t, double ji_t, double h, double sqrth, curandStateMtgp32* state,
Eigen::Vector2d & increment, pfmatrix sigma, cudaTextureObject_t tex_f, pfvector b, cudaTextureObject_t tex_c, pfscalarN psi,
pfscalarN varphi,  pfvector mu, pfvector F,double* params, int Nx, int Ny ){
  Eigen::Matrix2d sigma_aux = sigma(X,t);
  Increment_UpdateCUDA(increment, state, sqrth);
  float x=Nx*(X(0)-params[0])/(params[2]-params[0]);
  float y=Ny*(X(1)-params[1])/(params[3]-params[1]);
  Z += cubicTex2DSimple(tex_f,x,y)*Y*h + psi(X,N,t)*Y*ji_t;
  xi += Y *(F(X,t)).dot(increment);
  Y += cubicTex2DSimple(tex_c,x,y)*Y*h + Y *(mu(X,t)).dot(increment) + varphi(X,N,t) * Y * ji_t;
  X += (b(X,t)-sigma_aux*mu(X,t))*h + sigma_aux*increment;
  t -= h;
}
__device__ inline void StepCUDA(Eigen::Vector2d & X, Eigen::Vector2d & N, double & Y, double & Z, double & xi,
double & t, double ji_t, double h, double sqrth, curandStateMtgp32* state,
Eigen::Vector2d & increment, pfmatrix sigma, pfscalar f, pfvector b, pfscalar c, pfscalarN psi,
pfscalarN varphi,  pfvector mu, pfvector F ,double* params, int Nx, int Ny ){
  Eigen::Matrix2d sigma_aux = sigma(X,t);
  Increment_UpdateCUDA(increment, state, sqrth);
  Z += f(X,t)*Y*h + psi(X,N,t)*Y*ji_t;
  xi += Y *(F(X,t)).dot(increment);
  Y += c(X,t)*Y*h + Y *(mu(X,t)).dot(increment) + varphi(X,N,t) * Y * ji_t;
  X += (b(X,t)-sigma_aux*mu(X,t))*h + sigma_aux*increment;
  t -= h;
}

/*One step of the plain Euler's discretization with Control Variates for the Lepingle Algorithm*/
__device__ inline void StepCUDA(Eigen::Vector2d & X, Eigen::Vector2d & N, Eigen::Vector2d & Npro, double & Y, double & Z ,
double & xi, double & t, double & ji_t, double rho,  double h, Eigen::Vector2d & increment, pfmatrix sigma, cudaTextureObject_t tex_f,
pfvector b, cudaTextureObject_t tex_c, pfscalarN psi, pfscalarN varphi,cudaTextureObject_t tex_ux,cudaTextureObject_t tex_uy, pfdist distance, double *params,
double & d_k, curandStateMtgp32* state ,double & sqrth, unsigned int & N_rngcalls, int Nx, int Ny){
  Eigen::Matrix2d sigma_aux = sigma(X,t);
  Eigen::Vector2d Xp, Nprop, Np;
  double omega,uc,nu;
  int bid=blockIdx.x;
  Eigen::Vector2d gradient;
  float x=Nx*(X(0)-params[0])/(params[2]-params[0]);
  float y=Ny*(X(1)-params[1])/(params[3]-params[1]);
  gradient<< cubicTex2DSimple(tex_ux,x,y),cubicTex2DSimple(tex_uy,x,y);
  if (d_k > -rho){
        do{
            Increment_UpdateCUDA(increment, state, sqrth);
            Xp = X + b(X,t)*h + sigma_aux*increment;
            float x=curand_uniform(&state[bid]);
            omega =  log(1-x)*(-2*h); //Exponential distribution with parameter 1/(2*h)
            N_rngcalls += 3;
            uc = (N.transpose()*sigma_aux).dot(increment) +N.transpose().dot(b(X,t))*h;
            nu = 0.5 *(uc+sqrt(pow((N.transpose()*sigma_aux).norm(),2.0)*omega+pow(uc,2.0)));
            //d_k = bvp.boundary.Dist(params, Xp,E_Pp,Np);
            if (d_k < -0.0) d_k = 0.0;
            ji_t = std::max(0.0,nu+d_k);
            Xp = Xp - ji_t*N;
        }while((Xp - Nprop).dot(N)>0.0);
        d_k = distance(params, Xp,Nprop,Np);
        if(d_k > 0.0){
            //printf("WARNING: The particle didn't enter in the domain  after Lepingle step.\n");
            Xp = Nprop;
            ji_t += d_k;
            d_k = 0.0;
        }
    } else {
        Increment_UpdateCUDA(increment, state, sqrth);
        N_rngcalls += 2;
        Xp = X + b(X,t)*h + sigma_aux*increment;
        ji_t = 0.0;
        d_k = distance(params, Xp,Nprop,Np);
        if(d_k > -0.0){
            Xp = Nprop;
            ji_t = d_k;
            d_k = 0.0;
            //std::cout << ji_t << std::endl;
        }
    }
    Z += cubicTex2DSimple(tex_f,x,y)*Y*h + psi(X,N,t)*Y*ji_t;
    xi +=  Y *(-sigma_aux.transpose()*gradient).dot(increment);
    Y += cubicTex2DSimple(tex_c,x,y)*Y*h + varphi(X,N,t) * Y * ji_t;
    X = Xp;
    N = Np;
    Npro = Nprop;
    t += - h;
    ji_t = 0.0;
}
__device__ inline void StepCUDA(Eigen::Vector2d & X, Eigen::Vector2d & N, Eigen::Vector2d & Npro, double & Y, double & Z ,
double & xi, double & t, double & ji_t, double rho,  double h, Eigen::Vector2d & increment, pfmatrix sigma, cudaTextureObject_t tex_f,
pfvector b, cudaTextureObject_t tex_c, pfscalarN psi, pfscalarN varphi,pfvector gradient, pfdist distance, double *params,
double & d_k, curandStateMtgp32* state ,double & sqrth, unsigned int & N_rngcalls, int Nx, int Ny){
  Eigen::Matrix2d sigma_aux = sigma(X,t);
  Eigen::Vector2d Xp, Nprop, Np;
  double omega,uc,nu;
  int bid=blockIdx.x;
  float x=Nx*(X(0)-params[0])/(params[2]-params[0]);
  float y=Ny*(X(1)-params[1])/(params[3]-params[1]);
  if (d_k > -rho){
        do{
            Increment_UpdateCUDA(increment, state, sqrth);
            Xp = X + b(X,t)*h + sigma_aux*increment;
            float x=curand_uniform(&state[bid]);
            omega =  log(1-x)*(-2*h); //Exponential distribution with parameter 1/(2*h)
            N_rngcalls += 3;
            uc = (N.transpose()*sigma_aux).dot(increment) +N.transpose().dot(b(X,t))*h;
            nu = 0.5 *(uc+sqrt(pow((N.transpose()*sigma_aux).norm(),2.0)*omega+pow(uc,2.0)));
            //d_k = bvp.boundary.Dist(params, Xp,E_Pp,Np);
            if (d_k < -0.0) d_k = 0.0;
            ji_t = std::max(0.0,nu+d_k);
            Xp = Xp - ji_t*N;
        }while((Xp - Nprop).dot(N)>0.0);
        d_k = distance(params, Xp,Nprop,Np);
        if(d_k > 0.0){
            //printf("WARNING: The particle didn't enter in the domain  after Lepingle step.\n");
            Xp = Nprop;
            ji_t += d_k;
            d_k = 0.0;
        }
    } else {
        Increment_UpdateCUDA(increment, state, sqrth);
        N_rngcalls += 2;
        Xp = X + b(X,t)*h + sigma_aux*increment;
        ji_t = 0.0;
        d_k = distance(params, Xp,Nprop,Np);
        if(d_k > -0.0){
            Xp = Nprop;
            ji_t = d_k;
            d_k = 0.0;
            //std::cout << ji_t << std::endl;
        }
    }
    Z += cubicTex2DSimple(tex_f,x,y)*Y*h + psi(X,N,t)*Y*ji_t;
    xi +=  Y *(-sigma_aux.transpose()*gradient(X,t)).dot(increment);
    Y += cubicTex2DSimple(tex_c,x,y)*Y*h + varphi(X,N,t) * Y * ji_t;
    X = Xp;
    N = Np;
    Npro = Nprop;
    t += - h;
    ji_t = 0.0;
}
__device__ inline void StepCUDA(Eigen::Vector2d & X, Eigen::Vector2d & N, Eigen::Vector2d & Npro, double & Y, double & Z ,
double & xi, double & t, double & ji_t, double rho,  double h, Eigen::Vector2d & increment, pfmatrix sigma, cudaTextureObject_t tex_f,
pfvector b, pfscalar c, pfscalarN psi, pfscalarN varphi,pfvector gradient, pfdist distance, double *params,
double & d_k, curandStateMtgp32* state ,double & sqrth, unsigned int & N_rngcalls, int Nx, int Ny){
  Eigen::Matrix2d sigma_aux = sigma(X,t);
  Eigen::Vector2d Xp, Nprop, Np;
  double omega,uc,nu;
  int bid=blockIdx.x;
  float x=Nx*(X(0)-params[0])/(params[2]-params[0]);
  float y=Ny*(X(1)-params[1])/(params[3]-params[1]);
  if (d_k > -rho){
        do{
            Increment_UpdateCUDA(increment, state, sqrth);
            Xp = X + b(X,t)*h + sigma_aux*increment;
            float x=curand_uniform(&state[bid]);
            omega =  log(1-x)*(-2*h); //Exponential distribution with parameter 1/(2*h)
            N_rngcalls += 3;
            uc = (N.transpose()*sigma_aux).dot(increment) +N.transpose().dot(b(X,t))*h;
            nu = 0.5 *(uc+sqrt(pow((N.transpose()*sigma_aux).norm(),2.0)*omega+pow(uc,2.0)));
            //d_k = bvp.boundary.Dist(params, Xp,E_Pp,Np);
            if (d_k < -0.0) d_k = 0.0;
            ji_t = std::max(0.0,nu+d_k);
            Xp = Xp - ji_t*N;
        }while((Xp - Nprop).dot(N)>0.0);
        d_k = distance(params, Xp,Nprop,Np);
        if(d_k > 0.0){
            //printf("WARNING: The particle didn't enter in the domain  after Lepingle step.\n");
            Xp = Nprop;
            ji_t += d_k;
            d_k = 0.0;
        }
    } else {
        Increment_UpdateCUDA(increment, state, sqrth);
        N_rngcalls += 2;
        Xp = X + b(X,t)*h + sigma_aux*increment;
        ji_t = 0.0;
        d_k = distance(params, Xp,Nprop,Np);
        if(d_k > -0.0){
            Xp = Nprop;
            ji_t = d_k;
            d_k = 0.0;
            //std::cout << ji_t << std::endl;
        }
    }
    Z += cubicTex2DSimple(tex_f,x,y)*Y*h + psi(X,N,t)*Y*ji_t;
    xi +=  Y *(-sigma_aux.transpose()*gradient(X,t)).dot(increment);
    Y += c(X,t)*Y*h + varphi(X,N,t) * Y * ji_t;
    X = Xp;
    N = Np;
    Npro = Nprop;
    t += - h;
    ji_t = 0.0;
}
__device__ inline void StepCUDA(Eigen::Vector2d & X, Eigen::Vector2d & N, Eigen::Vector2d & Npro, double & Y, double & Z ,
double & xi, double & t, double & ji_t, double rho,  double h, Eigen::Vector2d & increment, pfmatrix sigma, pfscalar f,
pfvector b, cudaTextureObject_t tex_c, pfscalarN psi, pfscalarN varphi,pfvector gradient, pfdist distance, double *params,
double & d_k, curandStateMtgp32* state ,double & sqrth, unsigned int & N_rngcalls, int Nx, int Ny){
  Eigen::Matrix2d sigma_aux = sigma(X,t);
  Eigen::Vector2d Xp, Nprop, Np;
  double omega,uc,nu;
  int bid=blockIdx.x;
  float x=Nx*(X(0)-params[0])/(params[2]-params[0]);
  float y=Ny*(X(1)-params[1])/(params[3]-params[1]);
  if (d_k > -rho){
        do{
            Increment_UpdateCUDA(increment, state, sqrth);
            Xp = X + b(X,t)*h + sigma_aux*increment;
            float x=curand_uniform(&state[bid]);
            omega =  log(1-x)*(-2*h); //Exponential distribution with parameter 1/(2*h)
            N_rngcalls += 3;
            uc = (N.transpose()*sigma_aux).dot(increment) +N.transpose().dot(b(X,t))*h;
            nu = 0.5 *(uc+sqrt(pow((N.transpose()*sigma_aux).norm(),2.0)*omega+pow(uc,2.0)));
            //d_k = bvp.boundary.Dist(params, Xp,E_Pp,Np);
            if (d_k < -0.0) d_k = 0.0;
            ji_t = std::max(0.0,nu+d_k);
            Xp = Xp - ji_t*N;
        }while((Xp - Nprop).dot(N)>0.0);
        d_k = distance(params, Xp,Nprop,Np);
        if(d_k > 0.0){
            //printf("WARNING: The particle didn't enter in the domain  after Lepingle step.\n");
            Xp = Nprop;
            ji_t += d_k;
            d_k = 0.0;
        }
    } else {
        Increment_UpdateCUDA(increment, state, sqrth);
        N_rngcalls += 2;
        Xp = X + b(X,t)*h + sigma_aux*increment;
        ji_t = 0.0;
        d_k = distance(params, Xp,Nprop,Np);
        if(d_k > -0.0){
            Xp = Nprop;
            ji_t = d_k;
            d_k = 0.0;
            //std::cout << ji_t << std::endl;
        }
    }
    Z += f(X,t)*Y*h + psi(X,N,t)*Y*ji_t;
    xi +=  Y *(-sigma_aux.transpose()*gradient(X,t)).dot(increment);
    Y += cubicTex2DSimple(tex_c,x,y)*Y*h + varphi(X,N,t) * Y * ji_t;
    X = Xp;
    N = Np;
    Npro = Nprop;
    t += - h;
    ji_t = 0.0;
}
__device__ inline void StepCUDA(Eigen::Vector2d & X, Eigen::Vector2d & N, Eigen::Vector2d & Npro, double & Y, double & Z ,
double & xi, double & t, double & ji_t, double rho,  double h, Eigen::Vector2d & increment, pfmatrix sigma, pfscalar f,
pfvector b, pfscalar c, pfscalarN psi, pfscalarN varphi,  pfvector gradient, pfdist distance, double *params,
double & d_k, curandStateMtgp32* state ,double & sqrth, unsigned int & N_rngcalls, int Nx, int Ny){
  Eigen::Matrix2d sigma_aux = sigma(X,t);
  Eigen::Vector2d Xp, Nprop, Np;
  double omega,uc,nu;
  int bid=blockIdx.x;
  if (d_k > -rho){
        do{
            Increment_UpdateCUDA(increment, state, sqrth);
            Xp = X + b(X,t)*h + sigma_aux*increment;
            float x=curand_uniform(&state[bid]);
            omega =  log(1-x)*(-2*h); //Exponential distribution with parameter 1/(2*h)
            N_rngcalls += 3;
            uc = (N.transpose()*sigma_aux).dot(increment) +N.transpose().dot(b(X,t))*h;
            nu = 0.5 *(uc+sqrt(pow((N.transpose()*sigma_aux).norm(),2.0)*omega+pow(uc,2.0)));
            //d_k = bvp.boundary.Dist(params, Xp,E_Pp,Np);
            if (d_k < -0.0) d_k = 0.0;
            ji_t = std::max(0.0,nu+d_k);
            Xp = Xp - ji_t*N;
        }while((Xp - Nprop).dot(N)>0.0);
        d_k = distance(params, Xp,Nprop,Np);
        if(d_k > 0.0){
            //printf("WARNING: The particle didn't enter in the domain  after Lepingle step.\n");
            Xp = Nprop;
            ji_t += d_k;
            d_k = 0.0;
        }
    } else {
        Increment_UpdateCUDA(increment, state, sqrth);
        N_rngcalls += 2;
        Xp = X + b(X,t)*h + sigma_aux*increment;
        ji_t = 0.0;
        d_k = distance(params, Xp,Nprop,Np);
        if(d_k > -0.0){
            Xp = Nprop;
            ji_t = d_k;
            d_k = 0.0;
            //std::cout << ji_t << std::endl;
        }
    }
    Z += f(X,t)*Y*h + psi(X,N,t)*Y*ji_t;
    xi +=  Y *(-sigma_aux.transpose()*gradient(X,t)).dot(increment);
    Y += c(X,t)*Y*h + varphi(X,N,t) * Y * ji_t;
    X = Xp;
    N = Np;
    Npro = Nprop;
    t += - h;
    ji_t = 0.0;
}
//Status of the particle
enum outcome {in, stop, reflec, time_out};
__device__ inline outcome InsideCUDA(double & distance, bool & stoppingbc, bool & Neumannbc, Eigen::Vector2d & X, double & t, double &ji_t, double & T, double & sqrth,
Eigen::Vector2d & Npro, Eigen::Vector2d & N, double *params, double Gobet_Constant, pfdist dist, pfbtype stopf, pfbtype Neumannf, pfmatrix sigma){

    distance = dist(params, X, Npro, N);
    stoppingbc = stopf(Npro);
    Neumannbc = Neumannf(Npro);
    Eigen::Matrix2d sigma_aux = sigma(X,t);
    outcome status;
    if(stoppingbc){

      if(distance < Gobet_Constant*(N.transpose()*sigma_aux).norm()*sqrth){

        status = (t>0) ? in : time_out;

      } else {

        status = (t>0) ? stop : time_out;

      }

    }else{
      if( distance <= 0){

        status = (t>0) ? in : time_out;

        } else {

        status = (t>0) ? reflec : time_out;
        X = Npro;

      }
    }

    switch(status){

      case stop:
        ji_t = 0.0;
        X = Npro;
        break;

      case reflec:
        ji_t = distance;
        X = Npro;
        break;

      case time_out:
        ji_t = 0.0;
        t = 0.0;
        break;

      default:
        ji_t = 0.0;
        break;
    }
    return status;
}
