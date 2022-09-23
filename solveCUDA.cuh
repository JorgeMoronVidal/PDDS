#include "../BVPs/BVP.cuh"
//# include "../BVPs/initBVPdev.cuh"
//#include "../BVPs/LUT.hpp"
//#include "../BVPs/EDPs/Monegros_Poisson.hpp"
//#include "../Integrators/FKACIntegrator.cuh"
//#include <omp.h>
//#include <math.h>
//#include <eigen3/Eigen/Core>
//#include <vector>
//#include <iostream>
//#include <cuda.h>
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//#include <curand.h>
//#include <curand_kernel.h>
//#include <curand_mtgp32_host.h>
//#include <curand_mtgp32dc_p_11213.h>
#include "SolveLoop.cuh"
#define NUMBLOCKS 128
#define BLOCKSIZE 256
curandStateMtgp32* dev_curand_states;
mtgp32_kernel_params *devKernelParams;
#define ERROR_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %dn", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__host__ void initRNGCuda(int seed){
  cudaMalloc((void**)&dev_curand_states, NUMBLOCKS*sizeof(curandStateMtgp32));
  cudaMalloc((void**)&devKernelParams, sizeof(mtgp32_kernel_params));
  curandMakeMTGP32Constants(mtgp32dc_params_fast_11213, devKernelParams);
  curandMakeMTGP32KernelState(dev_curand_states, mtgp32dc_params_fast_11213, devKernelParams,NUMBLOCKS, seed);

}

template <typename Typef,typename Typec>
__host__ void SolveCUDA(Eigen::Vector2d X0,double T, double* boundary_parameters, double h,long long int N_tray, pfscalar g, int Nx, int Ny, Typef f, Typec c, bool VARC,
                        double* phi, double* phi2, double* phi3, double* phi4, double* phixi, double* phi_plus_xi2, double* xi, double* xi2, double* tau, double* tau2){
    bool EMFlag=false;
    //Calcular tiempo de ejecucións
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds=0;
    float time;
    //Time discretization of the trajectories and initial time of the trajectory T
    double sqrth= sqrt(h);
    double time_discretization=h,rho=0.2;
    //Number of trayectories
    long long int N=0; int added=NUMBLOCKS*BLOCKSIZE;
    bool seAcabo[2*NUMBLOCKS*BLOCKSIZE],dentro[NUMBLOCKS*BLOCKSIZE],cuenta[NUMBLOCKS*BLOCKSIZE];
    //Error objective
    double eps=1.0;
    //X0
      double X01=X0(0), X02=X0(1);
//	printf("\nX_1=%f, X_2=%f",X01,X02);
    //Errores
      cudaError_t errMalloc, errSync,errCpy;
      cudaError_t errMemset, errAsync;
    //Variables de device y copy
        double *d_sums, *d_X0, *d_X1, *d_Y, *d_Z, *d_xi, *d_ji_t;
        bool* d_seAcabo,*d_continua,*d_cuenta;
	const int sizeSum=23*BLOCKSIZE*NUMBLOCKS;
        double h_sums[sizeSum];
         bool seguir=true;
 //Variables auxiliares device
 cudaMalloc((void**) &d_X0,2*NUMBLOCKS*BLOCKSIZE*sizeof(double));
 cudaMalloc((void**) &d_X1,2*NUMBLOCKS*BLOCKSIZE*sizeof(double));
 cudaMalloc((void**) &d_Y,2*NUMBLOCKS*BLOCKSIZE*sizeof(double));
 cudaMalloc((void**) &d_Z,2*NUMBLOCKS*BLOCKSIZE*sizeof(double));
 cudaMalloc((void**) &d_xi,2*NUMBLOCKS*BLOCKSIZE*sizeof(double));
 cudaMalloc((void**) &d_ji_t,2*NUMBLOCKS*BLOCKSIZE*sizeof(double));
//Bucle en N_tray
     //Malloc e iniciacion variables device
       //DSUMS
           cudaMalloc((void**) &d_sums, sizeSum*sizeof(double));
           errMalloc= cudaGetLastError();
           cudaMemset(d_sums, 0, sizeSum * sizeof(double));
           errMemset= cudaGetLastError();
           if (errMalloc != cudaSuccess)
             printf("cudMalloc kernel error [d_sums]: %s\n", cudaGetErrorString(errMalloc));
           if (errMemset != cudaSuccess)
             printf("errMemset kernel error[d_sums]: %s\n", cudaGetErrorString(errMemset));
     //D_BOOL
       //seAcabo
           cudaMalloc((void**) &d_seAcabo,2*NUMBLOCKS*BLOCKSIZE*sizeof(bool));
           errMalloc= cudaGetLastError();
           if (errMalloc != cudaSuccess)
           printf("cudMalloc kernel error [d_seAcabo]: %s\n", cudaGetErrorString(errMalloc));
           cudaMemset(d_seAcabo, false, 2*NUMBLOCKS*BLOCKSIZE * sizeof(bool));
           errMemset= cudaGetLastError();
           if (errMemset != cudaSuccess)
           printf("cudaMemcpy error [d_seAcabo]: %s\n", cudaGetErrorString(errMemset));
     //La trayectoria está dentro o fuera
         cudaMalloc((void**) &d_continua,NUMBLOCKS*BLOCKSIZE*sizeof(bool));
           errMalloc= cudaGetLastError();
           if (errMalloc != cudaSuccess)
             printf("cudMalloc kernel error [d_continua]: %s\n", cudaGetErrorString(errMalloc));
             cudaMemset(d_continua, false, NUMBLOCKS*BLOCKSIZE*sizeof(bool));
             errMemset= cudaGetLastError();
             if (errMemset != cudaSuccess)
             printf("cudaMemcpy error [d_continua]: %s\n", cudaGetErrorString(errMemset));

       //La trayectoria cuenta
       cudaMalloc((void**) &d_cuenta,NUMBLOCKS*BLOCKSIZE*sizeof(bool));
         errMalloc= cudaGetLastError();
         if (errMalloc != cudaSuccess)
       printf("cudMalloc kernel error [d_cuenta]: %s\n", cudaGetErrorString(errMalloc));
     cudaMemset(d_cuenta, true, NUMBLOCKS*BLOCKSIZE*sizeof(bool));
     errMemset= cudaGetLastError();
     if (errMemset != cudaSuccess)
           printf("cudaMemcpy error [d_cuenta]: %s\n", cudaGetErrorString(errMemset));
while(seguir==true){
  /* code */
//do{    //Entramos en kernel
      cudaEventRecord(start);
        SolveLoop<<<NUMBLOCKS,BLOCKSIZE>>>(X01,X02,h,T,rho,sqrth,boundary_parameters[0],
                boundary_parameters[1],boundary_parameters[2],boundary_parameters[3],dev_curand_states,
                d_sums,d_seAcabo, d_continua,d_cuenta,d_X0, d_X1, d_Y, d_Z, d_xi, d_ji_t,N,N_tray,
                Nx,Ny,f, c, g,  false,VARC);
      errSync  = cudaGetLastError();
      errAsync = cudaDeviceSynchronize();
      cudaEventRecord(stop);
      if (errSync != cudaSuccess)
        printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
      if (errAsync != cudaSuccess)
        printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
//      printf("kernel ejecutado por primera vez\n" );
      N+=added;
     //CONTAR CON D_BOOL
//      cudaMemcpy(seAcabo, d_seAcabo, 2*NUMBLOCKS*BLOCKSIZE*sizeof(bool), cudaMemcpyDeviceToHost);
//      cudaMemcpy(cuenta, d_cuenta, NUMBLOCKS*BLOCKSIZE*sizeof(bool), cudaMemcpyDeviceToHost);
//      errCpy  = cudaGetLastError();
//      if (errCpy != cudaSuccess)
//      printf("cudaMemcpy error[bool]: %s\n", cudaGetErrorString(errCpy));
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&milliseconds, start, stop);
      time+=milliseconds;
      milliseconds=0;
      seguir=false;
      added=0;
//      for(int jjj=0;jjj<NUMBLOCKS*BLOCKSIZE;jjj++){
//        if(cuenta[jjj]==true){
//          seguir=true;
//          if(seAcabo[jjj]==true&&seAcabo[jjj+NUMBLOCKS*BLOCKSIZE]==true){
//            seAcabo[jjj]=false;
//            seAcabo[jjj+NUMBLOCKS*BLOCKSIZE]=false;
//            added++;
//      }}}
//      cudaMemcpy(d_seAcabo, seAcabo, 2*NUMBLOCKS*BLOCKSIZE*sizeof(bool), cudaMemcpyHostToDevice);
//////////////////////////
  //CONTAR CON continua
	int traye=0;
        cudaMemcpy(cuenta, d_cuenta, NUMBLOCKS*BLOCKSIZE*sizeof(bool), cudaMemcpyDeviceToHost);
        cudaMemcpy(dentro, d_continua, NUMBLOCKS*BLOCKSIZE*sizeof(bool), cudaMemcpyDeviceToHost);
              for(int jjj=0;jjj<NUMBLOCKS*BLOCKSIZE;jjj++){
                if(cuenta[jjj]==true){
                  seguir=true;
                  if(dentro[jjj]==false&& N<N_tray){
                    added++;
              }else{traye++;}}}
//	printf("\nTrayectorias activas=%i",traye);
}
cudaMemcpy(h_sums, d_sums, sizeSum*sizeof(double), cudaMemcpyDeviceToHost);
  *phi=0;
  *phi2=0;
  *phi3=0;
  *phi4=0;
  *phixi=0;
  *phi_plus_xi2=0;
  *xi=0;
  *xi2=0;
  *tau=0;
  *tau2=0;
for(int id=0; id<NUMBLOCKS*BLOCKSIZE;id++){
  *phi+=h_sums[23*id+0]/N;
  *phi2+=h_sums[23*id+2]/N;
  *phi3+=h_sums[23*id+17]/N;
  *phi4+=h_sums[23*id+18]/N;
  *phixi+=h_sums[23*id+12]/N;
  *phi_plus_xi2+=h_sums[23*id+10]/N;
  *xi+=h_sums[23*id+4]/N;
  *xi2+=h_sums[23*id+6]/N;
  *tau+=h_sums[23*id+15]/N;
  *tau2+=h_sums[23*id+19]/N;
}
errCpy  = cudaGetLastError();
if (errCpy != cudaSuccess)
printf("cudaMemcpy error: %s\n", cudaGetErrorString(errCpy));
//printf("\n %f;%f;%f;%f;%f;%f;%f;%f;%f;%f", *phi, *phi2, *phi3, *phi4, *phixi,*phi_plus_xi2,*xi,*xi2,*tau,*tau2);
FILE *out_file=fopen("output_cuda", "w");
fprintf(out_file,"\n N= %lld, Ntray= %lld", N, N_tray);
printf("\n N= %lld, Ntray= %lld", N, N_tray);
fprintf(out_file,"\n Tiempo consumido en CUDA=%f", time);
printf("\n Tiempo consumido en CUDA=%f", time);
	// Free memory
        cudaFree(d_sums);
        cudaFree(d_seAcabo);
        cudaFree(d_continua);
        cudaFree(d_X0);
        cudaFree(d_X1);
        cudaFree(d_xi);
        cudaFree(d_Y);
        cudaFree(d_Z);

}
template <typename Typef,typename Typec, typename Typeu>
__host__ void SolveCUDA(Eigen::Vector2d X0,double T, double* boundary_parameters, double h,long long int N_tray, pfscalar g, int Nx, int Ny, Typef f, Typec c, bool VARC,Typeu ux, Typeu uy,
                        double* phi, double* phi2, double* phi3, double* phi4, double* phixi, double* phi_plus_xi2, double* xi, double* xi2, double* tau, double* tau2){
    bool EMFlag=false;
    //Calcular tiempo de ejecucións
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds=0;
    float time;
    //Time discretization of the trajectories and initial time of the trajectory T
    double sqrth= sqrt(h);
    double time_discretization=h,rho=0.2;
    //Number of trayectories
    long long int N=0; int added=NUMBLOCKS*BLOCKSIZE;
    bool seAcabo[2*NUMBLOCKS*BLOCKSIZE],dentro[NUMBLOCKS*BLOCKSIZE],cuenta[NUMBLOCKS*BLOCKSIZE];
    //Error objective
    double eps=1.0;
    //X0
      double X01=X0(0), X02=X0(1);
//	printf("\nX_1=%f, X_2=%f",X01,X02);
    //Errores
      cudaError_t errMalloc, errSync,errCpy;
      cudaError_t errMemset, errAsync;
    //Variables de device y copy
        double *d_sums, *d_X0, *d_X1, *d_Y, *d_Z, *d_xi, *d_ji_t;
        bool* d_seAcabo,*d_continua,*d_cuenta;
	const int sizeSum=23*BLOCKSIZE*NUMBLOCKS;
        double h_sums[sizeSum];
         bool seguir=true;
 //Variables auxiliares device
 cudaMalloc((void**) &d_X0,2*NUMBLOCKS*BLOCKSIZE*sizeof(double));
 cudaMalloc((void**) &d_X1,2*NUMBLOCKS*BLOCKSIZE*sizeof(double));
 cudaMalloc((void**) &d_Y,2*NUMBLOCKS*BLOCKSIZE*sizeof(double));
 cudaMalloc((void**) &d_Z,2*NUMBLOCKS*BLOCKSIZE*sizeof(double));
 cudaMalloc((void**) &d_xi,2*NUMBLOCKS*BLOCKSIZE*sizeof(double));
 cudaMalloc((void**) &d_ji_t,2*NUMBLOCKS*BLOCKSIZE*sizeof(double));
//Bucle en N_tray
     //Malloc e iniciacion variables device
       //DSUMS
           cudaMalloc((void**) &d_sums, sizeSum*sizeof(double));
           errMalloc= cudaGetLastError();
           cudaMemset(d_sums, 0, sizeSum * sizeof(double));
           errMemset= cudaGetLastError();
           if (errMalloc != cudaSuccess)
             printf("cudMalloc kernel error [d_sums]: %s\n", cudaGetErrorString(errMalloc));
           if (errMemset != cudaSuccess)
             printf("errMemset kernel error[d_sums]: %s\n", cudaGetErrorString(errMemset));
     //D_BOOL
       //seAcabo
           cudaMalloc((void**) &d_seAcabo,2*NUMBLOCKS*BLOCKSIZE*sizeof(bool));
           errMalloc= cudaGetLastError();
           if (errMalloc != cudaSuccess)
           printf("cudMalloc kernel error [d_seAcabo]: %s\n", cudaGetErrorString(errMalloc));
           cudaMemset(d_seAcabo, false, 2*NUMBLOCKS*BLOCKSIZE * sizeof(bool));
           errMemset= cudaGetLastError();
           if (errMemset != cudaSuccess)
           printf("cudaMemcpy error [d_seAcabo]: %s\n", cudaGetErrorString(errMemset));
     //La trayectoria está dentro o fuera
         cudaMalloc((void**) &d_continua,NUMBLOCKS*BLOCKSIZE*sizeof(bool));
           errMalloc= cudaGetLastError();
           if (errMalloc != cudaSuccess)
             printf("cudMalloc kernel error [d_continua]: %s\n", cudaGetErrorString(errMalloc));
             cudaMemset(d_continua, false, NUMBLOCKS*BLOCKSIZE*sizeof(bool));
             errMemset= cudaGetLastError();
             if (errMemset != cudaSuccess)
             printf("cudaMemcpy error [d_continua]: %s\n", cudaGetErrorString(errMemset));

       //La trayectoria cuenta
       cudaMalloc((void**) &d_cuenta,NUMBLOCKS*BLOCKSIZE*sizeof(bool));
         errMalloc= cudaGetLastError();
         if (errMalloc != cudaSuccess)
       printf("cudMalloc kernel error [d_cuenta]: %s\n", cudaGetErrorString(errMalloc));
     cudaMemset(d_cuenta, true, NUMBLOCKS*BLOCKSIZE*sizeof(bool));
     errMemset= cudaGetLastError();
     if (errMemset != cudaSuccess)
           printf("cudaMemcpy error [d_cuenta]: %s\n", cudaGetErrorString(errMemset));
while(seguir==true){
  /* code */
//do{    //Entramos en kernel
      cudaEventRecord(start);
        SolveLoop<<<NUMBLOCKS,BLOCKSIZE>>>(X01,X02,h,T,rho,sqrth,boundary_parameters[0],
                boundary_parameters[1],boundary_parameters[2],boundary_parameters[3],dev_curand_states,
                d_sums,d_seAcabo, d_continua,d_cuenta,d_X0, d_X1, d_Y, d_Z, d_xi, d_ji_t,N,N_tray,
                Nx,Ny,f, c, g,  false,VARC,ux,uy);
      errSync  = cudaGetLastError();
      errAsync = cudaDeviceSynchronize();
      cudaEventRecord(stop);
      if (errSync != cudaSuccess)
        printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
      if (errAsync != cudaSuccess)
        printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
//      printf("kernel ejecutado por primera vez\n" );
      N+=added;
     //CONTAR CON D_BOOL
//      cudaMemcpy(seAcabo, d_seAcabo, 2*NUMBLOCKS*BLOCKSIZE*sizeof(bool), cudaMemcpyDeviceToHost);
//      cudaMemcpy(cuenta, d_cuenta, NUMBLOCKS*BLOCKSIZE*sizeof(bool), cudaMemcpyDeviceToHost);
//      errCpy  = cudaGetLastError();
//      if (errCpy != cudaSuccess)
//      printf("cudaMemcpy error[bool]: %s\n", cudaGetErrorString(errCpy));
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&milliseconds, start, stop);
      time+=milliseconds;
      milliseconds=0;
      seguir=false;
      added=0;
//      for(int jjj=0;jjj<NUMBLOCKS*BLOCKSIZE;jjj++){
//        if(cuenta[jjj]==true){
//          seguir=true;
//          if(seAcabo[jjj]==true&&seAcabo[jjj+NUMBLOCKS*BLOCKSIZE]==true){
//            seAcabo[jjj]=false;
//            seAcabo[jjj+NUMBLOCKS*BLOCKSIZE]=false;
//            added++;
//      }}}
//      cudaMemcpy(d_seAcabo, seAcabo, 2*NUMBLOCKS*BLOCKSIZE*sizeof(bool), cudaMemcpyHostToDevice);
//////////////////////////
  //CONTAR CON continua
	int traye=0;
        cudaMemcpy(cuenta, d_cuenta, NUMBLOCKS*BLOCKSIZE*sizeof(bool), cudaMemcpyDeviceToHost);
        cudaMemcpy(dentro, d_continua, NUMBLOCKS*BLOCKSIZE*sizeof(bool), cudaMemcpyDeviceToHost);
              for(int jjj=0;jjj<NUMBLOCKS*BLOCKSIZE;jjj++){
                if(cuenta[jjj]==true){
                  seguir=true;
                  if(dentro[jjj]==false&& N<N_tray){
                    added++;
              }else{traye++;}}}
//	printf("\nTrayectorias activas=%i",traye);
}
cudaMemcpy(h_sums, d_sums, sizeSum*sizeof(double), cudaMemcpyDeviceToHost);
  *phi=0;
  *phi2=0;
  *phi3=0;
  *phi4=0;
  *phixi=0;
  *phi_plus_xi2=0;
  *xi=0;
  *xi2=0;
  *tau=0;
  *tau2=0;
for(int id=0; id<NUMBLOCKS*BLOCKSIZE;id++){
  *phi+=h_sums[23*id+0]/N;
  *phi2+=h_sums[23*id+2]/N;
  *phi3+=h_sums[23*id+17]/N;
  *phi4+=h_sums[23*id+18]/N;
  *phixi+=h_sums[23*id+12]/N;
  *phi_plus_xi2+=h_sums[23*id+10]/N;
  *xi+=h_sums[23*id+4]/N;
  *xi2+=h_sums[23*id+6]/N;
  *tau+=h_sums[23*id+15]/N;
  *tau2+=h_sums[23*id+19]/N;
}
errCpy  = cudaGetLastError();
if (errCpy != cudaSuccess)
printf("cudaMemcpy error: %s\n", cudaGetErrorString(errCpy));
//printf("\n %f;%f;%f;%f;%f;%f;%f;%f;%f;%f", *phi, *phi2, *phi3, *phi4, *phixi,*phi_plus_xi2,*xi,*xi2,*tau,*tau2);
//printf("\n N=%i, Ntray=%i", N, N_tray);
	// Free memory
        cudaFree(d_sums);
        cudaFree(d_seAcabo);
        cudaFree(d_continua);
        cudaFree(d_X0);
        cudaFree(d_X1);
        cudaFree(d_xi);
        cudaFree(d_Y);
        cudaFree(d_Z);

}
///////////////////
//////////////////
///////////////////
// Euler-MAruyama
///////////////
template <typename Typef,typename Typec>
__host__ void SolveCUDA(Eigen::Vector2d X0,double T, double* boundary_parameters, double h,long long int N_tray, pfscalar g, int Nx, int Ny, Typef f, Typec c, bool VARC,
                        double* phi, double* phi2, double* phi3, double* phi4, double* phixi, double* phi_plus_xi2, double* xi, double* xi2, double* tau, double* tau2,
                        double* phiEM, double* phi2EM, double* phi3EM, double* phi4EM, double* tauEM, double* tau2EM){
    bool EMFlag=false;
    //Calcular tiempo de ejecucións
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds=0;
    float time;

    //Time discretization of the trajectories and initial time of the trajectory T
    double sqrth= sqrt(h);
    double time_discretization=h,rho=0.2;
    //Number of trayectories
    long long int N=0; int added=NUMBLOCKS*BLOCKSIZE;
    bool seAcabo[2*NUMBLOCKS*BLOCKSIZE],dentro[NUMBLOCKS*BLOCKSIZE],cuenta[NUMBLOCKS*BLOCKSIZE];
    //Error objective
    double eps=1.0;
    //X0
      double X01=X0(0), X02=X0(1);
//	printf("\nX_1=%f, X_2=%f",X01,X02);
    //Errores
      cudaError_t errMalloc, errSync,errCpy;
      cudaError_t errMemset, errAsync;
    //Variables de device y copy
        double *d_sums, *d_X0, *d_X1, *d_Y, *d_Z, *d_xi, *d_ji_t;
        bool* d_seAcabo,*d_continua,*d_cuenta;
	const int sizeSum=23*BLOCKSIZE*NUMBLOCKS;
        double h_sums[sizeSum];
         bool seguir=true;
 //Variables auxiliares device
 cudaMalloc((void**) &d_X0,2*NUMBLOCKS*BLOCKSIZE*sizeof(double));
 cudaMalloc((void**) &d_X1,2*NUMBLOCKS*BLOCKSIZE*sizeof(double));
 cudaMalloc((void**) &d_Y,2*NUMBLOCKS*BLOCKSIZE*sizeof(double));
 cudaMalloc((void**) &d_Z,2*NUMBLOCKS*BLOCKSIZE*sizeof(double));
 cudaMalloc((void**) &d_xi,2*NUMBLOCKS*BLOCKSIZE*sizeof(double));
 cudaMalloc((void**) &d_ji_t,2*NUMBLOCKS*BLOCKSIZE*sizeof(double));
//Bucle en N_tray

     //Malloc e iniciacion variables device
       //DSUMS
           cudaMalloc((void**) &d_sums, sizeSum*sizeof(double));
           errMalloc= cudaGetLastError();
           cudaMemset(d_sums, 0, sizeSum * sizeof(double));
           errMemset= cudaGetLastError();
           if (errMalloc != cudaSuccess)
             printf("cudMalloc kernel error [d_sums]: %s\n", cudaGetErrorString(errMalloc));
           if (errMemset != cudaSuccess)
             printf("errMemset kernel error[d_sums]: %s\n", cudaGetErrorString(errMemset));
     //D_BOOL
       //seAcabo
           cudaMalloc((void**) &d_seAcabo,2*NUMBLOCKS*BLOCKSIZE*sizeof(bool));
           errMalloc= cudaGetLastError();
           if (errMalloc != cudaSuccess)
           printf("cudMalloc kernel error [d_seAcabo]: %s\n", cudaGetErrorString(errMalloc));
           cudaMemset(d_seAcabo, false, 2*NUMBLOCKS*BLOCKSIZE * sizeof(bool));
           errMemset= cudaGetLastError();
           if (errMemset != cudaSuccess)
           printf("cudaMemcpy error [d_seAcabo]: %s\n", cudaGetErrorString(errMemset));
     //La trayectoria está dentro o fuera
         cudaMalloc((void**) &d_continua,NUMBLOCKS*BLOCKSIZE*sizeof(bool));
           errMalloc= cudaGetLastError();
           if (errMalloc != cudaSuccess)
             printf("cudMalloc kernel error [d_continua]: %s\n", cudaGetErrorString(errMalloc));
             cudaMemset(d_continua, false, NUMBLOCKS*BLOCKSIZE*sizeof(bool));
             errMemset= cudaGetLastError();
             if (errMemset != cudaSuccess)
             printf("cudaMemcpy error [d_continua]: %s\n", cudaGetErrorString(errMemset));

       //La trayectoria cuenta
       cudaMalloc((void**) &d_cuenta,NUMBLOCKS*BLOCKSIZE*sizeof(bool));
         errMalloc= cudaGetLastError();
         if (errMalloc != cudaSuccess)
       printf("cudMalloc kernel error [d_cuenta]: %s\n", cudaGetErrorString(errMalloc));
     cudaMemset(d_cuenta, true, NUMBLOCKS*BLOCKSIZE*sizeof(bool));
     errMemset= cudaGetLastError();
     if (errMemset != cudaSuccess)
           printf("cudaMemcpy error [d_cuenta]: %s\n", cudaGetErrorString(errMemset));
while(seguir==true){
  /* code */
//do{    //Entramos en kernel
      cudaEventRecord(start);
        SolveLoop<<<NUMBLOCKS,BLOCKSIZE>>>(X01,X02,h,T,rho,sqrth,boundary_parameters[0],
                boundary_parameters[1],boundary_parameters[2],boundary_parameters[3],dev_curand_states,
                d_sums,d_seAcabo, d_continua,d_cuenta,d_X0, d_X1, d_Y, d_Z, d_xi, d_ji_t,N,N_tray,
                Nx,Ny,f, c, g,  false,VARC);
      errSync  = cudaGetLastError();
      errAsync = cudaDeviceSynchronize();
      cudaEventRecord(stop);
      if (errSync != cudaSuccess)
        printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
      if (errAsync != cudaSuccess)
        printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
//      printf("kernel ejecutado por primera vez\n" );
      N+=added;
     //CONTAR CON D_BOOL
//      cudaMemcpy(seAcabo, d_seAcabo, 2*NUMBLOCKS*BLOCKSIZE*sizeof(bool), cudaMemcpyDeviceToHost);
//      cudaMemcpy(cuenta, d_cuenta, NUMBLOCKS*BLOCKSIZE*sizeof(bool), cudaMemcpyDeviceToHost);
//      errCpy  = cudaGetLastError();
//      if (errCpy != cudaSuccess)
//      printf("cudaMemcpy error[bool]: %s\n", cudaGetErrorString(errCpy));
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&milliseconds, start, stop);
      time+=milliseconds;
      milliseconds=0;
      seguir=false;
      added=0;
//      for(int jjj=0;jjj<NUMBLOCKS*BLOCKSIZE;jjj++){
//        if(cuenta[jjj]==true){
//          seguir=true;
//          if(seAcabo[jjj]==true&&seAcabo[jjj+NUMBLOCKS*BLOCKSIZE]==true){
//            seAcabo[jjj]=false;
//            seAcabo[jjj+NUMBLOCKS*BLOCKSIZE]=false;
//            added++;
//      }}}
//      cudaMemcpy(d_seAcabo, seAcabo, 2*NUMBLOCKS*BLOCKSIZE*sizeof(bool), cudaMemcpyHostToDevice);
//////////////////////////
  //CONTAR CON continua
	int traye=0;
        cudaMemcpy(cuenta, d_cuenta, NUMBLOCKS*BLOCKSIZE*sizeof(bool), cudaMemcpyDeviceToHost);
        cudaMemcpy(dentro, d_continua, NUMBLOCKS*BLOCKSIZE*sizeof(bool), cudaMemcpyDeviceToHost);
              for(int jjj=0;jjj<NUMBLOCKS*BLOCKSIZE;jjj++){
                if(cuenta[jjj]==true){
                  seguir=true;
                  if(dentro[jjj]==false&& N<N_tray){
                    added++;
              }else{traye++;}}}
//	printf("\nTrayectorias activas=%i",traye);
}
cudaMemcpy(h_sums, d_sums, sizeSum*sizeof(double), cudaMemcpyDeviceToHost);
*phi=0;
*phi2=0;
*phi3=0;
*phi4=0;
*phixi=0;
*phi_plus_xi2=0;
*xi=0;
*xi2=0;
*tau=0;
*tau2=0;
*phiEM=0;
*phi2EM=0;
*phi3EM=0;
*phi4EM=0;
*tauEM=0;
*tau2EM=0;
for(int id=0; id<NUMBLOCKS*BLOCKSIZE;id++){
*phi+=h_sums[23*id+0]/N;
*phi2+=h_sums[23*id+2]/N;
*phi3+=h_sums[23*id+17]/N;
*phi4+=h_sums[23*id+18]/N;
*phixi+=h_sums[23*id+12]/N;
*phi_plus_xi2+=h_sums[23*id+10]/N;
*xi+=h_sums[23*id+4]/N;
*xi2+=h_sums[23*id+6]/N;
*tau+=h_sums[23*id+15]/N;
*tau2+=h_sums[23*id+19]/N;
*phiEM+=h_sums[23*id+1]/N;
*phi2EM+=h_sums[23*id+3]/N;
*phi3EM+=h_sums[23*id+20]/N ;
*phi4EM+=h_sums[23*id+21]/N ;
*tauEM+= h_sums[23*id+16]/N ;
*tau2EM+= h_sums[23*id+22]/N ;
}
errCpy  = cudaGetLastError();
if (errCpy != cudaSuccess)
printf("cudaMemcpy error: %s\n", cudaGetErrorString(errCpy));
//printf("\n %f;%f;%f;%f;%f;%f;%f;%f;%f;%f", *phi, *phi2, *phi3, *phi4, *phixi,*phi_plus_xi2,*xi,*xi2,*tau,*tau2);
//printf("\n N=%i, Ntray=%i", N, N_tray);
	// Free memory
        cudaFree(d_sums);
        cudaFree(d_seAcabo);
        cudaFree(d_continua);
        cudaFree(d_X0);
        cudaFree(d_X1);
        cudaFree(d_xi);
        cudaFree(d_Y);
        cudaFree(d_Z);

}
template <typename Typef,typename Typec, typename Typeu>
__host__ void SolveCUDA(Eigen::Vector2d X0,double T, double* boundary_parameters, double h,long long int N_tray, pfscalar g, int Nx, int Ny, Typef f, Typec c, bool VARC,Typeu ux, Typeu uy,
                        double* phi, double* phi2, double* phi3, double* phi4, double* phixi, double* phi_plus_xi2, double* xi, double* xi2, double* tau, double* tau2,
                        double* phiEM, double* phi2EM, double* phi3EM, double* phi4EM, double* tauEM, double* tau2EM){
    bool EMFlag=false;
    //Calcular tiempo de ejecucións
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds=0;
    float time;

    //Time discretization of the trajectories and initial time of the trajectory T
    double sqrth= sqrt(h);
    double time_discretization=h,rho=0.2;
    //Number of trayectories
    long long int N=0; int added=NUMBLOCKS*BLOCKSIZE;
    bool seAcabo[2*NUMBLOCKS*BLOCKSIZE],dentro[NUMBLOCKS*BLOCKSIZE],cuenta[NUMBLOCKS*BLOCKSIZE];
    //Error objective
    double eps=1.0;
    //X0
      double X01=X0(0), X02=X0(1);
//	printf("\nX_1=%f, X_2=%f",X01,X02);
    //Errores
      cudaError_t errMalloc, errSync,errCpy;
      cudaError_t errMemset, errAsync;
    //Variables de device y copy
        double *d_sums, *d_X0, *d_X1, *d_Y, *d_Z, *d_xi, *d_ji_t;
        bool* d_seAcabo,*d_continua,*d_cuenta;
	const int sizeSum=23*BLOCKSIZE*NUMBLOCKS;
        double h_sums[sizeSum];
         bool seguir=true;
 //Variables auxiliares device
 cudaMalloc((void**) &d_X0,2*NUMBLOCKS*BLOCKSIZE*sizeof(double));
 cudaMalloc((void**) &d_X1,2*NUMBLOCKS*BLOCKSIZE*sizeof(double));
 cudaMalloc((void**) &d_Y,2*NUMBLOCKS*BLOCKSIZE*sizeof(double));
 cudaMalloc((void**) &d_Z,2*NUMBLOCKS*BLOCKSIZE*sizeof(double));
 cudaMalloc((void**) &d_xi,2*NUMBLOCKS*BLOCKSIZE*sizeof(double));
 cudaMalloc((void**) &d_ji_t,2*NUMBLOCKS*BLOCKSIZE*sizeof(double));
//Bucle en N_tray
     //Malloc e iniciacion variables device
       //DSUMS
           cudaMalloc((void**) &d_sums, sizeSum*sizeof(double));
           errMalloc= cudaGetLastError();
           cudaMemset(d_sums, 0, sizeSum * sizeof(double));
           errMemset= cudaGetLastError();
           if (errMalloc != cudaSuccess)
             printf("cudMalloc kernel error [d_sums]: %s\n", cudaGetErrorString(errMalloc));
           if (errMemset != cudaSuccess)
             printf("errMemset kernel error[d_sums]: %s\n", cudaGetErrorString(errMemset));
     //D_BOOL
       //seAcabo
           cudaMalloc((void**) &d_seAcabo,2*NUMBLOCKS*BLOCKSIZE*sizeof(bool));
           errMalloc= cudaGetLastError();
           if (errMalloc != cudaSuccess)
           printf("cudMalloc kernel error [d_seAcabo]: %s\n", cudaGetErrorString(errMalloc));
           cudaMemset(d_seAcabo, false, 2*NUMBLOCKS*BLOCKSIZE * sizeof(bool));
           errMemset= cudaGetLastError();
           if (errMemset != cudaSuccess)
           printf("cudaMemcpy error [d_seAcabo]: %s\n", cudaGetErrorString(errMemset));
     //La trayectoria está dentro o fuera
         cudaMalloc((void**) &d_continua,NUMBLOCKS*BLOCKSIZE*sizeof(bool));
           errMalloc= cudaGetLastError();
           if (errMalloc != cudaSuccess)
             printf("cudMalloc kernel error [d_continua]: %s\n", cudaGetErrorString(errMalloc));
             cudaMemset(d_continua, false, NUMBLOCKS*BLOCKSIZE*sizeof(bool));
             errMemset= cudaGetLastError();
             if (errMemset != cudaSuccess)
             printf("cudaMemcpy error [d_continua]: %s\n", cudaGetErrorString(errMemset));

       //La trayectoria cuenta
       cudaMalloc((void**) &d_cuenta,NUMBLOCKS*BLOCKSIZE*sizeof(bool));
         errMalloc= cudaGetLastError();
         if (errMalloc != cudaSuccess)
       printf("cudMalloc kernel error [d_cuenta]: %s\n", cudaGetErrorString(errMalloc));
     cudaMemset(d_cuenta, true, NUMBLOCKS*BLOCKSIZE*sizeof(bool));
     errMemset= cudaGetLastError();
     if (errMemset != cudaSuccess)
           printf("cudaMemcpy error [d_cuenta]: %s\n", cudaGetErrorString(errMemset));
while(seguir==true){
  /* code */
//do{    //Entramos en kernel
      cudaEventRecord(start);
        SolveLoop<<<NUMBLOCKS,BLOCKSIZE>>>(X01,X02,h,T,rho,sqrth,boundary_parameters[0],
                boundary_parameters[1],boundary_parameters[2],boundary_parameters[3],dev_curand_states,
                d_sums,d_seAcabo, d_continua,d_cuenta,d_X0, d_X1, d_Y, d_Z, d_xi, d_ji_t,N,N_tray,
                Nx,Ny,f, c, g,  false,VARC,ux,uy);
      errSync  = cudaGetLastError();
      errAsync = cudaDeviceSynchronize();
      cudaEventRecord(stop);
      if (errSync != cudaSuccess)
        printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
      if (errAsync != cudaSuccess)
        printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
//      printf("kernel ejecutado por primera vez\n" );
      N+=added;
     //CONTAR CON D_BOOL
//      cudaMemcpy(seAcabo, d_seAcabo, 2*NUMBLOCKS*BLOCKSIZE*sizeof(bool), cudaMemcpyDeviceToHost);
//      cudaMemcpy(cuenta, d_cuenta, NUMBLOCKS*BLOCKSIZE*sizeof(bool), cudaMemcpyDeviceToHost);
//      errCpy  = cudaGetLastError();
//      if (errCpy != cudaSuccess)
//      printf("cudaMemcpy error[bool]: %s\n", cudaGetErrorString(errCpy));
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&milliseconds, start, stop);
      time+=milliseconds;
      milliseconds=0;
      seguir=false;
      added=0;
//      for(int jjj=0;jjj<NUMBLOCKS*BLOCKSIZE;jjj++){
//        if(cuenta[jjj]==true){
//          seguir=true;
//          if(seAcabo[jjj]==true&&seAcabo[jjj+NUMBLOCKS*BLOCKSIZE]==true){
//            seAcabo[jjj]=false;
//            seAcabo[jjj+NUMBLOCKS*BLOCKSIZE]=false;
//            added++;
//      }}}
//      cudaMemcpy(d_seAcabo, seAcabo, 2*NUMBLOCKS*BLOCKSIZE*sizeof(bool), cudaMemcpyHostToDevice);
//////////////////////////
  //CONTAR CON continua
	int traye=0;
        cudaMemcpy(cuenta, d_cuenta, NUMBLOCKS*BLOCKSIZE*sizeof(bool), cudaMemcpyDeviceToHost);
        cudaMemcpy(dentro, d_continua, NUMBLOCKS*BLOCKSIZE*sizeof(bool), cudaMemcpyDeviceToHost);
              for(int jjj=0;jjj<NUMBLOCKS*BLOCKSIZE;jjj++){
                if(cuenta[jjj]==true){
                  seguir=true;
                  if(dentro[jjj]==false&& N<N_tray){
                    added++;
              }else{traye++;}}}
//	printf("\nTrayectorias activas=%i",traye);
}
cudaMemcpy(h_sums, d_sums, sizeSum*sizeof(double), cudaMemcpyDeviceToHost);
  *phi=0;
  *phi2=0;
  *phi3=0;
  *phi4=0;
  *phixi=0;
  *phi_plus_xi2=0;
  *xi=0;
  *xi2=0;
  *tau=0;
  *tau2=0;
  *phiEM=0;
  *phi2EM=0;
  *phi3EM=0;
  *phi4EM=0;
  *tauEM=0;
  *tau2EM=0;
for(int id=0; id<NUMBLOCKS*BLOCKSIZE;id++){
  *phi+=h_sums[23*id+0]/N;
  *phi2+=h_sums[23*id+2]/N;
  *phi3+=h_sums[23*id+17]/N;
  *phi4+=h_sums[23*id+18]/N;
  *phixi+=h_sums[23*id+12]/N;
  *phi_plus_xi2+=h_sums[23*id+10]/N;
  *xi+=h_sums[23*id+4]/N;
  *xi2+=h_sums[23*id+6]/N;
  *tau+=h_sums[23*id+15]/N;
  *tau2+=h_sums[23*id+19]/N;
  *phiEM+=h_sums[23*id+1]/N;
  *phi2EM+=h_sums[23*id+3]/N;
  *phi3EM+=h_sums[23*id+20]/N ;
  *phi4EM+=h_sums[23*id+21]/N ;
  *tauEM+= h_sums[23*id+16]/N ;
  *tau2EM+= h_sums[23*id+22]/N ;
}
errCpy  = cudaGetLastError();
if (errCpy != cudaSuccess)
printf("cudaMemcpy error: %s\n", cudaGetErrorString(errCpy));
//printf("\n %f;%f;%f;%f;%f;%f;%f;%f;%f;%f", *phi, *phi2, *phi3, *phi4, *phixi,*phi_plus_xi2,*xi,*xi2,*tau,*tau2);
//printf("\n N=%i, Ntray=%i", N, N_tray);
	// Free memory
        cudaFree(d_sums);

        cudaFree(d_seAcabo);
        cudaFree(d_continua);
        cudaFree(d_X0);
        cudaFree(d_X1);
        cudaFree(d_xi);
        cudaFree(d_Y);
        cudaFree(d_Z);

}




/////////////
/////////////
/////////////
///MODO XYZ
template <typename Typef,typename Typec>
__host__ void SolveCUDA(Eigen::Vector2d X0,double T, double* boundary_parameters, double h,long long int N_tray, int Nx, int Ny, Typef f, Typec c,bool VARC,
  double* X_tau_lin_1, double* X_tau_lin_2, double* Y_tau_lin, double* Z_tau_lin){
  bool EMFlag=false;
  //Calcular tiempo de ejecucións
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float milliseconds=0;
  float time;

  //Time discretization of the trajectories and initial time of the trajectory T
  double sqrth= sqrt(h);
  double time_discretization=h,rho=0.2;
  //Number of trayectories
  long long int N=0;
  bool seAcabo[2*NUMBLOCKS*BLOCKSIZE],dentro[NUMBLOCKS*BLOCKSIZE],cuenta[NUMBLOCKS*BLOCKSIZE];
  //Error objective
  double eps=1.0;
  //X0
    double X01=X0(0), X02=X0(1);
  //Errores
    cudaError_t errMalloc, errSync,errCpy;
    cudaError_t errMemset, errAsync;
  //Variables de device y copy
      double *d_sums, *d_X0, *d_X1, *d_Y, *d_Z, *d_xi, *d_ji_t;
      bool* d_seAcabo,*d_continua,*d_cuenta;
       const int sizeSum=23*BLOCKSIZE*NUMBLOCKS;
       double h_sums[sizeSum];
       bool seguir=true;
//Variables auxiliares device
cudaMalloc((void**) &d_X0,2*NUMBLOCKS*BLOCKSIZE*sizeof(double));
cudaMalloc((void**) &d_X1,2*NUMBLOCKS*BLOCKSIZE*sizeof(double));
cudaMalloc((void**) &d_Y,2*NUMBLOCKS*BLOCKSIZE*sizeof(double));
cudaMalloc((void**) &d_Z,2*NUMBLOCKS*BLOCKSIZE*sizeof(double));
cudaMalloc((void**) &d_xi,2*NUMBLOCKS*BLOCKSIZE*sizeof(double));
cudaMalloc((void**) &d_ji_t,2*NUMBLOCKS*BLOCKSIZE*sizeof(double));
int added= BLOCKSIZE*NUMBLOCKS;
//Bucle en N_tray

     //Malloc e iniciacion variables device
       //DSUMS
           cudaMalloc((void**) &d_sums, sizeSum*sizeof(double));
           errMalloc= cudaGetLastError();
           cudaMemset(d_sums, 0, sizeSum * sizeof(double));
           errMemset= cudaGetLastError();
           if (errMalloc != cudaSuccess)
             printf("cudMalloc kernel error [d_sums]: %s\n", cudaGetErrorString(errMalloc));
           if (errMemset != cudaSuccess)
             printf("errMemset kernel error[d_sums]: %s\n", cudaGetErrorString(errMemset));
     //D_BOOL
       //seAcabo
           cudaMalloc((void**) &d_seAcabo,2*NUMBLOCKS*BLOCKSIZE*sizeof(bool));
           errMalloc= cudaGetLastError();
           if (errMalloc != cudaSuccess)
           printf("cudMalloc kernel error [d_seAcabo]: %s\n", cudaGetErrorString(errMalloc));
           cudaMemset(d_seAcabo, false, 2*NUMBLOCKS*BLOCKSIZE * sizeof(bool));
           errMemset= cudaGetLastError();
           if (errMemset != cudaSuccess)
           printf("cudaMemcpy error [d_seAcabo]: %s\n", cudaGetErrorString(errMemset));
     //La trayectoria está dentro o fuera
         cudaMalloc((void**) &d_continua,NUMBLOCKS*BLOCKSIZE*sizeof(bool));
           errMalloc= cudaGetLastError();
           if (errMalloc != cudaSuccess)
             printf("cudMalloc kernel error [d_continua]: %s\n", cudaGetErrorString(errMalloc));
             cudaMemset(d_continua, false, NUMBLOCKS*BLOCKSIZE*sizeof(bool));
             errMemset= cudaGetLastError();
             if (errMemset != cudaSuccess)
             printf("cudaMemcpy error [d_continua]: %s\n", cudaGetErrorString(errMemset));

       //La trayectoria cuenta
       cudaMalloc((void**) &d_cuenta,NUMBLOCKS*BLOCKSIZE*sizeof(bool));
         errMalloc= cudaGetLastError();
         if (errMalloc != cudaSuccess)
       printf("cudMalloc kernel error [d_cuenta]: %s\n", cudaGetErrorString(errMalloc));
     cudaMemset(d_cuenta, true, NUMBLOCKS*BLOCKSIZE*sizeof(bool));
     errMemset= cudaGetLastError();
     if (errMemset != cudaSuccess)
           printf("cudaMemcpy error [d_cuenta]: %s\n", cudaGetErrorString(errMemset));
 int place=0;
//Bucle en N_tray
while(seguir==true){
  /* code */
//do{    //Entramos en kernel
      cudaEventRecord(start);
      SolveLoop<<<NUMBLOCKS,BLOCKSIZE>>>(X01,X02,h,T,rho,sqrth,boundary_parameters[0],
              boundary_parameters[1],boundary_parameters[2],boundary_parameters[3],dev_curand_states,
              d_sums,d_seAcabo, d_continua,d_cuenta,d_X0, d_X1, d_Y, d_Z, d_xi, d_ji_t,N,N_tray,
              Nx,Ny,f, c,  false,VARC);
      errSync  = cudaGetLastError();
      errAsync = cudaDeviceSynchronize();
      cudaEventRecord(stop);
      if (errSync != cudaSuccess)
        printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
      if (errAsync != cudaSuccess)
        printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
//      printf("kernel ejecutado por primera vez\n" );
      N+=added;
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&milliseconds, start, stop);
      time+=milliseconds;
      milliseconds=0;
      seguir=false;
      added=0;
  //CONTAR CON continua
        cudaMemcpy(cuenta, d_cuenta, NUMBLOCKS*BLOCKSIZE*sizeof(bool), cudaMemcpyDeviceToHost);
        cudaMemcpy(dentro, d_continua, NUMBLOCKS*BLOCKSIZE*sizeof(bool), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_sums, d_sums, sizeSum*sizeof(double), cudaMemcpyDeviceToHost);
        errCpy  = cudaGetLastError();
        if (errCpy != cudaSuccess)
        printf("cudaMemcpy error: %s\n", cudaGetErrorString(errCpy));
///////////////////////////////////////////
        //TAMAÑO OUTPUT=5+8*N_TRAY
///////////////////////////////////////////
              for(int jjj=0;jjj<NUMBLOCKS*BLOCKSIZE;jjj++){
                //RNGcalls=h_sums[13*jjj+0]+(jjj>0)*RNGcalls;
                //tau_lin=h_sums[13*jjj+1]+(jjj>0)*tau_lin;
                //tau_lin2=h_sums[13*jjj+3]+(jjj>0)*tau_lin2;
                if(cuenta[jjj]==true&&(place-1)<N_tray){
                  seguir=true;
                  X_tau_lin_1[place]=h_sums[13*jjj+5];
                  X_tau_lin_2[place]=h_sums[13*jjj+6];
                  Y_tau_lin[place]=h_sums[13*jjj+7];
                  Z_tau_lin[place]=h_sums[13*jjj+8];
                  place++;
                  if(dentro[jjj]==false&& (N-1)<N_tray){
                    added++;
              }}}
}

	// Free memory
        cudaFree(d_sums);

        cudaFree(d_seAcabo);
        cudaFree(d_continua);
        cudaFree(d_X0);
        cudaFree(d_X1);
        cudaFree(d_xi);
        cudaFree(d_Y);
        cudaFree(d_Z);

}
template <typename Typef,typename Typec, typename Typeu>
__host__ void SolveCUDA(Eigen::Vector2d X0,double T, double* boundary_parameters, double h,long long int N_tray, int Nx, int Ny, Typef f, Typec c,bool VARC, Typeu ux, Typeu uy,
                          double* X_tau_lin_1, double* X_tau_lin_2, double* Y_tau_lin, double* Z_tau_lin){
  bool EMFlag=false;
  //Calcular tiempo de ejecucións
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float milliseconds=0;
  float time;

  //Time discretization of the trajectories and initial time of the trajectory T
  double sqrth= sqrt(h);
  double time_discretization=h,rho=0.2;
  //Number of trayectories
  long long int N=0;
  bool seAcabo[2*NUMBLOCKS*BLOCKSIZE],dentro[NUMBLOCKS*BLOCKSIZE],cuenta[NUMBLOCKS*BLOCKSIZE];
  //Error objective
  double eps=1.0;
  //X0
    double X01=X0(0), X02=X0(1);
  //Errores
    cudaError_t errMalloc, errSync,errCpy;
    cudaError_t errMemset, errAsync;
  //Variables de device y copy
      double *d_sums, *d_X0, *d_X1, *d_Y, *d_Z, *d_xi, *d_ji_t;
      bool* d_seAcabo,*d_continua,*d_cuenta;
       const int sizeSum=23*BLOCKSIZE*NUMBLOCKS;
       double h_sums[sizeSum];
       bool seguir=true;
//Variables auxiliares device
cudaMalloc((void**) &d_X0,2*NUMBLOCKS*BLOCKSIZE*sizeof(double));
cudaMalloc((void**) &d_X1,2*NUMBLOCKS*BLOCKSIZE*sizeof(double));
cudaMalloc((void**) &d_Y,2*NUMBLOCKS*BLOCKSIZE*sizeof(double));
cudaMalloc((void**) &d_Z,2*NUMBLOCKS*BLOCKSIZE*sizeof(double));
cudaMalloc((void**) &d_xi,2*NUMBLOCKS*BLOCKSIZE*sizeof(double));
cudaMalloc((void**) &d_ji_t,2*NUMBLOCKS*BLOCKSIZE*sizeof(double));
int added= BLOCKSIZE*NUMBLOCKS;
//Bucle en N_tray

     //Malloc e iniciacion variables device
       //DSUMS
           cudaMalloc((void**) &d_sums, sizeSum*sizeof(double));
           errMalloc= cudaGetLastError();
           cudaMemset(d_sums, 0, sizeSum * sizeof(double));
           errMemset= cudaGetLastError();
           if (errMalloc != cudaSuccess)
             printf("cudMalloc kernel error [d_sums]: %s\n", cudaGetErrorString(errMalloc));
           if (errMemset != cudaSuccess)
             printf("errMemset kernel error[d_sums]: %s\n", cudaGetErrorString(errMemset));
     //D_BOOL
       //seAcabo
           cudaMalloc((void**) &d_seAcabo,2*NUMBLOCKS*BLOCKSIZE*sizeof(bool));
           errMalloc= cudaGetLastError();
           if (errMalloc != cudaSuccess)
           printf("cudMalloc kernel error [d_seAcabo]: %s\n", cudaGetErrorString(errMalloc));
           cudaMemset(d_seAcabo, false, 2*NUMBLOCKS*BLOCKSIZE * sizeof(bool));
           errMemset= cudaGetLastError();
           if (errMemset != cudaSuccess)
           printf("cudaMemcpy error [d_seAcabo]: %s\n", cudaGetErrorString(errMemset));
     //La trayectoria está dentro o fuera
         cudaMalloc((void**) &d_continua,NUMBLOCKS*BLOCKSIZE*sizeof(bool));
           errMalloc= cudaGetLastError();
           if (errMalloc != cudaSuccess)
             printf("cudMalloc kernel error [d_continua]: %s\n", cudaGetErrorString(errMalloc));
             cudaMemset(d_continua, false, NUMBLOCKS*BLOCKSIZE*sizeof(bool));
             errMemset= cudaGetLastError();
             if (errMemset != cudaSuccess)
             printf("cudaMemcpy error [d_continua]: %s\n", cudaGetErrorString(errMemset));

       //La trayectoria cuenta
       cudaMalloc((void**) &d_cuenta,NUMBLOCKS*BLOCKSIZE*sizeof(bool));
         errMalloc= cudaGetLastError();
         if (errMalloc != cudaSuccess)
       printf("cudMalloc kernel error [d_cuenta]: %s\n", cudaGetErrorString(errMalloc));
     cudaMemset(d_cuenta, true, NUMBLOCKS*BLOCKSIZE*sizeof(bool));
     errMemset= cudaGetLastError();
     if (errMemset != cudaSuccess)
           printf("cudaMemcpy error [d_cuenta]: %s\n", cudaGetErrorString(errMemset));
 int place=0;
//Bucle en N_tray
while(seguir==true){
  /* code */
//do{    //Entramos en kernel
      cudaEventRecord(start);
      SolveLoop<<<NUMBLOCKS,BLOCKSIZE>>>(X01,X02,h,T,rho,sqrth,boundary_parameters[0],
              boundary_parameters[1],boundary_parameters[2],boundary_parameters[3],dev_curand_states,
              d_sums,d_seAcabo, d_continua,d_cuenta,d_X0, d_X1, d_Y, d_Z, d_xi, d_ji_t,N,N_tray,
              Nx,Ny,f, c, false,VARC,ux,uy);
      errSync  = cudaGetLastError();
      errAsync = cudaDeviceSynchronize();
      cudaEventRecord(stop);
      if (errSync != cudaSuccess)
        printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
      if (errAsync != cudaSuccess)
        printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
//      printf("kernel ejecutado por primera vez\n" );
      N+=added;
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&milliseconds, start, stop);
      time+=milliseconds;
      milliseconds=0;
      seguir=false;
      added=0;
  //CONTAR CON continua
        cudaMemcpy(cuenta, d_cuenta, NUMBLOCKS*BLOCKSIZE*sizeof(bool), cudaMemcpyDeviceToHost);
        cudaMemcpy(dentro, d_continua, NUMBLOCKS*BLOCKSIZE*sizeof(bool), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_sums, d_sums, sizeSum*sizeof(double), cudaMemcpyDeviceToHost);
        errCpy  = cudaGetLastError();
        if (errCpy != cudaSuccess)
        printf("cudaMemcpy error: %s\n", cudaGetErrorString(errCpy));
///////////////////////////////////////////
        //TAMAÑO OUTPUT=5+8*N_TRAY
///////////////////////////////////////////
              for(int jjj=0;jjj<NUMBLOCKS*BLOCKSIZE;jjj++){
                //RNGcalls=h_sums[13*jjj+0]+(jjj>0)*RNGcalls;
                //tau_lin=h_sums[13*jjj+1]+(jjj>0)*tau_lin;
                //tau_lin2=h_sums[13*jjj+3]+(jjj>0)*tau_lin2;
                if(cuenta[jjj]==true&&(place-1)<N_tray){
                  seguir=true;
                  X_tau_lin_1[place]=h_sums[13*jjj+5];
                  X_tau_lin_2[place]=h_sums[13*jjj+6];
                  Y_tau_lin[place]=h_sums[13*jjj+7];
                  Z_tau_lin[place]=h_sums[13*jjj+8];
                  place++;
                  if(dentro[jjj]==false&& (N-1)<N_tray){
                    added++;
              }}}
}

	// Free memory
        cudaFree(d_sums);

        cudaFree(d_seAcabo);
        cudaFree(d_continua);
        cudaFree(d_X0);
        cudaFree(d_X1);
        cudaFree(d_xi);
        cudaFree(d_Y);
        cudaFree(d_Z);

}
///MODO XYZ
template <typename Typef,typename Typec>
__host__ void SolveCUDA(Eigen::Vector2d X0,double T, double* boundary_parameters, double h,long long int N_tray, int Nx, int Ny, Typef f, Typec c, bool VARC,
                          double* X_tau_lin_1, double* X_tau_lin_2, double* Y_tau_lin, double* Z_tau_lin,
                          double* X_tau_sublin_1, double* X_tau_sublin_2, double* Y_tau_sublin, double* Z_tau_sublin){
  bool EMFlag=true;
  //Calcular tiempo de ejecucións
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float milliseconds=0;
  float time;

  //Time discretization of the trajectories and initial time of the trajectory T
  double sqrth= sqrt(h);
  double time_discretization=h,rho=0.2;
  //Number of trayectories
  long long int N=0;
  bool seAcabo[2*NUMBLOCKS*BLOCKSIZE],dentro[NUMBLOCKS*BLOCKSIZE],cuenta[NUMBLOCKS*BLOCKSIZE];
  //Error objective
  double eps=1.0;
  //X0
    double X01=X0(0), X02=X0(1);
  //Errores
    cudaError_t errMalloc, errSync,errCpy;
    cudaError_t errMemset, errAsync;
  //Variables de device y copy
      double *d_sums, *d_X0, *d_X1, *d_Y, *d_Z, *d_xi, *d_ji_t;
      bool* d_seAcabo,*d_continua,*d_cuenta;
       const int sizeSum=23*BLOCKSIZE*NUMBLOCKS;
       bool seguir=true;
       double h_sums[sizeSum];

//Variables auxiliares device
cudaMalloc((void**) &d_X0,2*NUMBLOCKS*BLOCKSIZE*sizeof(double));
cudaMalloc((void**) &d_X1,2*NUMBLOCKS*BLOCKSIZE*sizeof(double));
cudaMalloc((void**) &d_Y,2*NUMBLOCKS*BLOCKSIZE*sizeof(double));
cudaMalloc((void**) &d_Z,2*NUMBLOCKS*BLOCKSIZE*sizeof(double));
cudaMalloc((void**) &d_xi,2*NUMBLOCKS*BLOCKSIZE*sizeof(double));
cudaMalloc((void**) &d_ji_t,2*NUMBLOCKS*BLOCKSIZE*sizeof(double));
int added= BLOCKSIZE*NUMBLOCKS;
//Bucle en N_tray

     //Malloc e iniciacion variables device
       //DSUMS
           cudaMalloc((void**) &d_sums, sizeSum*sizeof(double));
           errMalloc= cudaGetLastError();
           cudaMemset(d_sums, 0, sizeSum * sizeof(double));
           errMemset= cudaGetLastError();
           if (errMalloc != cudaSuccess)
             printf("cudMalloc kernel error [d_sums]: %s\n", cudaGetErrorString(errMalloc));
           if (errMemset != cudaSuccess)
             printf("errMemset kernel error[d_sums]: %s\n", cudaGetErrorString(errMemset));
     //D_BOOL
       //seAcabo
           cudaMalloc((void**) &d_seAcabo,2*NUMBLOCKS*BLOCKSIZE*sizeof(bool));
           errMalloc= cudaGetLastError();
           if (errMalloc != cudaSuccess)
           printf("cudMalloc kernel error [d_seAcabo]: %s\n", cudaGetErrorString(errMalloc));
           cudaMemset(d_seAcabo, false, 2*NUMBLOCKS*BLOCKSIZE * sizeof(bool));
           errMemset= cudaGetLastError();
           if (errMemset != cudaSuccess)
           printf("cudaMemcpy error [d_seAcabo]: %s\n", cudaGetErrorString(errMemset));
     //La trayectoria está dentro o fuera
         cudaMalloc((void**) &d_continua,NUMBLOCKS*BLOCKSIZE*sizeof(bool));
           errMalloc= cudaGetLastError();
           if (errMalloc != cudaSuccess)
             printf("cudMalloc kernel error [d_continua]: %s\n", cudaGetErrorString(errMalloc));
             cudaMemset(d_continua, false, NUMBLOCKS*BLOCKSIZE*sizeof(bool));
             errMemset= cudaGetLastError();
             if (errMemset != cudaSuccess)
             printf("cudaMemcpy error [d_continua]: %s\n", cudaGetErrorString(errMemset));

       //La trayectoria cuenta
       cudaMalloc((void**) &d_cuenta,NUMBLOCKS*BLOCKSIZE*sizeof(bool));
         errMalloc= cudaGetLastError();
         if (errMalloc != cudaSuccess)
       printf("cudMalloc kernel error [d_cuenta]: %s\n", cudaGetErrorString(errMalloc));
     cudaMemset(d_cuenta, true, NUMBLOCKS*BLOCKSIZE*sizeof(bool));
     errMemset= cudaGetLastError();
     if (errMemset != cudaSuccess)
           printf("cudaMemcpy error [d_cuenta]: %s\n", cudaGetErrorString(errMemset));
 int place=0;
//Bucle en N_tray
while(seguir==true){
  /* code */
//do{    //Entramos en kernel
      cudaEventRecord(start);
      SolveLoop<<<NUMBLOCKS,BLOCKSIZE>>>(X01,X02,h,T,rho,sqrth,boundary_parameters[0],
              boundary_parameters[1],boundary_parameters[2],boundary_parameters[3],dev_curand_states,
              d_sums,d_seAcabo, d_continua,d_cuenta,d_X0, d_X1, d_Y, d_Z, d_xi, d_ji_t,N,N_tray,
              Nx,Ny,f, c,  false,VARC);
      errSync  = cudaGetLastError();
      errAsync = cudaDeviceSynchronize();
      cudaEventRecord(stop);
      if (errSync != cudaSuccess)
        printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
      if (errAsync != cudaSuccess)
        printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
//      printf("kernel ejecutado por primera vez\n" );
      N+=added;
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&milliseconds, start, stop);
      time+=milliseconds;
      milliseconds=0;
      seguir=false;
      added=0;
  //CONTAR CON continua
        cudaMemcpy(cuenta, d_cuenta, NUMBLOCKS*BLOCKSIZE*sizeof(bool), cudaMemcpyDeviceToHost);
        cudaMemcpy(dentro, d_continua, NUMBLOCKS*BLOCKSIZE*sizeof(bool), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_sums, d_sums, sizeSum*sizeof(double), cudaMemcpyDeviceToHost);
        errCpy  = cudaGetLastError();
        if (errCpy != cudaSuccess)
        printf("cudaMemcpy error: %s\n", cudaGetErrorString(errCpy));
///////////////////////////////////////////
        //TAMAÑO OUTPUT=5+8*N_TRAY
///////////////////////////////////////////
              for(int jjj=0;jjj<NUMBLOCKS*BLOCKSIZE;jjj++){
                //RNGcalls=h_sums[13*jjj+0]+(jjj>0)*RNGcalls;
                //tau_lin=h_sums[13*jjj+1]+(jjj>0)*tau_lin;
                //tau_sublin=h_sums[13*jjj+2]+(jjj>0)*tau_sublin;
                //tau_lin2=h_sums[13*jjj+3]+(jjj>0)*tau_lin2;
                //tau_sublin2=h_sums[13*jjj+4]+(jjj>0)*tau_sublin2;
                if(cuenta[jjj]==true&&(place-1)<N_tray){
                  seguir=true;
                  X_tau_lin_1[place]=h_sums[13*jjj+5];
                  X_tau_lin_2[place]=h_sums[13*jjj+6];
                  Y_tau_lin[place]=h_sums[13*jjj+7];
                  Z_tau_lin[place]=h_sums[13*jjj+8];
                  X_tau_sublin_1[place]=h_sums[13*jjj+9];
                  X_tau_sublin_2[place]=h_sums[13*jjj+10];
                  Y_tau_sublin[place]=h_sums[13*jjj+11];
                  Z_tau_sublin[place]=  h_sums[13*jjj+12];
                  place++;
                  if(dentro[jjj]==false&& (N-1)<N_tray){
                    added++;
              }}}
}

	// Free memory
        cudaFree(d_sums);

        cudaFree(d_seAcabo);
        cudaFree(d_continua);
        cudaFree(d_X0);
        cudaFree(d_X1);
        cudaFree(d_xi);
        cudaFree(d_Y);
        cudaFree(d_Z);
}
///MODO XYZ
template <typename Typef,typename Typec, typename Typeu>
__host__ void SolveCUDA(Eigen::Vector2d X0,double T, double* boundary_parameters, double h,long long int N_tray, int Nx, int Ny, Typef f, Typec c,bool VARC, Typeu ux, Typeu uy,
                          double* X_tau_lin_1, double* X_tau_lin_2, double* Y_tau_lin, double* Z_tau_lin,
                          double* X_tau_sublin_1, double* X_tau_sublin_2, double* Y_tau_sublin, double* Z_tau_sublin){
  bool EMFlag=true;
  //Calcular tiempo de ejecucións
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float milliseconds=0;
  float time;

  //Time discretization of the trajectories and initial time of the trajectory T
  double sqrth= sqrt(h);
  double time_discretization=h,rho=0.2;
  //Number of trayectories
  long long int N=0;
  bool seAcabo[2*NUMBLOCKS*BLOCKSIZE],dentro[NUMBLOCKS*BLOCKSIZE],cuenta[NUMBLOCKS*BLOCKSIZE];
  //Error objective
  double eps=1.0;
  //X0
    double X01=X0(0), X02=X0(1);
  //Errores
    cudaError_t errMalloc, errSync,errCpy;
    cudaError_t errMemset, errAsync;
  //Variables de device y copy
      double *d_sums, *d_X0, *d_X1, *d_Y, *d_Z, *d_xi, *d_ji_t;
      bool* d_seAcabo,*d_continua,*d_cuenta;
       const int sizeSum=23*BLOCKSIZE*NUMBLOCKS;
       bool seguir=true;
       double h_sums[sizeSum];

//Variables auxiliares device
cudaMalloc((void**) &d_X0,2*NUMBLOCKS*BLOCKSIZE*sizeof(double));
cudaMalloc((void**) &d_X1,2*NUMBLOCKS*BLOCKSIZE*sizeof(double));
cudaMalloc((void**) &d_Y,2*NUMBLOCKS*BLOCKSIZE*sizeof(double));
cudaMalloc((void**) &d_Z,2*NUMBLOCKS*BLOCKSIZE*sizeof(double));
cudaMalloc((void**) &d_xi,2*NUMBLOCKS*BLOCKSIZE*sizeof(double));
cudaMalloc((void**) &d_ji_t,2*NUMBLOCKS*BLOCKSIZE*sizeof(double));
int added= BLOCKSIZE*NUMBLOCKS;
//Bucle en N_tray

     //Malloc e iniciacion variables device
       //DSUMS
           cudaMalloc((void**) &d_sums, sizeSum*sizeof(double));
           errMalloc= cudaGetLastError();
           cudaMemset(d_sums, 0, sizeSum * sizeof(double));
           errMemset= cudaGetLastError();
           if (errMalloc != cudaSuccess)
             printf("cudMalloc kernel error [d_sums]: %s\n", cudaGetErrorString(errMalloc));
           if (errMemset != cudaSuccess)
             printf("errMemset kernel error[d_sums]: %s\n", cudaGetErrorString(errMemset));
     //D_BOOL
       //seAcabo
           cudaMalloc((void**) &d_seAcabo,2*NUMBLOCKS*BLOCKSIZE*sizeof(bool));
           errMalloc= cudaGetLastError();
           if (errMalloc != cudaSuccess)
           printf("cudMalloc kernel error [d_seAcabo]: %s\n", cudaGetErrorString(errMalloc));
           cudaMemset(d_seAcabo, false, 2*NUMBLOCKS*BLOCKSIZE * sizeof(bool));
           errMemset= cudaGetLastError();
           if (errMemset != cudaSuccess)
           printf("cudaMemcpy error [d_seAcabo]: %s\n", cudaGetErrorString(errMemset));
     //La trayectoria está dentro o fuera
         cudaMalloc((void**) &d_continua,NUMBLOCKS*BLOCKSIZE*sizeof(bool));
           errMalloc= cudaGetLastError();
           if (errMalloc != cudaSuccess)
             printf("cudMalloc kernel error [d_continua]: %s\n", cudaGetErrorString(errMalloc));
             cudaMemset(d_continua, false, NUMBLOCKS*BLOCKSIZE*sizeof(bool));
             errMemset= cudaGetLastError();
             if (errMemset != cudaSuccess)
             printf("cudaMemcpy error [d_continua]: %s\n", cudaGetErrorString(errMemset));

       //La trayectoria cuenta
       cudaMalloc((void**) &d_cuenta,NUMBLOCKS*BLOCKSIZE*sizeof(bool));
         errMalloc= cudaGetLastError();
         if (errMalloc != cudaSuccess)
       printf("cudMalloc kernel error [d_cuenta]: %s\n", cudaGetErrorString(errMalloc));
     cudaMemset(d_cuenta, true, NUMBLOCKS*BLOCKSIZE*sizeof(bool));
     errMemset= cudaGetLastError();
     if (errMemset != cudaSuccess)
           printf("cudaMemcpy error [d_cuenta]: %s\n", cudaGetErrorString(errMemset));
 int place=0;
//Bucle en N_tray
while(seguir==true){
  /* code */
//do{    //Entramos en kernel
      cudaEventRecord(start);
      SolveLoop<<<NUMBLOCKS,BLOCKSIZE>>>(X01,X02,h,T,rho,sqrth,boundary_parameters[0],
              boundary_parameters[1],boundary_parameters[2],boundary_parameters[3],dev_curand_states,
              d_sums,d_seAcabo, d_continua,d_cuenta,d_X0, d_X1, d_Y, d_Z, d_xi, d_ji_t,N,N_tray,
              Nx,Ny,f, c,  false,VARC,ux,uy);
      errSync  = cudaGetLastError();
      errAsync = cudaDeviceSynchronize();
      cudaEventRecord(stop);
      if (errSync != cudaSuccess)
        printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
      if (errAsync != cudaSuccess)
        printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
//      printf("kernel ejecutado por primera vez\n" );
      N+=added;
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&milliseconds, start, stop);
      time+=milliseconds;
      milliseconds=0;
      seguir=false;
      added=0;
  //CONTAR CON continua
        cudaMemcpy(cuenta, d_cuenta, NUMBLOCKS*BLOCKSIZE*sizeof(bool), cudaMemcpyDeviceToHost);
        cudaMemcpy(dentro, d_continua, NUMBLOCKS*BLOCKSIZE*sizeof(bool), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_sums, d_sums, sizeSum*sizeof(double), cudaMemcpyDeviceToHost);
        errCpy  = cudaGetLastError();
        if (errCpy != cudaSuccess)
        printf("cudaMemcpy error: %s\n", cudaGetErrorString(errCpy));
///////////////////////////////////////////
        //TAMAÑO OUTPUT=5+8*N_TRAY
///////////////////////////////////////////
              for(int jjj=0;jjj<NUMBLOCKS*BLOCKSIZE;jjj++){
                ///RNGcalls=h_sums[13*jjj+0]+(jjj>0)*RNGcalls;
                ///tau_lin=h_sums[13*jjj+1]+(jjj>0)*tau_lin;
                ///tau_sublin=h_sums[13*jjj+2]+(jjj>0)*tau_sublin;
                ///tau_lin2=h_sums[13*jjj+3]+(jjj>0)*tau_lin2;
                ///tau_sublin2=h_sums[13*jjj+4]+(jjj>0)*tau_sublin2;
                if(cuenta[jjj]==true&&(place-1)<N_tray){
                  seguir=true;
                  X_tau_lin_1[place]=h_sums[13*jjj+5];
                  X_tau_lin_2[place]=h_sums[13*jjj+6];
                  Y_tau_lin[place]=h_sums[13*jjj+7];
                  Z_tau_lin[place]=h_sums[13*jjj+8];
                  X_tau_sublin_1[place]=h_sums[13*jjj+9];
                  X_tau_sublin_2[place]=h_sums[13*jjj+10];
                  Y_tau_sublin[place]=h_sums[13*jjj+11];
                  Z_tau_sublin[place]=  h_sums[13*jjj+12];
                  place++;
                  if(dentro[jjj]==false&& (N-1)<N_tray){
                    added++;
              }}}
}

	// Free memory
        cudaFree(d_sums);

        cudaFree(d_seAcabo);
        cudaFree(d_continua);
        cudaFree(d_X0);
        cudaFree(d_X1);
        cudaFree(d_xi);
        cudaFree(d_Y);
        cudaFree(d_Z);
}
