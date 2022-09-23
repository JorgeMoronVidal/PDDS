#include "solveCUDA.cuh"
#include "../BVPs/LUT2.cuh"
cudaTextureObject_t tex_g,tex_f,tex_c, tex_ux, tex_uy;
cudaArray_t* cuArray_g, cuArray_f, cuArray_c, cuArray_ux, cuArray_uy;
__device__ pfscalar f_ = Equation_dev_f;
__device__ pfscalar c_ = Equation_dev_c;
__device__ pfscalar g_ = Equation_dev_g;

__host__ void MCinCUDA(int deviceId,int RNGinit,int texMode, int seed,Eigen::Vector2d X0,double T, double* boundary_parameters, double h,long long int N_tray, int Nx, int Ny, bool VARC,
                        double* phi, double* phi2, double* phi3, double* phi4, double* phixi, double* phi_plus_xi2, double* xi, double* xi2, double* tau, double* tau2){
  cudaSetDevice(deviceId);if (RNGinit==0){initRNGCuda(seed);}
  pfscalar g;
  cudaMemcpyFromSymbol( &g, g_, sizeof(pfscalar));
  pfscalar f;
  cudaMemcpyFromSymbol( &f, f_, sizeof(pfscalar));
  pfscalar c;
  cudaMemcpyFromSymbol( &c, c_, sizeof(pfscalar));
  switch (texMode) {
    case 0:
    SolveCUDA(X0,T, boundary_parameters, h, N_tray, g, Nx, Ny, f, c, VARC,
                             phi,  phi2, phi3,  phi4, phixi, phi_plus_xi2, xi, xi2, tau, tau2);
    break;
case 1:
    SolveCUDA(X0,T, boundary_parameters, h, N_tray, g, Nx, Ny, tex_f, c, VARC,
                            phi,  phi2, phi3,  phi4, phixi, phi_plus_xi2, xi, xi2, tau, tau2);
    break;
case 2:
    SolveCUDA(X0,T, boundary_parameters, h, N_tray, g, Nx, Ny, f, tex_c, VARC,
                            phi,  phi2, phi3,  phi4, phixi, phi_plus_xi2, xi, xi2, tau, tau2);
        break;
case 3:
    SolveCUDA(X0,T, boundary_parameters, h, N_tray, g, Nx, Ny, tex_f, tex_c, VARC,
                                                    phi,  phi2, phi3,  phi4, phixi, phi_plus_xi2, xi, xi2, tau, tau2);
    break;
case 4:
    SolveCUDA(X0,T, boundary_parameters, h, N_tray, g, Nx, Ny, tex_f, tex_c, VARC, tex_ux, tex_uy,
    phi,  phi2, phi3,  phi4, phixi, phi_plus_xi2, xi, xi2, tau, tau2);
    break;

  }

}


__host__ void MCinCUDA(int deviceId,int RNGinit,int texMode,  int firstArrType, float* firstArr,int seed,Eigen::Vector2d X0,double T, double* boundary_parameters, double h,long long int N_tray, int Nx, int Ny, bool VARC,
                        double* phi, double* phi2, double* phi3, double* phi4, double* phixi, double* phi_plus_xi2, double* xi, double* xi2, double* tau, double* tau2) {
  cudaSetDevice(deviceId);if (RNGinit==0){initRNGCuda(seed);}
  pfscalar g;
  cudaMemcpyFromSymbol( &g, g_, sizeof(pfscalar));
  pfscalar f;
  cudaMemcpyFromSymbol( &f, f_, sizeof(pfscalar));
  pfscalar c;
  cudaMemcpyFromSymbol( &c, c_, sizeof(pfscalar));
  switch (firstArrType) {
    case 0:
      Init_tex_LUT(Nx,Ny,firstArr,tex_f, cuArray_f);
   break;
case 1:
      Init_tex_LUT(Nx,Ny,firstArr,tex_c, cuArray_c);
  }
  switch (texMode) {
    break;
case 0:
    SolveCUDA(X0,T, boundary_parameters, h, N_tray, g, Nx, Ny, f, c, VARC,
                             phi,  phi2, phi3,  phi4, phixi, phi_plus_xi2, xi, xi2, tau, tau2);
    break;
case 1:
    SolveCUDA(X0,T, boundary_parameters, h, N_tray, g, Nx, Ny, tex_f, c, VARC,
                            phi,  phi2, phi3,  phi4, phixi, phi_plus_xi2, xi, xi2, tau, tau2);
    break;
case 2:
    SolveCUDA(X0,T, boundary_parameters, h, N_tray, g, Nx, Ny, f, tex_c, VARC,
                            phi,  phi2, phi3,  phi4, phixi, phi_plus_xi2, xi, xi2, tau, tau2);
    break;
case 3:
    SolveCUDA(X0,T, boundary_parameters, h, N_tray, g, Nx, Ny, tex_f, tex_c, VARC,
                                                    phi,  phi2, phi3,  phi4, phixi, phi_plus_xi2, xi, xi2, tau, tau2);
    break;
case 4:
    SolveCUDA(X0,T, boundary_parameters, h, N_tray, g, Nx, Ny, tex_f, tex_c, VARC, tex_ux, tex_uy,
    phi,  phi2, phi3,  phi4, phixi, phi_plus_xi2, xi, xi2, tau, tau2);
    break;

  }
}

__host__ void MCinCUDA(int deviceId,int RNGinit, int texMode, int firstArrType, float* firstArr, float* secArr,int seed,Eigen::Vector2d X0,double T, double* boundary_parameters, double h,long long int N_tray, int Nx, int Ny, bool VARC,
                        double* phi, double* phi2, double* phi3, double* phi4, double* phixi, double* phi_plus_xi2, double* xi, double* xi2, double* tau, double* tau2){
  cudaSetDevice(deviceId);if (RNGinit==0){initRNGCuda(seed);}
  pfscalar g;
  cudaMemcpyFromSymbol( &g, g_, sizeof(pfscalar));
  pfscalar f;
  cudaMemcpyFromSymbol( &f, f_, sizeof(pfscalar));
  pfscalar c;
  cudaMemcpyFromSymbol( &c, c_, sizeof(pfscalar));
  switch (firstArrType) {
    case 0:
      Init_tex_LUT(Nx,Ny,firstArr,tex_f, cuArray_f);
      Init_tex_LUT(Nx,Ny,secArr,tex_c, cuArray_c);
    break;
case 1:
    Init_tex_LUT(Nx,Ny,firstArr,tex_ux, cuArray_ux);
    Init_tex_LUT(Nx,Ny,secArr,tex_uy, cuArray_uy);
  }
  switch (texMode) {
    break;
case 0:
    SolveCUDA(X0,T, boundary_parameters, h, N_tray, g, Nx, Ny, f, c, VARC,
                             phi,  phi2, phi3,  phi4, phixi, phi_plus_xi2, xi, xi2, tau, tau2);
    break;
case 1:
    SolveCUDA(X0,T, boundary_parameters, h, N_tray, g, Nx, Ny, tex_f, c, VARC,
                            phi,  phi2, phi3,  phi4, phixi, phi_plus_xi2, xi, xi2, tau, tau2);
    break;
case 2:
    SolveCUDA(X0,T, boundary_parameters, h, N_tray, g, Nx, Ny, f, tex_c, VARC,
                            phi,  phi2, phi3,  phi4, phixi, phi_plus_xi2, xi, xi2, tau, tau2);
    break;
case 3:
    SolveCUDA(X0,T, boundary_parameters, h, N_tray, g, Nx, Ny, tex_f, tex_c, VARC,
                                                    phi,  phi2, phi3,  phi4, phixi, phi_plus_xi2, xi, xi2, tau, tau2);
    break;
case 4:
    SolveCUDA(X0,T, boundary_parameters, h, N_tray, g, Nx, Ny, tex_f, tex_c, VARC, tex_ux, tex_uy,
    phi,  phi2, phi3,  phi4, phixi, phi_plus_xi2, xi, xi2, tau, tau2);
    break;

  }
}

__host__ void MCinCUDA(int deviceId,int RNGinit, float* f_arr, float* c_arr, float* ux_arr,  float* uy_arr,int seed,Eigen::Vector2d X0,double T, double* boundary_parameters, double h,long long int N_tray, int Nx, int Ny, bool VARC,
                        double* phi, double* phi2, double* phi3, double* phi4, double* phixi, double* phi_plus_xi2, double* xi, double* xi2, double* tau, double* tau2){
  cudaSetDevice(deviceId);if (RNGinit==0){initRNGCuda(seed);}
  pfscalar g;
  cudaMemcpyFromSymbol( &g, g_, sizeof(pfscalar));
      Init_tex_LUT(Nx,Ny,f_arr,tex_f, cuArray_f);
      Init_tex_LUT(Nx,Ny,c_arr,tex_c, cuArray_c);
    Init_tex_LUT(Nx,Ny,ux_arr,tex_ux, cuArray_ux);
    Init_tex_LUT(Nx,Ny,uy_arr,tex_uy, cuArray_uy);
  SolveCUDA(X0,T, boundary_parameters, h, N_tray, g, Nx, Ny, tex_f, tex_c, VARC, tex_ux, tex_uy,
                           phi,  phi2, phi3,  phi4, phixi, phi_plus_xi2, xi, xi2, tau, tau2);
}
/////
/////
/// Score with EM
__host__ void MCinCUDA(int deviceId,int RNGinit,int texMode,int seed,Eigen::Vector2d X0,double T, double* boundary_parameters, double h,long long int N_tray, int Nx, int Ny, bool VARC,
                        double* phi, double* phi2, double* phi3, double* phi4, double* phixi, double* phi_plus_xi2, double* xi, double* xi2, double* tau, double* tau2,
                        double* phiMC, double* phi2MC, double* phi3MC, double* phi4MC, double* tauMC, double* tau2MC){

  cudaSetDevice(deviceId);if (RNGinit==0){initRNGCuda(seed);}
  pfscalar g;
  cudaMemcpyFromSymbol( &g, g_, sizeof(pfscalar));
  pfscalar f;
  cudaMemcpyFromSymbol( &f, f_, sizeof(pfscalar));
  pfscalar c;
  cudaMemcpyFromSymbol( &c, c_, sizeof(pfscalar));
  switch (texMode) {
    case 0:
    SolveCUDA(X0,T, boundary_parameters, h, N_tray, g, Nx, Ny, f, c, VARC,
                             phi,  phi2, phi3,  phi4, phixi, phi_plus_xi2, xi, xi2, tau, tau2,
                           phiMC, phi2MC, phi3MC, phi4MC, tauMC, tau2MC);
    break;
case 1:
    SolveCUDA(X0,T, boundary_parameters, h, N_tray, g, Nx, Ny, tex_f, c, VARC,
      phi,  phi2, phi3,  phi4, phixi, phi_plus_xi2, xi, xi2, tau, tau2,
    phiMC, phi2MC, phi3MC, phi4MC, tauMC, tau2MC);
    break;
case 2:
    SolveCUDA(X0,T, boundary_parameters, h, N_tray, g, Nx, Ny, f, tex_c, VARC,
      phi,  phi2, phi3,  phi4, phixi, phi_plus_xi2, xi, xi2, tau, tau2,
    phiMC, phi2MC, phi3MC, phi4MC, tauMC, tau2MC);
    break;
case 3:
    SolveCUDA(X0,T, boundary_parameters, h, N_tray, g, Nx, Ny, tex_f, tex_c, VARC,
      phi,  phi2, phi3,  phi4, phixi, phi_plus_xi2, xi, xi2, tau, tau2,
    phiMC, phi2MC, phi3MC, phi4MC, tauMC, tau2MC);
    break;
case 4:
    SolveCUDA(X0,T, boundary_parameters, h, N_tray, g, Nx, Ny, tex_f, tex_c, VARC, tex_ux, tex_uy,
      phi,  phi2, phi3,  phi4, phixi, phi_plus_xi2, xi, xi2, tau, tau2,
    phiMC, phi2MC, phi3MC, phi4MC, tauMC, tau2MC);
    break;
  }

}


__host__ void MCinCUDA(int deviceId,int RNGinit, int texMode, int firstArrType, float* firstArr,int seed,Eigen::Vector2d X0,double T, double* boundary_parameters, double h,long long int N_tray, int Nx, int Ny, bool VARC,
  double* phi, double* phi2, double* phi3, double* phi4, double* phixi, double* phi_plus_xi2, double* xi, double* xi2, double* tau, double* tau2,
  double* phiMC, double* phi2MC, double* phi3MC, double* phi4MC, double* tauMC, double* tau2MC){
  cudaSetDevice(deviceId);if (RNGinit==0){initRNGCuda(seed);}
  pfscalar g;
  cudaMemcpyFromSymbol( &g, g_, sizeof(pfscalar));
  pfscalar f;
  cudaMemcpyFromSymbol( &f, f_, sizeof(pfscalar));
  pfscalar c;
  cudaMemcpyFromSymbol( &c, c_, sizeof(pfscalar));
  switch (firstArrType) {
    case 0:
      Init_tex_LUT(Nx,Ny,firstArr,tex_f, cuArray_f);
   break;
case 1:
      Init_tex_LUT(Nx,Ny,firstArr,tex_c, cuArray_c);
      break;

  }
  switch (texMode) {
    case 0:
    SolveCUDA(X0,T, boundary_parameters, h, N_tray, g, Nx, Ny, f, c, VARC,
      phi,  phi2, phi3,  phi4, phixi, phi_plus_xi2, xi, xi2, tau, tau2,
    phiMC, phi2MC, phi3MC, phi4MC, tauMC, tau2MC);
    break;
case 1:
    SolveCUDA(X0,T, boundary_parameters, h, N_tray, g, Nx, Ny, tex_f, c, VARC,
      phi,  phi2, phi3,  phi4, phixi, phi_plus_xi2, xi, xi2, tau, tau2,
    phiMC, phi2MC, phi3MC, phi4MC, tauMC, tau2MC);
    break;
case 2:
    SolveCUDA(X0,T, boundary_parameters, h, N_tray, g, Nx, Ny, f, tex_c, VARC,
      phi,  phi2, phi3,  phi4, phixi, phi_plus_xi2, xi, xi2, tau, tau2,
    phiMC, phi2MC, phi3MC, phi4MC, tauMC, tau2MC);
    break;
case 3:
    SolveCUDA(X0,T, boundary_parameters, h, N_tray, g, Nx, Ny, tex_f, tex_c, VARC,
      phi,  phi2, phi3,  phi4, phixi, phi_plus_xi2, xi, xi2, tau, tau2,
    phiMC, phi2MC, phi3MC, phi4MC, tauMC, tau2MC);
    break;
case 4:
    SolveCUDA(X0,T, boundary_parameters, h, N_tray, g, Nx, Ny, tex_f, tex_c, VARC, tex_ux, tex_uy,
      phi,  phi2, phi3,  phi4, phixi, phi_plus_xi2, xi, xi2, tau, tau2,
    phiMC, phi2MC, phi3MC, phi4MC, tauMC, tau2MC);
    break;

  }
}

__host__ void MCinCUDA(int deviceId,int RNGinit, int texMode, int firstArrType, float* firstArr, float* secArr,int seed,Eigen::Vector2d X0,double T, double* boundary_parameters, double h,long long int N_tray, int Nx, int Ny, bool VARC,
  double* phi, double* phi2, double* phi3, double* phi4, double* phixi, double* phi_plus_xi2, double* xi, double* xi2, double* tau, double* tau2,
  double* phiMC, double* phi2MC, double* phi3MC, double* phi4MC, double* tauMC, double* tau2MC){
  cudaSetDevice(deviceId);if (RNGinit==0){initRNGCuda(seed);}
  pfscalar g;
  cudaMemcpyFromSymbol( &g, g_, sizeof(pfscalar));
  pfscalar f;
  cudaMemcpyFromSymbol( &f, f_, sizeof(pfscalar));
  pfscalar c;
  cudaMemcpyFromSymbol( &c, c_, sizeof(pfscalar));
  switch (firstArrType) {
    case 0:
      Init_tex_LUT(Nx,Ny,firstArr,tex_f, cuArray_f);
      Init_tex_LUT(Nx,Ny,secArr,tex_c, cuArray_c);
    break;
case 1:
    Init_tex_LUT(Nx,Ny,firstArr,tex_ux, cuArray_ux);
    Init_tex_LUT(Nx,Ny,secArr,tex_uy, cuArray_uy);
    break;

  }
  switch (texMode) {
    case 0:
    SolveCUDA(X0,T, boundary_parameters, h, N_tray, g, Nx, Ny, f, c, VARC,
      phi,  phi2, phi3,  phi4, phixi, phi_plus_xi2, xi, xi2, tau, tau2,
    phiMC, phi2MC, phi3MC, phi4MC, tauMC, tau2MC);
    break;
case 1:
    SolveCUDA(X0,T, boundary_parameters, h, N_tray, g, Nx, Ny, tex_f, c, VARC,
      phi,  phi2, phi3,  phi4, phixi, phi_plus_xi2, xi, xi2, tau, tau2,
    phiMC, phi2MC, phi3MC, phi4MC, tauMC, tau2MC);
    break;
case 2:
    SolveCUDA(X0,T, boundary_parameters, h, N_tray, g, Nx, Ny, f, tex_c, VARC,
      phi,  phi2, phi3,  phi4, phixi, phi_plus_xi2, xi, xi2, tau, tau2,
    phiMC, phi2MC, phi3MC, phi4MC, tauMC, tau2MC);
    break;
case 3:
    SolveCUDA(X0,T, boundary_parameters, h, N_tray, g, Nx, Ny, tex_f, tex_c, VARC,
      phi,  phi2, phi3,  phi4, phixi, phi_plus_xi2, xi, xi2, tau, tau2,
    phiMC, phi2MC, phi3MC, phi4MC, tauMC, tau2MC);
    break;
case 4:
    SolveCUDA(X0,T, boundary_parameters, h, N_tray, g, Nx, Ny, tex_f, tex_c, VARC, tex_ux, tex_uy,
      phi,  phi2, phi3,  phi4, phixi, phi_plus_xi2, xi, xi2, tau, tau2,
    phiMC, phi2MC, phi3MC, phi4MC, tauMC, tau2MC);
    break;

  }
}

__host__ void MCinCUDA(int deviceId,int RNGinit, float* f_arr, float* c_arr, float* ux_arr,  float* uy_arr,int seed,Eigen::Vector2d X0,double T, double* boundary_parameters, double h,long long int N_tray, int Nx, int Ny, bool VARC,
  double* phi, double* phi2, double* phi3, double* phi4, double* phixi, double* phi_plus_xi2, double* xi, double* xi2, double* tau, double* tau2,
  double* phiMC, double* phi2MC, double* phi3MC, double* phi4MC, double* tauMC, double* tau2MC){
  cudaSetDevice(deviceId);if (RNGinit==0){initRNGCuda(seed);}
  pfscalar g;
  cudaMemcpyFromSymbol( &g, g_, sizeof(pfscalar));
      Init_tex_LUT(Nx,Ny,f_arr,tex_f, cuArray_f);
      Init_tex_LUT(Nx,Ny,c_arr,tex_c, cuArray_c);
    Init_tex_LUT(Nx,Ny,ux_arr,tex_ux, cuArray_ux);
    Init_tex_LUT(Nx,Ny,uy_arr,tex_uy, cuArray_uy);
  SolveCUDA(X0,T, boundary_parameters, h, N_tray, g, Nx, Ny, tex_f, tex_c, VARC, tex_ux, tex_uy,
    phi,  phi2, phi3,  phi4, phixi, phi_plus_xi2, xi, xi2, tau, tau2,
  phiMC, phi2MC, phi3MC, phi4MC, tauMC, tau2MC);
}////
////
//XYZ MODE
__host__ void MCinCUDA(int deviceId,int RNGinit,int texMode, int seed,Eigen::Vector2d X0,double T, double* boundary_parameters, double h,long long int N_tray, int Nx, int Ny, bool VARC,
                        double* phi, double* phi2, double* phi3, double* phi4, double* phixi, double* phi_plus_xi2, double* xi, double* xi2, double* tau, double* tau2){
  cudaSetDevice(deviceId);if (RNGinit==0){initRNGCuda(seed);}
  pfscalar g;
  cudaMemcpyFromSymbol( &g, g_, sizeof(pfscalar));
  pfscalar f;
  cudaMemcpyFromSymbol( &f, f_, sizeof(pfscalar));
  pfscalar c;
  cudaMemcpyFromSymbol( &c, c_, sizeof(pfscalar));
  switch (texMode) {
    case 0:
    SolveCUDA(X0,T, boundary_parameters, h, N_tray, g, Nx, Ny, f, c, VARC,
                            X_tau_lin_1, X_tau_lin_2, Y_tau_lin, Z_tau_lin);
    break;
case 1:
    SolveCUDA(X0,T, boundary_parameters, h, N_tray, g, Nx, Ny, tex_f, c, VARC,
                           X_tau_lin_1, X_tau_lin_2, Y_tau_lin, Z_tau_lin);
    break;
case 2:
    SolveCUDA(X0,T, boundary_parameters, h, N_tray, g, Nx, Ny, f, tex_c, VARC,
                           X_tau_lin_1, X_tau_lin_2, Y_tau_lin, Z_tau_lin);
        break;
case 3:
    SolveCUDA(X0,T, boundary_parameters, h, N_tray, g, Nx, Ny, tex_f, tex_c, VARC,
                                                   X_tau_lin_1, X_tau_lin_2, Y_tau_lin, Z_tau_lin);
    break;
case 4:
    SolveCUDA(X0,T, boundary_parameters, h, N_tray, g, Nx, Ny, tex_f, tex_c, VARC, tex_ux, tex_uy,
   X_tau_lin_1, X_tau_lin_2, Y_tau_lin, Z_tau_lin);
    break;

  }

}


__host__ void MCinCUDA(int deviceId,int RNGinit,int texMode,  int firstArrType, float* firstArr,int seed,Eigen::Vector2d X0,double T, double* boundary_parameters, double h,long long int N_tray, int Nx, int Ny, bool VARC,
                        double* phi, double* phi2, double* phi3, double* phi4, double* phixi, double* phi_plus_xi2, double* xi, double* xi2, double* tau, double* tau2) {
  cudaSetDevice(deviceId);if (RNGinit==0){initRNGCuda(seed);}
  pfscalar g;
  cudaMemcpyFromSymbol( &g, g_, sizeof(pfscalar));
  pfscalar f;
  cudaMemcpyFromSymbol( &f, f_, sizeof(pfscalar));
  pfscalar c;
  cudaMemcpyFromSymbol( &c, c_, sizeof(pfscalar));
  switch (firstArrType) {
    case 0:
      Init_tex_LUT(Nx,Ny,firstArr,tex_f, cuArray_f);
   break;
case 1:
      Init_tex_LUT(Nx,Ny,firstArr,tex_c, cuArray_c);
  }
  switch (texMode) {
    break;
case 0:
    SolveCUDA(X0,T, boundary_parameters, h, N_tray, g, Nx, Ny, f, c, VARC,
                            X_tau_lin_1, X_tau_lin_2, Y_tau_lin, Z_tau_lin);
    break;
case 1:
    SolveCUDA(X0,T, boundary_parameters, h, N_tray, g, Nx, Ny, tex_f, c, VARC,
                           X_tau_lin_1, X_tau_lin_2, Y_tau_lin, Z_tau_lin);
    break;
case 2:
    SolveCUDA(X0,T, boundary_parameters, h, N_tray, g, Nx, Ny, f, tex_c, VARC,
                           X_tau_lin_1, X_tau_lin_2, Y_tau_lin, Z_tau_lin);
    break;
case 3:
    SolveCUDA(X0,T, boundary_parameters, h, N_tray, g, Nx, Ny, tex_f, tex_c, VARC,
                                                   X_tau_lin_1, X_tau_lin_2, Y_tau_lin, Z_tau_lin);
    break;
case 4:
    SolveCUDA(X0,T, boundary_parameters, h, N_tray, g, Nx, Ny, tex_f, tex_c, VARC, tex_ux, tex_uy,
   X_tau_lin_1, X_tau_lin_2, Y_tau_lin, Z_tau_lin);
    break;

  }
}

__host__ void MCinCUDA(int deviceId,int RNGinit, int texMode, int firstArrType, float* firstArr, float* secArr,int seed,Eigen::Vector2d X0,double T, double* boundary_parameters, double h,long long int N_tray, int Nx, int Ny, bool VARC,
                        double* phi, double* phi2, double* phi3, double* phi4, double* phixi, double* phi_plus_xi2, double* xi, double* xi2, double* tau, double* tau2){
  cudaSetDevice(deviceId);if (RNGinit==0){initRNGCuda(seed);}
  pfscalar g;
  cudaMemcpyFromSymbol( &g, g_, sizeof(pfscalar));
  pfscalar f;
  cudaMemcpyFromSymbol( &f, f_, sizeof(pfscalar));
  pfscalar c;
  cudaMemcpyFromSymbol( &c, c_, sizeof(pfscalar));
  switch (firstArrType) {
    case 0:
      Init_tex_LUT(Nx,Ny,firstArr,tex_f, cuArray_f);
      Init_tex_LUT(Nx,Ny,secArr,tex_c, cuArray_c);
    break;
case 1:
    Init_tex_LUT(Nx,Ny,firstArr,tex_ux, cuArray_ux);
    Init_tex_LUT(Nx,Ny,secArr,tex_uy, cuArray_uy);
  }
  switch (texMode) {
    break;
case 0:
    SolveCUDA(X0,T, boundary_parameters, h, N_tray, g, Nx, Ny, f, c, VARC,
                            X_tau_lin_1, X_tau_lin_2, Y_tau_lin, Z_tau_lin);
    break;
case 1:
    SolveCUDA(X0,T, boundary_parameters, h, N_tray, g, Nx, Ny, tex_f, c, VARC,
                           X_tau_lin_1, X_tau_lin_2, Y_tau_lin, Z_tau_lin);
    break;
case 2:
    SolveCUDA(X0,T, boundary_parameters, h, N_tray, g, Nx, Ny, f, tex_c, VARC,
                           X_tau_lin_1, X_tau_lin_2, Y_tau_lin, Z_tau_lin);
    break;
case 3:
    SolveCUDA(X0,T, boundary_parameters, h, N_tray, g, Nx, Ny, tex_f, tex_c, VARC,
                                                   X_tau_lin_1, X_tau_lin_2, Y_tau_lin, Z_tau_lin);
    break;
case 4:
    SolveCUDA(X0,T, boundary_parameters, h, N_tray, g, Nx, Ny, tex_f, tex_c, VARC, tex_ux, tex_uy,
   X_tau_lin_1, X_tau_lin_2, Y_tau_lin, Z_tau_lin);
    break;

  }
}

__host__ void MCinCUDA(int deviceId,int RNGinit, float* f_arr, float* c_arr, float* ux_arr,  float* uy_arr,int seed,Eigen::Vector2d X0,double T, double* boundary_parameters, double h,long long int N_tray, int Nx, int Ny, bool VARC,
                        double* phi, double* phi2, double* phi3, double* phi4, double* phixi, double* phi_plus_xi2, double* xi, double* xi2, double* tau, double* tau2){
  cudaSetDevice(deviceId);if (RNGinit==0){initRNGCuda(seed);}
  pfscalar g;
  cudaMemcpyFromSymbol( &g, g_, sizeof(pfscalar));
      Init_tex_LUT(Nx,Ny,f_arr,tex_f, cuArray_f);
      Init_tex_LUT(Nx,Ny,c_arr,tex_c, cuArray_c);
    Init_tex_LUT(Nx,Ny,ux_arr,tex_ux, cuArray_ux);
    Init_tex_LUT(Nx,Ny,uy_arr,tex_uy, cuArray_uy);
  SolveCUDA(X0,T, boundary_parameters, h, N_tray, g, Nx, Ny, tex_f, tex_c, VARC, tex_ux, tex_uy,
                          X_tau_lin_1, X_tau_lin_2, Y_tau_lin, Z_tau_lin);
}
/////
/////
/// Score with EM
__host__ void MCinCUDA(int deviceId,int RNGinit,int texMode,int seed,Eigen::Vector2d X0,double T, double* boundary_parameters, double h,long long int N_tray, int Nx, int Ny, bool VARC,
                      double* X_tau_lin_1,double* X_tau_lin_2, double* Y_tau_lin,double* Z_tau_lin,
                        double* X_tau_sublin_1,double* X_tau_sublin_2, double* Y_tau_sublin,double* Z_tau_sublin){

  cudaSetDevice(deviceId);if (RNGinit==0){initRNGCuda(seed);}
  pfscalar g;
  cudaMemcpyFromSymbol( &g, g_, sizeof(pfscalar));
  pfscalar f;
  cudaMemcpyFromSymbol( &f, f_, sizeof(pfscalar));
  pfscalar c;
  cudaMemcpyFromSymbol( &c, c_, sizeof(pfscalar));
  switch (texMode) {
    case 0:
    SolveCUDA(X0,T, boundary_parameters, h, N_tray, g, Nx, Ny, f, c, VARC,
                            X_tau_lin_1, X_tau_lin_2, Y_tau_lin, Z_tau_lin,
                           X_tau_sublin_1, X_tau_sublin_2, Y_tau_sublin, Z_tau_sublin);
    break;
case 1:
    SolveCUDA(X0,T, boundary_parameters, h, N_tray, g, Nx, Ny, tex_f, c, VARC,
     X_tau_lin_1, X_tau_lin_2, Y_tau_lin, Z_tau_lin,
    X_tau_sublin_1, X_tau_sublin_2, Y_tau_sublin, Z_tau_sublin);
    break;
case 2:
    SolveCUDA(X0,T, boundary_parameters, h, N_tray, g, Nx, Ny, f, tex_c, VARC,
     X_tau_lin_1, X_tau_lin_2, Y_tau_lin, Z_tau_lin,
    X_tau_sublin_1, X_tau_sublin_2, Y_tau_sublin, Z_tau_sublin);
    break;
case 3:
    SolveCUDA(X0,T, boundary_parameters, h, N_tray, g, Nx, Ny, tex_f, tex_c, VARC,
     X_tau_lin_1, X_tau_lin_2, Y_tau_lin, Z_tau_lin,
    X_tau_sublin_1, X_tau_sublin_2, Y_tau_sublin, Z_tau_sublin);
    break;
case 4:
    SolveCUDA(X0,T, boundary_parameters, h, N_tray, g, Nx, Ny, tex_f, tex_c, VARC, tex_ux, tex_uy,
     X_tau_lin_1, X_tau_lin_2, Y_tau_lin, Z_tau_lin,
    X_tau_sublin_1, X_tau_sublin_2, Y_tau_sublin, Z_tau_sublin);
    break;
  }

}


__host__ void MCinCUDA(int deviceId,int RNGinit, int texMode, int firstArrType, float* firstArr,int seed,Eigen::Vector2d X0,double T, double* boundary_parameters, double h,long long int N_tray, int Nx, int Ny, bool VARC,
double* X_tau_lin_1,double* X_tau_lin_2, double* Y_tau_lin,double* Z_tau_lin,
  double* X_tau_sublin_1,double* X_tau_sublin_2, double* Y_tau_sublin,double* Z_tau_sublin){
  cudaSetDevice(deviceId);if (RNGinit==0){initRNGCuda(seed);}
  pfscalar g;
  cudaMemcpyFromSymbol( &g, g_, sizeof(pfscalar));
  pfscalar f;
  cudaMemcpyFromSymbol( &f, f_, sizeof(pfscalar));
  pfscalar c;
  cudaMemcpyFromSymbol( &c, c_, sizeof(pfscalar));
  switch (firstArrType) {
    case 0:
      Init_tex_LUT(Nx,Ny,firstArr,tex_f, cuArray_f);
   break;
case 1:
      Init_tex_LUT(Nx,Ny,firstArr,tex_c, cuArray_c);
      break;

  }
  switch (texMode) {
    case 0:
    SolveCUDA(X0,T, boundary_parameters, h, N_tray, g, Nx, Ny, f, c, VARC,
     X_tau_lin_1, X_tau_lin_2, Y_tau_lin, Z_tau_lin,
    X_tau_sublin_1, X_tau_sublin_2, Y_tau_sublin, Z_tau_sublin);
    break;
case 1:
    SolveCUDA(X0,T, boundary_parameters, h, N_tray, g, Nx, Ny, tex_f, c, VARC,
     X_tau_lin_1, X_tau_lin_2, Y_tau_lin, Z_tau_lin,
    X_tau_sublin_1, X_tau_sublin_2, Y_tau_sublin, Z_tau_sublin);
    break;
case 2:
    SolveCUDA(X0,T, boundary_parameters, h, N_tray, g, Nx, Ny, f, tex_c, VARC,
     X_tau_lin_1, X_tau_lin_2, Y_tau_lin, Z_tau_lin,
    X_tau_sublin_1, X_tau_sublin_2, Y_tau_sublin, Z_tau_sublin);
    break;
case 3:
    SolveCUDA(X0,T, boundary_parameters, h, N_tray, g, Nx, Ny, tex_f, tex_c, VARC,
     X_tau_lin_1, X_tau_lin_2, Y_tau_lin, Z_tau_lin,
    X_tau_sublin_1, X_tau_sublin_2, Y_tau_sublin, Z_tau_sublin);
    break;
case 4:
    SolveCUDA(X0,T, boundary_parameters, h, N_tray, g, Nx, Ny, tex_f, tex_c, VARC, tex_ux, tex_uy,
     X_tau_lin_1, X_tau_lin_2, Y_tau_lin, Z_tau_lin,
    X_tau_sublin_1, X_tau_sublin_2, Y_tau_sublin, Z_tau_sublin);
    break;

  }
}

__host__ void MCinCUDA(int deviceId,int RNGinit, int texMode, int firstArrType, float* firstArr, float* secArr,int seed,Eigen::Vector2d X0,double T, double* boundary_parameters, double h,long long int N_tray, int Nx, int Ny, bool VARC,
double* X_tau_lin_1,double* X_tau_lin_2, double* Y_tau_lin,double* Z_tau_lin,
  double* X_tau_sublin_1,double* X_tau_sublin_2, double* Y_tau_sublin,double* Z_tau_sublin){
  cudaSetDevice(deviceId);if (RNGinit==0){initRNGCuda(seed);}
  pfscalar g;
  cudaMemcpyFromSymbol( &g, g_, sizeof(pfscalar));
  pfscalar f;
  cudaMemcpyFromSymbol( &f, f_, sizeof(pfscalar));
  pfscalar c;
  cudaMemcpyFromSymbol( &c, c_, sizeof(pfscalar));
  switch (firstArrType) {
    case 0:
      Init_tex_LUT(Nx,Ny,firstArr,tex_f, cuArray_f);
      Init_tex_LUT(Nx,Ny,secArr,tex_c, cuArray_c);
    break;
case 1:
    Init_tex_LUT(Nx,Ny,firstArr,tex_ux, cuArray_ux);
    Init_tex_LUT(Nx,Ny,secArr,tex_uy, cuArray_uy);
    break;

  }
  switch (texMode) {
    case 0:
    SolveCUDA(X0,T, boundary_parameters, h, N_tray, g, Nx, Ny, f, c, VARC,
     X_tau_lin_1, X_tau_lin_2, Y_tau_lin, Z_tau_lin,
    X_tau_sublin_1, X_tau_sublin_2, Y_tau_sublin, Z_tau_sublin);
    break;
case 1:
    SolveCUDA(X0,T, boundary_parameters, h, N_tray, g, Nx, Ny, tex_f, c, VARC,
     X_tau_lin_1, X_tau_lin_2, Y_tau_lin, Z_tau_lin,
    X_tau_sublin_1, X_tau_sublin_2, Y_tau_sublin, Z_tau_sublin);
    break;
case 2:
    SolveCUDA(X0,T, boundary_parameters, h, N_tray, g, Nx, Ny, f, tex_c, VARC,
     X_tau_lin_1, X_tau_lin_2, Y_tau_lin, Z_tau_lin,
    X_tau_sublin_1, X_tau_sublin_2, Y_tau_sublin, Z_tau_sublin);
    break;
case 3:
    SolveCUDA(X0,T, boundary_parameters, h, N_tray, g, Nx, Ny, tex_f, tex_c, VARC,
     X_tau_lin_1, X_tau_lin_2, Y_tau_lin, Z_tau_lin,
    X_tau_sublin_1, X_tau_sublin_2, Y_tau_sublin, Z_tau_sublin);
    break;
case 4:
    SolveCUDA(X0,T, boundary_parameters, h, N_tray, g, Nx, Ny, tex_f, tex_c, VARC, tex_ux, tex_uy,
     X_tau_lin_1, X_tau_lin_2, Y_tau_lin, Z_tau_lin,
    X_tau_sublin_1, X_tau_sublin_2, Y_tau_sublin, Z_tau_sublin);
    break;

  }
}

__host__ void MCinCUDA(int deviceId,int RNGinit, float* f_arr, float* c_arr, float* ux_arr,  float* uy_arr,int seed,Eigen::Vector2d X0,double T, double* boundary_parameters, double h,long long int N_tray, int Nx, int Ny, bool VARC,
double* X_tau_lin_1,double* X_tau_lin_2, double* Y_tau_lin,double* Z_tau_lin,
  double* X_tau_sublin_1,double* X_tau_sublin_2, double* Y_tau_sublin,double* Z_tau_sublin){
  cudaSetDevice(deviceId);if (RNGinit==0){initRNGCuda(seed);}
  pfscalar g;
  cudaMemcpyFromSymbol( &g, g_, sizeof(pfscalar));
      Init_tex_LUT(Nx,Ny,f_arr,tex_f, cuArray_f);
      Init_tex_LUT(Nx,Ny,c_arr,tex_c, cuArray_c);
    Init_tex_LUT(Nx,Ny,ux_arr,tex_ux, cuArray_ux);
    Init_tex_LUT(Nx,Ny,uy_arr,tex_uy, cuArray_uy);
  SolveCUDA(X0,T, boundary_parameters, h, N_tray, g, Nx, Ny, tex_f, tex_c, VARC, tex_ux, tex_uy,
   X_tau_lin_1, X_tau_lin_2, Y_tau_lin, Z_tau_lin,
  X_tau_sublin_1, X_tau_sublin_2, Y_tau_sublin, Z_tau_sublin);
}////
////
