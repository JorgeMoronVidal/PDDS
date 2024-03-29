#include <iostream>
#include <stdio.h>
#include <eigen3/Eigen/Core>
#include <vector>
#include <math.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_interp2d.h>
#include <gsl/gsl_spline2d.h>
#include <gsl/gsl_errno.h>
#ifndef BVP
#define BVP
//#include "LUT.hpp"
typedef double (*pfscalar)(Eigen::Vector2d, double );
typedef double (*pfscalarLUT)(Eigen::Vector2d, double ,
                gsl_spline2d *, gsl_interp_accel *,
                gsl_interp_accel *);
typedef double (*pfscalar2LUT)(Eigen::Vector2d, double ,
                gsl_spline2d *, gsl_interp_accel *,
                gsl_interp_accel *,gsl_spline2d *, gsl_interp_accel *,
                gsl_interp_accel *);
typedef double (*pfscalarti)(Eigen::Vector2d);
typedef double (*pfscalarN)(Eigen::Vector2d, Eigen::Vector2d, double);
typedef Eigen::Vector2d (*pfvector)(Eigen::Vector2d, double);
typedef Eigen::Vector2d (*pfvectorLUT)(Eigen::Vector2d, double ,
                gsl_spline2d *, gsl_interp_accel *,
                gsl_interp_accel *);
typedef Eigen::Vector2d (*pfvector2LUT)(Eigen::Vector2d, double ,
                gsl_spline2d *, gsl_interp_accel *, gsl_interp_accel *, 
                gsl_spline2d *, gsl_interp_accel *, gsl_interp_accel *);
typedef Eigen::Matrix2d (*pfmatrix)(Eigen::Vector2d, double);
typedef double (*pfdist)(double *,Eigen::Vector2d, Eigen::Vector2d &, Eigen::Vector2d &);
typedef bool (*pfbtype)(Eigen::Vector2d);
typedef double (*pfRBF)(Eigen::Vector2d, Eigen::Vector2d, double);
//Default scalar function which always returns  0.0
inline double Default_Scalar(Eigen::Vector2d position,double t){
        return 0.0;
}
//Default scalar function with LUT which always returns  0.0
inline double Default_ScalarLUT(Eigen::Vector2d position,double t,
                gsl_spline2d *LUT, gsl_interp_accel *xacc,
                gsl_interp_accel *yacc){
        return 0.0;
}
inline double Default_Scalar2LUT(Eigen::Vector2d position,double t,
                gsl_spline2d *LUT_u, gsl_interp_accel *xacc_u,
                gsl_interp_accel *yacc_u, gsl_spline2d *LUT_v, 
                gsl_interp_accel *xacc_v, gsl_interp_accel *yacc_v){
        return 0.0;
}
//Default scalar (Time independent) function which always returns  0.0
inline double Default_Scalarti(Eigen::Vector2d position){
        return 0.0;
}
//Default scalar (Normal dependent) function which always returns  0.0
inline double Default_ScalarN(Eigen::Vector2d position,Eigen::Vector2d normal, double t){
        return 0.0;
}
//Default vector function which always returns  a vector of the same size as position with all 0's
inline Eigen::Vector2d Default_Vector(Eigen::Vector2d position, double t){
        return position*0.0;
}
//Default vector function  with LUT which always returns  a vector of the same size as position with all 0's
inline Eigen::Vector2d Default_VectorLUT(Eigen::Vector2d position, double t,
        gsl_spline2d *LUT, gsl_interp_accel *xacc, gsl_interp_accel *yacc){
         return Eigen::Vector2d(
         gsl_spline2d_eval_deriv_x(LUT, position(0), position(1), xacc, yacc)
        ,gsl_spline2d_eval_deriv_y(LUT, position(0), position(1), xacc, yacc));
}
//Default matrix function which always returns  a matrix of the same size as position with all 0's
inline Eigen::Matrix2d Default_Matrix(Eigen::Vector2d position, double t){
        return Eigen::Matrix2d::Identity();
}
//Default distance function which always returns everything 0
inline double Default_Distance(double  *parameters,Eigen::Vector2d position, Eigen::Vector2d & normal, Eigen::Vector2d & nProjection){
        normal = 0.0*position;
        nProjection = normal;
        return 0.0;
}
//Default boundary type function which always returns true
inline bool Default_Btype(Eigen::Vector2d position){
        return true;
}
//Default Radial Basis Function inverse multiquadric
inline double Default_RBF(Eigen::Vector2d X, Eigen::Vector2d X_j, double c){
        //return 1.0/sqrt(pow((X-X_j).norm(),2) + c);
        return exp(-pow((X-X_j).norm(),2)/(c*c));
}       
/*This structure stores the functions that define a given BVP.*/
struct bvp{
        //-u(Eigen::Vector2d position, double t) is the solution of the problem.
        pfscalar u = Default_Scalar;
        //-u(Eigen::Vector2d position, double t) is the solution of the problem.
        pfscalarLUT num_u = Default_ScalarLUT;
        //-g(Eigen::Vector2d position, double t) is the value of the problem on the dirichlet BC's.
        pfscalar g = Default_Scalar;
        //-p(Eigen::Vector2d position) is the initial condition [solution when t= 0] of the BVP.
        pfscalarti p = Default_Scalarti;
        //-f(Eigen::Vector2d position, double t) is the source term of the BVP's PDE.
        pfscalar f = Default_Scalar;
        //Numerical f
        pfscalarLUT num_f = Default_ScalarLUT;
        //Numerical f two LUT
        pfscalar2LUT num_f_2LUT = Default_Scalar2LUT;
        //-c(Eigen::Vector2d position, double t) is the reactions term in the BVP's PDE. Has to be negative.
        pfscalar c = Default_Scalar;
        //Numerical c
        pfscalarLUT num_c = Default_ScalarLUT;
        //-\sigma(Eigen::Vector2d position, double t) diffusion matrix.
        pfmatrix sigma = Default_Matrix;
        //-b(Eigen::Vector2d position, double t) drift of the diffusion. 
        pfvector b = Default_Vector;
        //-grad_u(Eigen::Vector2d position, double t) gradient of the solution. It is important for variance reduction purposes.
        pfvector gradient = Default_Vector;
        //-grad_u(Eigen::Vector2d position, double t) gradient of the solution. It is important for variance reduction purposes.
        pfvectorLUT num_gradient_LUT = Default_VectorLUT;
        //-grad_u(Eigen::Vector2d position, double t) gradient of the solution. It is important for variance reduction purposes.
        //pfvector2LUT num_gradient_2LUT = Default_Vector;
        //-F function for control variates
        pfvector F = Default_Vector;
        //-mu function for variance reduction
        pfvector mu = Default_Vector;
        //-\psi(Eigen::Vector2d position, Eigen::Vector2d N, double t). 
        pfscalarN psi = Default_ScalarN;
        //-\varphi(Eigen::Vector2d position, Eigen::Vector2d N, double t)
        //Has to be negative. Different from 0 if Robin BC's are present.
        pfscalarN varphi = Default_ScalarN;
        //-\distance(std::vector<double> domain parameters, Eigen::Vector2d position, Eigen::Vector2d & normal, Eigen::Vector2d & conormal projection)
        //Distance to the domain. 
        pfdist distance = Default_Distance;
        //absorbing(Eigen::Vector2d Bposition) True if BC's are absorbing in Bposition.   
        pfbtype absorbing = Default_Btype;
        //Neumann(Eigen::Vector2d Bposition) True if BC's Neumann in Bposition.   
        pfbtype Neumann = Default_Btype;
        //RBF for interpolation   
        pfRBF RBF = Default_RBF;
        
};
#endif