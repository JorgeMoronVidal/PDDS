#include <eigen3/Eigen/Core>
#include <math.h>
#include <iostream>
#include <gsl/gsl_math.h>
#include <gsl/gsl_interp2d.h>
#include <gsl/gsl_spline2d.h>
#include <gsl/gsl_errno.h>
#define C2_iteration 1.0
#define Alpha_iteration 3.0
//#include "../LUT.hpp"
inline double Equation_u(Eigen::Vector2d X, double t){
    return  sin(2*M_PI*X(0)+0.5)*cos(M_PI*M_PI*(X(0)+X(1)));
}
inline double EquationLI_d2udx2(Eigen::Vector2d X){
    return -(4*M_PI*M_PI + pow(M_PI,4))*Equation_u(X,0.0)-4.0*pow(M_PI,3)*cos(2*M_PI*X(0)+0.5)*sin(M_PI*M_PI*(X(0)+X(1)));
}
inline double EquationLI_d2udy2(Eigen::Vector2d X){
    return - pow(M_PI,4)*Equation_u(X,0.0);
}
inline Eigen::Matrix2d EquationLI_sigma(Eigen::Vector2d X, double t){
    return Eigen::Matrix2d::Identity() * 1.41421356237;
}

inline double EquationLI_c(Eigen::Vector2d X, double t){
    return C2_iteration - Alpha_iteration;
}
inline double EquationLI_c_LUT(Eigen::Vector2d X, double t,gsl_spline2d *LUT_ui, gsl_interp_accel *xacc_ui,
                gsl_interp_accel *yacc_ui){
    return  C2_iteration - Alpha_iteration;
}
inline double EquationLI_f(Eigen::Vector2d X, double t){
    return -EquationLI_d2udx2(X)-EquationLI_d2udy2(X) - C2_iteration*Equation_u(X,t);
}
inline double EquationLI_f_LUT(Eigen::Vector2d X, double t,gsl_spline2d *LUT_ui, gsl_interp_accel *xacc_ui,
                gsl_interp_accel *yacc_ui,gsl_spline2d *LUT_u0, gsl_interp_accel *xacc_u0,
                gsl_interp_accel *yacc_u0){
    return -EquationLI_d2udx2(X)-EquationLI_d2udy2(X) - C2_iteration*Equation_u(X,t) +Alpha_iteration * gsl_spline2d_eval(LUT_ui, X(0), X(1), xacc_ui, yacc_ui);
}
inline double EquationLI_u_LUT(Eigen::Vector2d X, double t,gsl_spline2d *LUT, gsl_interp_accel *xacc,
                gsl_interp_accel *yacc){
    int status;
    status = gsl_spline2d_eval(LUT, X(0), X(1), xacc, yacc);
    if(status){
       status = gsl_spline2d_eval(LUT, X(0)-1E-03, X(1), xacc, yacc);
       if(status){
           status = gsl_spline2d_eval(LUT, X(0)+1E-03, X(1), xacc, yacc);
           if(status){
               status = gsl_spline2d_eval(LUT, X(0), X(1)-1E-03, xacc, yacc);
               if(status){
                   status = gsl_spline2d_eval(LUT, X(0), X(1)+1E-03, xacc, yacc);
                   if(status){
                       return 0.0;
                       std::cout << "EquationLI_u_LUT if chain is not working properly\n";
                   }else{
                       return  gsl_spline2d_eval(LUT, X(0), X(1)+1E-03, xacc, yacc);
                   }
               }else{
                   return gsl_spline2d_eval(LUT, X(0), X(1)-1E-03, xacc, yacc);
               }
           } else {
               return gsl_spline2d_eval(LUT, X(0)+1E-03, X(1), xacc, yacc);
           }
       }else{
           return gsl_spline2d_eval(LUT, X(0)-1E-03, X(1), xacc, yacc);
       }
    } else {
        return gsl_spline2d_eval(LUT, X(0), X(1), xacc, yacc);
    }
    
}
inline double EquationLI_g(Eigen::Vector2d X, double t){
    return Equation_u(X,t);
}
Eigen::Vector2d Equation_grad(Eigen::Vector2d X, double t){
    Eigen::Vector2d grad;
    grad(0) = cos(X(0)*M_PI)*sin(X(1)*M_PI);
    grad(1) = cos(X(1)*M_PI)*sin(X(0)*M_PI);
    return M_PI*grad;
}
Eigen::Vector2d EquationLI_grad_LUT(Eigen::Vector2d X, double t,gsl_spline2d *LUT, gsl_interp_accel *xacc,
                gsl_interp_accel *yacc){
    Eigen::Vector2d grad;
    grad(0) = gsl_spline2d_eval_deriv_x(LUT, X(0), X(1), xacc, yacc);
    grad(1) = gsl_spline2d_eval_deriv_y(LUT, X(0), X(1), xacc, yacc);
    return grad;
}