#include <eigen3/Eigen/Core>
#include <math.h>
#include <iostream>
#include <gsl/gsl_math.h>
#include <gsl/gsl_interp2d.h>
#include <gsl/gsl_spline2d.h>
#include <gsl/gsl_errno.h>
#define C2_iteration 1.0
#define Alpha_iteration 10.0
//#include "../LUT.hpp"
inline double EquationLI_u(Eigen::Vector2d X, double t){
    return  1 + sin(M_PI*X(0))*sin(M_PI*X(1));
}
inline double EquationLI_d2udx2(Eigen::Vector2d X){
    return - M_PI*M_PI*EquationLI_u(X,0.0);
}
inline double EquationLI_d2udy2(Eigen::Vector2d X){
    return - M_PI*M_PI*EquationLI_u(X,0.0);
}
inline Eigen::Matrix2d EquationLI_sigma(Eigen::Vector2d X, double t){
    return Eigen::Matrix2d::Identity() * 1.41421356237;
}

inline double EquationLI_c_FirstIt(Eigen::Vector2d X, double t){
    return C2_iteration - Alpha_iteration;
}
inline double EquationLI_c(Eigen::Vector2d X, double t,gsl_spline2d *LUT_ui, gsl_interp_accel *xacc_ui,
                gsl_interp_accel *yacc_ui){
    return  C2_iteration - Alpha_iteration;
}
inline double EquationLI_f(Eigen::Vector2d X, double t){
    return +2*M_PI*M_PI*sin(M_PI*X(0))*sin(M_PI*X(1)) - C2_iteration*(1 + sin(M_PI*X(0))*sin(M_PI*X(1)));
}
inline double EquationLI_f_LUT(Eigen::Vector2d X, double t,gsl_spline2d *LUT_ui, gsl_interp_accel *xacc_ui,
                gsl_interp_accel *yacc_ui,gsl_spline2d *LUT_u0, gsl_interp_accel *xacc_u0,
                gsl_interp_accel *yacc_u0){
    return +2*M_PI*M_PI*sin(M_PI*X(0))*sin(M_PI*X(1)) - C2_iteration*(1 + sin(M_PI*X(0))*sin(M_PI*X(1))) +Alpha_iteration * gsl_spline2d_eval(LUT_ui, X(0), X(1), xacc_ui, yacc_ui);
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
    return 1.0;
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