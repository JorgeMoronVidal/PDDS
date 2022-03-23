#include <eigen3/Eigen/Core>
#include <math.h>
#include <iostream>
#include "../LUT.hpp"
inline double EquationSM_u(Eigen::Vector2d X, double t){
    return sin(M_PI*X(0))*sin(M_PI*X(1));
}
inline double EquationSM_d2udx2(Eigen::Vector2d X){
    return - M_PI*M_PI*EquationSM_u(X,0.0);
}
inline double EquationSM_d2udy2(Eigen::Vector2d X){
    return - M_PI*M_PI*EquationSM_u(X,0.0);
}
inline Eigen::Matrix2d EquationSM_sigma(Eigen::Vector2d X, double t){
    return Eigen::Matrix2d::Identity() * 1.41421356237;
}
inline double EquationSM_c(Eigen::Vector2d X, double t){
    return -3.0*pow(gsl_spline2d_eval(spline_u, X(0), X(1), xacc_u, yacc_u),2);
}
inline double EquationSM_f(Eigen::Vector2d X, double t){
    return +2*M_PI*M_PI*EquationSM_u(X,0.0) + pow(EquationSM_u(X,t),3);
}
inline double EquationSM_g(Eigen::Vector2d X, double t){
    return 0.0;
}
Eigen::Vector2d Equation_grad(Eigen::Vector2d X, double t){
    Eigen::Vector2d grad;
    grad(0) = cos(X(0)*M_PI)*sin(X(1)*M_PI);
    grad(1) = cos(X(1)*M_PI)*sin(X(0)*M_PI);
    return M_PI*grad;
}
inline double EquationSM_Residual(Eigen::Vector2d X, double t){
    return gsl_spline2d_eval_deriv_xx(spline_u, X(0), X(1), xacc_u, yacc_u)
    + gsl_spline2d_eval_deriv_yy(spline_u, X(0), X(1), xacc_u, yacc_u)
    -pow(gsl_spline2d_eval(spline_u, X(0), X(1), xacc_u, yacc_u),3)
    +EquationSM_f(X,t);
}