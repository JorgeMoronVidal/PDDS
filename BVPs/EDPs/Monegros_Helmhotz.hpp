#include <eigen3/Eigen/Core>
#include <math.h>
#include <iostream>
#include <gsl/gsl_math.h>
#include <gsl/gsl_interp2d.h>
#include <gsl/gsl_spline2d.h>
#include <gsl/gsl_errno.h>
#define k_1 1.0
#define k_2 0.01 
#define k_3 0.02 
#define k_4 0.12
#define k_5 0.05
#define k_6 0.05 
#define k_7 -0.12
#define scaling_factor 0.333
#define C2_iteration 0.1
#define Alpha_iteration 0.3
inline double Equation_u(Eigen::Vector2d X, double t){
    return 3.0 + scaling_factor*(sin(sqrt(k_1+k_2*X(0)*X(0)+k_3*X(1)*X(1))) + tanh(sin(k_4*X(0)+k_5*X(1))+sin(k_6*X(0)+k_7*X(1))));
}
inline double Equation_dudx(Eigen::Vector2d X){
    return scaling_factor*((cos(sqrt(k_1+k_2*X(0)*X(0)+k_3*X(1)*X(1)))*k_2*X(0)/sqrt(k_1 + k_2*X(0)*X(0)+k_3*X(1)*X(1))) -
    (k_4*cos(k_4*X(0)+k_5*X(1))+k_6*cos(k_6*X(0)+k_7*X(1)))*(pow(tanh(sin(k_4*X(0)+k_5*X(1))+sin(k_6*X(0)+k_7*X(1))),2)-1));
}
inline double Equation_dudy(Eigen::Vector2d X){
    return scaling_factor*(cos(sqrt(k_1+k_2*X(0)*X(0)+k_3*X(1)*X(1)))*k_1*k_3*X(1)/sqrt(k_1+k_2*X(0)*X(0)+k_3*X(1)*X(1)) -
    (k_5*cos(k_4*X(0)+k_5*X(1))+k_7*cos(k_6*X(0)+k_7*X(1)))*(pow(tanh(sin(k_4*X(0)+k_5*X(1))+sin(k_6*X(0)+k_7*X(1))),2)-1));
}
inline double Equation_d2udx2(Eigen::Vector2d X){
    double aux1 = sqrt(k_1+k_2*X(0)*X(0)+k_3*X(1)*X(1)),aux2 = tanh(sin(k_4*X(0)+k_5*X(1))+sin(k_6*X(0)+k_7*X(1)));
    return scaling_factor*(((k_2/aux1)-(k_2*k_2*X(0)*X(0)/pow(aux1,3)))*cos(aux1)-(X(0)*X(0)*k_2*k_2/(aux1*aux1))*sin(aux1)
    +2.0*pow(k_4*cos(k_4*X(0)+k_5*X(1))+k_6*cos(k_6*X(0)+k_7*X(1)),2)*(pow(aux2,2)-1)*aux2
    +(k_4*k_4*sin(k_4*X(0)+k_5*X(1))+k_6*k_6*sin(k_6*X(0)+k_7*X(1)))*(pow(aux2,2)-1));
}
inline double Equation_d2udy2(Eigen::Vector2d X){
    double aux1 = sqrt(k_1+k_2*X(0)*X(0)+k_3*X(1)*X(1)),aux2 = tanh(sin(k_4*X(0)+k_5*X(1))+sin(k_6*X(0)+k_7*X(1)));
    return scaling_factor*(((k_3/aux1)-(k_3*k_3*X(1)*X(1)/pow(aux1,3)))*cos(aux1)-(X(1)*X(1)*k_3*k_3/(aux1*aux1))*sin(aux1)
    +2.0*pow(k_5*cos(k_4*X(0)+k_5*X(1))+k_7*cos(k_6*X(0)+k_7*X(1)),2)*(pow(aux2,2)-1)*aux2
    +(k_5*k_5*sin(k_4*X(0)+k_5*X(1))+k_7*k_7*sin(k_6*X(0)+k_7*X(1)))*(pow(aux2,2)-1));
}
inline double Equation_d2udxdy(Eigen::Vector2d X){
    double aux1 = sqrt(k_1+k_2*X(0)*X(0)+k_3*X(1)*X(1)),aux2 = tanh(sin(k_4*X(0)+k_5*X(1))+sin(k_6*X(0)+k_7*X(1)));
    return scaling_factor*(-(k_2*k_3*X(0)*X(1)*sin(aux1)/(aux1*aux1)) - (k_2*k_3*X(0)*X(1)/pow(aux1,3))*cos(aux1) +
    2.0*(k_4*cos(k_4*X(0)+k_5*X(1))+k_6*cos(k_6*X(0)+k_7*X(1)))*(k_5*cos(k_4*X(0)+k_5*X(1))+k_7*cos(k_6*X(0)+k_7*X(1)))*
    (aux2*aux2 -1)*aux2 + (k_4*k_5*sin(k_4*X(0)+k_5*X(1)) +  k_6*k_7*sin(k_6*X(0)+k_7*X(1)))*(aux2*aux2-1));
}
inline double EquationLI_c(Eigen::Vector2d X, double t){
    return C2_iteration - Alpha_iteration;
}
inline double EquationLI_c_LUT(Eigen::Vector2d X, double t,gsl_spline2d *LUT_ui, gsl_interp_accel *xacc_ui,
                gsl_interp_accel *yacc_ui){
    return  C2_iteration - Alpha_iteration;
}
inline double EquationLI_f(Eigen::Vector2d X, double t){
    return -Equation_d2udx2(X)-Equation_d2udy2(X) - C2_iteration*Equation_u(X,t) +Alpha_iteration*0.0;
}
inline double EquationLI_f_LUT(Eigen::Vector2d X, double t,gsl_spline2d *LUT_ui, gsl_interp_accel *xacc_ui,
                gsl_interp_accel *yacc_ui,gsl_spline2d *LUT_u0, gsl_interp_accel *xacc_u0,
                gsl_interp_accel *yacc_u0){
    return -Equation_d2udx2(X)-Equation_d2udy2(X) - C2_iteration*Equation_u(X,t) +Alpha_iteration * gsl_spline2d_eval(LUT_ui, X(0), X(1), xacc_ui, yacc_ui);
}
inline double EquationLI_g(Eigen::Vector2d X, double t){
    return Equation_u(X,t);
}
inline double Equation_p(Eigen::Vector2d X, double t){
    return Equation_u(X,t);
}
inline Eigen::Vector2d Equation_b(Eigen::Vector2d X, double t){
    Eigen::Vector2d Output(2);
    return Output* 0.0;
}
inline Eigen::Matrix2d EquationLI_sigma(Eigen::Vector2d X, double t){
    return Eigen::Matrix2d::Identity() * 1.41421356237;
}
inline double Equation_Varphi(Eigen::Vector2d X,Eigen::Vector2d normal, double t){
    return 0.0;
}
inline double Equation_Psi(Eigen::Vector2d X, Eigen::Vector2d normal, double t){
    return 0.0;
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
inline Eigen::Vector2d Equation_F(Eigen::Vector2d X, double t){
    Eigen::Vector2d F;
    F(0) = Equation_dudx(X);
    F(1) = Equation_dudy(X);
    F = -EquationLI_sigma(X,t).transpose()*F;
    return F;
}
inline Eigen::Vector2d Equation_grad(Eigen::Vector2d X, double t){
    Eigen::Vector2d grad;
    grad(0) = Equation_dudx(X);
    grad(1) = Equation_dudy(X);
    return grad;
}
Eigen::Vector2d EquationLI_grad_LUT(Eigen::Vector2d X, double t,gsl_spline2d *LUT, gsl_interp_accel *xacc,
                gsl_interp_accel *yacc){
    Eigen::Vector2d grad;
    grad(0) = gsl_spline2d_eval_deriv_x(LUT, X(0), X(1), xacc, yacc);
    grad(1) = gsl_spline2d_eval_deriv_y(LUT, X(0), X(1), xacc, yacc);
    return grad;
}
inline bool Stopping_mix(Eigen::Vector2d X){
    //if(fabs(X(0) + 1.0) < 1E-8) return false;
    return true;
}
inline double Equation_RBF(Eigen::Vector2d x , Eigen::Vector2d xj, double c2){
    return 1/sqrt(pow((x-xj).norm(),2) + c2);
}