#include <eigen3/Eigen/Core>
#include <math.h>
#include <iostream>
#define omegax 0.23
#define omegay 0.49
#define omegapx 0.331
#define omegapy 0.667 
inline double Equation_c(Eigen::VectorXd X, double t){
    return 0.0;
}
inline double Equation_f(Eigen::VectorXd X, double t){
    return M_PI*M_PI*((omegax*omegax + omegay*omegay)*sin(omegax * M_PI *X(0) + omegay * M_PI *X(1)) +
    (omegapx*omegapx + omegapy*omegapy)*cos(omegapx * M_PI *X(0) + omegapy * M_PI *X(1)));
}
inline double Equation_u(Eigen::VectorXd X, double t){
    return sin(omegax * M_PI *X(0) + omegay * M_PI *X(1)) +
    cos(omegapx * M_PI *X(0) + omegapy * M_PI *X(1));

}
inline double Equation_g(Eigen::VectorXd X, double t){
    return Equation_u(X,t);
}
inline Eigen::VectorXd Equation_b(Eigen::VectorXd X, double t){
    return X*0.0;
}
inline Eigen::MatrixXd Equation_sigma(Eigen::VectorXd, double){
    return Eigen::MatrixXd::Identity(2,2) * 1.41421356237;
}
inline double Equation_Varphi(Eigen::VectorXd X, Eigen::VectorXd, double t){
    return -2.0 + sin(X(1)*M_PI);
    //return 0.0;
}
inline double Equation_Psi(Eigen::VectorXd X, Eigen::VectorXd normal, double t){
    return normal(0) * M_PI*((omegax)*cos(omegax * M_PI *X(0) + omegay * M_PI *X(1)) -
    omegapx*sin(omegapx * M_PI *X(0) + omegapy * M_PI *X(1))) + normal(1) * M_PI*
    ((omegay)*cos(omegax * M_PI *X(0) + omegay * M_PI *X(1)) - (omegapy)*sin(omegapx
     * M_PI *X(0) + omegapy * M_PI *X(1))) - Equation_u(X,t)*Equation_Varphi(X,normal,t);
}
inline Eigen::VectorXd Equation_F(Eigen::VectorXd X, double t){
    Eigen::VectorXd F(2);
    F(0) =  M_PI*((omegax)*cos(omegax * M_PI *X(0) + omegay * M_PI *X(1)) -
    (omegapx)*sin(omegapx * M_PI *X(0) + omegapy * M_PI *X(1)));
    F(1) =  M_PI*((omegay)*cos(omegax * M_PI *X(0) + omegay * M_PI *X(1)) -
    (omegapy)*sin(omegapx * M_PI *X(0) + omegapy * M_PI *X(1)));
    F = -Equation_sigma(X,t).transpose()*F;
    return F;
}
inline bool Stopping_mix(Eigen::VectorXd X){
    if(fabs(X(0) + 1.0) < 1E-8) return false;
    return true;
}
inline double Equation_RBF(Eigen::VectorXd x , Eigen::VectorXd xj, double c2){
    return 1/(pow((x-xj).norm(),2) + c2);
}