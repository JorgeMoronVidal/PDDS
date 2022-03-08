#include <eigen3/Eigen/Core>
#include <math.h>
#include <iostream>
#define k_1 1.0
#define k_2 0.01 
#define k_3 0.02 
#define k_4 0.12
#define k_5 0.05
#define k_6 0.05 
#define k_7 -0.12
#define scaling_factor 0.333
inline double Equation_u(Eigen::VectorXd X, double t){
    return 3.0 + scaling_factor*(sin(sqrt(k_1+k_2*X(0)*X(0)+k_3*X(1)*X(1))) + tanh(sin(k_4*X(0)+k_5*X(1))+sin(k_6*X(0)+k_7*X(1))));
}
inline double Equation_dudx(Eigen::VectorXd X){
    return scaling_factor*((cos(sqrt(k_1+k_2*X(0)*X(0)+k_3*X(1)*X(1)))*k_2*X(0)/sqrt(k_1 + k_2*X(0)*X(0)+k_3*X(1)*X(1))) -
    (k_4*cos(k_4*X(0)+k_5*X(1))+k_6*cos(k_6*X(0)+k_7*X(1)))*(pow(tanh(sin(k_4*X(0)+k_5*X(1))+sin(k_6*X(0)+k_7*X(1))),2)-1));
}
inline double Equation_dudy(Eigen::VectorXd X){
    return scaling_factor*(cos(sqrt(k_1+k_2*X(0)*X(0)+k_3*X(1)*X(1)))*k_1*k_3*X(1)/sqrt(k_1+k_2*X(0)*X(0)+k_3*X(1)*X(1)) -
    (k_5*cos(k_4*X(0)+k_5*X(1))+k_7*cos(k_6*X(0)+k_7*X(1)))*(pow(tanh(sin(k_4*X(0)+k_5*X(1))+sin(k_6*X(0)+k_7*X(1))),2)-1));
}
inline double Equation_d2udx2(Eigen::VectorXd X){
    double aux1 = sqrt(k_1+k_2*X(0)*X(0)+k_3*X(1)*X(1)),aux2 = tanh(sin(k_4*X(0)+k_5*X(1))+sin(k_6*X(0)+k_7*X(1)));
    return scaling_factor*(((k_2/aux1)-(k_2*k_2*X(0)*X(0)/pow(aux1,3)))*cos(aux1)-(X(0)*X(0)*k_2*k_2/(aux1*aux1))*sin(aux1)
    +2.0*pow(k_4*cos(k_4*X(0)+k_5*X(1))+k_6*cos(k_6*X(0)+k_7*X(1)),2)*(pow(aux2,2)-1)*aux2
    +(k_4*k_4*sin(k_4*X(0)+k_5*X(1))+k_6*k_6*sin(k_6*X(0)+k_7*X(1)))*(pow(aux2,2)-1));
}
inline double Equation_d2udy2(Eigen::VectorXd X){
    double aux1 = sqrt(k_1+k_2*X(0)*X(0)+k_3*X(1)*X(1)),aux2 = tanh(sin(k_4*X(0)+k_5*X(1))+sin(k_6*X(0)+k_7*X(1)));
    return scaling_factor*(((k_3/aux1)-(k_3*k_3*X(1)*X(1)/pow(aux1,3)))*cos(aux1)-(X(1)*X(1)*k_3*k_3/(aux1*aux1))*sin(aux1)
    +2.0*pow(k_5*cos(k_4*X(0)+k_5*X(1))+k_7*cos(k_6*X(0)+k_7*X(1)),2)*(pow(aux2,2)-1)*aux2
    +(k_5*k_5*sin(k_4*X(0)+k_5*X(1))+k_7*k_7*sin(k_6*X(0)+k_7*X(1)))*(pow(aux2,2)-1));
}
inline double Equation_d2udxdy(Eigen::VectorXd X){
    double aux1 = sqrt(k_1+k_2*X(0)*X(0)+k_3*X(1)*X(1)),aux2 = tanh(sin(k_4*X(0)+k_5*X(1))+sin(k_6*X(0)+k_7*X(1)));
    return scaling_factor*(-(k_2*k_3*X(0)*X(1)*sin(aux1)/(aux1*aux1)) - (k_2*k_3*X(0)*X(1)/pow(aux1,3))*cos(aux1) +
    2.0*(k_4*cos(k_4*X(0)+k_5*X(1))+k_6*cos(k_6*X(0)+k_7*X(1)))*(k_5*cos(k_4*X(0)+k_5*X(1))+k_7*cos(k_6*X(0)+k_7*X(1)))*
    (aux2*aux2 -1)*aux2 + (k_4*k_5*sin(k_4*X(0)+k_5*X(1)) +  k_6*k_7*sin(k_6*X(0)+k_7*X(1)))*(aux2*aux2-1));
}
inline double Equation_c(Eigen::VectorXd X, double t){
    return -0.0;
}
inline double Equation_g(Eigen::VectorXd X, double t){
    return Equation_u(X,t);
}
inline double Equation_p(Eigen::VectorXd X, double t){
    return Equation_u(X,t);
}
inline Eigen::VectorXd Equation_b(Eigen::VectorXd X, double t){
    Eigen::VectorXd Output(2);
    return Output* 0.0;
}
inline Eigen::MatrixXd Equation_sigma(Eigen::VectorXd X, double t){
    return Eigen::MatrixXd::Identity(2,2) * 1.41421356237;
}
inline double Equation_f(Eigen::VectorXd X, double t){
    return -Equation_d2udx2(X)-Equation_d2udy2(X);
}
inline double Equation_Varphi(Eigen::VectorXd X,Eigen::VectorXd normal, double t){
    return 0.0;
}
inline double Equation_Psi(Eigen::VectorXd X, Eigen::VectorXd normal, double t){
    return 0.0;
}
inline Eigen::VectorXd Equation_F(Eigen::VectorXd X, double t){
    Eigen::VectorXd F(2);
    F(0) = Equation_dudx(X);
    F(1) = Equation_dudy(X);
    F = -Equation_sigma(X,t).transpose()*F;
    return F;
}
inline bool Stopping_mix(Eigen::VectorXd X){
    //if(fabs(X(0) + 1.0) < 1E-8) return false;
    return true;
}
inline double Equation_RBF(Eigen::VectorXd x , Eigen::VectorXd xj, double c2){
    return 1/(pow((x-xj).norm(),2) + c2);
}