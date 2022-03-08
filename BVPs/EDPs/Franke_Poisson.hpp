#include <eigen3/Eigen/Core>
#include <math.h>
#include <iostream>
inline double Franke_1 (Eigen::VectorXd & X){
    return exp(-0.25*(pow(9.0*X(0)-2.0,2)+pow(9.0*X(1)-2.0,2)));
}
inline double Franke_2 (Eigen::VectorXd & X){
    return exp((-1.0/49.0)*pow(9.0*X(0)+1.0,2)-0.1*(9.0*X(1)+1.0));
}
inline double Franke_3 (Eigen::VectorXd & X){
    return exp(-0.25*(pow(9.0*X(0)-7.0,2)+pow(9.0*X(1)-3.0,2)));
}
inline double Franke_4 (Eigen::VectorXd & X){
    return exp(-1.0*(pow(9.0*X(0)-4.0,2)+pow(9.0*X(1)-7.0,2)));
}
inline double Equation_u(Eigen::VectorXd X, double t){
    return 0.75*Franke_1(X) + 0.75*Franke_2(X) +0.5*Franke_3(X) - 0.2*Franke_4(X);
}
inline double Equation_dudx(Eigen::VectorXd & X){
    return -3.375*(9.0*X(0)-2.0)*Franke_1(X)-(27.0/98.0)*(9.0*X(0)+1.0)*Franke_2(X)
    -2.25*(9.0*X(0)-7.0)*Franke_3(X) + 3.6*(9.0*X(0)-4)*Franke_4(X);
}
inline double Equation_dudy(Eigen::VectorXd & X){
    return -3.375*(9.0*X(1)-2.0)*Franke_1(X)-0.675*Franke_2(X)
    -2.25*(9.0*X(1)-3.0)*Franke_3(X)+3.6*(9.0*X(1)-7.0)*Franke_4(X);
}
inline double Equation_d2udxdy(Eigen::VectorXd & X){
    return 15.1875*(9.0*X(1)-2.0)*(9.0*X(0)-2.0)*Franke_1(X)+(243.0/980.0)*(9.0*X(0)+1.0)*Franke_2(X)
    +10.125*(9.0*X(0)-7.0)*(9.0*X(1)-3.0)*Franke_3(X)-64.8*(9.0*X(0)-4.0)*(9.0*X(1)-7.0)*Franke_4(X);
}
inline double Equation_d2udx2(Eigen::VectorXd & X){
    return -3.375*(9.0-4.5*pow(9.0*X(0)-2.0,2))*Franke_1(X)
    -(27.0/98.0)*(9.0-(18.0/49.0)*pow(9.0*X(0)+1.0,2))*Franke_2(X)
    -2.25*(9.0-4.5*pow(9.0*X(0)-7.0,2))*Franke_3(X)
    +3.6*(9.0-18.0*pow(9.0*X(0)-4.0,2))*Franke_4(X);
}
inline double Equation_d2udy2(Eigen::VectorXd & X){
    return -3.375*(9.0-4.5*pow((9.0*X(1)-2.0),2))*Franke_1(X)
    +(243.0/400.0)*Franke_2(X) -2.25*(9.0-4.5*pow(9.0*X(1)-3.0,2))*Franke_3(X)
    +3.6*(9.0-18.0*pow(9.0*X(1)-7.0,2))*Franke_4(X);
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
