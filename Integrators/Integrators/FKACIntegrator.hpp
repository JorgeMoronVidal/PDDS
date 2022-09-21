#include <stdlib.h>
#include <iostream> 
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "../BVPs/BVP.hpp"
#include <boost/random.hpp>
#include <eigen3/Eigen/Core>

/*Updates increment vector*/
inline void Increment_Update( Eigen::Vector2d & increment, boost::mt19937 & rng, 
boost::normal_distribution<double> & normalrng, double sqrth){
  increment(0) = sqrth*normalrng(rng);
  increment(1) = sqrth*normalrng(rng);
};
/*One step of plain Euler Maruyama*/
inline void Step(Eigen::Vector2d & X, Eigen::Vector2d & N, double & Y, double & Z, double & t, 
double ji_t, double h, double sqrth, boost::mt19937 & rng, boost::normal_distribution<double> & normalrng,
Eigen::Vector2d & increment, pfmatrix sigma, pfscalar f, pfvector b, pfscalar c, pfscalarN psi,
pfscalarN varphi){
  Increment_Update(increment, rng, normalrng, sqrth);
  Z += f(X,t)*Y*h + psi(X,N,t)*Y*ji_t;
  Y += c(X,t)*Y*h + varphi(X,N,t) * Y * ji_t;
  X += b(X,t)*h + sigma(X,t)*increment;
  t -= h;
}
/*One step of the plain Euler's discretization with Variance Reduction*/
inline void Step(Eigen::Vector2d & X, Eigen::Vector2d & N, double & Y, double & Z, double & t, 
double ji_t, double h, double sqrth, boost::mt19937 & rng, boost::normal_distribution<double> & normalrng,
Eigen::Vector2d & increment, pfmatrix sigma, pfscalar f,  pfvector b, pfscalar c, pfscalarN psi, pfscalarN varphi,
pfvector gradient, pfscalar u){
  Eigen::Matrix2d sigma_aux = sigma(X,t);
  Increment_Update(increment, rng, normalrng, sqrth);
  Z += f(X,t)*Y*h + psi(X,N,t)*Y*ji_t;
  Y += c(X,t)*Y*h + 
  Y *(-sigma_aux.transpose()*gradient(X,t)/u(X,t)).transpose().dot(increment) +
  varphi(X,N,t) * Y * ji_t;
  X += (b(X,t)-sigma_aux*(-sigma_aux.transpose()*gradient(X,t)/u(X,t)))*h + sigma_aux*increment;
  t -= h;
}
/*One step of the plain Euler's discretization with Control Variates*/
inline void Step(Eigen::Vector2d & X, Eigen::Vector2d & N, double & Y, double & Z , double & xi,
double & t, double ji_t, double h, double sqrth, boost::mt19937 & rng, boost::normal_distribution<double> & normalrng,
Eigen::Vector2d & increment, pfmatrix sigma, pfscalar f, pfvector b, pfscalar c, 
pfscalarN psi, pfscalarN varphi,  pfvector gradient){
  Eigen::Matrix2d sigma_aux = sigma(X,t);
  Increment_Update(increment, rng, normalrng, sqrth);
  xi +=  Y *(-sigma_aux.transpose()*gradient(X,t)).dot(increment);
  Z += f(X,t)*Y*h + psi(X,N,t)*Y*ji_t;
  Y += c(X,t)*Y*h + varphi(X,N,t) * Y * ji_t;
  X += b(X,t)*h + sigma_aux*increment;
  t -= h;
}

/*One step of the plain Euler's discretization with Variance Reduction and Control variates*/
inline void Step(Eigen::Vector2d & X, Eigen::Vector2d & N, double & Y, double & Z, double & xi,
double & t, double ji_t, double h, double sqrth, boost::mt19937 & rng, boost::normal_distribution<double> & normalrng,
Eigen::Vector2d & increment, pfmatrix sigma, pfscalar f, pfvector b, pfscalar c, pfscalarN psi,
pfscalarN varphi,  pfvector mu, pfvector F){
  Eigen::Matrix2d sigma_aux = sigma(X,t);
  Increment_Update(increment, rng, normalrng, sqrth);
  Z += f(X,t)*Y*h + psi(X,N,t)*Y*ji_t;
  xi += Y *(F(X,t)).dot(increment);
  Y += c(X,t)*Y*h + Y *(mu(X,t)).dot(increment) + varphi(X,N,t) * Y * ji_t;
  X += (b(X,t)-sigma_aux*mu(X,t))*h + sigma_aux*increment;
  t -= h;
}

/*One step of the plain Euler's discretization with Control Variates for the Lepingle Algorithm*/
inline void Step(Eigen::Vector2d & X, Eigen::Vector2d & N, Eigen::Vector2d & Npro, double & Y, double & Z ,
double & xi, double & t, double & ji_t, double rho,  double h, Eigen::Vector2d & increment, pfmatrix sigma, pfscalar f, 
pfvector b, pfscalar c, pfscalarN psi, pfscalarN varphi,  pfvector gradient, pfdist distance, double *params,
double & d_k, boost::mt19937 & rng, boost::normal_distribution<double> & normalrng, boost::exponential_distribution<double> & exprng
,double & sqrth, unsigned int & N_rngcalls){
  Eigen::Matrix2d sigma_aux = sigma(X,t);
  Eigen::Vector2d Xp, Nprop, Np;
  double omega,uc,nu;
  if (d_k > -rho){
        do{
            Increment_Update(increment, rng, normalrng, sqrth);
            Xp = X + b(X,t)*h + sigma_aux*increment;
            omega =  exprng(rng); //Exponential distribution with parameter 1/(2*h)
            N_rngcalls += 3;
            uc = (N.transpose()*sigma_aux).dot(increment) +N.transpose().dot(b(X,t))*h;
            nu = 0.5 *(uc+sqrt(pow((N.transpose()*sigma_aux).norm(),2.0)*omega+pow(uc,2.0)));
            //d_k = bvp.boundary.Dist(params, Xp,E_Pp,Np);
            if (d_k < -0.0) d_k = 0.0;
            ji_t = std::max(0.0,nu+d_k);
            Xp = Xp - ji_t*N;
        }while((Xp - Nprop).dot(N)>0.0);
        d_k = distance(params, Xp,Nprop,Np);
        if(d_k > 0.0){
            //printf("WARNING: The particle didn't enter in the domain  after Lepingle step.\n");
            Xp = Nprop;
            ji_t += d_k;
            d_k = 0.0;
        }
    } else {
        Increment_Update(increment, rng, normalrng, sqrth);
        N_rngcalls += 2;
        Xp = X + b(X,t)*h + sigma_aux*increment;
        ji_t = 0.0;
        d_k = distance(params, Xp,Nprop,Np);
        if(d_k > -0.0){
            Xp = Nprop;
            ji_t = d_k;
            d_k = 0.0;
            //std::cout << ji_t << std::endl;
        }
    }
    Z += f(X,t)*Y*h + psi(X,N,t)*Y*ji_t;
    xi +=  Y *(-sigma_aux.transpose()*gradient(X,t)).dot(increment);
    Y += c(X,t)*Y*h + varphi(X,N,t) * Y * ji_t;
    X = Xp;
    N = Np;
    Npro = Nprop;
    t += - h;
    ji_t = 0.0;
}
//Status of the particle 
enum outcome {in, stop, reflect, time_out};
inline outcome Inside(double & distance, bool & stoppingbc, bool & Neumannbc, Eigen::Vector2d & X, double & t, double &ji_t, double T, double & sqrth,
Eigen::Vector2d & Npro, Eigen::Vector2d & N, double *params, double Gobet_Constant, pfdist dist, pfbtype stopf, pfbtype Neumannf, pfmatrix sigma){

    distance = dist(params, X, Npro, N);
    stoppingbc = stopf(Npro);
    Neumannbc = Neumannf(Npro);
    Eigen::Matrix2d sigma_aux = sigma(X,t);
    outcome status;
    if(stoppingbc){

      if(distance < Gobet_Constant*(N.transpose()*sigma_aux).norm()*sqrth){

        status = (t>0) ? in : time_out;

      } else {

        status = (t>0) ? stop : time_out;

      }

    }else{
      if( distance <= 0){

        status = (t>0) ? in : time_out;

        } else {
          
        status = (t>0) ? reflect : time_out;
        X = Npro;

      }
    }

    switch(status){

      case stop:
        ji_t = 0.0;
        break;

      case reflect:
        ji_t = distance;
        X = Npro;
        break;

      case time_out:
        ji_t = 0.0;
        t = 0.0;
        break;

      default:
        ji_t = 0.0;
        break;
    }
    return status;
}