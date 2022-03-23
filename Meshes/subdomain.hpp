#ifndef SUBDOMAIN
#define SUBDOMAIN
#include <string> 
#include <iostream>  
#include <sstream> 
#include <vector>
#include <eigen3/Eigen/Core>
#include "mpi.h"
#include "interface.hpp"
#include "stencil.hpp"
#define ASK_SERVER 200
#define REPLY_WORKER 201
#define SEND_WORK 202
#define END_WORKER 203
#define SEND_LABEL 210
#define SEND_DIRECTION 211
#define SEND_X 212
#define SEND_Y 213
#define SEND_SOL 214
#define SEND_SOL_NOISY 215
//enum direction {North, South, East, West};
class Subdomain{
    private:
    //North, east, south and west interfaces
    std::map<direction, Interface> interfaces;
    //Mesh of points inside the subdomain
    std::vector<Eigen::VectorXd> mesh;
    //Solution of the nodes of the Mesh
    std::vector<double> solution;
    public:
    //True if the subdomain is solved
    bool solved;
    //Label of the subdomain
    std::vector<int> label;
    //Default initialization
    Subdomain(void);
    //Initialization of the variables of an instance
    void Init(std::vector<int> subdomain_label, std::map<direction, Interface> interfaces, 
              std::vector<double> boundary_params, bvp BoundValProb);
    //Solves the BVP in the subdomain
    void Solve(MPI_Comm & world);
    //Solves the BVP in the subdomain
    void Solve_NL(MPI_Comm & world);
    //Sends the subdomain information to the worker
    void Send_To_Worker(MPI_Status & status, MPI_Comm & world);
    void Send_direction(MPI_Status & status, MPI_Comm & world, direction dir);
    //Receives the subdomain information from the server
    void Recieve_From_Server(int server, MPI_Comm & world);
    void Recieve_direction(int server, MPI_Comm & world, direction dir);
    //Prints the solution in a given filename
    void Print_Solution(char* filename);
};
#endif