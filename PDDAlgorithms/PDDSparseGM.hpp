//mpiexec --mca orte_base_help_aggregate 0  -np 4 ./main
#ifndef PDDSPARSEGM
#define PDDSPARSEGM
//Fraction of the interface the stencil is elongated
#define STEN_ELONG 0.0
//Minimum value of N
#define N_min 500
//Minimum value of h
#define h_min 1E-05
//MPI TAGS
#define REQUEST_NODE 1
#define REPLY_NODE 2
#define NODE_POSITION 3
#define PARAMETERS 4
#define SIZES 5
#define IND_NORTH 6
#define IND_SOUTH 7
#define IND_EAST 8
#define IND_WEST 9
#define X_NORTH 10
#define X_SOUTH 12
#define X_EAST 14
#define X_WEST 16
#define Y_NORTH 11
#define Y_SOUTH 13
#define Y_EAST 15
#define Y_WEST 17
#define INTER_NUMBER 18
#define INTER_LABELS 19
#define TAG_Gi 20
#define TAG_Gj 21
#define TAG_Gval 22
#define TAG_Gvar 23
#define TAG_GCT 24
#define TAG_Bi 30
#define TAG_Bval 31
#define TAG_Bvar 32
#define TAG_BCT 33
#define TAG_APL 34
#define TAG_time 35
#define TAG_Estvar 36
#define TAG_RNGCalls 37
#define TAG_xi 38
#define TAG_phi 39
#define TAG_xixi 40
#define TAG_phiphi 41
#define TAG_phixi 42
#define TAG_phip 43
#define TAG_phiphip 44
#define TAG_phipxi 45
#define TAG_phi_trap 46
#define TAG_phiphi_trap 47
#define TAG_phi_trap_VR 48
#define TAG_phiphi_trap_VR 49
#define TAG_bias_num 50
#define TAG_bias_trap 51
#define TAG_Estvar_VR 52
#define TODO_JOB_UINT  101
#define TODO_JOB_INT  102
#define TODO_JOB_DOUBLE 103
#define DONE_JOB_UINT 111
#define DONE_JOB_INT 112
#define DONE_JOB_DOUBLE 113
#define REQUEST_INTERFACE 150
#define REPLY_INTERFACE 151
#define TAG_INTERFACE_X 160
#define TAG_INTERFACE_Y 161
#define TAG_INTERFACE_INDEX 162
//Interfaces
#define INTERSECTIONS_YES
//Stencil 
//#define SQUARE_PATCHES
//MPI Implementation
//#define LOCAL_SERVERS
#include "../Solvers/EMFKACSolver.hpp"
#include "../Meshes/interface.hpp"
#include "../Meshes/stencil.hpp"
#include "../Meshes/subdomain.hpp"
#include <mpi.h>
#include <time.h>
#include <string.h>
#include <fstream>
#include <forward_list>
#include <gsl/gsl_math.h>
#include <gsl/gsl_interp2d.h>
#include <gsl/gsl_spline2d.h>
#include <gsl/gsl_errno.h>
//enum direction {North, South, East, West};
typedef Eigen::Triplet<double,int> T;
class PDDSJob {
    public:
    int index[1];
    unsigned int N[2];
    double h[1];
    PDDSJob(void);
    void Send_To_Worker(MPI_Status & status, MPI_Comm & world);
    void Recieve_From_Server(int server, MPI_Comm & world);
    //void Send_To_Server(int server, MPI_Comm & world);
    //void Recieve_From_Worker(MPI_Status status, MPI_Comm & world);
};

class PDDSparseGM{
    friend class GMSolver;
    private:
        /*
        MPI-related variables
        */
        MPI_Comm world, workers;
        MPI_Group world_group, worker_group;
        MPI_Status status;
        int ranks[1],work_control[2],numprocs, myid, server, local_server, workerid;
        /*
        Interfaces of the problem
        */
        std::vector<Interface> interfaces;
        /*
        Subdomains of the problem
        */
        std::vector<Subdomain> subdomains;
        /*-maps each interface 2D index in its 1D index*/
        std::map<std::vector<int>, int> interface_map;
        /*
         -Number of interfaces (And subdomains) per direction.
         -Nuber of nodes per interface.
        */
        std::vector<int> iN,nN;
        /*File were the parameters of the problem are written*/
        std::string configuration_file;
        /*-h0 is the value of time discretization for the lowest level.
          -T_start is the time value where we want to compute dhe solution
          -eps represents the rse objective of the stimator
          -parameters stores the boundary parameters
          -factor that multiplies the internodal distance
          -constant of the RBF interpolator 
          */
        double h0, T_start, eps, fac, c2,wclock_start,wclock_measure,wclock_prev;
        std::vector<double> parameters;
        /*-SW is the south west point of the domain.
          -NE is the north east point of the domain.
          -Sol is the vector of solutions
        */
       Eigen::Vector2d SW, NE;
       /*G and B storage vector*/
       std::vector<double> G, G_CT, B, B_CT, G_var, B_var, APL, times, xi, phi, xixi, phiphi,
                           phixi, phip, phiphip, phipxi, Est_var, Est_var_VR, RNGCalls, PCoeff_o, PCoeff_n,
                           phi_trap, phiphi_trap, phi_trap_VR, phiphi_trap_VR,var_phi_trap,
                           var_phi_trap_VR, bias_num, bias_trap;
       std::vector<int>  G_j, G_i, B_i;
       /*Triplet's vector*/
       std::vector<T> T_vec_G, T_vec_Gvar, T_vec_GCT, T_vec_B, T_vec_Bvar, T_vec_BCT,
                      T_vec_APL, T_vec_times, T_vec_xi, T_vec_phi, T_vec_xixi, T_vec_phiphi,
                      T_vec_phixi, T_vec_phip, T_vec_phiphip, T_vec_phipxi, T_vec_Estvar,
                      T_vec_RNGCalls,T_vec_phi_trap,T_vec_phiphi_trap,T_vec_phi_trap_VR,
                      T_vec_phiphi_trap_VR, T_vec_bias_num, T_vec_bias_trap, T_vec_Estvar_VR;
        /*
          -N is the initial number of trayectories 
        */
       int N, N_job, nNodes;
       /*Stencil of the node*/
       Stencil node_stencil;
       /*Position of the Node being solver*/
       Eigen::Vector2d position;
       /*Filename of the Flux DEBUG File*/
       char debug_fname[256];
       Eigen::SparseMatrix<double> G_sparse_var, G_sparse_GM, G_sparse_InvApprox;
       Eigen::Vector2d bias, knot_solutions;
       /*Starts the MPI values and structures*/
       void MPI_Configuration(int argc, char *argv[]);
       /*Returns the interfaces the node belongst to*/
       std::vector<std::vector<int> > Get_Interfaces(int index);
       /*Returns the stencils labels*/
       std::map<direction, std::vector<std::vector <int> > > Labels_Stencil(int index);
       std::map<direction, std::vector<std::vector <int> > > Labels_Stencil_Square(int index);
       /*Sends node information*/
       void Send_Node(PDDSJob job);
       /*Send node information*/
       void Send_Node_Loop(PDDSJob job);
        /*Sends interface information*/
       void Send_Interface(std::vector<Eigen::Vector2d> positions, std::vector<int> indexes);
       /*Receives node information*/
       bool Receive_Node(PDDSJob & job);
       /*Receives interface information*/
       bool Receive_Interface(std::vector<double> & x, std::vector<double> &y, std::vector<int> &indexes);
       /*Sends the stencil data to the worker*/
       void Send_Stencil_Data(int index);
       void Send_Stencil_Data_Loop(int index);
       /*Receives the stencil data from the server*/
       Stencil Recieve_Stencil_Data(void);
       void Set_LUT_FILE_Solution(int index, std::vector<double> & aux_x,std::vector<double> & aux_y,std::vector<double> & aux_sol);
       void Set_LUT_FILE_Correction(int index, std::vector<double> & aux_x,std::vector<double> & aux_y,std::vector<double> & aux_sol);
       void Init_LUT(unsigned int Nx, unsigned int Ny, double *x, double *y, double *value, 
       gsl_spline2d *spline, gsl_interp_accel *xaccel, gsl_interp_accel *yaccel);
       Stencil Recieve_Stencil_Data_Loop(void);
       Stencil Compute_Stencil(int index);
       /*Send G matrix and B Matrix*/
       void Send_G_B(void);
       /*Send B vector to server*/
       void Send_B(void);
       /*Send Metadata from worker to server*/
       void Send_Metadata(int aux_server);
       /*Send 2 Gmatrix and 2 BMatrix*/
       void Send_G_B_2(int aux_server);
       /*Receive G matrix and B Matrix*/
       void Receive_G_B(void);
       /*Receive B vector from worker*/
       void Receive_B(void);
       /*Send Metadata from worker to server*/
       void Receive_Metadata(void);
       /*Process Metadata and prints it*/ 
       void Process_Metadata(void);
       /*Send 2 Gmatrix and 2 BMatrix*/
       void Receive_G_B_2(void);
       /*Reduce G and B*/
       //void Reduce_G_B(void);
       /*Computes the inner elements of B with the pseudospectral method*/
       //void Compute_B_Deterministic(void);
       /*Having G and B as triplets, computes the solution for the PDDS problem*/
       void Compute_Solution(bvp BoundValProb);
       /*Having 2 Gs and 2 Bs as triplets, computes the solution for the PDDS problem*/
       //void Compute_Solution_2(bvp BoundValProb);
       /*Position of a given node*/
       Eigen::Vector2d Node_Position(int node_index);
       /*Prints metadata file*/
       void Print_metadata(void);
       /*Prints solution file*/
       void Print_solution(bvp BoundValProb);
       /*Prints Domains files*/
       void Print_subdomain(bvp BoundValProb);
       /*Reading subdomain files routines*/
       std::vector<double> Read_File(char fname[256]);
    public:
       /*Reads the problem information from a file.*/
       void ReadFromFile(std::string file);
       /*Interface and node structure intialization*/
       void Fullfill_interfaces(void);
       /*Interface and node structure intialization*/
       void Fullfill_subdomains(bvp BoundValProb);
       /*Initialization of the class*/
       PDDSparseGM(int argc, char *argv[]);
       /*Initialization of the class with a configuration file*/
       PDDSparseGM(int argc, char *argv[], std::string file);
       /*Solves the problem given by the configuration file*/
       void Solve(bvp BoundValProb, std::string file);
       /*Solves the problem given by the configuration file*/
       void Solve(bvp BoundValProb); 
       /*Solves the problem given by the configuration file*/
       void Solve_CUDA(bvp BoundValProb); 
        /*Prints the problem parameters on screen*/
       void Print_Problem(void); 
       /*Prints the content of the Interface vector on screen*/
       void Print_Interface(void);
       /*Reads solution file*/
       void Read_Solution(void);
       /*Random solution into interfaces*/
       void Random_Solution(double constant);
       /*Fullfills subdomains with random numbers*/
       void Fullfill_Subdomains_Random(bvp BoundValProb, double constant);
       /*Solves subdomains*/
       void Solve_Subdomains(bvp BoundValProb);
       /*Solves a non-linear problem*/
       void Solve_SemiLin(int iteration, bvp Lin_BVP); 
       /*Solves Subdomains of a Semilinear problem*/
       void Solve_Subdomains_SemiLin(int iteration, bvp BoundValProb);
       /*Solves Subdomains in the first iteration of a linear problem iterative solver*/
       void Solve_Subdomains_LinIt_First(bvp BoundValProb);
       /*Solves Subdomains in the not-first iterations of a linear problem iterative solver*/
       void Solve_Subdomains_LinIt(bvp BoundValProb);
       /*Solves semilinear equation with Variance Reduction*/
       void Solve_SemiLin_numVR(int iteration, bvp Lin_BVP);
       /*Solves a linear BVP with the iterative algorithm*/
       void Solve_Iterative_numVR(int iteration, bvp Lin_BVP);
       /*Computes the optimal h and N given a bias and variance estimation*/
       //void Compute_h_N(bvp BoundValProb, double eps, std::vector<double> & h, std::vector<int> & N);
       /*Solves with numerical Variance Reduction after a warm-up fase*/
       //void Solve_NumVR(bvp BoundValProb, std::vector<double> h_vec, std::vector<int> N_vec);
       /*Updates Times File*/
       void Update_TimeFile(std::string task,int N_processors);
};
#endif