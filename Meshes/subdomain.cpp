#include "subdomain.hpp"
Subdomain::Subdomain(void){
    solved = false;
}
void Subdomain::Init(std::vector<int> subdomain_label, std::map<direction, 
                    Interface> input_interfaces, std::vector<double> boundary_params,
                    bvp boundary_value_problem){
    label = subdomain_label;
    //True if interface is on the boundary, false if isnt
    std::map<direction, bool> on_boundary;
    std::vector<direction> directions{North, South, East, West};
    for(std::vector<direction>::iterator direction_iterator = directions.begin();
    direction_iterator != directions.end();
    direction_iterator ++){
        if(input_interfaces.count(*direction_iterator)){
            interfaces[*direction_iterator] = input_interfaces[*direction_iterator];
            on_boundary[*direction_iterator] = false;
        } else {
            on_boundary[*direction_iterator] = true;
        }
    }
    Interface aux_interface;
    for(std::vector<direction>::iterator direction_iterator = directions.begin();
    direction_iterator != directions.end();
    direction_iterator ++){
        if(on_boundary[*direction_iterator]){
            switch (*direction_iterator){
            case North :
                aux_interface.position = interfaces[South].position;
                aux_interface.solution.resize(aux_interface.position.size());
                aux_interface.solution_noisy.resize(aux_interface.position.size());
                for(uint32_t index = 0; index < aux_interface.position.size();
                    index ++){
                    aux_interface.position[index][1] = boundary_params[3];
                    aux_interface.solution[index] = boundary_value_problem.g(aux_interface.position[index],0.0);
                    aux_interface.solution_noisy[index] = boundary_value_problem.g(aux_interface.position[index],0.0);
                }
            break;
            case South :
                aux_interface.position = interfaces[North].position;
                aux_interface.solution.resize(aux_interface.position.size());
                aux_interface.solution_noisy.resize(aux_interface.position.size());
                for(uint32_t index = 0; index < aux_interface.position.size();
                    index ++){
                    aux_interface.position[index][1] = boundary_params[1];
                    aux_interface.solution[index] = boundary_value_problem.g(aux_interface.position[index],0.0);
                    aux_interface.solution_noisy[index] = boundary_value_problem.g(aux_interface.position[index],0.0);
                }
            break;
            case West :
                aux_interface.position = interfaces[East].position;
                aux_interface.solution.resize(aux_interface.position.size());
                aux_interface.solution_noisy.resize(aux_interface.position.size());
                for(uint32_t index = 0; index < aux_interface.position.size();
                    index ++){
                    aux_interface.position[index][0] = boundary_params[0];
                    aux_interface.solution[index] = boundary_value_problem.g(aux_interface.position[index],0.0);
                    aux_interface.solution_noisy[index] = boundary_value_problem.g(aux_interface.position[index],0.0);
                }
            break;
            case East :
                aux_interface.position = interfaces[West].position;
                aux_interface.solution.resize(aux_interface.position.size());
                aux_interface.solution_noisy.resize(aux_interface.position.size());
                for(uint32_t index = 0; index < aux_interface.position.size();
                    index ++){
                    aux_interface.position[index][0] = boundary_params[2];
                    aux_interface.solution[index] = boundary_value_problem.g(aux_interface.position[index],0.0);
                    aux_interface.solution_noisy[index] = boundary_value_problem.g(aux_interface.position[index],0.0);
                }
            break;
            default :
                std::cout << "Something went wrong\n";
            }
            interfaces[*direction_iterator] = aux_interface;
        }
    }
}
//Solves the BVP in the subdomain
void Subdomain::Solve(MPI_Comm & world){
    //id of the process that it is solving the subdomain
    int myid;
    MPI_Comm_rank(world, &myid);
    std::stringstream aux_ss;
    FILE *pf;
    std::vector<direction> directions{North, South, East, West};
    char summon_Octave[256];
    //sprintf(summon_Octave,"octave-cli Poisson_eq_3.m %d %d %d",myid,label[0],label[1]);
    sprintf(summon_Octave,"octave-cli Solvers/SemLin_first_VR.m %d %d %d",myid,label[0],label[1]);
    for(std::vector<direction>::iterator direction_iterator = directions.begin();
    direction_iterator != directions.end();
    direction_iterator ++){
        aux_ss.str("");
        switch (*direction_iterator){
            case North:
                aux_ss << "Input/Interfaces/North_" << myid << ".txt";
            break;
            case South:
                aux_ss << "Input/Interfaces/South_" << myid << ".txt";
            break;
            case East:
                aux_ss << "Input/Interfaces/East_" << myid << ".txt";
            break;
            case West:
                aux_ss << "Input/Interfaces/West_" << myid << ".txt";
            break;
            default:
            std::cout << "Something went wrong writing files during Subdomain.Solve()" << std::endl;
        }
        //std::cout << aux_ss.str() << std::endl;
        pf = fopen(aux_ss.str().c_str(),"w");
        for(uint32_t i = 0;
            i < interfaces[*direction_iterator].position.size();
            i++){
            fprintf(pf,"%f,%f,%f,%f\n",interfaces[*direction_iterator].position[i][0]
            ,interfaces[*direction_iterator].position[i][1],interfaces[*direction_iterator].solution[i],
            interfaces[*direction_iterator].solution_noisy[i]);
        }
        fclose(pf);
    }
    system(summon_Octave);
}
void Subdomain::Solve_NL(MPI_Comm & world){
    //id of the process that it is solving the subdomain
    int myid;
    MPI_Comm_rank(world, &myid);
    std::stringstream aux_ss;
    FILE *pf;
    std::vector<direction> directions{North, South, East, West};
    char summon_Octave[256];
    //sprintf(summon_Octave,"octave-cli Poisson_eq_3.m %d %d %d",myid,label[0],label[1]);
    sprintf(summon_Octave,"octave-cli Solvers/SemLin_VR.m %d %d %d",myid,label[0],label[1]);
    for(std::vector<direction>::iterator direction_iterator = directions.begin();
    direction_iterator != directions.end();
    direction_iterator ++){
        aux_ss.str("");
        switch (*direction_iterator){
            case North:
                aux_ss << "Input/Interfaces/North_" << myid << ".txt";
            break;
            case South:
                aux_ss << "Input/Interfaces/South_" << myid << ".txt";
            break;
            case East:
                aux_ss << "Input/Interfaces/East_" << myid << ".txt";
            break;
            case West:
                aux_ss << "Input/Interfaces/West_" << myid << ".txt";
            break;
            default:
            std::cout << "Something went wrong writing files during Subdomain.Solve()" << std::endl;
        }
        //std::cout << aux_ss.str() << std::endl;
        pf = fopen(aux_ss.str().c_str(),"w");
        for(uint32_t i = 0;
            i < interfaces[*direction_iterator].position.size();
            i++){
            fprintf(pf,"%f,%f,%f,%f\n",interfaces[*direction_iterator].position[i][0]
            ,interfaces[*direction_iterator].position[i][1],interfaces[*direction_iterator].solution[i],
            interfaces[*direction_iterator].solution_noisy[i]);
        }
        fclose(pf);
    }
    system(summon_Octave);
}
void Subdomain::Solve_LinIt_First(MPI_Comm & world){
    //id of the process that it is solving the subdomain
    int myid;
    MPI_Comm_rank(world, &myid);
    std::stringstream aux_ss;
    FILE *pf;
    std::vector<direction> directions{North, South, East, West};
    char summon_Octave[256];
    //sprintf(summon_Octave,"octave-cli Poisson_eq_3.m %d %d %d",myid,label[0],label[1]);
    sprintf(summon_Octave,"octave-cli Solvers/LinIt_first.m %d %d %d",myid,label[0],label[1]);
    for(std::vector<direction>::iterator direction_iterator = directions.begin();
    direction_iterator != directions.end();
    direction_iterator ++){
        aux_ss.str("");
        switch (*direction_iterator){
            case North:
                aux_ss << "Input/Interfaces/North_" << myid << ".txt";
            break;
            case South:
                aux_ss << "Input/Interfaces/South_" << myid << ".txt";
            break;
            case East:
                aux_ss << "Input/Interfaces/East_" << myid << ".txt";
            break;
            case West:
                aux_ss << "Input/Interfaces/West_" << myid << ".txt";
            break;
            default:
            std::cout << "Something went wrong writing files during Subdomain.Solve()" << std::endl;
        }
        //std::cout << aux_ss.str() << std::endl;
        pf = fopen(aux_ss.str().c_str(),"w");
        for(uint32_t i = 0;
            i < interfaces[*direction_iterator].position.size();
            i++){
            fprintf(pf,"%f,%f,%f,%f\n",interfaces[*direction_iterator].position[i][0]
            ,interfaces[*direction_iterator].position[i][1],interfaces[*direction_iterator].solution[i],
            interfaces[*direction_iterator].solution_noisy[i]);
        }
        fclose(pf);
    }
    system(summon_Octave);
}
void Subdomain::Solve_LinIt(MPI_Comm & world){
    //id of the process that it is solving the subdomain
    int myid;
    MPI_Comm_rank(world, &myid);
    std::stringstream aux_ss;
    FILE *pf;
    std::vector<direction> directions{North, South, East, West};
    char summon_Octave[256];
    //sprintf(summon_Octave,"octave-cli Poisson_eq_3.m %d %d %d",myid,label[0],label[1]);
    sprintf(summon_Octave,"octave-cli Solvers/LinIt.m %d %d %d",myid,label[0],label[1]);
    for(std::vector<direction>::iterator direction_iterator = directions.begin();
    direction_iterator != directions.end();
    direction_iterator ++){
        aux_ss.str("");
        switch (*direction_iterator){
            case North:
                aux_ss << "Input/Interfaces/North_" << myid << ".txt";
            break;
            case South:
                aux_ss << "Input/Interfaces/South_" << myid << ".txt";
            break;
            case East:
                aux_ss << "Input/Interfaces/East_" << myid << ".txt";
            break;
            case West:
                aux_ss << "Input/Interfaces/West_" << myid << ".txt";
            break;
            default:
            std::cout << "Something went wrong writing files during Subdomain.Solve()" << std::endl;
        }
        //std::cout << aux_ss.str() << std::endl;
        pf = fopen(aux_ss.str().c_str(),"w");
        for(uint32_t i = 0;
            i < interfaces[*direction_iterator].position.size();
            i++){
            fprintf(pf,"%f,%f,%f,%f\n",interfaces[*direction_iterator].position[i][0]
            ,interfaces[*direction_iterator].position[i][1],interfaces[*direction_iterator].solution[i],
            interfaces[*direction_iterator].solution_noisy[i]);
        }
        fclose(pf);
    }
    system(summon_Octave);
}
void Subdomain::Fullfill_Random(MPI_Comm & world, double constant){
    //id of the process that it is solving the subdomain
    int myid;
    MPI_Comm_rank(world, &myid);
    std::stringstream aux_ss;
    FILE *pf;
    std::vector<direction> directions{North, South, East, West};
    char summon_Octave[256];
    //sprintf(summon_Octave,"octave-cli Poisson_eq_3.m %d %d %d",myid,label[0],label[1]);
    sprintf(summon_Octave,"octave-cli Solvers/Init_Rand.m %d %d %d %f",myid,label[0],label[1], constant);
    for(std::vector<direction>::iterator direction_iterator = directions.begin();
    direction_iterator != directions.end();
    direction_iterator ++){
        aux_ss.str("");
        switch (*direction_iterator){
            case North:
                aux_ss << "Input/Interfaces/North_" << myid << ".txt";
            break;
            case South:
                aux_ss << "Input/Interfaces/South_" << myid << ".txt";
            break;
            case East:
                aux_ss << "Input/Interfaces/East_" << myid << ".txt";
            break;
            case West:
                aux_ss << "Input/Interfaces/West_" << myid << ".txt";
            break;
            default:
            std::cout << "Something went wrong writing files during Subdomain.Solve()" << std::endl;
        }
        //std::cout << aux_ss.str() << std::endl;
        pf = fopen(aux_ss.str().c_str(),"w");
        for(uint32_t i = 0;
            i < interfaces[*direction_iterator].position.size();
            i++){
            fprintf(pf,"%f,%f,%f,%f\n",interfaces[*direction_iterator].position[i][0]
            ,interfaces[*direction_iterator].position[i][1],interfaces[*direction_iterator].solution[i],
            interfaces[*direction_iterator].solution_noisy[i]);
        }
        fclose(pf);
    }
    system(summon_Octave);
}
void Subdomain::Send_To_Worker(MPI_Status & status, MPI_Comm & world){
    std::vector<direction> directions{North, South, East, West};
    int aux_int[2];
    aux_int[0] = label[0]; aux_int[1] = label[1];
    MPI_Send(aux_int, 2, MPI_INT, status.MPI_SOURCE, SEND_LABEL, world);
    for(std::vector<direction>::iterator direction_iterator = directions.begin();
    direction_iterator != directions.end();
    direction_iterator ++){
        Send_direction(status, world, *direction_iterator);
    }
}
void Subdomain::Send_direction(MPI_Status & status, MPI_Comm & world, direction dir){
    int aux_int[2];
    aux_int[0] = dir;
    aux_int[1] = (int) interfaces[dir].position.size();
    MPI_Send(aux_int, 2, MPI_INT, status.MPI_SOURCE, SEND_DIRECTION, world);
    //printf("Direction %d sent from server to %d\n", dir, status.MPI_SOURCE);
    double *aux_y, *aux_x, *aux_sol, *aux_sol_noisy;
    aux_y = new double[aux_int[1]];
    aux_x = new double[aux_int[1]];
    aux_sol = new double[aux_int[1]];
    aux_sol_noisy = new double[aux_int[1]];
    for(int i = 0; i < aux_int[1]; i++){
        aux_x[i] = interfaces[dir].position[i][0];
        aux_y[i] = interfaces[dir].position[i][1];
        aux_sol[i] = interfaces[dir].solution[i];
        aux_sol_noisy[i] = interfaces[dir].solution_noisy[i];
    }
    MPI_Send(aux_x, aux_int[1], MPI_DOUBLE, status.MPI_SOURCE, SEND_X, world);
    MPI_Send(aux_y, aux_int[1], MPI_DOUBLE, status.MPI_SOURCE, SEND_Y, world);
    MPI_Send(aux_sol, aux_int[1], MPI_DOUBLE, status.MPI_SOURCE, SEND_SOL, world);
    MPI_Send(aux_sol_noisy, aux_int[1], MPI_DOUBLE, status.MPI_SOURCE, SEND_SOL_NOISY, world);
}
void Subdomain::Recieve_From_Server(int server, MPI_Comm & world){
    std::vector<direction> directions{North, South, East, West};
    int aux_int[2];
    MPI_Recv(aux_int, 2, MPI_INT, server, SEND_LABEL, world, MPI_STATUS_IGNORE);
    label.resize(2);
    label[0] = aux_int[0]; label[1] = aux_int[1];
    for(std::vector<direction>::iterator direction_iterator = directions.begin();
    direction_iterator != directions.end();
    direction_iterator ++){
        Recieve_direction(server, world, *direction_iterator);
    }
}
void Subdomain::Recieve_direction(int server, MPI_Comm & world, direction dir){
    int aux_int[2];
    int myid;
    std::vector<direction> directions{North, South, East, West};
    MPI_Comm_rank(world, &myid);
    MPI_Recv(aux_int, 2, MPI_INT, server, SEND_DIRECTION, world, MPI_STATUS_IGNORE);
    //printf("Direction %d received in %d from server\n", aux_int[0], myid);
    double *aux_y, *aux_x, *aux_sol, *aux_sol_noisy;
    aux_y = new double[aux_int[1]];
    aux_x = new double[aux_int[1]];
    aux_sol = new double[aux_int[1]];
    aux_sol_noisy = new double[aux_int[1]];
    MPI_Recv(aux_x, aux_int[1], MPI_DOUBLE, server, SEND_X, world, MPI_STATUS_IGNORE);
    MPI_Recv(aux_y, aux_int[1], MPI_DOUBLE, server, SEND_Y, world, MPI_STATUS_IGNORE);
    MPI_Recv(aux_sol, aux_int[1], MPI_DOUBLE, server, SEND_SOL, world, MPI_STATUS_IGNORE);
    MPI_Recv(aux_sol_noisy, aux_int[1], MPI_DOUBLE, server, SEND_SOL_NOISY, world, MPI_STATUS_IGNORE);
    interfaces[dir].position.resize(aux_int[1]);
    interfaces[dir].solution.resize(aux_int[1]);
    interfaces[dir].solution_noisy.resize(aux_int[1]);
    for(int i = 0; i < aux_int[1]; i++){
        interfaces[dir].position[i].resize(2);
        interfaces[dir].position[i][0] = aux_x[i];
        interfaces[dir].position[i][1] = aux_y[i];
        interfaces[dir].solution[i] = aux_sol[i];
        interfaces[dir].solution_noisy[i] = aux_sol_noisy[i];
    }
}
//Prints the solution in a given filename
void Subdomain::Print_Solution(char* filename){
    printf("To be coded\n");
}