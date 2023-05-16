#include "PDDSparseGM.hpp"
PDDSJob::PDDSJob(void){
    index[0] = -1;
    N[0] = 0;
    N[1] = 0;
}
void PDDSJob::Send_To_Worker(MPI_Status & status, MPI_Comm & world){
    MPI_Send(N, 2, MPI_UNSIGNED, status.MPI_SOURCE, TODO_JOB_UINT, world);
    //printf("N sent from server to %d\n", status.MPI_SOURCE);
    MPI_Send(index, 1, MPI_INT, status.MPI_SOURCE, TODO_JOB_INT, world);
    //printf("Index sent from server to %d\n", status.MPI_SOURCE);
    MPI_Send(h, 1, MPI_DOUBLE, status.MPI_SOURCE, TODO_JOB_DOUBLE, world);
    //printf("h sent from server to %d\n", status.MPI_SOURCE);
}
void PDDSJob::Recieve_From_Server( int server, MPI_Comm & world){
    int myid;
    MPI_Comm_rank(world, &myid);
    MPI_Recv(N, 2, MPI_UNSIGNED, server, TODO_JOB_UINT, world, MPI_STATUS_IGNORE);
    //printf("N received from server in %d\n", myid);
    MPI_Recv(index, 1, MPI_INT, server, TODO_JOB_INT, world, MPI_STATUS_IGNORE);
    //printf("Index received from server in  %d\n", myid);
    MPI_Recv(h, 1, MPI_DOUBLE, server, TODO_JOB_DOUBLE, world, MPI_STATUS_IGNORE);
    //printf("h received from server in  %d\n", myid);
}
/*void PDDSJob::Send_To_Server(int server, MPI_Comm & world){
    int myid;
    MPI_Comm_rank(world, &myid);
    MPI_Send(N, 2, MPI_UNSIGNED, server, DONE_JOB_UINT, world);
    printf("N sent to server from %d\n", myid);
    MPI_Send(index, 1, MPI_INT, server, DONE_JOB_INT, world);
    printf("Index sent to server from %d\n", myid);
}
void PDDSJob::Recieve_From_Worker(MPI_Status status, MPI_Comm & world){
    MPI_Recv(N, 2, MPI_UNSIGNED, status.MPI_SOURCE, DONE_JOB_UINT, world, MPI_STATUS_IGNORE);
    printf("N received in server from %d\n", status.MPI_SOURCE);
    MPI_Recv(index, 1, MPI_INT, status.MPI_SOURCE, DONE_JOB_INT, world, MPI_STATUS_IGNORE);
    printf("Index received in server from %d\n", status.MPI_SOURCE);
}
*/
void PDDSparseGM::MPI_Configuration(int argc, char *argv[]){
    MPI_Init(&argc, &argv);
    world = MPI_COMM_WORLD;
    MPI_Comm_size(world, &numprocs);
    MPI_Comm_rank(world, &myid);
    server = numprocs-1;
    MPI_Comm_group(world, &world_group);
    ranks[0] = server;
    local_server = (myid/48)*48;
    //We exclude a group in world to create workers
    MPI_Group_excl(world_group, 1, ranks, &worker_group);
    //Workers comunicator is created
    MPI_Comm_create(world, worker_group, &workers);
    //We free both comunicators groups
    MPI_Group_free(&worker_group);
    MPI_Group_free(&world_group);
    sprintf(debug_fname,"Output/Debug/Debug_output_p%d.txt",myid);
    FILE *dfile;
    dfile = fopen(debug_fname,"w");
    fclose(dfile);
    if(myid == server){
      wclock_start = MPI_Wtime();
      wclock_prev = MPI_Wtime();
      wclock_measure = MPI_Wtime();
      FILE *pfile;
      pfile = fopen("Output/Debug/times.txt","w");
      fprintf(pfile,"task,total_tike,task_time,N_process\n");
      fclose(pfile);
  }
}
PDDSparseGM::PDDSparseGM(int argc, char *argv[]){
  iN.resize(2);
  nN.resize(2);
  h0 = 0.0;
  T_start =INFINITY;
  eps = 0.5;
  SW.resize(2);
  NE.resize(2);
  N = 0;
  fac = 1;
  MPI_Configuration(argc, argv);
}
PDDSparseGM::PDDSparseGM(int argc, char *argv[],std::string file){ 
  MPI_Configuration(argc, argv);
  ReadFromFile(file);
  if(myid == server) Update_TimeFile("Reading configuration File",server+1);
  parameters.resize(4);
  parameters[0] = SW[0];
  parameters[1] = SW[1];
  parameters[2] = NE[0];
  parameters[3] = NE[1];
  Fullfill_interfaces();
  if(myid == server){
    Update_TimeFile("Fullfilling interfaces",1);
  }
}
void PDDSparseGM::ReadFromFile(std::string file){
  std::ifstream myfile(file.c_str());
    std::string line, cent;
    //If T is not specified then the problem is suposed to be elliptic
    T_start = INFINITY;
    if (myfile)  // same as: if (myfile.good())
    {   while (getline( myfile, line )){  // same as: while (getline( myfile, line ).good())

            if (line.c_str()[0]!='%') //First of all we check if the line is a comment
            { 
                cent.clear(); //Centinel needs to be empty

                for(std::string::iterator it = line.begin(); it != line.end(); it++){
                //Loop over all the characters of the line
                    
                    if(*it != ' ' && *it != '\t'){
                    //If it is not a space nor a tabulation then we concatenate the term to cent
                        cent += *it;
                    }
                    else{
                    //If we get a space or a tabulation then we check the string
                        if(cent == "h0="){
                            while( *it == ' ' or *it == '\t'){

                                        it++;

                            }

                            cent.clear();

                            while( it != line.end()){

                                cent += *it;
                                it++;

                            }
                            h0 = atof(cent.c_str());
                            break;
                        }
                        if(cent == "eps="){
                            while( *it == ' ' or *it == '\t'){

                                        it++;

                            }

                            cent.clear();

                            while( it != line.end()){

                                cent += *it;
                                it++;

                            }
                            
                            eps = atof(cent.c_str());
                            break;
                        }
                        if(cent == "fac="){
                            while( *it == ' ' or *it == '\t'){

                                        it++;

                            }

                            cent.clear();

                            while( it != line.end()){

                                cent += *it;
                                it++;

                            }
                            
                            fac = atof(cent.c_str());
                            break;
                        }
                        if(cent == "T="){
                         
                            while( *it == ' ' or *it == '\t'){

                                        it++;

                            }

                            cent.clear();

                            while( it != line.end()){

                                cent += *it;
                                it++;

                            }
                            
                            T_start = atof(cent.c_str());
                            break;
                        }
                        if(cent == "SW="){
                        //South West point of the domain 
                            SW.resize(2);
                            while( *it == ' ' or *it == '\t'){
                                        it++;

                            }
                            cent.clear();
                            for(int i = 0; i < 2; i++){
                                while((*it != ' ') && (*it != '\t') && (it != line.end())){

                                cent += *it;
                                it++;

                                }
                                SW(i) = atof(cent.c_str());
                                cent.clear();
                                while( *it == ' ' or *it == '\t'){
                                        it++;
                                }
                    
                            }

                            break;
                        }

                        if(cent == "NE="){
                        //South West point of the domain 
                            NE.resize(2);
                            while( *it == ' ' or *it == '\t'){
                                        it++;

                            }
                            cent.clear();
                            for(int i = 0; i < 2; i++){
                            while((*it != ' ') && (*it != '\t') && (it != line.end())){

                                cent += *it;
                                it++;

                            }
                            NE(i) = atof(cent.c_str());
                            cent.clear();
                            while( *it == ' ' or *it == '\t'){
                                        it++;
                            }
                    
                            }

                            break;
                        }

                        if(cent == "N="){
                        //Number of trayectories per node

                            while( *it == ' ' or *it == '\t'){

                                        it++;
                            }

                            cent.clear();

                            while( it != line.end()){

                                cent += *it;
                                it++;

                            }
                                N = atoi(cent.c_str()); 
                            break;
                        }
                        if(cent == "N_job="){
                        //Number of trayectories per node

                            while( *it == ' ' or *it == '\t'){

                                        it++;
                            }

                            cent.clear();

                            while( it != line.end()){

                                cent += *it;
                                it++;

                            }
                            
                            N_job = atoi(cent.c_str());
                            break;
                        }
                        if(cent == "N_subdo="){
                        //Number of domains in each direction

                            while( *it == ' ' or *it == '\t'){

                                        it++;
                            }

                            cent.clear();

                            while( it != line.end()){

                                cent += *it;
                                it++;

                            }
                            iN.resize(2);
                            iN[0] = atoi(cent.c_str());
                            iN[1] = atoi(cent.c_str());
                            break;
                        }

                        if(cent == "N_node="){
                        //Number nodes per interface

                            while( *it == ' ' or *it == '\t'){

                                        it++;
                            }

                            cent.clear();

                            while( it != line.end()){

                                cent += *it;
                                it++;

                            }
                            
                            nN.resize(2);
                            nN[0] = atoi(cent.c_str());
                            nN[1] = atoi(cent.c_str());
                            break;
                        }
                        
                    }
                }
            }
        }
        myfile.close();

    }

    else {
    
        std::cout << "WARNING: Cofiguration file was not found\n";
        std::cout << "Make sure it exist a configuration.txt file in the program's directory.\n";
    }    
}
void PDDSparseGM::Fullfill_interfaces(void){
   /*Initial values, long and  and shot increments for each direction*/
   std::vector<Eigen::Vector2d> P_cent, lincrement, sincrement, start, end;
   Eigen::Vector2d vaux;
   std::vector<std::vector<int>> node_index;
   interfaces.resize(iN[0]*(iN[1]-1)+ iN[1]*( iN[0]-1));
   subdomains.resize(iN[0]*(iN[1]-1)+ iN[1]*( iN[0]-1));
   node_index.resize(iN[0]*(iN[1]-1)+ iN[1]*( iN[0]-1));
   vaux.resize(SW.size());

   //Increments are defined
   //ShortIncrement
   for (int i = 0; i < SW.size() ; i++){
        for(int j = 0; j < vaux.size(); j++){
            if (i == j){

                vaux(j) = (NE(j) - SW(j))/(1.0* ((nN[j]-1)*iN[j]));
            } else {

                vaux(j) = 0.0;

            }
        }
        sincrement.push_back(vaux);
    }
    //LongIncrement
    for (int i = 0; i < SW.size() ; i++){

        for(int j = 0; j < vaux.size(); j++){

            if (i == j){

                vaux(j) = (NE(j) - SW(j))/(1.0* (iN[j]));

            } else {

                vaux(j) = 0.0;

            }
        }

        lincrement.push_back(vaux);
    }
    //Initial Point
   
    for (int i = 0; i < SW.size(); i++){
            vaux = SW;
            P_cent.push_back(vaux);
    }
    //c2 is computed as the square of fac*internodal space.
    c2 = pow(fac*sincrement[0].norm(),2.0);
    //This section is not valid for + 2D problems

    P_cent[0] += lincrement[1];
    P_cent[1] += lincrement[0];

    int cen = 0;

    for (int i = 0; i < iN[1] - 1; i++){

        for (int j = 0; j < iN[0]; j++){

            start.push_back(P_cent[0] + lincrement[0]*j);
            end.push_back(P_cent[0] + lincrement[0]*(j+1));

            interfaces[cen].label.push_back(j);
            interfaces[cen].label.push_back(2*i + 1);
            subdomains[cen].label.push_back(j);
            subdomains[cen].label.push_back(2*i + 1);
            cen++;

            start.push_back(P_cent[1] + lincrement[1]*j);
            end.push_back(P_cent[1] + lincrement[1]*(j+1));

            interfaces[cen].label.push_back(i);
            interfaces[cen].label.push_back(2*j);
            subdomains[cen].label.push_back(i);
            subdomains[cen].label.push_back(2*j);

            cen++;
        }


        P_cent[0] += lincrement[1];
        P_cent[1] += lincrement[0];

    }

    for(int i = 0; i < (int)interfaces.size(); i++){
        interface_map[interfaces[i].label] = i;
    }
    #ifdef INTERSECTIONS_YES
    //stvaux stores the current interface index.
    //cross stores the index of the nodes that constitutes 
    std::vector<int> stvaux, cross;
    unsigned int index, cent = 0, cross_cent = 0;
    cross.resize((iN[0]-1)*(iN[1]-1));
    //As It is done in Paco's original code we start by fullfilling the vertical interfaces
    cross.clear();
    for(int i = 0; i < iN[0] - 1; i++){
        for(int j = 0; j <= iN[1]*2 - 2; j += 2){
            stvaux.clear();
            stvaux.push_back(i);
            stvaux.push_back(j);
            index = interface_map[stvaux];
            node_index[index].push_back(cent);
            if(j != 0) cross[(j/2-1)*(iN[1]-1) + i] = cent;
            for(int k = 0; k < nN[1]-1; k ++){
                cent ++;
                node_index[index].push_back(cent);
            }
            if(j == iN[1]*2 -2) {
                cent ++;
            }
        }
    }
    //Horizontal interfaces
    for(int j = 1; j <= 2*iN[1]-3; j +=2 ){
        for(int i = 0; i < iN[0]; i ++){
            stvaux.clear();
            stvaux.push_back(i);
            stvaux.push_back(j);
            index = interface_map[stvaux];
            if(i != 0){
                node_index[index].push_back(cross[cross_cent]);
                cross_cent ++;
            } else {
                node_index[index].push_back(cent);
                cent ++;
            }
            for(int k = 1; k < nN[0] - 1; k ++){
                node_index[index].push_back(cent);
                cent ++;
            }
            if(i != iN[0] -1 ) {
                node_index[index].push_back(cross[cross_cent]);
            } else {
                node_index[index].push_back(cent);
                cent ++;
            }
        }
    }
    #else 
    //stvaux stores the current interface index.
    std::vector<int> stvaux;
    unsigned int index, cent = 0;
    for(int i = 0; i < iN[0] - 1; i++){
        for(int j = 0; j <= iN[1]*2 - 2; j += 2){
            stvaux.clear();
            stvaux.push_back(i);
            stvaux.push_back(j);
            index = interface_map[stvaux];
            node_index[index].push_back(cent);
            cent ++;
            //printf("[%d % d] nN = %d \n", i,j, nN[1]);
            for(int k = 0; k < nN[1] -1; k ++){
                node_index[index].push_back(cent);
                cent ++;
            }
        }
    }
    //Horizontal interfaces
    for(int j = 1; j <= 2*iN[1]-3; j +=2 ){
        for(int i = 0; i < iN[0]; i ++){
            stvaux.clear();
            stvaux.push_back(i);
            stvaux.push_back(j);
            index = interface_map[stvaux];
            node_index[index].push_back(cent);
            cent ++;
            //printf("[%d % d] nN = %d \n", i,j, nN[0]);
            for(int k = 0; k < nN[0] - 1; k ++){
                node_index[index].push_back(cent);
                cent ++;
            }
        }
    }
    #endif
    nNodes = (int)cent;
    FILE *dfile;
    dfile = fopen(debug_fname,"a");
    fprintf(dfile,"nNodes is %d\n",nNodes);
    fclose(dfile);
    //Loop over the points in the system
    //double disp;
    //disp = 0.5*(end[0]-start[0]).norm()/(node_index[0].size());
    for(std::vector<Interface>::iterator it = interfaces.begin();
    it != interfaces.end();
    it++){
        index = interface_map[(*it).label];
        (*it).index.resize(node_index[index].size());
        (*it).solution.resize(node_index[index].size());
        (*it).solution_noisy.resize(node_index[index].size());
        (*it).position.resize(node_index[index].size());
        sincrement[0] = (end[index]-start[index])/(node_index[index].size()-1);
        #ifdef INTERSECTIONS_YES
        vaux = start[index];
        #else
        if((*it).label[1]%2 == 0){
            //Vertical
            if((*it).label[1] == 0){
                end[index][1] = end[index][1] - disp;
            } else {
                if((*it).label[1] == 2*(iN[1] - 1)){
                    start[index][1] = start[index][1] + disp;
                } else {
                    start[index][1] = start[index][1] + disp;
                    end[index][1] = end[index][1] - disp;
                }
            }
        } else {
            //Horizontal
            if((*it).label[0] == 0){
                end[index][0] = end[index][0] - disp;
            } else {
                if((*it).label[0] == iN[0] - 1){
                start[index][0] = start[index][0] + disp;
                } else {
                    start[index][0] = start[index][0] + disp;
                    end[index][0] = end[index][0] - disp;
                }
            }
        }
        sincrement[0] = (end[index]-start[index])/(node_index[index].size()-1);
        vaux = start[index];
        #endif
        for(int i = 0; i < (int) node_index[index].size(); i++){
            (*it).index[i] = node_index[index][i];
            (*it).position[i] = vaux;
            //printf("%d %f %f\n", node_index[index][i],(*it).position[i][0],(*it).position[i][1]);
            vaux += sincrement[0];
        }
    }
    //getchar();
}

void PDDSparseGM::Solve(bvp BoundValProb){
    bool done=true;
    //double start = MPI_Wtime();
    if(myid==server) Print_Problem();
    //Compute the PDDSparse Matrix
    if(myid==server){
        FILE *pFile;
        pFile = fopen("Output/Debug/times.txt","a");
        fprintf(pFile,"************WARMING UP***********\n");
        fclose(pFile);
        std::vector<int> aux_vec;
        std::vector<Eigen::Vector2d> positions;
        std::vector<int> indexes;
        aux_vec.resize(2);
        for(int Iy = 0; Iy <= 2*iN[1]-2; Iy ++){
            if(Iy%2 == 0){
                for(int Ix = 0; Ix <= iN[0]-2; Ix ++){
                    aux_vec[0] = Ix;
                    aux_vec[1] = Iy;
                    if(Iy == 0){
                        indexes.assign(interfaces[interface_map[aux_vec]].index.begin(),
                        interfaces[interface_map[aux_vec]].index.end());
                        positions.assign(interfaces[interface_map[aux_vec]].position.begin(),
                        interfaces[interface_map[aux_vec]].position.end());
                    } else {
                        indexes.assign(interfaces[interface_map[aux_vec]].index.begin()+1,
                        interfaces[interface_map[aux_vec]].index.end());
                        positions.assign(interfaces[interface_map[aux_vec]].position.begin()+1,
                        interfaces[interface_map[aux_vec]].position.end());
                    }
                    //std::cout << "Sending interface [" << aux_vec[0] << ","<< aux_vec[1] << "]"<< std::endl;
                    Send_Interface(positions, indexes);
                    //Send_Stencil_Data(indexes[0]);
                    //std::cout << "Interface [" << aux_vec[0] << ","<< aux_vec[1] << "] sent"<< std::endl;
                }
            } else {
                for(int Ix = 0; Ix <= iN[0]-1; Ix ++){
                    aux_vec[0] = Ix;
                    aux_vec[1] = Iy;
                    if(Ix == 0){
                        indexes.assign(interfaces[interface_map[aux_vec]].index.begin(),
                        interfaces[interface_map[aux_vec]].index.end()-1);
                        positions.assign(interfaces[interface_map[aux_vec]].position.begin(),
                        interfaces[interface_map[aux_vec]].position.end()-1);
                    } else {
                        if(Ix == iN[0]-1){
                            indexes.assign(interfaces[interface_map[aux_vec]].index.begin()+1,
                            interfaces[interface_map[aux_vec]].index.end());
                            positions.assign(interfaces[interface_map[aux_vec]].position.begin()+1,
                            interfaces[interface_map[aux_vec]].position.end());
                        }else{
                            indexes.assign(interfaces[interface_map[aux_vec]].index.begin()+1,
                            interfaces[interface_map[aux_vec]].index.end()-1);
                            positions.assign(interfaces[interface_map[aux_vec]].position.begin()+1,
                            interfaces[interface_map[aux_vec]].position.end()-1);
                        }
                    }
                    //std::cout << "Sending interface [" << aux_vec[0] << ","<< aux_vec[1] << "]"<< std::endl;
                    Send_Interface(positions, indexes);
                    //Send_Stencil_Data(indexes[0]);
                    //std::cout << "Interface [" << aux_vec[0] << ","<< aux_vec[1] << "] sent"<< std::endl;
                }
                
            }
        }
        Update_TimeFile("MC Simulations",server+1);
        //G and B are received from the workers
        #ifdef LOCAL_SERVERS
        #else
        for(int process = 0; process < server; process++){
             Receive_G_B();
        }
        Update_TimeFile("Receiving G and B",server+1);
        B.clear();
        B_i.clear();
        #endif
        //Compute_B_Deterministic();
        Compute_Solution(BoundValProb);
        //system("octave-cli SmoothingSplines/SmoothingSplines.m");
        //pFile = fopen(debug_fname,"a");
        //double end = MPI_Wtime();
    } else {
        //Start and end time for the node
        //double knot_start, knot_end;
        //G and B storage vectors for each node
        double B_temp, BB_temp;
        std::vector<double> G_temp, G_var_temp, x, y;
        std::vector<int>  G_j_temp, indexes;
        Eigen::Vector2d X0;
        //Stencil
        Stencil stencil;
        EMFKACSolver solver;
        c2 = pow(fac*(1.0/(nN[0]-1)),2.0);
        G_i.clear();G_j.clear(); G.clear(); G_CT.clear(); G_var_temp.clear(); 
        B.clear();B_CT.clear();B_i.clear();B_var.clear(); G_var.clear();
        while(! Receive_Interface(x,y,indexes)){
            //stencil.Print(indexes[0]);
            std::cout << "Interface received"<< std::endl;
            for(unsigned int knot = 0; knot < x.size(); knot ++){
                //knot_start = MPI_Wtime();
               
                stencil = Compute_Stencil(indexes[knot]);
                stencil.Compute_ipsi(BoundValProb,c2,debug_fname);
                for(int i = 0; i < 4; i++) stencil.global_parameters[i] = parameters[i];
                std::cout << "Solving knot " << indexes[knot] << std::endl;
                B_temp = 0.0;
                BB_temp = 0.0;
                G_j_temp.clear();
                G_temp.clear();
                G_var_temp.clear();
                X0(0) = x[knot]; X0(1) = y[knot];
                stencil.Reset();
                solver.Reset_sums();
                solver.N = 0;
                solver.Solve_OMP_Analytic(X0,(unsigned int) N,h0, 0.1,
                BoundValProb, stencil, c2, G_temp,G_var_temp,G_j_temp,B_temp,
                BB_temp);
                B.push_back(B_temp);
                B_i.push_back(indexes[knot]);
                B_var.push_back(B_temp*B_temp - BB_temp);
                for(unsigned int k = 0;k < G_temp.size(); k ++){
                    G_i.push_back(indexes[knot]);
                    G_j.push_back(G_j_temp[k]);
                    G.push_back(G_temp[k]);
                    G_var.push_back(G_var_temp[k]);
                }
                //knot_end = MPI_Wtime();
                
            }
        }
        std::cout << "OUT \n";
        Send_G_B();
        //Compute_B_Deterministic();
        //double end = MPI_Wtime();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    //MPI_Finalize();
}
/*void PDDSparseGM::Solve_CUDA(bvp BoundValProb){
    bool done=true;
    //double start = MPI_Wtime();
    if(myid==server) Print_Problem();
    //Compute the PDDSparse Matrix
    if(myid==server){
        FILE *pFile;
        pFile = fopen("Output/Debug/times.txt","a");
        fprintf(pFile,"************WARMING UP***********\n");
        fclose(pFile);
        std::vector<int> aux_vec;
        std::vector<Eigen::Vector2d> positions;
        std::vector<int> indexes;
        aux_vec.resize(2);
        for(int Iy = 0; Iy <= 2*iN[1]-2; Iy ++){
            if(Iy%2 == 0){
                for(int Ix = 0; Ix <= iN[0]-2; Ix ++){
                    aux_vec[0] = Ix;
                    aux_vec[1] = Iy;
                    if(Iy == 0){
                        indexes.assign(interfaces[interface_map[aux_vec]].index.begin(),
                        interfaces[interface_map[aux_vec]].index.end());
                        positions.assign(interfaces[interface_map[aux_vec]].position.begin(),
                        interfaces[interface_map[aux_vec]].position.end());
                    } else {
                        indexes.assign(interfaces[interface_map[aux_vec]].index.begin()+1,
                        interfaces[interface_map[aux_vec]].index.end());
                        positions.assign(interfaces[interface_map[aux_vec]].position.begin()+1,
                        interfaces[interface_map[aux_vec]].position.end());
                    }
                    std::cout << "Sending interface [" << aux_vec[0] << ","<< aux_vec[1] << "]"<< std::endl;
                    Send_Interface(positions, indexes);
                    //Send_Stencil_Data(indexes[0]);
                    std::cout << "Interface [" << aux_vec[0] << ","<< aux_vec[1] << "] sent"<< std::endl;
                }
            } else {
                for(int Ix = 0; Ix <= iN[0]-1; Ix ++){
                    aux_vec[0] = Ix;
                    aux_vec[1] = Iy;
                    if(Ix == 0){
                        indexes.assign(interfaces[interface_map[aux_vec]].index.begin(),
                        interfaces[interface_map[aux_vec]].index.end()-1);
                        positions.assign(interfaces[interface_map[aux_vec]].position.begin(),
                        interfaces[interface_map[aux_vec]].position.end()-1);
                    } else {
                        if(Ix == iN[0]-1){
                            indexes.assign(interfaces[interface_map[aux_vec]].index.begin()+1,
                            interfaces[interface_map[aux_vec]].index.end());
                            positions.assign(interfaces[interface_map[aux_vec]].position.begin()+1,
                            interfaces[interface_map[aux_vec]].position.end());
                        }else{
                            indexes.assign(interfaces[interface_map[aux_vec]].index.begin()+1,
                            interfaces[interface_map[aux_vec]].index.end()-1);
                            positions.assign(interfaces[interface_map[aux_vec]].position.begin()+1,
                            interfaces[interface_map[aux_vec]].position.end()-1);
                        }
                    }
                    std::cout << "Sending interface [" << aux_vec[0] << ","<< aux_vec[1] << "]"<< std::endl;
                    Send_Interface(positions, indexes);
                    //Send_Stencil_Data(indexes[0]);
                    std::cout << "Interface [" << aux_vec[0] << ","<< aux_vec[1] << "] sent"<< std::endl;
                }
                
            }
        }
        Update_TimeFile("MC Simulations",server+1);
        //G and B are received from the workers
        #ifdef LOCAL_SERVERS
        #else
        for(int process = 0; process < server; process++){
             Receive_G_B();
        }
        Update_TimeFile("Receiving G and B",server+1);
        B.clear();
        B_i.clear();
        #endif
        //Compute_B_Deterministic();
        Compute_Solution(BoundValProb);
        //system("octave-cli SmoothingSplines/SmoothingSplines.m");
        //pFile = fopen(debug_fname,"a");
        //double end = MPI_Wtime();
    } else {
        //Start and end time for the node
        //double knot_start, knot_end;
        //G and B storage vectors for each node
        double B_temp, BB_temp;
        std::vector<double> G_temp, G_var_temp, x, y;
        std::vector<int>  G_j_temp, indexes;
        Eigen::Vector2d X0;
        //Stencil
        Stencil stencil;
        EMFKACSolver solver;
        c2 = pow(fac*(1.0/(nN[0]-1)),2.0);
        G_i.clear();G_j.clear(); G.clear(); G_CT.clear(); G_var_temp.clear(); 
        B.clear();B_CT.clear();B_i.clear();B_var.clear(); G_var.clear();
        while(! Receive_Interface(x,y,indexes)){
            //stencil.Print(indexes[0]);
            std::cout << "Interface received"<< std::endl;
            for(unsigned int knot = 0; knot < x.size(); knot ++){
                //knot_start = MPI_Wtime();
                stencil = Compute_Stencil(indexes[knot]);
                stencil.Compute_ipsi(BoundValProb,c2,debug_fname);
                for(int i = 0; i < 4; i++) stencil.global_parameters[i] = parameters[i];
                std::cout << "Solving knot " << indexes[knot] << std::endl;
                B_temp = 0.0;
                BB_temp = 0.0;
                G_j_temp.clear();
                G_temp.clear();
                G_var_temp.clear();
                X0(0) = x[knot]; X0(1) = y[knot];
                stencil.Reset();
                solver.Reset_sums();
                solver.N = 0;
                solver.Solve_CUDA_Analytic(X0,(unsigned int) N,h0, 0.1,
                BoundValProb, stencil, c2, G_temp,G_var_temp,G_j_temp,B_temp,
                BB_temp);
                B.push_back(B_temp);
                B_i.push_back(indexes[knot]);
                B_var.push_back(B_temp*B_temp - BB_temp);
                for(unsigned int k = 0;k < G_temp.size(); k ++){
                    G_i.push_back(indexes[knot]);
                    G_j.push_back(G_j_temp[k]);
                    G.push_back(G_temp[k]);
                    G_var.push_back(G_var_temp[k]);
                }
                //knot_end = MPI_Wtime();
                
            }
        }
        std::cout << "OUT \n";
        Send_G_B();
        //Compute_B_Deterministic();
        //double end = MPI_Wtime();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    //MPI_Finalize();
}
*/
void PDDSparseGM::Solve_SemiLin(int iteration, bvp BoundValProb){
    bool done=true;
    //double start = MPI_Wtime();
    if(myid==server) Print_Problem();
    //Compute the PDDSparse Matrix
    if(myid==server){ 
        char order[256];
        sprintf(order,"cp -r Output Output_%d",iteration);
        system(order);
        FILE *pFile;
        pFile = fopen("Output/Debug/times.txt","a");
        fprintf(pFile,"************WARMING UP***********\n");
        fclose(pFile);
        std::vector<int> aux_vec;
        std::vector<Eigen::Vector2d> positions;
        std::vector<int> indexes;
        aux_vec.resize(2);
        for(int Iy = 0; Iy <= 2*iN[1]-2; Iy ++){
            if(Iy%2 == 0){
                for(int Ix = 0; Ix <= iN[0]-2; Ix ++){
                    aux_vec[0] = Ix;
                    aux_vec[1] = Iy;
                    if(Iy == 0){
                        indexes.assign(interfaces[interface_map[aux_vec]].index.begin(),
                        interfaces[interface_map[aux_vec]].index.end());
                        positions.assign(interfaces[interface_map[aux_vec]].position.begin(),
                        interfaces[interface_map[aux_vec]].position.end());
                    } else {
                        indexes.assign(interfaces[interface_map[aux_vec]].index.begin()+1,
                        interfaces[interface_map[aux_vec]].index.end());
                        positions.assign(interfaces[interface_map[aux_vec]].position.begin()+1,
                        interfaces[interface_map[aux_vec]].position.end());
                    }
                    std::cout << "Sending interface [" << aux_vec[0] << ","<< aux_vec[1] << "]"<< std::endl;
                    Send_Interface(positions, indexes);
                    //Send_Stencil_Data(indexes[0]);
                    std::cout << "Interface [" << aux_vec[0] << ","<< aux_vec[1] << "] sent"<< std::endl;
                }
            } else {
                for(int Ix = 0; Ix <= iN[0]-1; Ix ++){
                    aux_vec[0] = Ix;
                    aux_vec[1] = Iy;
                    if(Ix == 0){
                        indexes.assign(interfaces[interface_map[aux_vec]].index.begin(),
                        interfaces[interface_map[aux_vec]].index.end()-1);
                        positions.assign(interfaces[interface_map[aux_vec]].position.begin(),
                        interfaces[interface_map[aux_vec]].position.end()-1);
                    } else {
                        if(Ix == iN[0]-1){
                            indexes.assign(interfaces[interface_map[aux_vec]].index.begin()+1,
                            interfaces[interface_map[aux_vec]].index.end());
                            positions.assign(interfaces[interface_map[aux_vec]].position.begin()+1,
                            interfaces[interface_map[aux_vec]].position.end());
                        }else{
                            indexes.assign(interfaces[interface_map[aux_vec]].index.begin()+1,
                            interfaces[interface_map[aux_vec]].index.end()-1);
                            positions.assign(interfaces[interface_map[aux_vec]].position.begin()+1,
                            interfaces[interface_map[aux_vec]].position.end()-1);
                        }
                    }
                    std::cout << "Sending interface [" << aux_vec[0] << ","<< aux_vec[1] << "]"<< std::endl;
                    Send_Interface(positions, indexes);
                    //Send_Stencil_Data(indexes[0]);
                    std::cout << "Interface [" << aux_vec[0] << ","<< aux_vec[1] << "] sent"<< std::endl;
                }
                
            }
        }
        Update_TimeFile("MC Simulations",server+1);
        //G and B are received from the workers
        #ifdef LOCAL_SERVERS
        #else
        for(int process = 0; process < server; process++){
             Receive_G_B();
        }
        Update_TimeFile("Receiving G and B",server+1);
        B.clear();
        B_i.clear();
        #endif
        //Compute_B_Deterministic();
        Compute_Solution(BoundValProb);
        //pFile = fopen(debug_fname,"a");
        //double end = MPI_Wtime();
    } else {
        //Start and end time for the node
        //double knot_start, knot_end;
        //G and B storage vectors for each node
        double B_temp, BB_temp;
        std::vector<double> G_temp, G_var_temp, x, y, x_LUT,y_LUT,u_LUT,v_LUT;
        double *array_x, *array_y, *array_u, *array_v;
        std::vector<int>  G_j_temp, indexes;
        Eigen::Vector2d X0;
        //Stencil
        Stencil stencil;
        EMFKACSolver solver;
        //spline
        gsl_spline2d *LUT_u, *LUT_v;
        gsl_interp_accel *xacc_u, *yacc_u, *xacc_v, *yacc_v;
        c2 = pow(fac*(1.0/(nN[0]-1)),2.0);
        G_i.clear();G_j.clear(); G.clear(); G_CT.clear(); G_var_temp.clear(); 
        B.clear();B_CT.clear();B_i.clear();B_var.clear(); G_var.clear();
        while(! Receive_Interface(x,y,indexes)){

            //stencil.Print(indexes[0]);
            //std::cout << "Interface received"<< std::endl;
            for(unsigned int knot = 0; knot < x.size(); knot ++){
                //knot_start = MPI_Wtime();
                stencil = Compute_Stencil(indexes[knot]);
                stencil.Compute_ipsi(BoundValProb,c2,debug_fname);
                for(int i = 0; i < 4; i++) stencil.global_parameters[i] = parameters[i];
                Set_LUT_FILE_Solution(indexes[knot],x_LUT,y_LUT,u_LUT);
                array_x = new double[x_LUT.size()];for(unsigned int i = 0; i < x_LUT.size(); i++) array_x[i] = x_LUT[i];
                array_y = new double[y_LUT.size()];for(unsigned int i = 0; i < y_LUT.size(); i++) array_y[i] = y_LUT[i];
                array_u = new double[u_LUT.size()];for(unsigned int i = 0; i < u_LUT.size(); i++) array_u[i] = u_LUT[i];
                LUT_u = gsl_spline2d_alloc(gsl_interp2d_bicubic, x_LUT.size(), y_LUT.size());
                xacc_u = gsl_interp_accel_alloc();
                yacc_u = gsl_interp_accel_alloc();
                gsl_spline2d_init(LUT_u, array_x, array_y, array_u, x_LUT.size(), y_LUT.size());
                Set_LUT_FILE_Correction(indexes[knot],x_LUT,y_LUT,v_LUT);
                //array_x = new double[x_LUT.size()];for(unsigned int i = 0; i < x_LUT.size(); i++) array_x[i] = x_LUT[i];
                //array_y = new double[y_LUT.size()];for(unsigned int i = 0; i < y_LUT.size(); i++) array_y[i] = y_LUT[i];
                array_v = new double[v_LUT.size()];for(unsigned int i = 0; i < v_LUT.size(); i++) array_v[i] = v_LUT[i];
                LUT_v = gsl_spline2d_alloc(gsl_interp2d_bicubic, x_LUT.size(), y_LUT.size());
                xacc_v = gsl_interp_accel_alloc();
                yacc_v = gsl_interp_accel_alloc();
                gsl_spline2d_init(LUT_v, array_x, array_y, array_v, x_LUT.size(), y_LUT.size());
                //std::cout << "Solving knot " << indexes[knot] << std::endl;
                B_temp = 0.0;
                BB_temp = 0.0;
                G_j_temp.clear();
                G_temp.clear();
                G_var_temp.clear();
                X0(0) = x[knot]; X0(1) = y[knot];
                //printf("LUT defined in [%f %f], [%f %f]\n",array_x[0],array_y[0],array_x[x_LUT.size()-1],array_y[y_LUT.size()-1]);
                //printf("Point [%f, %f]\n",X0(0),X0(1));
                //std::cout << gsl_spline2d_eval(LUT,X0(0),X0(1),xacc,yacc) << std::endl;
                stencil.Reset();
                solver.Reset_sums();
                solver.N = 0;
                solver.Solve_OMP_Semilinear(X0,(unsigned int) N,h0, 0.1,
                BoundValProb, stencil, c2, G_temp,G_var_temp,G_j_temp,B_temp,
                BB_temp,LUT_u,xacc_u,yacc_u,LUT_v,xacc_v,yacc_v);
                B.push_back(B_temp);
                B_i.push_back(indexes[knot]);
                std::cout << B_i.size() << "\t" << indexes[knot];
                B_var.push_back(B_temp*B_temp - BB_temp);
                for(unsigned int k = 0;k < G_temp.size(); k ++){
                    G_i.push_back(indexes[knot]);
                    G_j.push_back(G_j_temp[k]);
                    G.push_back(G_temp[k]);
                    G_var.push_back(G_var_temp[k]);
                }
                delete array_x; delete array_y; delete array_u; delete array_v;
                //knot_end = MPI_Wtime();
                
            }
        }
        Send_G_B();
        //Compute_B_Deterministic();
        //double end = MPI_Wtime();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    //MPI_Finalize();
}
void PDDSparseGM::Solve(bvp BoundValProb, std::string file){

    ReadFromFile(file);
    if(myid == server){
          Fullfill_interfaces();
    }
    Solve(BoundValProb);
}
void PDDSparseGM::Send_Node(PDDSJob job){
    int work_control_aux[2];
    MPI_Recv(work_control_aux, 2, MPI_INT, MPI_ANY_SOURCE, REQUEST_NODE, world, &status);
    MPI_Send(work_control, 2, MPI_INT, status.MPI_SOURCE, REPLY_NODE, world);
    job.Send_To_Worker(status, world);
    double pos[2];
    pos[0] = Node_Position(job.index[0])[0];
    pos[1] = Node_Position(job.index[0])[1];
    MPI_Send(pos, 2, MPI_DOUBLE, status.MPI_SOURCE, NODE_POSITION, world);
    if(work_control[1] == 1){
        /*FILE *pFile;
        pFile = fopen ("Output/Debug/Node_position.txt","a");
        fprintf(pFile, "%d,%.4f,%.4f \n",job.index[0],pos[0],pos[1]);
        fclose(pFile);
        pFile = fopen(debug_fname, "a");
        fprintf(pFile,"Sending node %d.\n",job.index[0]);
        fclose(pFile);*/
        Send_Stencil_Data(job.index[0]);
    } else {
        printf("Ending process.\n");
    }
}
void PDDSparseGM::Send_Interface(std::vector<Eigen::Vector2d> positions, std::vector<int> indexes){
    int work_control_aux[2];
    //std::cout << "Expecting demand \n";
    MPI_Recv(work_control_aux, 2, MPI_INT, MPI_ANY_SOURCE, REQUEST_INTERFACE, world, &status);
    //std::cout << "Demand received \n";
    work_control[0] = (int)  positions.size();
    work_control[1] = 1;
    MPI_Send(work_control, 2, MPI_INT, status.MPI_SOURCE, REPLY_INTERFACE, world);
    //std::cout << "Centinel sent\n";
    double *aux_double;
    int *aux_int;
    aux_double = new double[work_control[0]];
    aux_int = new int[work_control[0]];
    for(int i = 0; i < work_control[0];i++) aux_double[i] = positions[i][0];
    MPI_Send(aux_double, work_control[0], MPI_DOUBLE, status.MPI_SOURCE, TAG_INTERFACE_X, world);
    for(int i = 0; i < work_control[0];i++) aux_double[i] = positions[i][1];
    MPI_Send(aux_double, work_control[0], MPI_DOUBLE, status.MPI_SOURCE, TAG_INTERFACE_Y, world);
    for(int i = 0; i < work_control[0];i++) aux_int[i] = indexes[i];
    MPI_Send(aux_int, work_control[0], MPI_INT, status.MPI_SOURCE, TAG_INTERFACE_INDEX, world);
    //std::cout << "Quantities sent\n";
}
/*void PDDSparseGM::Compute_B_Deterministic(void){
    if(myid == server){
        //Server
        std::vector<int> aux_vec;
        std::vector<Eigen::Vector2d> positions;
        std::vector<int> indexes;
        aux_vec.resize(2);
        for(int Iy = 2; Iy <= 2*(iN[1]-2); Iy ++){
            if(Iy%2 == 0){
                for(int Ix = 1; Ix <= iN[0]-2; Ix ++){
                    aux_vec[0] = Ix;
                    aux_vec[1] = Iy;
                    indexes.assign(interfaces[interface_map[aux_vec]].index.begin()+1,
                    interfaces[interface_map[aux_vec]].index.end()-1);
                    positions.assign(interfaces[interface_map[aux_vec]].position.begin()+1,
                    interfaces[interface_map[aux_vec]].position.end()-1);
                    Send_Interface(positions, indexes);
                    Send_Stencil_Data(indexes[0]);
                    if(Iy != 2*(iN[1]-2)){
                        indexes.assign(interfaces[interface_map[aux_vec]].index.end()-1,
                        interfaces[interface_map[aux_vec]].index.end());
                        positions.assign(interfaces[interface_map[aux_vec]].position.end()-1,
                        interfaces[interface_map[aux_vec]].position.end());
                        Send_Interface(positions, indexes);
                        Send_Stencil_Data(indexes[0]);
                    }
                }
            } else {
                for(int Ix = 1; Ix < iN[0]-1; Ix ++){
                    aux_vec[0] = Ix;
                    aux_vec[1] = Iy;
                    indexes.assign(interfaces[interface_map[aux_vec]].index.begin()+1,
                    interfaces[interface_map[aux_vec]].index.end()-1);
                    positions.assign(interfaces[interface_map[aux_vec]].position.begin()+1,
                    interfaces[interface_map[aux_vec]].position.end()-1);
                    Send_Interface(positions, indexes);
                    Send_Stencil_Data(indexes[0]);
                }
            }
        }
        B.clear();
        B_i.clear();
        for(int process = 0; process < server; process ++){
            Receive_B();
        }
    } else {
        //Worker
        std::vector<double> x,y;
        std::vector<int> indexes;
        Stencil stencil;
        FILE *pfile;
        std::ifstream myfile;
        std::string line;
        char filename[256];
        int cent;
        B_i.clear();
        B.clear();
        while(! Receive_Interface(x,y,indexes)){
            stencil = Recieve_Stencil_Data();
            sprintf(filename,"Input/Buffer/parameters_%d.csv",myid);
            pfile = fopen(filename,"w");
            for(int i = 0; i < 4; i++) fprintf(pfile,"%e\n",stencil.stencil_parameters[i]);
            fclose(pfile);
            sprintf(filename,"Input/Buffer/positions_%d.csv",myid);
            pfile = fopen(filename,"w");
            for(int i = 0; i < (int)x.size(); i++) fprintf(pfile,"%e,%e\n",x[i],y[i]);
            fclose(pfile);
            sprintf(filename,"octave-cli Poisson_eq_3_B.m %d",myid);
            system(filename);
            sprintf(filename,"Input/Buffer/B_%d.csv",myid);
            myfile.open(filename);
            if (myfile.is_open())
            {
            cent = 0;
            while ( getline (myfile,line) )
            {
                B.push_back(atof(line.c_str()));
                B_i.push_back(indexes[cent]);
                cent ++;
            }
            myfile.close();
            }else{
                std::cout << "There was a problem opening " << filename << std::endl;
            }
        }
        Send_B();
    }
}*/
void PDDSparseGM::Send_Node_Loop(PDDSJob job){
    int work_control_aux[2];
    MPI_Recv(work_control_aux, 2, MPI_INT, MPI_ANY_SOURCE, REQUEST_NODE, world, &status);
    MPI_Send(work_control, 2, MPI_INT, status.MPI_SOURCE, REPLY_NODE, world);
    job.Send_To_Worker(status, world);
    double pos[2];
    pos[0] = Node_Position(job.index[0])[0];
    pos[1] = Node_Position(job.index[0])[1];
    MPI_Send(pos, 2, MPI_DOUBLE, status.MPI_SOURCE, NODE_POSITION, world);
    if(work_control[1] == 1){
        /*FILE *pFile;
        pFile = fopen ("Output/Debug/Node_position.txt","a");
        fprintf(pFile, "%d,%.4f,%.4f \n",job.index[0],pos[0],pos[1]);
        fclose(pFile);
        pFile = fopen(debug_fname, "a");
        fprintf(pFile,"Sending node %d.\n",job.index[0]);
        fclose(pFile);
        */
        Send_Stencil_Data_Loop(job.index[0]);
    } else {
        printf("Ending process.\n");
    }
}
std::vector<std::vector<int> > PDDSparseGM::Get_Interfaces(int index){
    std::vector<std::vector<int> > output;
    //printf("Node %d is in interfaces: ",index);
    for(std::vector<Interface>::iterator it = interfaces.begin();
    it != interfaces.end();
    it++){
        if((*it).In(index) == true) {
            output.push_back((*it).label);
            //printf("[%d %d]", (*it).label[0],(*it).label[1]);
        }
    }
    //printf("\n");
    return output;
}
std::map<direction, std::vector<std::vector <int> > > PDDSparseGM::Labels_Stencil(int index){
    std::map<direction, std::vector<std::vector <int> > > output;
    std::vector< std::vector<int> > node_interfaces;
    Interface inter;
    std::vector< int > interface_east, interface_west,interface_north, interface_south;
    int max_v = 0, max_h = 0;
    node_interfaces = Get_Interfaces(index);
    FILE *dfile;
    switch(node_interfaces.size()){
        case 1:
            inter = interfaces[interface_map[node_interfaces[0]]];
            for(int i = 0; i < (int) inter.N().size(); i ++){
                if(interface_map.count(inter.N()[i]) > 0) output[North].push_back(inter.N()[i]);
            }
            for(int i = 0; i < (int) inter.S().size(); i ++){
                if(interface_map.count(inter.S()[i]) > 0) output[South].push_back(inter.S()[i]);
            }
            for(int i = 0; i < (int) inter.E().size(); i ++){
                if(interface_map.count(inter.E()[i]) > 0) output[East].push_back(inter.E()[i]);
            }
            for(int i = 0; i < (int) inter.W().size(); i ++){
                if(interface_map.count(inter.W()[i]) > 0) output[West].push_back(inter.W()[i]);
            }
            break;
        case 4:
            max_h = std::max(node_interfaces[0][0], node_interfaces[1][0]);
            max_h = std::max(max_h, node_interfaces[2][0]);
            max_h = std::max(max_h, node_interfaces[3][0]);
            max_v = std::max(node_interfaces[0][1], node_interfaces[1][1]);
            max_v = std::max(max_v, node_interfaces[2][1]);
            max_v = std::max(max_v, node_interfaces[3][1]);
            if(node_interfaces[0][1]%2 == 1){
                //Horizontal interface
                if(node_interfaces[0][0] == max_h) interface_east = node_interfaces[0];
                else interface_west = node_interfaces[0];
            } else {
                //Vertical interface
                if(node_interfaces[0][1] == max_v) interface_north = node_interfaces[0];
                else interface_south = node_interfaces[0];
            }
            if(node_interfaces[1][1]%2 == 1){
                //Horizontal interface
                if(node_interfaces[1][0] == max_h) interface_east = node_interfaces[1];
                else interface_west = node_interfaces[1];
            } else {
                //Vertical interface
                if(node_interfaces[1][1] == max_v) interface_north = node_interfaces[1];
                else interface_south = node_interfaces[1];
            }
            if(node_interfaces[2][1]%2 == 1){
                //Horizontal interface
                if(node_interfaces[2][0] == max_h) interface_east = node_interfaces[2];
                else interface_west = node_interfaces[2];
            } else {
                //Vertical interface
                if(node_interfaces[2][1] == max_v) interface_north = node_interfaces[2];
                else interface_south = node_interfaces[2];
            }
            if(node_interfaces[3][1]%2 == 1){
                //Horizontal interface
                if(node_interfaces[3][0] == max_h) interface_east = node_interfaces[3];
                else interface_west = node_interfaces[3];
            } else {
                //Vertical interface
                if(node_interfaces[3][1] == max_v) interface_north = node_interfaces[3];
                else interface_south = node_interfaces[3];
            }
            inter = interfaces[interface_map[interface_west]];
            if(interface_map.count(inter.N()[0]) > 0) output[North].push_back(inter.N()[0]);
            inter = interfaces[interface_map[interface_east]];
            if(interface_map.count(inter.N()[0]) > 0) output[North].push_back(inter.N()[0]);
            inter = interfaces[interface_map[interface_west]];
            if(interface_map.count(inter.S()[0]) > 0) output[South].push_back(inter.S()[0]);
            inter = interfaces[interface_map[interface_east]];
            if(interface_map.count(inter.S()[0]) > 0) output[South].push_back(inter.S()[0]);
            inter = interfaces[interface_map[interface_south]];
            if(interface_map.count(inter.E()[0]) > 0) output[East].push_back(inter.E()[0]);
            inter = interfaces[interface_map[interface_north]];
            if(interface_map.count(inter.E()[0]) > 0) output[East].push_back(inter.E()[0]);
            inter = interfaces[interface_map[interface_south]];
            if(interface_map.count(inter.W()[0]) > 0) output[West].push_back(inter.W()[0]);
            inter = interfaces[interface_map[interface_north]];
            if(interface_map.count(inter.W()[0]) > 0) output[West].push_back(inter.W()[0]);
            break;
        default:
            dfile = fopen(debug_fname,"a");
            fprintf(dfile,"Something went wrong in Labels_Stencil.\n");
            fclose(dfile);
    }
    /*
    printf("South: ");
    for(int i = 0; i < (int) output[South].size(); i++) printf("[%d %d] ",output[South][i][0],output[South][i][1]);
    printf("\n");
    printf("North: ");
    for(int i = 0; i < (int) output[North].size(); i++) printf("[%d %d] ",output[North][i][0],output[North][i][1]);
    printf("\n");
    printf("East: ");
    for(int i = 0; i < (int) output[East].size(); i++) printf("[%d %d] ",output[East][i][0],output[East][i][1]);
    printf("\n");
    printf("West: ");
    for(int i = 0; i < (int) output[West].size(); i++) printf("[%d %d] ",output[West][i][0],output[West][i][1]);
    printf("\n"); */
    return output;
}
std::map<direction, std::vector<std::vector <int> > > PDDSparseGM::Labels_Stencil_Square(int index){
    std::map<direction, std::vector<std::vector <int> > > output;
    std::vector< std::vector<int> > node_interfaces;
    Interface inter;
    std::vector< int > interface_east, interface_west,interface_north, interface_south;
    int max_v = 0, max_h = 0;
    node_interfaces = Get_Interfaces(index);
    FILE *dfile;
    int cent = -1;
    switch(node_interfaces.size()){
        case 1:
            /*for(int i = 0; i < (int) inter.N().size(); i ++){
                if(interface_map.count(inter.N()[i]) > 0) output[North].push_back(inter.N()[i]);
            }
            for(int i = 0; i < (int) inter.S().size(); i ++){
                if(interface_map.count(inter.S()[i]) > 0) output[South].push_back(inter.S()[i]);
            }
            for(int i = 0; i < (int) inter.E().size(); i ++){
                if(interface_map.count(inter.E()[i]) > 0) output[East].push_back(inter.E()[i]);
            }
            for(int i = 0; i < (i nt) inter.W().size(); i ++){
                if(interface_map.count(inter.W()[i]) > 0) output[West].push_back(inter.W()[i]);
            }*/
            inter = interfaces[interface_map[node_interfaces[0]]];
            for(unsigned int i = 0; i < inter.index.size(); i ++){
                if(inter.index[i] == index)cent = (int) i;
            }
            if(cent < 0){
                std::cout << "Labels Stencil Square is not working properly\n";
                std::cout << "Index " << index << "\n";
            }else{
                if((1.0*cent/inter.index.size())<0.5){
                    return Labels_Stencil(inter.index.front());
                }else {
                    return Labels_Stencil(inter.index.back());
                }
            }
            break;
        case 4:
            max_h = std::max(node_interfaces[0][0], node_interfaces[1][0]);
            max_h = std::max(max_h, node_interfaces[2][0]);
            max_h = std::max(max_h, node_interfaces[3][0]);
            max_v = std::max(node_interfaces[0][1], node_interfaces[1][1]);
            max_v = std::max(max_v, node_interfaces[2][1]);
            max_v = std::max(max_v, node_interfaces[3][1]);
            if(node_interfaces[0][1]%2 == 1){
                //Horizontal interface
                if(node_interfaces[0][0] == max_h) interface_east = node_interfaces[0];
                else interface_west = node_interfaces[0];
            } else {
                //Vertical interface
                if(node_interfaces[0][1] == max_v) interface_north = node_interfaces[0];
                else interface_south = node_interfaces[0];
            }
            if(node_interfaces[1][1]%2 == 1){
                //Horizontal interface
                if(node_interfaces[1][0] == max_h) interface_east = node_interfaces[1];
                else interface_west = node_interfaces[1];
            } else {
                //Vertical interface
                if(node_interfaces[1][1] == max_v) interface_north = node_interfaces[1];
                else interface_south = node_interfaces[1];
            }
            if(node_interfaces[2][1]%2 == 1){
                //Horizontal interface
                if(node_interfaces[2][0] == max_h) interface_east = node_interfaces[2];
                else interface_west = node_interfaces[2];
            } else {
                //Vertical interface
                if(node_interfaces[2][1] == max_v) interface_north = node_interfaces[2];
                else interface_south = node_interfaces[2];
            }
            if(node_interfaces[3][1]%2 == 1){
                //Horizontal interface
                if(node_interfaces[3][0] == max_h) interface_east = node_interfaces[3];
                else interface_west = node_interfaces[3];
            } else {
                //Vertical interface
                if(node_interfaces[3][1] == max_v) interface_north = node_interfaces[3];
                else interface_south = node_interfaces[3];
            }
            inter = interfaces[interface_map[interface_west]];
            if(interface_map.count(inter.N()[0]) > 0) output[North].push_back(inter.N()[0]);
            inter = interfaces[interface_map[interface_east]];
            if(interface_map.count(inter.N()[0]) > 0) output[North].push_back(inter.N()[0]);
            inter = interfaces[interface_map[interface_west]];
            if(interface_map.count(inter.S()[0]) > 0) output[South].push_back(inter.S()[0]);
            inter = interfaces[interface_map[interface_east]];
            if(interface_map.count(inter.S()[0]) > 0) output[South].push_back(inter.S()[0]);
            inter = interfaces[interface_map[interface_south]];
            if(interface_map.count(inter.E()[0]) > 0) output[East].push_back(inter.E()[0]);
            inter = interfaces[interface_map[interface_north]];
            if(interface_map.count(inter.E()[0]) > 0) output[East].push_back(inter.E()[0]);
            inter = interfaces[interface_map[interface_south]];
            if(interface_map.count(inter.W()[0]) > 0) output[West].push_back(inter.W()[0]);
            inter = interfaces[interface_map[interface_north]];
            if(interface_map.count(inter.W()[0]) > 0) output[West].push_back(inter.W()[0]);
            break;
        default:
            dfile = fopen(debug_fname,"a");
            fprintf(dfile,"Something went wrong in Labels_Stencil.\n");
            fclose(dfile);
    }
    /*
    printf("South: ");
    for(int i = 0; i < (int) output[South].size(); i++) printf("[%d %d] ",output[South][i][0],output[South][i][1]);
    printf("\n");
    printf("North: ");
    for(int i = 0; i < (int) output[North].size(); i++) printf("[%d %d] ",output[North][i][0],output[North][i][1]);
    printf("\n");
    printf("East: ");
    for(int i = 0; i < (int) output[East].size(); i++) printf("[%d %d] ",output[East][i][0],output[East][i][1]);
    printf("\n");
    printf("West: ");
    for(int i = 0; i < (int) output[West].size(); i++) printf("[%d %d] ",output[West][i][0],output[West][i][1]);
    printf("\n"); */
    return output;
}
void PDDSparseGM::Send_Stencil_Data(int index){
    #ifdef SQUARE_PATCHES
        std::map<direction, std::vector<std::vector <int> > > labels = Labels_Stencil_Square(index);
    #else
        std::map<direction, std::vector<std::vector <int> > > labels = Labels_Stencil(index);
    #endif
    //The parameters of the stencil are computed
    double s_params[4];
    for(int i = 0; i < 4; i++) s_params[i] = parameters[i];
    if(labels[South].size()>0){
        //s_params[0] = interfaces[interface_map[labels[South][0]]].position[0][0];
        s_params[1] = interfaces[interface_map[labels[South][0]]].position[0][1];
    }
    if(labels[West].size()>0){
        s_params[0] = interfaces[interface_map[labels[West][0]]].position[0][0];
        //s_params[1] = interfaces[interface_map[labels[West][0]]].position[0][1];
    }
    int aux_size;
    if(labels[North].size()>0){
        aux_size = (int) interfaces[interface_map[labels[North][labels[North].size()-1]]].position.size();
        //s_params[2] = interfaces[interface_map[labels[North][labels[North].size()-1]]].position[aux_size - 1][0];
        s_params[3] = interfaces[interface_map[labels[North][labels[North].size()-1]]].position[aux_size - 1][1];
    }
    if(labels[East].size()>0){
        aux_size = (int) interfaces[interface_map[labels[East][labels[East].size()-1]]].position.size();
        s_params[2] = interfaces[interface_map[labels[East][labels[East].size()-1]]].position[aux_size - 1][0];
        //s_params[3] = interfaces[interface_map[labels[East][labels[East].size()-1]]].position[aux_size - 1][1];
    }
    //The indexes and positions of the stencils are computed
    std::map<direction, std::vector<int> > s_index;
    std::map<direction, std::vector<double> > s_x, s_y;
    std::vector<direction> s_dir;
    Interface s_inter, aux_inter;
    std::vector<int> aux_label;
    FILE *dfile;
    s_dir.resize(4);
    s_dir[0] = North; s_dir[1] = South; s_dir[2] = East; s_dir[3] = West;
    for(std::vector<direction>::iterator it = s_dir.begin(); it != s_dir.end(); it++){
    if(labels[*it].size() > 0){
        s_inter = interfaces[interface_map[labels[*it][0]]];
        aux_label.resize(2);
        aux_label[0] = s_inter.label[0]; aux_label[1] = s_inter.label[1];
        switch(*it){
            case North:
                aux_label[0] --;
                if(interface_map.count(aux_label)>0){
                aux_inter = interfaces[interface_map[aux_label]];
                for(int i = (int)((1-STEN_ELONG)*(aux_inter.index.size()-1));
                    i < (int) aux_inter.index.size(); i ++){
                        s_index[*it].push_back(aux_inter.index[i]);
                        s_x[*it].push_back(aux_inter.position[i][0]);
                        s_y[*it].push_back(aux_inter.position[i][1]);
                    }
                }
            break;
            case South:
                aux_label[0] --;
                if(interface_map.count(aux_label)>0){
                aux_inter = interfaces[interface_map[aux_label]];
                for(int i = (int)((1-STEN_ELONG)*(aux_inter.index.size()-1));
                    i < (int) aux_inter.index.size(); i ++){
                        s_index[*it].push_back(aux_inter.index[i]);
                        s_x[*it].push_back(aux_inter.position[i][0]);
                        s_y[*it].push_back(aux_inter.position[i][1]);
                    }
                }
                break;
            case East:
                aux_label[1] += -2;
                if(interface_map.count(aux_label)>0){
                aux_inter = interfaces[interface_map[aux_label]];
                for(int i = (int)((1-STEN_ELONG)*(aux_inter.index.size()-1));
                    i < (int) aux_inter.index.size(); i ++){
                        s_index[*it].push_back(aux_inter.index[i]);
                        s_x[*it].push_back(aux_inter.position[i][0]);
                        s_y[*it].push_back(aux_inter.position[i][1]);
                    }
                }
            break;
            case West:
                aux_label[1] += -2;
                if(interface_map.count(aux_label)>0){
                aux_inter = interfaces[interface_map[aux_label]];
                for(int i = (int)((1-STEN_ELONG)*(aux_inter.index.size()-1));
                    i < (int) aux_inter.index.size(); i ++){
                        s_index[*it].push_back(aux_inter.index[i]);
                        s_x[*it].push_back(aux_inter.position[i][0]);
                        s_y[*it].push_back(aux_inter.position[i][1]);
                    }
                }
            break;
            default:
                dfile = fopen(debug_fname,"a");
                fprintf(dfile,"Something went wrong during stencil formation. 1\n");
                fclose(dfile);
        }
        for(int i = 0; i < (int) labels[*it].size(); i++){
            s_inter = interfaces[interface_map[labels[*it][i]]];
            for(int j = 0;
            j < (int) s_inter.index.size(); j ++){
                s_index[*it].push_back(s_inter.index[j]);
                s_x[*it].push_back(s_inter.position[j][0]);
                s_y[*it].push_back(s_inter.position[j][1]);
            }
        }
        aux_label.resize(2);
        aux_label[0] = s_inter.label[0]; aux_label[1] = s_inter.label[1];
        switch(*it){
            case North:
                aux_label[0] ++;
                if(interface_map.count(aux_label)>0){
                aux_inter = interfaces[interface_map[aux_label]];
                for(int i = 0;
                    i < (int) ((STEN_ELONG)*(aux_inter.index.size())); i ++){
                        s_index[*it].push_back(aux_inter.index[i]);
                        s_x[*it].push_back(aux_inter.position[i][0]);
                        s_y[*it].push_back(aux_inter.position[i][1]);
                    }
                }
            break;
            case South:
                aux_label[0] ++;
                if(interface_map.count(aux_label)>0){
                aux_inter = interfaces[interface_map[aux_label]];
                for(int i = 0;
                    i < (int) ((STEN_ELONG)*(aux_inter.index.size())); i ++){
                        s_index[*it].push_back(aux_inter.index[i]);
                        s_x[*it].push_back(aux_inter.position[i][0]);
                        s_y[*it].push_back(aux_inter.position[i][1]);
                    }
                }
                break;
            case East:
                aux_label[1] += 2;
                if(interface_map.count(aux_label)>0){
                aux_inter = interfaces[interface_map[aux_label]];
                for(int i = 0;
                    i < (int) ((STEN_ELONG)*(aux_inter.index.size())); i ++){
                        s_index[*it].push_back(aux_inter.index[i]);
                        s_x[*it].push_back(aux_inter.position[i][0]);
                        s_y[*it].push_back(aux_inter.position[i][1]);
                    }
                }
            break;
            case West:
                aux_label[1] += 2;
                if(interface_map.count(aux_label)>0){
                aux_inter = interfaces[interface_map[aux_label]];
                for(int i = 0;
                    i < (int) ((STEN_ELONG)*(aux_inter.index.size())); i ++){
                        s_index[*it].push_back(aux_inter.index[i]);
                        s_x[*it].push_back(aux_inter.position[i][0]);
                        s_y[*it].push_back(aux_inter.position[i][1]);
                    }
                }
            break;
            default:
                dfile = fopen(debug_fname,"w");
                fprintf(dfile,"Something went wrong during stencil formation. 2\n");
                fclose(dfile);
        }
    }
    }
    
    //Parameters of the stencils are sent to the worker
    MPI_Send(s_params, 4, MPI_DOUBLE, status.MPI_SOURCE, PARAMETERS, world);
    int sizes[4];
    sizes[0] = (int) s_index[North].size();sizes[1] = (int) s_index[South].size();
    sizes[2] = (int) s_index[East].size();sizes[3] = (int) s_index[West].size();
    //Sizes of the stencil's vectors are sent to the worker
    MPI_Send(sizes, 4, MPI_INT, status.MPI_SOURCE, SIZES, world);
    //Stencil indexes and positions are sent to the worker 
    int *i_north, *i_south, *i_west, *i_east;
    double *x_north, *x_south, *x_west, *x_east, 
           *y_north, *y_south, *y_west, *y_east;
    //North stencil
    i_north = new int[sizes[0]];
    x_north = new double[sizes[0]];
    y_north = new double[sizes[0]];
    for(int i = 0; i < sizes[0]; i++){
        i_north[i] = s_index[North][i];
        x_north[i] = s_x[North][i];
        y_north[i] = s_y[North][i];
    }
    MPI_Send(i_north, sizes[0], MPI_INT, status.MPI_SOURCE, IND_NORTH, world);
    MPI_Send(x_north, sizes[0], MPI_DOUBLE, status.MPI_SOURCE, X_NORTH, world);
    MPI_Send(y_north, sizes[0], MPI_DOUBLE, status.MPI_SOURCE, Y_NORTH, world);
    delete i_north; delete x_north; delete y_north;
    //South stencil
    i_south = new int[sizes[1]];
    x_south = new double[sizes[1]];
    y_south = new double[sizes[1]];
    for(int i = 0; i < sizes[1]; i++){
        i_south[i] = s_index[South][i];
        x_south[i] = s_x[South][i];
        y_south[i] = s_y[South][i];
    }
    MPI_Send(i_south, sizes[1], MPI_INT, status.MPI_SOURCE, IND_SOUTH, world);
    MPI_Send(x_south, sizes[1], MPI_DOUBLE, status.MPI_SOURCE, X_SOUTH, world);
    MPI_Send(y_south, sizes[1], MPI_DOUBLE, status.MPI_SOURCE, Y_SOUTH, world);
    delete i_south; delete x_south; delete y_south;
    //East stencil
    i_east = new int[sizes[2]];
    x_east = new double[sizes[2]];
    y_east = new double[sizes[2]];
    for(int i = 0; i < sizes[2]; i++){
        i_east[i] = s_index[East][i];
        x_east[i] = s_x[East][i];
        y_east[i] = s_y[East][i];
    }
    MPI_Send(i_east, sizes[2], MPI_INT, status.MPI_SOURCE, IND_EAST, world);
    MPI_Send(x_east, sizes[2], MPI_DOUBLE, status.MPI_SOURCE, X_EAST, world);
    MPI_Send(y_east, sizes[2], MPI_DOUBLE, status.MPI_SOURCE, Y_EAST, world);
    delete i_east; delete x_east; delete y_east;
    //West stencil
    i_west = new int[sizes[3]];
    x_west = new double[sizes[3]];
    y_west = new double[sizes[3]];
    for(int i = 0; i < sizes[3]; i++){
        i_west[i] = s_index[West][i];
        x_west[i] = s_x[West][i];
        y_west[i] = s_y[West][i];
    }
    MPI_Send(i_west, sizes[3], MPI_INT, status.MPI_SOURCE, IND_WEST, world);
    MPI_Send(x_west, sizes[3], MPI_DOUBLE, status.MPI_SOURCE, X_WEST, world);
    MPI_Send(y_west, sizes[3], MPI_DOUBLE, status.MPI_SOURCE, Y_WEST, world);
    delete i_west; delete x_west; delete y_west;
}
void PDDSparseGM::Send_Stencil_Data_Loop(int index){
    std::vector< std::vector<int> > node_interfaces;
    node_interfaces = Get_Interfaces(index);
    int aux_size[1];
    if(node_interfaces.size() == 1){
        int aux_int[2] = {node_interfaces[0][0], node_interfaces[0][1]};
        aux_size[0]=2;
        MPI_Send(aux_size,1,MPI_INT,status.MPI_SOURCE,INTER_NUMBER,world);
        MPI_Send(aux_int,2,MPI_INT,status.MPI_SOURCE,INTER_LABELS,world);
    }else{
        int aux_int[8] = {node_interfaces[0][0], node_interfaces[0][1],
                          node_interfaces[1][0], node_interfaces[1][1],
                          node_interfaces[2][0], node_interfaces[2][1],
                          node_interfaces[3][0], node_interfaces[3][1],
                        };
        aux_size[0]=8;
        MPI_Send(aux_size,1,MPI_INT,status.MPI_SOURCE,INTER_NUMBER,world);
        MPI_Send(aux_int,8,MPI_INT,status.MPI_SOURCE,INTER_LABELS,world);
    }
    Send_Stencil_Data(index);
}
bool PDDSparseGM::Receive_Node(PDDSJob & job){
    //FILE *dfile;
    MPI_Comm_rank(workers, &workerid);
    MPI_Send(work_control, 2, MPI_INT, server, REQUEST_NODE, world);
    MPI_Recv(work_control, 2, MPI_INT, server, REPLY_NODE, world, MPI_STATUS_IGNORE);
    if(work_control[1] == 1){
        job.Recieve_From_Server(server, world);
        //dfile = fopen(debug_fname,"a");
        //fprintf(dfile,"Process %d is solving node %d .\n",workerid,job.index[0]);
        //fclose(dfile);
        double pos[2];
        MPI_Recv(pos, 2, MPI_DOUBLE, server, NODE_POSITION, world, MPI_STATUS_IGNORE);
        position.resize(2);
        position[0] = pos[0]; position[1] = pos[1];
        return false;
    } else {
        return true;
    }
}
bool PDDSparseGM::Receive_Interface(std::vector<double> & x, std::vector<double> &y, std::vector<int> &indexes){
    //FILE *dfile;
    //MPI_Comm_rank(workers, &workerid);

    MPI_Send(work_control, 2, MPI_INT, server, REQUEST_INTERFACE, world);
    //std::cout << "Sending request\n";
    MPI_Recv(work_control, 2, MPI_INT, server, REPLY_INTERFACE, world, MPI_STATUS_IGNORE);
    //std::cout << "Request answered\n";
    if(work_control[1] == 1){
        double *aux_double;
        int *aux_int;
        aux_double = new double[work_control[0]];
        aux_int = new int[work_control[0]];
        x.resize(work_control[0]);y.resize(work_control[0]);indexes.resize(work_control[0]);
        MPI_Recv(aux_double, work_control[0], MPI_DOUBLE, server, TAG_INTERFACE_X, world, MPI_STATUS_IGNORE);
        for(int i = 0; i < work_control[0]; i++) x[i] = aux_double[i];
        MPI_Recv(aux_double, work_control[0], MPI_DOUBLE, server, TAG_INTERFACE_Y, world, MPI_STATUS_IGNORE);
        for(int i = 0; i < work_control[0]; i++) y[i] = aux_double[i];
        MPI_Recv(aux_int, work_control[0], MPI_INT, server, TAG_INTERFACE_INDEX, world, MPI_STATUS_IGNORE);
        for(int i = 0; i < work_control[0]; i++) indexes[i] = aux_int[i];
        return false;
    } else {
        return true;
    }
}
Stencil PDDSparseGM::Compute_Stencil(int index){
    #ifdef SQUARE_PATCHES
        std::map<direction, std::vector<std::vector <int> > > labels = Labels_Stencil_Square(index);
    #else
        std::map<direction, std::vector<std::vector <int> > > labels = Labels_Stencil(index);
    #endif
    //The parameters of the stencil are computed
    double s_params[4];
    for(int i = 0; i < 4; i++) s_params[i] = parameters[i];
    if(labels[South].size()>0){
        //s_params[0] = interfaces[interface_map[labels[South][0]]].position[0][0];
        s_params[1] = interfaces[interface_map[labels[South][0]]].position[0][1];
    }
    if(labels[West].size()>0){
        s_params[0] = interfaces[interface_map[labels[West][0]]].position[0][0];
        //s_params[1] = interfaces[interface_map[labels[West][0]]].position[0][1];
    }
    int aux_size;
    if(labels[North].size()>0){
        aux_size = (int) interfaces[interface_map[labels[North][labels[North].size()-1]]].position.size();
        //s_params[2] = interfaces[interface_map[labels[North][labels[North].size()-1]]].position[aux_size - 1][0];
        s_params[3] = interfaces[interface_map[labels[North][labels[North].size()-1]]].position[aux_size - 1][1];
    }
    if(labels[East].size()>0){
        aux_size = (int) interfaces[interface_map[labels[East][labels[East].size()-1]]].position.size();
        s_params[2] = interfaces[interface_map[labels[East][labels[East].size()-1]]].position[aux_size - 1][0];
        //s_params[3] = interfaces[interface_map[labels[East][labels[East].size()-1]]].position[aux_size - 1][1];
    }
    //The indexes and positions of the stencils are computed
    std::map<direction, std::vector<int> > s_index;
    std::map<direction, std::vector<double> > s_x, s_y;
    std::vector<direction> s_dir;
    Interface s_inter, aux_inter;
    std::vector<int> aux_label;
    FILE *dfile;
    s_dir.resize(4);
    s_dir[0] = North; s_dir[1] = South; s_dir[2] = East; s_dir[3] = West;
    for(std::vector<direction>::iterator it = s_dir.begin(); it != s_dir.end(); it++){
    if(labels[*it].size() > 0){
        s_inter = interfaces[interface_map[labels[*it][0]]];
        aux_label.resize(2);
        aux_label[0] = s_inter.label[0]; aux_label[1] = s_inter.label[1];
        switch(*it){
            case North:
                aux_label[0] --;
                if(interface_map.count(aux_label)>0){
                aux_inter = interfaces[interface_map[aux_label]];
                for(int i = (int)((1-STEN_ELONG)*(aux_inter.index.size()-1));
                    i < (int) aux_inter.index.size(); i ++){
                        s_index[*it].push_back(aux_inter.index[i]);
                        s_x[*it].push_back(aux_inter.position[i][0]);
                        s_y[*it].push_back(aux_inter.position[i][1]);
                    }
                }
            break;
            case South:
                aux_label[0] --;
                if(interface_map.count(aux_label)>0){
                aux_inter = interfaces[interface_map[aux_label]];
                for(int i = (int)((1-STEN_ELONG)*(aux_inter.index.size()-1));
                    i < (int) aux_inter.index.size(); i ++){
                        s_index[*it].push_back(aux_inter.index[i]);
                        s_x[*it].push_back(aux_inter.position[i][0]);
                        s_y[*it].push_back(aux_inter.position[i][1]);
                    }
                }
                break;
            case East:
                aux_label[1] += -2;
                if(interface_map.count(aux_label)>0){
                aux_inter = interfaces[interface_map[aux_label]];
                for(int i = (int)((1-STEN_ELONG)*(aux_inter.index.size()-1));
                    i < (int) aux_inter.index.size(); i ++){
                        s_index[*it].push_back(aux_inter.index[i]);
                        s_x[*it].push_back(aux_inter.position[i][0]);
                        s_y[*it].push_back(aux_inter.position[i][1]);
                    }
                }
            break;
            case West:
                aux_label[1] += -2;
                if(interface_map.count(aux_label)>0){
                aux_inter = interfaces[interface_map[aux_label]];
                for(int i = (int)((1-STEN_ELONG)*(aux_inter.index.size()-1));
                    i < (int) aux_inter.index.size(); i ++){
                        s_index[*it].push_back(aux_inter.index[i]);
                        s_x[*it].push_back(aux_inter.position[i][0]);
                        s_y[*it].push_back(aux_inter.position[i][1]);
                    }
                }
            break;
            default:
                dfile = fopen(debug_fname,"a");
                fprintf(dfile,"Something went wrong during stencil formation. 1\n");
                fclose(dfile);
        }
        for(int i = 0; i < (int) labels[*it].size(); i++){
            s_inter = interfaces[interface_map[labels[*it][i]]];
            for(int j = 0;
            j < (int) s_inter.index.size(); j ++){
                s_index[*it].push_back(s_inter.index[j]);
                s_x[*it].push_back(s_inter.position[j][0]);
                s_y[*it].push_back(s_inter.position[j][1]);
            }
        }
        aux_label.resize(2);
        aux_label[0] = s_inter.label[0]; aux_label[1] = s_inter.label[1];
        switch(*it){
            case North:
                aux_label[0] ++;
                if(interface_map.count(aux_label)>0){
                aux_inter = interfaces[interface_map[aux_label]];
                for(int i = 0;
                    i < (int) ((STEN_ELONG)*(aux_inter.index.size())); i ++){
                        s_index[*it].push_back(aux_inter.index[i]);
                        s_x[*it].push_back(aux_inter.position[i][0]);
                        s_y[*it].push_back(aux_inter.position[i][1]);
                    }
                }
            break;
            case South:
                aux_label[0] ++;
                if(interface_map.count(aux_label)>0){
                aux_inter = interfaces[interface_map[aux_label]];
                for(int i = 0;
                    i < (int) ((STEN_ELONG)*(aux_inter.index.size())); i ++){
                        s_index[*it].push_back(aux_inter.index[i]);
                        s_x[*it].push_back(aux_inter.position[i][0]);
                        s_y[*it].push_back(aux_inter.position[i][1]);
                    }
                }
                break;
            case East:
                aux_label[1] += 2;
                if(interface_map.count(aux_label)>0){
                aux_inter = interfaces[interface_map[aux_label]];
                for(int i = 0;
                    i < (int) ((STEN_ELONG)*(aux_inter.index.size())); i ++){
                        s_index[*it].push_back(aux_inter.index[i]);
                        s_x[*it].push_back(aux_inter.position[i][0]);
                        s_y[*it].push_back(aux_inter.position[i][1]);
                    }
                }
            break;
            case West:
                aux_label[1] += 2;
                if(interface_map.count(aux_label)>0){
                aux_inter = interfaces[interface_map[aux_label]];
                for(int i = 0;
                    i < (int) ((STEN_ELONG)*(aux_inter.index.size())); i ++){
                        s_index[*it].push_back(aux_inter.index[i]);
                        s_x[*it].push_back(aux_inter.position[i][0]);
                        s_y[*it].push_back(aux_inter.position[i][1]);
                    }
                }
            break;
            default:
                dfile = fopen(debug_fname,"w");
                fprintf(dfile,"Something went wrong during stencil formation. 2\n");
                fclose(dfile);
        }
    }
    }
    Stencil output;
    output.Init(s_index,s_x,s_y,s_params);

    return output;
}
Stencil PDDSparseGM::Recieve_Stencil_Data(void){
    double s_params[4];
    MPI_Recv(s_params, 4, MPI_DOUBLE, server, PARAMETERS, world, MPI_STATUS_IGNORE);
    int sizes[4];
    MPI_Recv(sizes, 4, MPI_INT, server, SIZES, world, MPI_STATUS_IGNORE);
    int *i_north, *i_south, *i_west, *i_east;
    double *x_north, *x_south, *x_west, *x_east, 
           *y_north, *y_south, *y_west, *y_east;
    std::map<direction, std::vector<int> > s_index;
    std::map<direction, std::vector<double> > s_x, s_y;
    i_north = new int[sizes[0]]; x_north = new double[sizes[0]]; y_north = new double[sizes[0]];
    MPI_Recv(i_north, sizes[0], MPI_INT, server, IND_NORTH, world, MPI_STATUS_IGNORE);
    MPI_Recv(x_north, sizes[0], MPI_DOUBLE, server, X_NORTH, world, MPI_STATUS_IGNORE);
    MPI_Recv(y_north, sizes[0], MPI_DOUBLE, server, Y_NORTH, world, MPI_STATUS_IGNORE);
    for(int i = 0; i < sizes[0]; i++){
        s_index[North].push_back(i_north[i]);
        s_x[North].push_back(x_north[i]);
        s_y[North].push_back(y_north[i]);
    }
    delete i_north; delete x_north; delete y_north;
    i_south = new int[sizes[1]]; x_south = new double[sizes[1]]; y_south = new double[sizes[1]]; 
    MPI_Recv(i_south, sizes[1], MPI_INT, server, IND_SOUTH, world, MPI_STATUS_IGNORE);
    MPI_Recv(x_south, sizes[1], MPI_DOUBLE, server, X_SOUTH, world, MPI_STATUS_IGNORE);
    MPI_Recv(y_south, sizes[1], MPI_DOUBLE, server, Y_SOUTH, world, MPI_STATUS_IGNORE);
    for(int i = 0; i < sizes[1]; i++){
        s_index[South].push_back(i_south[i]);
        s_x[South].push_back(x_south[i]);
        s_y[South].push_back(y_south[i]);
    }
    delete i_south; delete x_south; delete y_south;
    i_east = new int[sizes[2]]; x_east = new double[sizes[2]]; y_east = new double[sizes[2]];
    MPI_Recv(i_east, sizes[2], MPI_INT, server, IND_EAST, world, MPI_STATUS_IGNORE);
    MPI_Recv(x_east, sizes[2], MPI_DOUBLE, server, X_EAST, world, MPI_STATUS_IGNORE);
    MPI_Recv(y_east, sizes[2], MPI_DOUBLE, server, Y_EAST, world, MPI_STATUS_IGNORE);
    for(int i = 0; i < sizes[2]; i++){
        s_index[East].push_back(i_east[i]);
        s_x[East].push_back(x_east[i]);
        s_y[East].push_back(y_east[i]);
    }
    delete i_east; delete x_east; delete y_east;
    i_west = new int[sizes[3]]; x_west = new double[sizes[3]]; y_west = new double[sizes[3]];
    MPI_Recv(i_west, sizes[3], MPI_INT, server, IND_WEST, world, MPI_STATUS_IGNORE);
    MPI_Recv(x_west, sizes[3], MPI_DOUBLE, server, X_WEST, world, MPI_STATUS_IGNORE);
    MPI_Recv(y_west, sizes[3], MPI_DOUBLE, server, Y_WEST, world, MPI_STATUS_IGNORE);
    for(int i = 0; i < sizes[3]; i++){
        s_index[West].push_back(i_west[i]);
        s_x[West].push_back(x_west[i]);
        s_y[West].push_back(y_west[i]);
    }
    delete i_west; delete x_west; delete y_west;
    Stencil output;
    output.Init(s_index,s_x,s_y,s_params);
    return output;
}
void PDDSparseGM::Init_LUT(unsigned int Nx, unsigned int Ny, double *x, double *y, double *value, 
gsl_spline2d *spline, gsl_interp_accel *xaccel, gsl_interp_accel *yaccel){
    spline = gsl_spline2d_alloc(gsl_interp2d_bicubic, Nx, Ny);
    xaccel = gsl_interp_accel_alloc();
    yaccel = gsl_interp_accel_alloc();
    gsl_spline2d_init(spline, x, y, value, Nx, Ny);
    //();
}
void PDDSparseGM::Set_LUT_FILE_Solution(int index, std::vector<double> & aux_x,std::vector<double> & aux_y,std::vector<double> & aux_sol){
    std::vector<std::vector<int>> subd_index;
    std::vector< std::vector<int> > Iindexes_prev  = Get_Interfaces(index);
    #ifdef SQUARE_PATCHES
        Interface inter = interfaces[interface_map[Iindexes_prev[0]]];
        int cent = -1;
        for(unsigned int i = 0; i < inter.index.size(); i ++){
                if(inter.index[i] == index)cent = (int) i;
        }
        if(cent < 0){
            std::cout << "Labels Stencil Square is not working properly\n";
            std::cout << "Index " << index << "\n";
        }else{
            if((1.0*cent/inter.index.size())<0.5){
                    Iindexes_prev = Get_Interfaces(inter.index.front());
            }else {
                    Iindexes_prev = Get_Interfaces(inter.index.back());
            }
        }
    #endif
    std::vector<int> Iindexes;
    for(unsigned int i = 0; i < Iindexes_prev.size(); i++){ Iindexes.push_back(Iindexes_prev[i][0]); Iindexes.push_back(Iindexes_prev[i][1]);}
    char aux_fname[256];
    if(Iindexes.size() == 2){
        subd_index.resize(2);
        subd_index[0].resize(2);
        subd_index[1].resize(2);
        if(Iindexes[1]%2== 0){
            //horizontal patch
            //std::cout << "Horizontal patch\n";
            subd_index[0][0] = Iindexes[0]; subd_index[0][1] = Iindexes[1]/2;
            subd_index[1][0] = Iindexes[0]+1; subd_index[1][1] = Iindexes[1]/2;
            std::vector<double> x_W, x_E;
            std::vector<double> y_W, y_E;
            std::vector<double> z_W, z_W_n, z_E, z_E_n;
            unsigned int i,j;
            sprintf(aux_fname,"Output/Subdomains/X_%d_%d.txt",subd_index[0][0],subd_index[0][1]);
            x_W = Read_File(aux_fname);
            sprintf(aux_fname,"Output/Subdomains/Y_%d_%d.txt",subd_index[0][0],subd_index[0][1]);
            y_W = Read_File(aux_fname);
            sprintf(aux_fname,"Output/Subdomains/Sol_%d_%d.txt",subd_index[0][0],subd_index[0][1]);
            z_W = Read_File(aux_fname);
            sprintf(aux_fname,"Output/Subdomains/X_%d_%d.txt",subd_index[1][0],subd_index[1][1]);
            x_E = Read_File(aux_fname);
            sprintf(aux_fname,"Output/Subdomains/Y_%d_%d.txt",subd_index[1][0],subd_index[1][1]);
            y_E = Read_File(aux_fname);
            sprintf(aux_fname,"Output/Subdomains/Sol_%d_%d.txt",subd_index[1][0],subd_index[1][1]);
            z_E = Read_File(aux_fname);
            aux_x.resize(0); aux_y.resize(0); aux_sol.resize(0);
            for(j = 0; j < y_W.size(); j++){
                aux_y.push_back(y_W[j]);
                for(i = 0; i < x_W.size(); i++){
                    aux_sol.push_back(z_W[x_W.size()*j + i]);
                }
                for(i = 1; i < x_E.size(); i++){
                    aux_sol.push_back(z_E[x_W.size()*j + i]);
                }
            }
            for(i = 0; i < x_W.size(); i++){
                    aux_x.push_back(x_W[i]);
            }
            for(i = 1; i < x_E.size(); i++){
                    aux_x.push_back(x_E[i]);
            }
        }else{
            //vertical patch
            //std::cout << "Vertical patch\n";
            subd_index[0][0] = Iindexes[0]; subd_index[0][1] = (Iindexes[1]-1)/2;
            subd_index[1][0] = Iindexes[0]; subd_index[1][1] = 1+((Iindexes[1]-1)/2);
            std::vector<double> x_N, x_S;
            std::vector<double> y_N, y_S;
            std::vector<double> z_N, z_S;
            unsigned int i,j;
            sprintf(aux_fname,"Output/Subdomains/X_%d_%d.txt",subd_index[0][0],subd_index[0][1]);
            x_S = Read_File(aux_fname);
            sprintf(aux_fname,"Output/Subdomains/Y_%d_%d.txt",subd_index[0][0],subd_index[0][1]);
            y_S = Read_File(aux_fname);
            sprintf(aux_fname,"Output/Subdomains/Sol_%d_%d.txt",subd_index[0][0],subd_index[0][1]);
            z_S = Read_File(aux_fname);
            sprintf(aux_fname,"Output/Subdomains/X_%d_%d.txt",subd_index[1][0],subd_index[1][1]);
            x_N = Read_File(aux_fname);
            sprintf(aux_fname,"Output/Subdomains/Y_%d_%d.txt",subd_index[1][0],subd_index[1][1]);
            y_N = Read_File(aux_fname);
            sprintf(aux_fname,"Output/Subdomains/Sol_%d_%d.txt",subd_index[1][0],subd_index[1][1]);
            z_N = Read_File(aux_fname);
            aux_x.resize(0); aux_y.resize(0); aux_sol.resize(0);
            for(j = 0; j < y_S.size(); j++){
                aux_y.push_back(y_S[j]);
                for(i = 0; i < x_S.size(); i++){
                    aux_sol.push_back(z_S[x_S.size()*j + i]);
                }
            }
            for(j = 1; j < y_N.size(); j++){
                aux_y.push_back(y_N[j]);
                for(i = 0; i < x_S.size(); i++){
                    aux_sol.push_back(z_N[x_S.size()*j + i]);
                }
            }
            for(i = 0; i < x_S.size(); i++){
                    aux_x.push_back(x_S[i]);
            }
        }
    }else{
        //Knot belongs to 4 interfaces (It is on an intersection)
        //std::cout << "Squared patch\n";
        std::vector<int> aux_index;
        aux_index.resize(2);
        subd_index.resize(2);
        std::vector<double> x_W, x_E;
        std::vector<double> y_W, y_E;
        std::vector<double> z_W,z_E;
        unsigned int i,j,cent = 0;
        aux_x.resize(0); aux_y.resize(0); aux_sol.resize(0);
        for(int l = 0; l < 8; l = l +2){
            if(Iindexes[l+1]%2==0){
                aux_index[0] = Iindexes[l];
                aux_index[1] = Iindexes[l+1]/2;
                subd_index[0] = aux_index;
                aux_index[0] = Iindexes[l] + 1;
                aux_index[1] = Iindexes[l+1]/2;
                subd_index[1] = aux_index;
                i = 0; j = 0;
                sprintf(aux_fname,"Output/Subdomains/X_%d_%d.txt",subd_index[0][0],subd_index[0][1]);
                x_W = Read_File(aux_fname);
                //std::cout << aux_fname << std::endl;
                sprintf(aux_fname,"Output/Subdomains/Y_%d_%d.txt",subd_index[0][0],subd_index[0][1]);
                y_W = Read_File(aux_fname);
                sprintf(aux_fname,"Output/Subdomains/Sol_%d_%d.txt",subd_index[0][0],subd_index[0][1]);
                z_W = Read_File(aux_fname);
                sprintf(aux_fname,"Output/Subdomains/X_%d_%d.txt",subd_index[1][0],subd_index[1][1]);
                x_E = Read_File(aux_fname);
                //std::cout << aux_fname << std::endl;
                sprintf(aux_fname,"Output/Subdomains/Y_%d_%d.txt",subd_index[1][0],subd_index[1][1]);
                y_E = Read_File(aux_fname);
                sprintf(aux_fname,"Output/Subdomains/Sol_%d_%d.txt",subd_index[1][0],subd_index[1][1]);
                z_E = Read_File(aux_fname);
                //std::cout << y_W.size() << std::endl;
                //std::cout << aux_y.size() << std::endl;
                for(j = cent; j < y_W.size(); j++){
                    //std::cout << y_W[j] << std::endl;
                    aux_y.push_back(y_W[j]);
                    for(i = 0; i < x_W.size(); i++){
                        aux_sol.push_back(z_W[x_W.size()*j + i]);
                    }
                    for(i = 1; i < x_E.size(); i++){
                       aux_sol.push_back(z_E[x_W.size()*j + i]);
                    }
                }
                cent ++;
            }
        }
        for(i = 0; i < x_W.size(); i++){
                aux_x.push_back(x_W[i]);
        }
        for(i = 1; i < x_E.size(); i++){
                aux_x.push_back(x_E[i]);
        }
    }
    /*double *array_x, *array_y, *array_sol;
    array_x = new double[aux_x.size()];for(unsigned int i = 0; i < aux_x.size(); i++) array_x[i] = aux_x[i];
    array_y = new double[aux_y.size()];for(unsigned int i = 0; i < aux_y.size(); i++) array_y[i] = aux_y[i];
    array_sol = new double[aux_sol.size()];for(unsigned int i = 0; i < aux_sol.size(); i++) array_sol[i] = aux_sol[i];
    Init_LUT(aux_x.size(),aux_y.size(),array_x,array_y,array_sol,spline_u,xacc_u,yacc_u);
    delete array_x; delete array_y; delete array_sol;
    */
}
void PDDSparseGM::Set_LUT_FILE_Correction(int index, std::vector<double> & aux_x,std::vector<double> & aux_y,std::vector<double> & aux_sol){
    std::vector<std::vector<int>> subd_index;
    std::vector< std::vector<int> > Iindexes_prev  = Get_Interfaces(index);
    #ifdef SQUARE_PATCHES
        Interface inter = interfaces[interface_map[Iindexes_prev[0]]];
        int cent = -1;
        for(unsigned int i = 0; i < inter.index.size(); i ++){
                if(inter.index[i] == index)cent = (int) i;
        }
        if(cent < 0){
            std::cout << "Labels Stencil Square is not working properly\n";
            std::cout << "Index " << index << "\n";
        }else{
            if((1.0*cent/inter.index.size())<0.5){
                    Iindexes_prev = Get_Interfaces(inter.index.front());
            }else {
                    Iindexes_prev = Get_Interfaces(inter.index.back());
            }
        }
    #endif
    std::vector<int> Iindexes;
    for(unsigned int i = 0; i < Iindexes_prev.size(); i++){ Iindexes.push_back(Iindexes_prev[i][0]); Iindexes.push_back(Iindexes_prev[i][1]);}
    char aux_fname[256];
    if(Iindexes.size() == 2){
        subd_index.resize(2);
        subd_index[0].resize(2);
        subd_index[1].resize(2);
        if(Iindexes[1]%2== 0){
            //horizontal patch
            //std::cout << "Horizontal patch\n";
            subd_index[0][0] = Iindexes[0]; subd_index[0][1] = Iindexes[1]/2;
            subd_index[1][0] = Iindexes[0]+1; subd_index[1][1] = Iindexes[1]/2;
            std::vector<double> x_W, x_E;
            std::vector<double> y_W, y_E;
            std::vector<double> z_W, z_W_n, z_E, z_E_n;
            unsigned int i,j;
            sprintf(aux_fname,"Output/Subdomains/X_%d_%d.txt",subd_index[0][0],subd_index[0][1]);
            x_W = Read_File(aux_fname);
            sprintf(aux_fname,"Output/Subdomains/Y_%d_%d.txt",subd_index[0][0],subd_index[0][1]);
            y_W = Read_File(aux_fname);
            sprintf(aux_fname,"Output/Subdomains/Correction_%d_%d.txt",subd_index[0][0],subd_index[0][1]);
            z_W = Read_File(aux_fname);
            sprintf(aux_fname,"Output/Subdomains/X_%d_%d.txt",subd_index[1][0],subd_index[1][1]);
            x_E = Read_File(aux_fname);
            sprintf(aux_fname,"Output/Subdomains/Y_%d_%d.txt",subd_index[1][0],subd_index[1][1]);
            y_E = Read_File(aux_fname);
            sprintf(aux_fname,"Output/Subdomains/Correction_%d_%d.txt",subd_index[1][0],subd_index[1][1]);
            z_E = Read_File(aux_fname);
            aux_x.resize(0); aux_y.resize(0); aux_sol.resize(0);
            for(j = 0; j < y_W.size(); j++){
                aux_y.push_back(y_W[j]);
                for(i = 0; i < x_W.size(); i++){
                    aux_sol.push_back(z_W[x_W.size()*j + i]);
                }
                for(i = 1; i < x_E.size(); i++){
                    aux_sol.push_back(z_E[x_W.size()*j + i]);
                }
            }
            for(i = 0; i < x_W.size(); i++){
                    aux_x.push_back(x_W[i]);
            }
            for(i = 1; i < x_E.size(); i++){
                    aux_x.push_back(x_E[i]);
            }
        }else{
            //vertical patch
            //std::cout << "Vertical patch\n";
            subd_index[0][0] = Iindexes[0]; subd_index[0][1] = (Iindexes[1]-1)/2;
            subd_index[1][0] = Iindexes[0]; subd_index[1][1] = 1+((Iindexes[1]-1)/2);
            std::vector<double> x_N, x_S;
            std::vector<double> y_N, y_S;
            std::vector<double> z_N, z_S;
            unsigned int i,j;
            sprintf(aux_fname,"Output/Subdomains/X_%d_%d.txt",subd_index[0][0],subd_index[0][1]);
            x_S = Read_File(aux_fname);
            sprintf(aux_fname,"Output/Subdomains/Y_%d_%d.txt",subd_index[0][0],subd_index[0][1]);
            y_S = Read_File(aux_fname);
            sprintf(aux_fname,"Output/Subdomains/Correction_%d_%d.txt",subd_index[0][0],subd_index[0][1]);
            z_S = Read_File(aux_fname);
            sprintf(aux_fname,"Output/Subdomains/X_%d_%d.txt",subd_index[1][0],subd_index[1][1]);
            x_N = Read_File(aux_fname);
            sprintf(aux_fname,"Output/Subdomains/Y_%d_%d.txt",subd_index[1][0],subd_index[1][1]);
            y_N = Read_File(aux_fname);
            sprintf(aux_fname,"Output/Subdomains/Correction_%d_%d.txt",subd_index[1][0],subd_index[1][1]);
            z_N = Read_File(aux_fname);
            aux_x.resize(0); aux_y.resize(0); aux_sol.resize(0);
            for(j = 0; j < y_S.size(); j++){
                aux_y.push_back(y_S[j]);
                for(i = 0; i < x_S.size(); i++){
                    aux_sol.push_back(z_S[x_S.size()*j + i]);
                }
            }
            for(j = 1; j < y_N.size(); j++){
                aux_y.push_back(y_N[j]);
                for(i = 0; i < x_S.size(); i++){
                    aux_sol.push_back(z_N[x_S.size()*j + i]);
                }
            }
            for(i = 0; i < x_S.size(); i++){
                    aux_x.push_back(x_S[i]);
            }
        }
    }else{
        //Knot belongs to 4 interfaces (It is on an intersection)
        //std::cout << "Squared patch\n";
        std::vector<int> aux_index;
        aux_index.resize(2);
        subd_index.resize(2);
        std::vector<double> x_W, x_E;
        std::vector<double> y_W, y_E;
        std::vector<double> z_W,z_E;
        unsigned int i,j,cent = 0;
        aux_x.resize(0); aux_y.resize(0); aux_sol.resize(0);
        for(int l = 0; l < 8; l = l +2){
            if(Iindexes[l+1]%2==0){
                aux_index[0] = Iindexes[l];
                aux_index[1] = Iindexes[l+1]/2;
                subd_index[0] = aux_index;
                aux_index[0] = Iindexes[l] + 1;
                aux_index[1] = Iindexes[l+1]/2;
                subd_index[1] = aux_index;
                i = 0; j = 0;
                sprintf(aux_fname,"Output/Subdomains/X_%d_%d.txt",subd_index[0][0],subd_index[0][1]);
                x_W = Read_File(aux_fname);
                //std::cout << aux_fname << std::endl;
                sprintf(aux_fname,"Output/Subdomains/Y_%d_%d.txt",subd_index[0][0],subd_index[0][1]);
                y_W = Read_File(aux_fname);
                sprintf(aux_fname,"Output/Subdomains/Correction_%d_%d.txt",subd_index[0][0],subd_index[0][1]);
                z_W = Read_File(aux_fname);
                sprintf(aux_fname,"Output/Subdomains/X_%d_%d.txt",subd_index[1][0],subd_index[1][1]);
                x_E = Read_File(aux_fname);
                //std::cout << aux_fname << std::endl;
                sprintf(aux_fname,"Output/Subdomains/Y_%d_%d.txt",subd_index[1][0],subd_index[1][1]);
                y_E = Read_File(aux_fname);
                sprintf(aux_fname,"Output/Subdomains/Correction_%d_%d.txt",subd_index[1][0],subd_index[1][1]);
                z_E = Read_File(aux_fname);
                //std::cout << y_W.size() << std::endl;
                //std::cout << aux_y.size() << std::endl;
                for(j = cent; j < y_W.size(); j++){
                    //std::cout << y_W[j] << std::endl;
                    aux_y.push_back(y_W[j]);
                    for(i = 0; i < x_W.size(); i++){
                        aux_sol.push_back(z_W[x_W.size()*j + i]);
                    }
                    for(i = 1; i < x_E.size(); i++){
                       aux_sol.push_back(z_E[x_W.size()*j + i]);
                    }
                }
                cent ++;
            }
        }
        for(i = 0; i < x_W.size(); i++){
                aux_x.push_back(x_W[i]);
        }
        for(i = 1; i < x_E.size(); i++){
                aux_x.push_back(x_E[i]);
        }
    }
}
Stencil PDDSparseGM::Recieve_Stencil_Data_Loop(void){
    int size[1];
    MPI_Recv(size,1,MPI_INT, server, INTER_NUMBER, world, MPI_STATUS_IGNORE);
    int *Iindexes = new int[size[0]];
    MPI_Recv(Iindexes,size[0],MPI_INT,server, INTER_LABELS,world,MPI_STATUS_IGNORE);
    std::vector<std::vector<int>> subd_index;
    FILE *ox_file, *oy_file, *osol_file,
         *ox_noisy_file, *oy_noisy_file, *osol_noisy_file;
    char aux_fname[256];
    sprintf(aux_fname,"Input/Patch_%d/x_0.txt",myid);
    if((ox_file = fopen(aux_fname,"w"))==NULL){
        sprintf(aux_fname,"mkdir -p Input/Patch_%d",myid);
        system(aux_fname);
        sprintf(aux_fname,"Input/Patch_%d/x_0.txt",myid);
        ox_file = fopen(aux_fname,"w");
    }
    sprintf(aux_fname,"Input/Patch_noisy_%d/x_0.txt",myid);
    if((ox_noisy_file = fopen(aux_fname,"w"))==NULL){
        sprintf(aux_fname,"mkdir -p Input/Patch_noisy_%d",myid);
        system(aux_fname);
        sprintf(aux_fname,"Input/Patch_noisy_%d/x_0.txt",myid);
        ox_noisy_file = fopen(aux_fname,"w");
    }
    sprintf(aux_fname,"Input/Patch_%d/x_1.txt",myid);
    oy_file = fopen(aux_fname,"w");
    sprintf(aux_fname,"Input/Patch_noisy_%d/x_1.txt",myid);
    oy_noisy_file = fopen(aux_fname,"w");
    sprintf(aux_fname,"Input/Patch_%d/value.txt",myid);
    osol_file = fopen(aux_fname,"w");
    sprintf(aux_fname,"Input/Patch_noisy_%d/value.txt",myid);
    osol_noisy_file = fopen(aux_fname,"w");
    if(size[0] == 2){
        subd_index.resize(2);
        subd_index[0].resize(2);
        subd_index[1].resize(2);
        if(Iindexes[1]%2== 0){
            //horizontal patch
            //std::cout << "Horizontal patch\n";
            subd_index[0][0] = Iindexes[0]; subd_index[0][1] = Iindexes[1]/2;
            subd_index[1][0] = Iindexes[0]+1; subd_index[1][1] = Iindexes[1]/2;
            std::vector<double> x_W, x_E;
            std::vector<double> y_W, y_E;
            std::vector<double> z_W, z_W_n, z_E, z_E_n;
            unsigned int i,j;
            sprintf(aux_fname,"Output/Subdomains/X_%d_%d.txt",subd_index[0][0],subd_index[0][1]);
            x_W = Read_File(aux_fname);
            sprintf(aux_fname,"Output/Subdomains/Y_%d_%d.txt",subd_index[0][0],subd_index[0][1]);
            y_W = Read_File(aux_fname);
            sprintf(aux_fname,"Output/Subdomains/Sol_%d_%d.txt",subd_index[0][0],subd_index[0][1]);
            z_W = Read_File(aux_fname);
            sprintf(aux_fname,"Output/Subdomains/Sol_noisy_%d_%d.txt",subd_index[0][0],subd_index[0][1]);
            z_W_n = Read_File(aux_fname);
            sprintf(aux_fname,"Output/Subdomains/X_%d_%d.txt",subd_index[1][0],subd_index[1][1]);
            x_E = Read_File(aux_fname);
            sprintf(aux_fname,"Output/Subdomains/Y_%d_%d.txt",subd_index[1][0],subd_index[1][1]);
            y_E = Read_File(aux_fname);
            sprintf(aux_fname,"Output/Subdomains/Sol_%d_%d.txt",subd_index[1][0],subd_index[1][1]);
            z_E = Read_File(aux_fname);
            sprintf(aux_fname,"Output/Subdomains/Sol_noisy_%d_%d.txt",subd_index[1][0],subd_index[1][1]);
            z_E_n = Read_File(aux_fname);
            for(j = 0; j < y_W.size(); j++){
                fprintf(oy_file,"%e\n",y_W[j]);
                fprintf(oy_noisy_file,"%e\n",y_W[j]);
                for(i = 0; i < x_W.size(); i++){
                    fprintf(osol_file," %e",z_W[x_W.size()*j + i]);
                    fprintf(osol_noisy_file," %e",z_W_n[x_W.size()*j + i]);
                }
                for(i = 1; i < x_E.size(); i++){
                    fprintf(osol_file," %e",z_E[x_W.size()*j + i]);
                    fprintf(osol_noisy_file," %e",z_E_n[x_W.size()*j + i]);
                }
                fprintf(osol_file,"\n");
                fprintf(osol_noisy_file,"\n");
            }
            fclose(osol_file);
            fclose(osol_noisy_file);
            fclose(oy_file);
            fclose(oy_noisy_file);
            for(i = 0; i < x_W.size(); i++){
                    fprintf(ox_file,"%e\n",x_W[i]);
                    fprintf(ox_noisy_file,"%e\n",x_W[i]);
            }
            for(i = 1; i < x_E.size(); i++){
                    fprintf(ox_file,"%e\n",x_E[i]);
                    fprintf(ox_noisy_file,"%e\n",x_E[i]);
            }
            fclose(ox_file);
            fclose(ox_noisy_file);
        }else{
            //vertical patch
            //std::cout << "Vertical patch\n";
            subd_index[0][0] = Iindexes[0]; subd_index[0][1] = (Iindexes[1]-1)/2;
            subd_index[1][0] = Iindexes[0]; subd_index[1][1] = 1+((Iindexes[1]-1)/2);
            std::vector<double> x_N, x_S;
            std::vector<double> y_N, y_S;
            std::vector<double> z_N, z_N_n, z_S, z_S_n;
            unsigned int i,j;
            sprintf(aux_fname,"Output/Subdomains/X_%d_%d.txt",subd_index[0][0],subd_index[0][1]);
            x_S = Read_File(aux_fname);
            sprintf(aux_fname,"Output/Subdomains/Y_%d_%d.txt",subd_index[0][0],subd_index[0][1]);
            y_S = Read_File(aux_fname);
            sprintf(aux_fname,"Output/Subdomains/Sol_%d_%d.txt",subd_index[0][0],subd_index[0][1]);
            z_S = Read_File(aux_fname);
            sprintf(aux_fname,"Output/Subdomains/Sol_noisy_%d_%d.txt",subd_index[0][0],subd_index[0][1]);
            z_S_n = Read_File(aux_fname);
            sprintf(aux_fname,"Output/Subdomains/X_%d_%d.txt",subd_index[1][0],subd_index[1][1]);
            x_N = Read_File(aux_fname);
            sprintf(aux_fname,"Output/Subdomains/Y_%d_%d.txt",subd_index[1][0],subd_index[1][1]);
            y_N = Read_File(aux_fname);
            sprintf(aux_fname,"Output/Subdomains/Sol_%d_%d.txt",subd_index[1][0],subd_index[1][1]);
            z_N = Read_File(aux_fname);
            sprintf(aux_fname,"Output/Subdomains/Sol_noisy_%d_%d.txt",subd_index[1][0],subd_index[1][1]);
            z_N_n = Read_File(aux_fname);
            for(j = 0; j < y_S.size(); j++){
                fprintf(oy_file,"%e\n",y_S[j]);
                fprintf(oy_noisy_file,"%e\n",y_S[j]);
                for(i = 0; i < x_S.size(); i++){
                    fprintf(osol_file," %e",z_S[x_S.size()*j + i]);
                    fprintf(osol_noisy_file," %e",z_S_n[x_S.size()*j + i]);
                }
                fprintf(osol_file,"\n");
                fprintf(osol_noisy_file,"\n");
            }
            for(j = 1; j < y_N.size(); j++){
                fprintf(oy_file,"%e\n",y_N[j]);
                fprintf(oy_noisy_file,"%e\n",y_N[j]);
                for(i = 0; i < x_S.size(); i++){
                    fprintf(osol_file," %e",z_N[x_S.size()*j + i]);
                    fprintf(osol_noisy_file," %e",z_N_n[x_S.size()*j + i]);
                }
                fprintf(osol_file,"\n");
                fprintf(osol_noisy_file,"\n");
            }
            fclose(osol_file);
            fclose(osol_noisy_file);
            fclose(oy_file);
            fclose(oy_noisy_file);
            for(i = 0; i < x_S.size(); i++){
                    fprintf(ox_file," %e\n",x_S[i]);
                    fprintf(ox_noisy_file," %e\n",x_S[i]);
            }
            fclose(ox_file);
            fclose(ox_noisy_file);
        }
    }else{
        //Knot belongs to 4 interfaces (It is on an intersection)
        //std::cout << "Squared patch\n";
        std::vector<int> aux_index;
        aux_index.resize(2);
        subd_index.resize(2);
        std::vector<double> x_W, x_E;
        std::vector<double> y_W, y_E;
        std::vector<double> z_W, z_W_n,z_E,z_E_n;
        unsigned int i,j,cent = 0;
        for(int l = 0; l < 8; l = l +2){
            if(Iindexes[l+1]%2==0){
                aux_index[0] = Iindexes[l];
                aux_index[1] = Iindexes[l+1]/2;
                subd_index[0] = aux_index;
                aux_index[0] = Iindexes[l] + 1;
                aux_index[1] = Iindexes[l+1]/2;
                subd_index[1] = aux_index;
                i = 0; j = 0;
                sprintf(aux_fname,"Output/Subdomains/X_%d_%d.txt",subd_index[0][0],subd_index[0][1]);
                x_W = Read_File(aux_fname);
                sprintf(aux_fname,"Output/Subdomains/Y_%d_%d.txt",subd_index[0][0],subd_index[0][1]);
                y_W = Read_File(aux_fname);
                sprintf(aux_fname,"Output/Subdomains/Sol_%d_%d.txt",subd_index[0][0],subd_index[0][1]);
                z_W = Read_File(aux_fname);
                sprintf(aux_fname,"Output/Subdomains/Sol_noisy_%d_%d.txt",subd_index[0][0],subd_index[0][1]);
                z_W_n = Read_File(aux_fname);
                sprintf(aux_fname,"Output/Subdomains/X_%d_%d.txt",subd_index[1][0],subd_index[1][1]);
                x_E = Read_File(aux_fname);
                sprintf(aux_fname,"Output/Subdomains/Y_%d_%d.txt",subd_index[1][0],subd_index[1][1]);
                y_E = Read_File(aux_fname);
                sprintf(aux_fname,"Output/Subdomains/Sol_%d_%d.txt",subd_index[1][0],subd_index[1][1]);
                z_E = Read_File(aux_fname);
                sprintf(aux_fname,"Output/Subdomains/Sol_noisy_%d_%d.txt",subd_index[1][0],subd_index[1][1]);
                z_E_n = Read_File(aux_fname);
                for(j = cent; j < y_W.size(); j++){
                    fprintf(oy_file,"%e\n",y_W[j]);
                    fprintf(oy_noisy_file,"%e\n",y_W[j]);
                    for(i = 0; i < x_W.size(); i++){
                        fprintf(osol_file," %e",z_W[x_W.size()*j + i]);
                        fprintf(osol_noisy_file," %e",z_W_n[x_W.size()*j + i]);
                    }
                    for(i = 1; i < x_E.size(); i++){
                        fprintf(osol_file," %e",z_E[x_W.size()*j + i]);
                        fprintf(osol_noisy_file," %e",z_E_n[x_W.size()*j + i]);
                    }
                    fprintf(osol_file,"\n");
                    fprintf(osol_noisy_file,"\n");
                }
                cent ++;
            }
        }
        fclose(osol_file);
        fclose(osol_noisy_file);
        fclose(oy_file);
        fclose(oy_noisy_file);
        for(i = 0; i < x_W.size(); i++){
                fprintf(ox_file,"%e\n",x_W[i]);
                fprintf(ox_noisy_file,"%e\n",x_W[i]);
        }
        for(i = 1; i < x_E.size(); i++){
                fprintf(ox_file,"%e\n",x_E[i]);
                fprintf(ox_noisy_file,"%e\n",x_E[i]);
        }
        fclose(ox_file);
        fclose(ox_noisy_file);
    }
    return Recieve_Stencil_Data();
}
void PDDSparseGM::Send_G_B(void){
    work_control[0] = (int) G.size();
    work_control[1] = (int) B.size();
    MPI_Send(work_control, 2, MPI_INT, server, REQUEST_INTERFACE, world);
    //G_i is sent
    int *G_i_aux;
    G_i_aux = new int[work_control[0]];
    for(int i = 0; i < work_control[0]; i++) G_i_aux[i] = G_i[i];
    G_i.resize(0);
    MPI_Send(G_i_aux, work_control[0], MPI_INT, server, TAG_Gi, world);
    delete G_i_aux;
    //G_j is sent
    int *G_j_aux;
    G_j_aux = new int[work_control[0]];
    for(int i = 0; i < work_control[0]; i++) G_j_aux[i] = G_j[i];
    G_j.resize(0);
    MPI_Send(G_j_aux, work_control[0], MPI_INT, server, TAG_Gj, world);
    delete G_j_aux;
    //G_val is sent
    double *G_aux;
    G_aux = new double[work_control[0]];
    for(int i = 0; i < work_control[0]; i++){
         G_aux[i] = G[i];
    }
    G.resize(0);
    MPI_Send(G_aux, work_control[0], MPI_DOUBLE, server,TAG_Gval, world);
    for(int i = 0; i < work_control[0]; i++){
         G_aux[i] = G_var[i];
    }
    MPI_Send(G_aux, work_control[0], MPI_DOUBLE, server,TAG_Gvar, world);
    delete G_aux;
    //B_i is sent
    int *B_i_aux;
    B_i_aux = new int[work_control[1]];
    for(int i = 0; i < work_control[1]; i++) B_i_aux[i] = B_i[i];
    B_i.resize(0);
    MPI_Send(B_i_aux, work_control[1], MPI_INT, server, TAG_Bi, world);
    delete B_i_aux;
    //B is sent
    double *B_aux;
    B_aux = new double[work_control[1]];
    for(int i = 0; i < work_control[1]; i++) B_aux[i] = B[i];
    B.resize(0);
    MPI_Send(B_aux, work_control[1], MPI_DOUBLE, server, TAG_Bval, world);
    for(int i = 0; i < work_control[1]; i++) B_aux[i] = B_var[i];
    B.resize(0);
    MPI_Send(B_aux, work_control[1], MPI_DOUBLE, server, TAG_Bvar, world);
    delete B_aux;
    FILE *dfile;
    dfile = fopen(debug_fname,"a");
    fprintf(dfile,"Processor %d ended sending its contribution to G and B\n", workerid);
    fclose(dfile);
    //MPI_Comm_free(&workers);
}
void PDDSparseGM::Send_G_B_2(int aux_server){
    work_control[0] = (int) G.size();
    work_control[1] = (int) B.size();
    MPI_Send(work_control, 2, MPI_INT, aux_server, REQUEST_NODE, world);
    //G_i is sent
    int *G_i_aux;
    G_i_aux = new int[work_control[0]];
    for(int i = 0; i < work_control[0]; i++) G_i_aux[i] = G_i[i];
    G_i.resize(0);
    MPI_Send(G_i_aux, work_control[0], MPI_INT, aux_server, TAG_Gi, world);
    delete G_i_aux;
    //G_j is sent
    int *G_j_aux;
    G_j_aux = new int[work_control[0]];
    for(int i = 0; i < work_control[0]; i++) G_j_aux[i] = G_j[i];
    G_j.resize(0);
    MPI_Send(G_j_aux, work_control[0], MPI_INT, aux_server, TAG_Gj, world);
    delete G_j_aux;
    //G_val is sent
    double *G_aux;
    G_aux = new double[work_control[0]];
    for(int i = 0; i < work_control[0]; i++){
         G_aux[i] = G[i];
    }
    G.resize(0);
    MPI_Send(G_aux, work_control[0], MPI_DOUBLE, aux_server,TAG_Gval, world);
    for(int i = 0; i < work_control[0]; i++){
         G_aux[i] = G_var[i];
    }
    G_var.resize(0);
    MPI_Send(G_aux, work_control[0], MPI_DOUBLE, aux_server,TAG_Gvar, world);
    for(int i = 0; i < work_control[0]; i++){
         G_aux[i] = G_CT[i];
    }
    G_CT.resize(0);
    MPI_Send(G_aux, work_control[0], MPI_DOUBLE, aux_server,TAG_GCT, world);
    delete G_aux;
    //B_i is sent
    int *B_i_aux;
    B_i_aux = new int[work_control[1]];
    for(int i = 0; i < work_control[1]; i++){
        B_i_aux[i] = B_i[i];
    } 
    B_i.resize(0);
    MPI_Send(B_i_aux, work_control[1], MPI_INT, aux_server, TAG_Bi, world);
    delete B_i_aux;
    //B is sent
    double *B_aux;
    B_aux = new double[work_control[1]];
    for(int i = 0; i < work_control[1]; i++) B_aux[i] = B[i];
    B.resize(0);
    MPI_Send(B_aux, work_control[1], MPI_DOUBLE, aux_server, TAG_Bval, world);
    for(int i = 0; i < work_control[1]; i++) B_aux[i] = B_var[i];
    B_var.resize(0);
    MPI_Send(B_aux, work_control[1], MPI_DOUBLE, aux_server, TAG_Bvar, world);
    for(int i = 0; i < work_control[1]; i++) B_aux[i] = B_CT[i];
    B_CT.resize(0);
    MPI_Send(B_aux, work_control[1], MPI_DOUBLE, aux_server, TAG_BCT, world);
    delete B_aux;
    //FILE *dfile;
    //dfile = fopen(debug_fname,"a");
    //fprintf(dfile,"Processor %d ended sending its contribution to G and B\n", workerid);
    //fclose(dfile);
    //MPI_Comm_free(&workers);
}
void PDDSparseGM::Send_B(void){
    work_control[0] = 0;
    work_control[1] = (int) B.size();
    //std::cout << "Sending \n";
    //for(int i = 0; i < work_control[1]; i++) std::cout << B_i[i] << " " << B[i] << std::endl;
    MPI_Send(work_control, 2, MPI_INT, server, REQUEST_INTERFACE, world);
    //B_i is sent
    int *B_i_aux;
    B_i_aux = new int[work_control[1]];
    for(int i = 0; i < work_control[1]; i++) B_i_aux[i] = B_i[i];
    B_i.resize(0);
    MPI_Send(B_i_aux, work_control[1], MPI_INT, server, TAG_Bi, world);
    //for(int i = 0; i < work_control[1]; i++) std::cout << B_i_aux[i] << std::endl;
    delete B_i_aux;
    //B is sent
    double *B_aux;
    B_aux = new double[work_control[1]];
    for(int i = 0; i < work_control[1]; i++) B_aux[i] = B[i];
    B.resize(0);
    MPI_Send(B_aux, work_control[1], MPI_DOUBLE, server, TAG_Bval, world);
    delete B_aux;
}
void PDDSparseGM::Send_Metadata(int aux_server){
    double *double_aux;
    double_aux = new double[work_control[1]];
    for(int i = 0; i < work_control[1]; i++) double_aux[i] = APL[i];
    MPI_Send(double_aux, work_control[1], MPI_DOUBLE, aux_server, TAG_APL, world);
    APL.clear();
    //for(int i = 0; i < work_control[1]; i++) double_aux[i] = times[i];
    //MPI_Send(double_aux, work_control[1], MPI_DOUBLE, aux_server, TAG_time, world);
    times.clear();
    for(int i = 0; i < work_control[1]; i++) double_aux[i] = Est_var[i];
    MPI_Send(double_aux, work_control[1], MPI_DOUBLE, aux_server, TAG_Estvar, world);
    Est_var.clear();
    for(int i = 0; i < work_control[1]; i++) double_aux[i] = RNGCalls[i];
    MPI_Send(double_aux, work_control[1], MPI_DOUBLE, aux_server, TAG_RNGCalls, world);
    RNGCalls.clear();
    for(int i = 0; i < work_control[1]; i++) double_aux[i] = xi[i];
    MPI_Send(double_aux, work_control[1], MPI_DOUBLE, aux_server, TAG_xi, world);
    xi.clear();
    for(int i = 0; i < work_control[1]; i++) double_aux[i] = phi[i];
    MPI_Send(double_aux, work_control[1], MPI_DOUBLE, aux_server, TAG_phi, world);
    phi.clear();
    for(int i = 0; i < work_control[1]; i++) double_aux[i] = xixi[i];
    MPI_Send(double_aux, work_control[1], MPI_DOUBLE, aux_server, TAG_xixi, world);
    xixi.clear();
    for(int i = 0; i < work_control[1]; i++) double_aux[i] = phiphi[i];
    MPI_Send(double_aux, work_control[1], MPI_DOUBLE, aux_server, TAG_phiphi, world);
    phiphi.clear();
    for(int i = 0; i < work_control[1]; i++) double_aux[i] = phixi[i];
    MPI_Send(double_aux, work_control[1], MPI_DOUBLE, aux_server, TAG_phixi, world);
    phixi.clear();
    for(int i = 0; i < work_control[1]; i++) double_aux[i] = phip[i];
    MPI_Send(double_aux, work_control[1], MPI_DOUBLE, aux_server, TAG_phip, world);
    phip.clear();
    for(int i = 0; i < work_control[1]; i++) double_aux[i] = phiphip[i];
    MPI_Send(double_aux, work_control[1], MPI_DOUBLE, aux_server, TAG_phiphip, world);
    phiphip.clear();
    for(int i = 0; i < work_control[1]; i++) double_aux[i] = phipxi[i];
    MPI_Send(double_aux, work_control[1], MPI_DOUBLE, aux_server, TAG_phipxi, world);
    phipxi.clear();
    for(int i = 0; i < work_control[1]; i++) double_aux[i] = phi_trap[i];
    MPI_Send(double_aux, work_control[1], MPI_DOUBLE, aux_server, TAG_phi_trap, world);
    phi_trap.clear();
    for(int i = 0; i < work_control[1]; i++) double_aux[i] = phiphi_trap[i];
    MPI_Send(double_aux, work_control[1], MPI_DOUBLE, aux_server, TAG_phiphi_trap, world);
    phiphi_trap.clear();
    for(int i = 0; i < work_control[1]; i++) double_aux[i] = phi_trap_VR[i];
    MPI_Send(double_aux, work_control[1], MPI_DOUBLE, aux_server, TAG_phi_trap_VR, world);
    phi_trap_VR.clear();
    for(int i = 0; i < work_control[1]; i++) double_aux[i] = phiphi_trap_VR[i];
    MPI_Send(double_aux, work_control[1], MPI_DOUBLE, aux_server, TAG_phiphi_trap_VR, world);
    phiphi_trap_VR.clear();
    for(int i = 0; i < work_control[1]; i++) double_aux[i] = bias_num[i];
    MPI_Send(double_aux, work_control[1], MPI_DOUBLE, aux_server, TAG_bias_num, world);
    bias_num.clear();
    for(int i = 0; i < work_control[1]; i++) double_aux[i] = bias_trap[i];
    MPI_Send(double_aux, work_control[1], MPI_DOUBLE, aux_server, TAG_bias_trap, world);
    bias_trap.clear();
    for(int i = 0; i < work_control[1]; i++)double_aux[i] = Est_var_VR[i];
    MPI_Send(double_aux, work_control[1], MPI_DOUBLE, aux_server, TAG_Estvar_VR, world);
    Est_var_VR.clear();
    delete double_aux;
}
void PDDSparseGM::Receive_G_B(void){
    //It ends worker process solving
    MPI_Recv(work_control, 2, MPI_INT, MPI_ANY_SOURCE, 
    REQUEST_INTERFACE, world, &status);
    work_control[0] = 0;
    work_control[1] = 0;
    MPI_Send(work_control, 2, MPI_INT, status.MPI_SOURCE, REPLY_INTERFACE, world);
    //The server accumulates the values of G and B
    MPI_Recv(work_control, 2, MPI_INT, status.MPI_SOURCE, REQUEST_INTERFACE, world, &status);
    G.resize(work_control[0]);
    G_i.resize(work_control[0]);
    G_j.resize(work_control[0]);
    //G_i is received
    int *G_i_aux, *G_j_aux;
    double *G_aux;
    G_i_aux = new int[work_control[0]];
    MPI_Recv(G_i_aux, work_control[0], MPI_INT, status.MPI_SOURCE, TAG_Gi, world, &status);
    for(int i = 0; i < work_control[0]; i++) G_i[i] = G_i_aux[i];
    delete G_i_aux;
    //G_j is received
    G_j_aux = new int[work_control[0]];
    MPI_Recv(G_j_aux, work_control[0], MPI_INT, status.MPI_SOURCE, TAG_Gj, world, &status);
    for(int i = 0; i < work_control[0]; i++) G_j[i] = G_j_aux[i];
    delete G_j_aux;
    //G is received
    G_aux = new double[work_control[0]];
    MPI_Recv(G_aux, work_control[0], MPI_DOUBLE, status.MPI_SOURCE, TAG_Gval, world, &status);
    for(int i = 0; i < work_control[0]; i++) G[i] = G_aux[i];
    //Triplets vector is fullfilled
    for(int j = 0; j < work_control[0]; j++){
        T_vec_G.push_back(T(G_i[j], G_j[j], G[j]));
    }
    //G_var is received 
    MPI_Recv(G_aux, work_control[0], MPI_DOUBLE, status.MPI_SOURCE, TAG_Gvar, world, &status);
    for(int i = 0; i < work_control[0]; i++) G[i] = G_aux[i];
    for(int j = 0; j < work_control[0]; j++){
        T_vec_Gvar.push_back(T(G_i[j], G_j[j], G[j]));
    }
    delete G_aux;
    G.resize(0);
    G_i.resize(0);
    G_j.resize(0);
    B.resize(work_control[1]);
    B_i.resize(work_control[1]);
    double *B_aux;
    int *B_i_aux;
    //B_i is received
    B_i_aux = new int[work_control[1]];
    MPI_Recv(B_i_aux, work_control[1], MPI_INT, status.MPI_SOURCE, TAG_Bi, world, &status);
    for(int i = 0; i< work_control[1]; i++) B_i[i] = B_i_aux[i];
    delete B_i_aux;
    //B is received
    B_aux = new double[work_control[1]];
    MPI_Recv(B_aux, work_control[1], MPI_DOUBLE, status.MPI_SOURCE, TAG_Bval, world, &status);
    for(int i = 0; i < work_control[1]; i++) B[i] = B_aux[i];
    for(int j = 0; j < work_control[1]; j++){
        T_vec_B.push_back(T(B_i[j], 0, B[j]));
    }
    MPI_Recv(B_aux, work_control[1], MPI_DOUBLE, status.MPI_SOURCE, TAG_Bvar, world, &status);
    for(int i = 0; i < work_control[1]; i++) B[i] = B_aux[i];
    for(int j = 0; j < work_control[1]; j++){
        T_vec_Bvar.push_back(T(B_i[j], 0, B[j]));
    }
    delete B_aux;
}
void PDDSparseGM::Receive_G_B_2(void){
    //It ends worker process solving
    MPI_Recv(work_control, 2, MPI_INT, MPI_ANY_SOURCE, 
    REQUEST_NODE, world, &status);
    work_control[0] = 0;
    work_control[1] = 0;
    MPI_Send(work_control, 2, MPI_INT, status.MPI_SOURCE, REPLY_NODE, world);
    //The server accumulates the values of G and B
    MPI_Recv(work_control, 2, MPI_INT, status.MPI_SOURCE, REQUEST_NODE, world, &status);
    G.resize(work_control[0]);
    G_i.resize(work_control[0]);
    G_j.resize(work_control[0]);
    //G_i is received
    int *G_i_aux, *G_j_aux;
    double *G_aux;
    G_i_aux = new int[work_control[0]];
    MPI_Recv(G_i_aux, work_control[0], MPI_INT, status.MPI_SOURCE, TAG_Gi, world, &status);
    for(int i = 0; i < work_control[0]; i++) G_i[i] = G_i_aux[i];
    delete G_i_aux;
    //G_j is received
    G_j_aux = new int[work_control[0]];
    MPI_Recv(G_j_aux, work_control[0], MPI_INT, status.MPI_SOURCE, TAG_Gj, world, &status);
    for(int i = 0; i < work_control[0]; i++) G_j[i] = G_j_aux[i];
    delete G_j_aux;
    //G is received
    G_aux = new double[work_control[0]];
    MPI_Recv(G_aux, work_control[0], MPI_DOUBLE, status.MPI_SOURCE, TAG_Gval, world, &status);
    for(int i = 0; i < work_control[0]; i++) G[i] = G_aux[i];
    //Triplets vector is fullfilled
    for(int j = 0; j < work_control[0]; j++){
        T_vec_G.push_back(T(G_i[j], G_j[j], G[j]));
    }
    //G_var is received 
    MPI_Recv(G_aux, work_control[0], MPI_DOUBLE, status.MPI_SOURCE, TAG_Gvar, world, &status);
    for(int i = 0; i < work_control[0]; i++) G[i] = G_aux[i];
    for(int j = 0; j < work_control[0]; j++){
        T_vec_Gvar.push_back(T(G_i[j], G_j[j], G[j]));
    }
    //G_CT is received 
     MPI_Recv(G_aux, work_control[0], MPI_DOUBLE, status.MPI_SOURCE, TAG_GCT, world, &status);
    for(int i = 0; i < work_control[0]; i++) G[i] = G_aux[i];
    for(int j = 0; j < work_control[0]; j++){
        T_vec_GCT.push_back(T(G_i[j], G_j[j], G[j]));
    }
    delete G_aux;
    G.resize(0);
    G_i.resize(0);
    G_j.resize(0);
    B.resize(work_control[1]);
    B_i.resize(work_control[1]);
    double *B_aux;
    int *B_i_aux;
    //B_i is received
    B_i_aux = new int[work_control[1]];
    MPI_Recv(B_i_aux, work_control[1], MPI_INT, status.MPI_SOURCE, TAG_Bi, world, &status);
    for(int i = 0; i< work_control[1]; i++) B_i[i] = B_i_aux[i];
    delete B_i_aux;
    //B is received
    B_aux = new double[work_control[1]];
    MPI_Recv(B_aux, work_control[1], MPI_DOUBLE, status.MPI_SOURCE, TAG_Bval, world, &status);
    for(int i = 0; i < work_control[1]; i++) B[i] = B_aux[i];
    for(int j = 0; j < work_control[1]; j++){
        T_vec_B.push_back(T(B_i[j], 0, B[j]));
    }
    //B_var is received
    MPI_Recv(B_aux, work_control[1], MPI_DOUBLE, status.MPI_SOURCE, TAG_Bvar, world, &status);
    for(int i = 0; i < work_control[1]; i++) B[i] = B_aux[i];
    for(int j = 0; j < work_control[1]; j++){
        T_vec_Bvar.push_back(T(B_i[j], 0, B[j]));
    }
    //B_CT is received
    MPI_Recv(B_aux, work_control[1], MPI_DOUBLE, status.MPI_SOURCE, TAG_BCT, world, &status);
    for(int i = 0; i < work_control[1]; i++) B[i] = B_aux[i];
    for(int j = 0; j < work_control[1]; j++){
        T_vec_BCT.push_back(T(B_i[j], 0, B[j]));
    }
    delete B_aux;
}
void PDDSparseGM::Receive_B(void){
    MPI_Recv(work_control, 2, MPI_INT, MPI_ANY_SOURCE, REQUEST_INTERFACE, world, &status);
    work_control[0] = 0;
    work_control[1] = 0;
    MPI_Send(work_control, 2, MPI_INT, status.MPI_SOURCE, REPLY_INTERFACE, world);
    MPI_Recv(work_control, 2, MPI_INT, status.MPI_SOURCE, REQUEST_INTERFACE, world, &status);
    double *B_aux;
    int *B_i_aux;
    //B_i is received
    B_i_aux = new int[work_control[1]];
    MPI_Recv(B_i_aux, work_control[1], MPI_INT, status.MPI_SOURCE, TAG_Bi, world, &status);
    //B is received
    B_aux = new double[work_control[1]];
    MPI_Recv(B_aux, work_control[1], MPI_DOUBLE, status.MPI_SOURCE, TAG_Bval, world, &status);
    for(int j = 0; j < work_control[1]; j++){
        B.push_back(B_aux[j]);
        B_i.push_back(B_i_aux[j]);
    }
    delete B_aux;
    delete B_i_aux;
}
void PDDSparseGM::Receive_Metadata(void){
    double *double_aux;
    double_aux = new double[work_control[1]];
    MPI_Recv(double_aux, work_control[1], MPI_DOUBLE, status.MPI_SOURCE, TAG_APL, world, &status);
    for(int j = 0; j < work_control[1]; j++){
        T_vec_APL.push_back(T(B_i[j], 0, double_aux[j]));
    }
    //MPI_Recv(double_aux, work_control[1], MPI_DOUBLE, status.MPI_SOURCE, TAG_time, world, &status);
    //for(int j = 0; j < work_control[1]; j++){
    //    T_vec_times.push_back(T(B_i[j], 0, double_aux[j]));
    //}
    MPI_Recv(double_aux, work_control[1], MPI_DOUBLE, status.MPI_SOURCE, TAG_Estvar, world, &status);
    for(int j = 0; j < work_control[1]; j++){
        T_vec_Estvar.push_back(T(B_i[j], 0, double_aux[j]));
    }
    MPI_Recv(double_aux, work_control[1], MPI_DOUBLE, status.MPI_SOURCE, TAG_RNGCalls, world, &status);
    for(int j = 0; j < work_control[1]; j++){
        T_vec_RNGCalls.push_back(T(B_i[j], 0, double_aux[j]));
    }
    MPI_Recv(double_aux, work_control[1], MPI_DOUBLE, status.MPI_SOURCE, TAG_xi, world, &status);
    for(int j = 0; j < work_control[1]; j++){
        T_vec_xi.push_back(T(B_i[j], 0, double_aux[j]));
    }
    MPI_Recv(double_aux, work_control[1], MPI_DOUBLE, status.MPI_SOURCE, TAG_phi, world, &status);
    for(int j = 0; j < work_control[1]; j++){
        T_vec_phi.push_back(T(B_i[j], 0, double_aux[j]));
    }
    MPI_Recv(double_aux, work_control[1], MPI_DOUBLE, status.MPI_SOURCE, TAG_xixi, world, &status);
    for(int j = 0; j < work_control[1]; j++){
        T_vec_xixi.push_back(T(B_i[j], 0, double_aux[j]));
    }
    MPI_Recv(double_aux, work_control[1], MPI_DOUBLE, status.MPI_SOURCE, TAG_phiphi, world, &status);
    for(int j = 0; j < work_control[1]; j++){
        T_vec_phiphi.push_back(T(B_i[j], 0, double_aux[j]));
    }
    MPI_Recv(double_aux, work_control[1], MPI_DOUBLE, status.MPI_SOURCE, TAG_phixi, world, &status);
    for(int j = 0; j < work_control[1]; j++){
        T_vec_phixi.push_back(T(B_i[j], 0, double_aux[j]));
    }
    MPI_Recv(double_aux, work_control[1], MPI_DOUBLE, status.MPI_SOURCE, TAG_phip, world, &status);
    for(int j = 0; j < work_control[1]; j++){
        T_vec_phip.push_back(T(B_i[j], 0, double_aux[j]));
    }
    MPI_Recv(double_aux, work_control[1], MPI_DOUBLE, status.MPI_SOURCE, TAG_phiphip, world, &status);
    for(int j = 0; j < work_control[1]; j++){
        T_vec_phiphip.push_back(T(B_i[j], 0, double_aux[j]));
    }
    MPI_Recv(double_aux, work_control[1], MPI_DOUBLE, status.MPI_SOURCE, TAG_phipxi, world, &status);
    for(int j = 0; j < work_control[1]; j++){
        T_vec_phipxi.push_back(T(B_i[j], 0, double_aux[j]));
    }
    MPI_Recv(double_aux, work_control[1], MPI_DOUBLE, status.MPI_SOURCE, TAG_phi_trap, world, &status);
    for(int j = 0; j < work_control[1]; j++){
        T_vec_phi_trap.push_back(T(B_i[j], 0, double_aux[j]));
    }
    MPI_Recv(double_aux, work_control[1], MPI_DOUBLE, status.MPI_SOURCE, TAG_phiphi_trap, world, &status);
    for(int j = 0; j < work_control[1]; j++){
        T_vec_phiphi_trap.push_back(T(B_i[j], 0, double_aux[j]));
    }
    MPI_Recv(double_aux, work_control[1], MPI_DOUBLE, status.MPI_SOURCE, TAG_phi_trap_VR, world, &status);
    for(int j = 0; j < work_control[1]; j++){
        T_vec_phi_trap_VR.push_back(T(B_i[j], 0, double_aux[j]));
    }
    MPI_Recv(double_aux, work_control[1], MPI_DOUBLE, status.MPI_SOURCE, TAG_phiphi_trap_VR, world, &status);
    for(int j = 0; j < work_control[1]; j++){
        T_vec_phiphi_trap_VR.push_back(T(B_i[j], 0, double_aux[j]));
    }
    MPI_Recv(double_aux, work_control[1], MPI_DOUBLE, status.MPI_SOURCE, TAG_bias_num, world, &status);
    for(int j = 0; j < work_control[1]; j++){
        T_vec_bias_num.push_back(T(B_i[j], 0, double_aux[j]));
    }
    MPI_Recv(double_aux, work_control[1], MPI_DOUBLE, status.MPI_SOURCE, TAG_bias_trap, world, &status);
    for(int j = 0; j < work_control[1]; j++){
        T_vec_bias_trap.push_back(T(B_i[j], 0, double_aux[j]));
    }
    MPI_Recv(double_aux, work_control[1], MPI_DOUBLE, status.MPI_SOURCE, TAG_Estvar_VR, world, &status);
    for(int j = 0; j < work_control[1]; j++){
        T_vec_Estvar_VR.push_back(T(B_i[j], 0, double_aux[j]));
        //std::cout << double_aux[j] << std::endl;
    }
    delete double_aux;
    B_i.resize(0);
}
void PDDSparseGM::Process_Metadata(void){
    Eigen::SparseMatrix<double> APL_sparse, times_sparse, Estvar_sparse, RNGCalls_sparse
    , phi_sparse, xi_sparse, xixi_sparse, phiphi_sparse, phixi_sparse, phip_sparse, phiphip_sparse,
    phipxi_sparse, PCoeff_sparse, phi_trap_sparse, phiphi_trap_sparse, phi_trap_VR_sparse, 
    phiphi_trap_VR_sparse, bias_num_sparse, bias_trap_sparse, Estvar_VR_sparse;
    unsigned int size = (unsigned int) nNodes;
    APL_sparse.resize(size, 1); times_sparse.resize(size, 1); Estvar_sparse.resize(size,1);
    PCoeff_sparse.resize(size,1); RNGCalls_sparse.resize(size,1);
    APL_sparse.setFromTriplets(T_vec_APL.begin(), T_vec_APL.end());
    times_sparse.setFromTriplets(T_vec_times.begin(), T_vec_times.end());
    RNGCalls_sparse.setFromTriplets(T_vec_RNGCalls.begin(),T_vec_RNGCalls.end());
    Estvar_sparse.setFromTriplets(T_vec_Estvar.begin(), T_vec_Estvar.end());
    T_vec_APL.clear(); T_vec_times.clear(); T_vec_RNGCalls.clear(); T_vec_Estvar.clear();
    phi_sparse.resize(size,1); xi_sparse.resize(size,1); xixi_sparse.resize(size,1);
    phiphi_sparse.resize(size,1); phixi_sparse.resize(size,1); phip_sparse.resize(size,1);
    phiphip_sparse.resize(size,1); phipxi_sparse.resize(size,1); bias_num_sparse.resize(size,1);
    bias_trap_sparse.resize(size,1); Estvar_VR_sparse.resize(size,1);
    phi_sparse.setFromTriplets(T_vec_phi.begin(),T_vec_phi.end());
    xi_sparse.setFromTriplets(T_vec_xi.begin(),T_vec_xi.end());
    xixi_sparse.setFromTriplets(T_vec_xixi.begin(),T_vec_xixi.end());
    phiphi_sparse.setFromTriplets(T_vec_phiphi.begin(),T_vec_phiphi.end());
    phixi_sparse.setFromTriplets(T_vec_phixi.begin(),T_vec_phixi.end());
    phip_sparse.setFromTriplets(T_vec_phip.begin(),T_vec_phip.end());
    phiphip_sparse.setFromTriplets(T_vec_phiphip.begin(),T_vec_phiphip.end());
    phipxi_sparse.setFromTriplets(T_vec_phipxi.begin(),T_vec_phipxi.end());
    bias_num_sparse.setFromTriplets(T_vec_bias_num.begin(),T_vec_bias_num.end());
    bias_trap_sparse.setFromTriplets(T_vec_bias_trap.begin(),T_vec_bias_trap.end());
    T_vec_phi.clear(); T_vec_xi.clear(); T_vec_xixi.clear();T_vec_phiphi.clear();
    T_vec_bias_num.clear(); T_vec_bias_trap.clear();
    T_vec_phixi.clear(); T_vec_phip.clear();T_vec_phiphip.clear(); 
    T_vec_phipxi.clear();
    phi_trap_sparse.resize(size,1); phiphi_trap_sparse.resize(size,1); phi_trap_VR_sparse.resize(size,1);
    phiphi_trap_VR_sparse.resize(size,1);
    phi_trap_sparse.setFromTriplets(T_vec_phi_trap.begin(),T_vec_phi_trap.end());
    phiphi_trap_sparse.setFromTriplets(T_vec_phiphi_trap.begin(),T_vec_phiphi_trap.end()); 
    phi_trap_VR_sparse.setFromTriplets(T_vec_phi_trap_VR.begin(),T_vec_phi_trap_VR.end());
    phiphi_trap_VR_sparse.setFromTriplets(T_vec_phiphi_trap_VR.begin(),T_vec_phiphi_trap_VR.end());
    Estvar_VR_sparse.setFromTriplets(T_vec_Estvar_VR.begin(),T_vec_Estvar_VR.end());
    T_vec_phi_trap.clear(); T_vec_phiphi_trap.clear(); T_vec_phi_trap_VR.clear();
    T_vec_phiphi_trap_VR.clear(); T_vec_Estvar_VR.clear();
    FILE *pf;
    pf = fopen("Output/Debug/knot_metadata.csv","w");
    fprintf(pf,"index,APL,WCtime,RNGCalls,Estimated_Variance,Estimated_Variance_VR,PCoeff_old,PCoeff_new,Var_Plain,Var_VR,bias_num,bias_trap\n");
    Est_var.clear();
    PCoeff_o.clear();
    PCoeff_n.clear();
    bias_num.clear();
    for(unsigned int i = 0; i < nNodes; i++){
        Est_var.push_back(Estvar_sparse.coeff(i,0));
        PCoeff_o.push_back((phixi_sparse.coeff(i,0)-phi_sparse.coeff(i,0)*xi_sparse.coeff(i,0))/
        (sqrt(phiphi_sparse.coeff(i,0)-pow(phi_sparse.coeff(i,0),2))*sqrt(xixi_sparse.coeff(i,0)-pow(xi_sparse.coeff(i,0),2))));
        PCoeff_n.push_back((phipxi_sparse.coeff(i,0)-phip_sparse.coeff(i,0)*xi_sparse.coeff(i,0))/
        (sqrt(phiphip_sparse.coeff(i,0)-pow(phip_sparse.coeff(i,0),2))*sqrt(xixi_sparse.coeff(i,0)-pow(xi_sparse.coeff(i,0),2))));
        var_phi_trap.push_back(phiphi_trap_sparse.coeff(i,0)-pow(phi_trap_sparse.coeff(i,0),2));
        var_phi_trap_VR.push_back(phiphi_trap_VR_sparse.coeff(i,0)-pow(phi_trap_VR_sparse.coeff(i,0),2));
        bias_num.push_back(bias_num_sparse.coeff(i,0));
        fprintf(pf,"%u,%e,%e,%e,%e,%e,%e,%e,%e,%e,%e,%e\n",i
        ,APL_sparse.coeff(i,0),times_sparse.coeff(i,0),RNGCalls_sparse.coeff(i,0),
        Estvar_sparse.coeff(i,0), Estvar_VR_sparse.coeff(i,0), PCoeff_o[i],PCoeff_n[i],var_phi_trap[i],var_phi_trap_VR[i],
        bias_num_sparse.coeff(i,0), bias_trap_sparse.coeff(i,0));
        
    }
    fclose(pf);
}
Eigen::Vector2d PDDSparseGM::Node_Position(int node_index){
    Eigen::Vector2d output;
    output.resize(2);
    for(std::vector<Interface>::iterator it = interfaces.begin();
    it != interfaces.end();
    it++){
        if((*it).In(node_index)){
            output = (*it).Node_Position(node_index);
        }
    }
    return output;
}
void PDDSparseGM::Compute_Solution(bvp BoundValProb){
    Eigen::SparseMatrix<double> G_sparse, I_sparse, B_sparse;
    unsigned int size = (unsigned int) nNodes;
    FILE *ofile;
    G_sparse.resize(size, size);
    G_sparse.setFromTriplets(T_vec_Gvar.begin(),T_vec_Gvar.end());
    T_vec_Gvar.resize(0);
    
    ofile = fopen("Output/Debug/G_var.csv","w");
    fprintf(ofile, "Gvar_i,Gvar_j,Gvar_ij\n");
    for (int k=0; k<G_sparse.outerSize(); ++k)
        for (Eigen::SparseMatrix<double>::InnerIterator it(G_sparse,k); it; ++it)
        {
            fprintf(ofile,"%ld,%ld,%e\n",it.row(),it.col(),it.value());
        }
    
    G_sparse.resize(size, size);
    G_sparse.setFromTriplets(T_vec_G.begin(),T_vec_G.end());
    T_vec_G.resize(0);
    I_sparse.resize(size, size);
    I_sparse.setIdentity();
    G_sparse+=I_sparse;
    I_sparse.resize(0,0);
    ofile = fopen("Output/Debug/G.csv","w");
    fprintf(ofile, "G_i,G_j,G_ij\n");
    for (int k=0; k<G_sparse.outerSize(); ++k)
        for (Eigen::SparseMatrix<double>::InnerIterator it(G_sparse,k); it; ++it)
        {
            fprintf(ofile,"%ld,%ld,%e\n",it.row(),it.col(),it.value());
        }
    fclose(ofile);
    B_sparse.resize(size, 1);
    B_sparse.setFromTriplets(T_vec_Bvar.begin(), T_vec_Bvar.end());
    T_vec_Bvar.resize(0);
    ofile = fopen("Output/Debug/B_var.csv","w");
    fprintf(ofile, "B_i,B_j,B_ij\n");
    for (int k=0; k<B_sparse.outerSize(); ++k)
        for (Eigen::SparseMatrix<double>::InnerIterator it(B_sparse,k); it; ++it)
        {
            fprintf(ofile,"%ld,%ld,%e\n",it.row(),it.col(),it.value());
        }
    fclose(ofile);
    B_sparse.resize(size, 1);
    B_sparse.setFromTriplets(T_vec_B.begin(), T_vec_B.end());
    T_vec_B.resize(0);
    ofile = fopen("Output/Debug/B.csv","w");
    fprintf(ofile, "B_i,B_j,B_ij\n");
    for (int k=0; k<B_sparse.outerSize(); ++k)
        for (Eigen::SparseMatrix<double>::InnerIterator it(B_sparse,k); it; ++it)
        {
            fprintf(ofile,"%ld,%ld,%e\n",it.row(),it.col(),it.value());
        }
    fclose(ofile);
    Eigen::VectorXd ud,Bd,guess;
    Bd = Eigen::VectorXd(B_sparse);
    guess = Bd*0;
    B_sparse.resize(0,0);
    Update_TimeFile("Building Up G and B",1);
    //Eigen::SparseLU<Eigen::SparseMatrix<double> > solver_LU;
    //solver_LU.compute(G_sparse);
    //solver_LU.factorize(G_sparse);
    //ud = solver_LU.solve(Bd);
    Eigen::BiCGSTAB<Eigen::SparseMatrix<double> > solver_IT;
    solver_IT.compute(G_sparse);
    ud = solver_IT.solveWithGuess(Bd,guess);
    Update_TimeFile("Solving system",1);
    ofile = fopen("Output/solution.csv","w");
    fprintf(ofile,"Knot_index,x,y,sol_analytic,sol_PDDS,err,rerr\n");
    double sol, err, rerr;
    for(int i = 0; i< (int)ud.size(); i++){
        sol = BoundValProb.u(Node_Position(i),T_start);
        err = ud(i) - sol;
        rerr = fabs(err)/sol;
        fprintf(ofile,"%d,%e,%e,%e,%e,%e,%e\n",i,Node_Position(i)[0],Node_Position(i)[1],
        sol,ud(i),err,rerr);
    }
    fclose(ofile);
    Update_TimeFile("Writing Solution",1);
}
/*void PDDSparseGM::Compute_Solution_2(bvp BoundValProb){
    Eigen::SparseMatrix<double> G_sparse, I_sparse, B_sparse,C_sparse;
    unsigned int size = (unsigned int) nNodes;
    double epsilon_C;
    FILE *ofile;
    G_sparse.resize(size, size);
    G_sparse.setFromTriplets(T_vec_Gvar.begin(),T_vec_Gvar.end());
    T_vec_Gvar.resize(0);
    G_sparse_var = G_sparse;
    ofile = fopen("Output/Debug/G_var.csv","w");
    fprintf(ofile, "Gvar_i,Gvar_j,Gvar_ij\n");
    for (int k=0; k<G_sparse.outerSize(); ++k)
        for (Eigen::SparseMatrix<double>::InnerIterator it(G_sparse,k); it; ++it)
        {
            fprintf(ofile,"%ld,%ld,%e\n",it.row(),it.col(),it.value());
        }
    for (int i = 0; i<nNodes; i++)
        {
            fprintf(ofile,"%d,%d,%e\n",i,i,0.0);
        }
    fclose(ofile);
    G_sparse.resize(size, size);
    G_sparse_InvApprox.resize(size,size);
    G_sparse.setFromTriplets(T_vec_G.begin(),T_vec_G.end());
    T_vec_G.resize(0);
    I_sparse.resize(size, size);
    I_sparse.setIdentity();
    G_sparse+=I_sparse;
    G_sparse_GM.resize(nNodes,nNodes);
    G_sparse_GM = G_sparse;
    ofile = fopen("Output/Debug/G.csv","w");
    fprintf(ofile, "G_i,G_j,G_ij\n");
    epsilon_C = 0.0;
    for (int k=0; k<G_sparse.outerSize(); ++k){
        for (Eigen::SparseMatrix<double>::InnerIterator it(G_sparse,k); it; ++it)
        {
            fprintf(ofile,"%ld,%ld,%e\n",it.row(),it.col(),it.value());
        }
    }
    fclose(ofile);
    Update_TimeFile("Building up G and var_G GM",1);
    C_sparse.resize(nNodes,nNodes);
    C_sparse += G_sparse.cwiseAbs();
    eps = std::max((C_sparse*Eigen::Vector2d::Ones(nNodes)).maxCoeff()-2,0.0);
    std::cout << "Eps is equal to " << eps << std::endl;
    //Compute approx G considering it an M Matrix
    G_sparse_InvApprox = I_sparse + (1.0/(2.0+eps))*C_sparse;
    for(int i = 0; i < 10; i++){
        G_sparse_InvApprox = I_sparse + (1.0/(2.0+eps))*C_sparse*G_sparse_InvApprox;
    }
    G_sparse_InvApprox = (1.0/(2.0+eps))*G_sparse_InvApprox;
    C_sparse.resize(0,0);
    std::cout << "Error commited by M-Matrix inverse approximation " << 
    ((I_sparse - G_sparse_GM*G_sparse_InvApprox).cwiseAbs()*Eigen::Vector2d::Ones(nNodes)).maxCoeff() << std::endl;
    std::cout << "Approx skeel condition number" << 
    ((G_sparse_GM.cwiseAbs()*G_sparse_InvApprox.cwiseAbs())*Eigen::Vector2d::Ones(nNodes)).maxCoeff() << std::endl;
    std::cout << "Approx Inf condition number " << 
    (G_sparse_GM.cwiseAbs()*Eigen::Vector2d::Ones(nNodes)).maxCoeff()*(G_sparse_InvApprox.cwiseAbs()*Eigen::Vector2d::Ones(nNodes)).maxCoeff() << std::endl;
    Update_TimeFile("Computing Inv G approximation",1);
    B_sparse.resize(size, 1);
    B_sparse.setFromTriplets(T_vec_Bvar.begin(), T_vec_Bvar.end());
    T_vec_Bvar.resize(0);
    B_var.resize(nNodes);
    for (int i = 0; i<(int)B_i.size(); i++) B_sparse.coeffRef(B_i[i],0) = 0.0;
    ofile = fopen("Output/Debug/B_var.csv","w");
    fprintf(ofile, "B_i,B_j,B_ij\n");
    for (int k=0; k<B_sparse.outerSize(); ++k)
        for (Eigen::SparseMatrix<double>::InnerIterator it(B_sparse,k); it; ++it)
        {
            fprintf(ofile,"%ld,%ld,%e\n",it.row(),it.col(),it.value());
            B_var[it.row()] = it.value();
        }
    fclose(ofile);
    B_sparse.resize(size, 1);
    B_sparse.setFromTriplets(T_vec_B.begin(), T_vec_B.end());
    T_vec_B.resize(0);
    for (int i = 0; i<(int)B_i.size(); i++) B_sparse.coeffRef(B_i[i],0) = B[i];
    ofile = fopen("Output/Debug/B.csv","w");
    fprintf(ofile, "B_i,B_j,B_ij\n");
    for (int k=0; k<B_sparse.outerSize(); ++k)
        for (Eigen::SparseMatrix<double>::InnerIterator it(B_sparse,k); it; ++it)
        {
            fprintf(ofile,"%ld,%ld,%e\n",it.row(),it.col(),it.value());
        }
    fclose(ofile);
    Update_TimeFile("Building up B and var_B GM",1);
    Eigen::Vector2d ud,ud_CT,Bd,guess;
    Bd = Eigen::Vector2d(B_sparse);
    guess = Bd*0;
    B_sparse.resize(0,0);
    Eigen::SparseLU<Eigen::SparseMatrix<double> > solver_LU;
    solver_LU.compute(G_sparse);
    solver_LU.factorize(G_sparse);
    ud = solver_LU.solve(Bd);
    Eigen::BiCGSTAB<Eigen::SparseMatrix<double> > solver_IT;
    solver_IT.compute(G_sparse_InvApprox*G_sparse);
    ud = solver_IT.solveWithGuess(G_sparse_InvApprox*Bd,guess);
    std::cout << "BiCGSTAB iterations with preconditioner " << solver_IT.iterations() << std::endl;
    Update_TimeFile("Solving system with preconditioner",1);
    solver_IT.compute(G_sparse);
    ud = solver_IT.solveWithGuess(Bd,guess);
    std::cout << "BiCGSTAB  iterations without preconditioner " << solver_IT.iterations() << std::endl;
    Update_TimeFile("Solving system without preconditioner",1);
    knot_solutions = ud;
    const gsl_rng_type * T;
    gsl_rng * rng;
    T = gsl_rng_default; //Mt1997
    rng = gsl_rng_alloc(T);
    gsl_rng_set(rng, 17);
    ofile = fopen("Output/Debug/G_noisy.csv","w");
    fprintf(ofile, "G_i,G_j,G_ij\n");
    for (int k=0; k<G_sparse.outerSize(); ++k){
        for (Eigen::SparseMatrix<double>::InnerIterator it(G_sparse,k); it; ++it)
        {
            G_sparse.coeffRef(it.row(),it.col()) +=  gsl_ran_gaussian_ziggurat(rng,sqrt(G_sparse_var.coeff(it.row(),it.col())*(1.0/N)));
            fprintf(ofile,"%ld,%ld,%e\n",it.row(),it.col(),it.value());
        }
    }
    fclose(ofile);
    ofile = fopen("Output/Debug/B_noisy.csv","w");
    fprintf(ofile, "B_i,B_j,B_ij\n");
    for(int k = 0; k < Bd.size(); k++){
        Bd[k] +=  gsl_ran_gaussian_ziggurat(rng,1*sqrt(B_var[k]*(1.0/N)));
        fprintf(ofile,"%d,%d,%e\n",k,0,Bd[k]);
    }
    fclose(ofile);
    Update_TimeFile("Defining noisy system",1);
    solver_IT.compute(G_sparse);
    Eigen::Vector2d ud_noisy(nNodes);
    ud_noisy = solver_IT.solveWithGuess(Bd,ud);
    Update_TimeFile("Solving noisy system",1);
    G_sparse.resize(size, size);
    G_sparse.setFromTriplets(T_vec_GCT.begin(),T_vec_GCT.end());
    G_sparse+=I_sparse;
    I_sparse.resize(0,0);
    ofile = fopen("Output/Debug/G_CT.csv","w");
    fprintf(ofile, "G_i,G_j,G_ij\n");
    for (int k=0; k<G_sparse.outerSize(); ++k)
        for (Eigen::SparseMatrix<double>::InnerIterator it(G_sparse,k); it; ++it)
        {
            fprintf(ofile,"%ld,%ld,%e\n",it.row(),it.col(),it.value());
        }
    fclose(ofile);
    T_vec_GCT.resize(0);
    Update_TimeFile("Building up G EM",1);
    B_sparse.resize(size, 1);
    B_sparse.setFromTriplets(T_vec_BCT.begin(), T_vec_BCT.end());
    T_vec_BCT.resize(0);
    for (int i = 0; i<(int)B_i.size(); i++) B_sparse.coeffRef(B_i[i],0) = B[i];
    ofile = fopen("Output/Debug/B_CT.csv","w");
    fprintf(ofile, "B_i,B_j,B_ij\n");
    for (int k=0; k<B_sparse.outerSize(); ++k)
        for (Eigen::SparseMatrix<double>::InnerIterator it(B_sparse,k); it; ++it)
        {
            fprintf(ofile,"%ld,%ld,%e\n",it.row(),it.col(),it.value());
        }
    fclose(ofile);
    Update_TimeFile("Building up B EM",1);
    solver_IT.compute(G_sparse);
    Bd = Eigen::Vector2d(B_sparse);
    ud_CT = solver_IT.solveWithGuess(Bd,ud);
    Update_TimeFile("Solving EM system (u_GB as initial guess)",1);
    bias = (ud-ud_CT).cwiseAbs();
    ofile = fopen("Output/solution.csv","w");
    fprintf(ofile,"Knot_index,x,y,sol_analytic,sol_PDDS_GM,sol_PDDS_EM,err,rerr,sbias\n");
    double sol, err, rerr;
    for(int i = 0; i< (int)ud.size(); i++){
        sol = bvp.u.Value(Node_Position(i),T_start);
        err = ud(i) - sol;
        rerr = fabs(err)/sol;
        fprintf(ofile,"%d,%e,%e,%e,%e,%e,%e,%e,%e\n",i,Node_Position(i)[0],Node_Position(i)[1],
        sol,ud(i),ud_CT(i),err,rerr,bias(i));
    }
    fclose(ofile);
    Update_TimeFile("Writing results",1);
}*/
void PDDSparseGM::Print_Interface(void){
    if(myid==server){
    int cent = 0;
    for(std::vector<Interface>::iterator it = interfaces.begin();
    it != interfaces.end();
    it++){
        (*it).Print(cent);
        cent ++;
    }
    }
}
void PDDSparseGM::Print_Problem(void){
    FILE *dfile;
    dfile = fopen(debug_fname,"w");
    fprintf(dfile,"**********PDDSparse Solver*************\n");
    fprintf(dfile,"***************Ver 1.0******************\n");
    fprintf(dfile,"***********Jorge Moron Vidal************\n");
    fprintf(dfile,"**********jmoron@math.uc3m.es***********\n");
    fprintf(dfile,"h0 = %e \t T_start = %0.2f\n",h0, T_start);
    fprintf(dfile,"Number of trayectories per knot %d\n",N);
    fprintf(dfile,"Number of trayectories per job %d\n", N_job);
    fprintf(dfile,"Number of subdomains/interfaces: %d x %d \n", iN[0], iN[1]);
    fprintf(dfile,"Number of knots per interface: %d\n",nN[0]);
    fprintf(dfile,"Total number of knots: %d\n",nNodes);
    fprintf(dfile,"Position of South-West corner of the domain: (%.2f,%.2f)\n",SW(0), SW(1));
    fprintf(dfile,"Position of North-East corner of the domain: (%.2f,%.2f)\n",NE(0), NE(1));
    fclose(dfile);
}
void PDDSparseGM::Read_Solution(void){
   std::ifstream solution_file("Output/solution.csv");
   std::string line, aux_string;
   std::vector<double> csv_row;
   uint16_t aux_index;
   //bias.resize(nNodes);
   if(solution_file.is_open()){
       //std::cout << "Reading solution file" << std::endl;
       csv_row.clear();
       getline(solution_file, line);
       csv_row.resize(std::count(line.begin(),line.end(),',') +1);
       aux_string.clear(); 
       while(getline(solution_file, line)){
           aux_index = 0;
           for(std::string::iterator character = line.begin();
           character != line.end();
           character ++){
               if (*character == ','){
                   csv_row[aux_index] = atof(aux_string.c_str());
                   aux_string.clear();
                   aux_index ++;
               } else {
                   aux_string += *character;
               }
           }
           csv_row[aux_index] = atof(aux_string.c_str());
           aux_string.clear();
           for(std::vector<Interface>::iterator interface = interfaces.begin();
               interface != interfaces.end();
               interface ++){
                if( (*interface).In((int) csv_row[0])){
                    for(uint32_t interface_index = 0;
                        interface_index < (*interface).index.size();
                        interface_index ++){
                        if((*interface).index[interface_index] == (int) csv_row[0]){
                            (*interface).solution[interface_index] = csv_row[4];
                            (*interface).solution_noisy[interface_index] = csv_row[5];
                            //bias[(int) csv_row[0]] = csv_row[8];
                            /* printf("Node %d is in interface [%d %d] with solution %f \n",
                            (int) csv_row[0],(*interface).label[0],(*interface).label[1],
                            csv_row[4]);*/
                        }
                    }
                }
            }
        }

   } else {
       std::cout << "Output/solution.csv couldn't be opened" << std::endl;
   }
   solution_file.close();
}
void PDDSparseGM::Random_Solution(double constant){
    FILE *ofile;
    ofile = fopen("Output/solution.csv","w");
    fprintf(ofile,"Knot_index,x,y,sol_analytic,sol_PDDS,err,rerr\n");
    for(std::vector<Interface>::iterator interface = interfaces.begin();
    interface != interfaces.end();
    interface ++){
        for(uint32_t interface_index = 0;
            interface_index < (*interface).index.size();
            interface_index ++){
            (*interface).solution[interface_index] = (constant*std::rand())/RAND_MAX;
            fprintf(ofile,"%d,%e,%e,%e,0.0,0.0,0.0\n",(*interface).index[interface_index],
            Node_Position((*interface).index[interface_index])[0],
            Node_Position((*interface).index[interface_index])[1],(*interface).solution[interface_index]);
        }
    }
    fclose(ofile);
}
void PDDSparseGM::Fullfill_subdomains(bvp BoundValProb){
    subdomains.clear();
    Subdomain subdomain_aux;
    std::vector<int> label_aux;
    std::map<direction, Interface> interfaces_map_aux;
    label_aux.resize(2);
    for(int subd_y = 0; subd_y < iN[1]; subd_y ++){
        for(int subd_x = 0; subd_x < iN[0]; subd_x++){
            label_aux[0] = subd_x;
            interfaces_map_aux.clear();
            label_aux[1] = subd_y*2;
            if(interface_map.count(label_aux) > 0 ){
                interfaces_map_aux[East] = interfaces[interface_map[label_aux]];
            }
            label_aux[0] = subd_x-1;
            label_aux[1] = subd_y*2;
            if(interface_map.count(label_aux) > 0 ){
                interfaces_map_aux[West] = interfaces[interface_map[label_aux]];
            }
            label_aux[0] = subd_x;
            label_aux[1] = subd_y*2 +1;
            if(interface_map.count(label_aux) > 0 ){
                interfaces_map_aux[North] = interfaces[interface_map[label_aux]];
            }
            label_aux[0] = subd_x;
            label_aux[1] = subd_y*2 -1;
            if(interface_map.count(label_aux) > 0 ){
                interfaces_map_aux[South] = interfaces[interface_map[label_aux]];
            }
            label_aux[0] = subd_x;
            label_aux[1] = subd_y;
            subdomain_aux.Init(label_aux, interfaces_map_aux, parameters, BoundValProb);
            subdomains.push_back(subdomain_aux);
        }
    }
}
void PDDSparseGM::Solve_Subdomains(bvp BoundValProb){
    if(myid == server){
        Read_Solution();
        Fullfill_subdomains(BoundValProb);
        Update_TimeFile("Fullfilling subdomains",1);
        for(std::vector<Subdomain>::iterator it_subdomain = subdomains.begin();
            it_subdomain != subdomains.end();
            it_subdomain ++){
            MPI_Recv(work_control, 2, MPI_INT, MPI_ANY_SOURCE, ASK_SERVER, world, &status);
            work_control[0] = SEND_WORK;
            MPI_Send(work_control, 2, MPI_INT, status.MPI_SOURCE, REPLY_WORKER, world);
            (*it_subdomain).Send_To_Worker(status,world);
        }
        for(int i = 0; i<server; i++){
            MPI_Recv(work_control, 2, MPI_INT, MPI_ANY_SOURCE, ASK_SERVER, world, &status);
            work_control[0] = END_WORKER;
            MPI_Send(work_control, 2, MPI_INT, status.MPI_SOURCE, REPLY_WORKER, world);
        }
        Update_TimeFile("Solving subdomains",server+1);
    } else {
        Subdomain subdomain;
        do{
            MPI_Send(work_control, 2, MPI_INT, server, ASK_SERVER, world);
            MPI_Recv(work_control, 2, MPI_INT, server, REPLY_WORKER, world, &status);
            if(work_control[0] == SEND_WORK){
                subdomain.Recieve_From_Server(server, world);
                subdomain.Solve(world);
            }
        }while(work_control[0] == SEND_WORK);
    }
    //MPI_Finalize();
}
void PDDSparseGM::Solve_Subdomains_LinIt_First(bvp BoundValProb){
    if(myid == server){
        Read_Solution();
        Fullfill_subdomains(BoundValProb);
        Update_TimeFile("Fullfilling subdomains",1);
        for(std::vector<Subdomain>::iterator it_subdomain = subdomains.begin();
            it_subdomain != subdomains.end();
            it_subdomain ++){
            MPI_Recv(work_control, 2, MPI_INT, MPI_ANY_SOURCE, ASK_SERVER, world, &status);
            work_control[0] = SEND_WORK;
            MPI_Send(work_control, 2, MPI_INT, status.MPI_SOURCE, REPLY_WORKER, world);
            (*it_subdomain).Send_To_Worker(status,world);
        }
        for(int i = 0; i<server; i++){
            MPI_Recv(work_control, 2, MPI_INT, MPI_ANY_SOURCE, ASK_SERVER, world, &status);
            work_control[0] = END_WORKER;
            MPI_Send(work_control, 2, MPI_INT, status.MPI_SOURCE, REPLY_WORKER, world);
        }
        Update_TimeFile("Solving subdomains",server+1);
    } else {
        Subdomain subdomain;
        do{
            MPI_Send(work_control, 2, MPI_INT, server, ASK_SERVER, world);
            MPI_Recv(work_control, 2, MPI_INT, server, REPLY_WORKER, world, &status);
            if(work_control[0] == SEND_WORK){
                subdomain.Recieve_From_Server(server, world);
                subdomain.Solve_LinIt_First(world);
            }
        }while(work_control[0] == SEND_WORK);
    }
    //MPI_Finalize();
}
void PDDSparseGM::Solve_Subdomains_LinIt(bvp BoundValProb){
    if(myid == server){
        Read_Solution();
        Fullfill_subdomains(BoundValProb);
        Update_TimeFile("Fullfilling subdomains",1);
        for(std::vector<Subdomain>::iterator it_subdomain = subdomains.begin();
            it_subdomain != subdomains.end();
            it_subdomain ++){
            MPI_Recv(work_control, 2, MPI_INT, MPI_ANY_SOURCE, ASK_SERVER, world, &status);
            work_control[0] = SEND_WORK;
            MPI_Send(work_control, 2, MPI_INT, status.MPI_SOURCE, REPLY_WORKER, world);
            (*it_subdomain).Send_To_Worker(status,world);
        }
        for(int i = 0; i<server; i++){
            MPI_Recv(work_control, 2, MPI_INT, MPI_ANY_SOURCE, ASK_SERVER, world, &status);
            work_control[0] = END_WORKER;
            MPI_Send(work_control, 2, MPI_INT, status.MPI_SOURCE, REPLY_WORKER, world);
        }
        Update_TimeFile("Solving subdomains",server+1);
    } else {
        Subdomain subdomain;
        do{
            MPI_Send(work_control, 2, MPI_INT, server, ASK_SERVER, world);
            MPI_Recv(work_control, 2, MPI_INT, server, REPLY_WORKER, world, &status);
            if(work_control[0] == SEND_WORK){
                subdomain.Recieve_From_Server(server, world);
                subdomain.Solve_LinIt(world);
            }
        }while(work_control[0] == SEND_WORK);
    }
    //MPI_Finalize();
}
void PDDSparseGM::Solve_Subdomains_SemiLin(int iteration, bvp BoundValProb){
    if(myid == server){
        Read_Solution();
        Fullfill_subdomains(BoundValProb);
        Update_TimeFile("FullfillinFullfill_subg subdomains",1);
        for(std::vector<Subdomain>::iterator it_subdomain = subdomains.begin();
            it_subdomain != subdomains.end();
            it_subdomain ++){
            MPI_Recv(work_control, 2, MPI_INT, MPI_ANY_SOURCE, ASK_SERVER, world, &status);
            work_control[0] = SEND_WORK;
            MPI_Send(work_control, 2, MPI_INT, status.MPI_SOURCE, REPLY_WORKER, world);
            (*it_subdomain).Send_To_Worker(status,world);
        }
        for(int i = 0; i<server; i++){
            MPI_Recv(work_control, 2, MPI_INT, MPI_ANY_SOURCE, ASK_SERVER, world, &status);
            work_control[0] = END_WORKER;
            MPI_Send(work_control, 2, MPI_INT, status.MPI_SOURCE, REPLY_WORKER, world);
        }
        Update_TimeFile("Solving subdomains",server+1);
    } else {
        Subdomain subdomain;
        do{
            MPI_Send(work_control, 2, MPI_INT, server, ASK_SERVER, world);
            MPI_Recv(work_control, 2, MPI_INT, server, REPLY_WORKER, world, &status);
            if(work_control[0] == SEND_WORK){
                subdomain.Recieve_From_Server(server, world);
                subdomain.Solve_NL(world);
            }
        }while(work_control[0] == SEND_WORK);
    }

}
void PDDSparseGM::Fullfill_Subdomains_Random(bvp BoundValProb,double constant){
    if(myid == server){
        Random_Solution(constant);
        Fullfill_subdomains(BoundValProb);
        for(std::vector<Subdomain>::iterator it_subdomain = subdomains.begin();
            it_subdomain != subdomains.end();
            it_subdomain ++){
            MPI_Recv(work_control, 2, MPI_INT, MPI_ANY_SOURCE, ASK_SERVER, world, &status);
            work_control[0] = SEND_WORK;
            MPI_Send(work_control, 2, MPI_INT, status.MPI_SOURCE, REPLY_WORKER, world);
            (*it_subdomain).Send_To_Worker(status,world);
        }
        for(int i = 0; i<server; i++){
            MPI_Recv(work_control, 2, MPI_INT, MPI_ANY_SOURCE, ASK_SERVER, world, &status);
            work_control[0] = END_WORKER;
            MPI_Send(work_control, 2, MPI_INT, status.MPI_SOURCE, REPLY_WORKER, world);
        }
        
        Update_TimeFile("Fullfilling subdomains Random",1);
    } else {
        Subdomain subdomain;
        do{
            MPI_Send(work_control, 2, MPI_INT, server, ASK_SERVER, world);
            MPI_Recv(work_control, 2, MPI_INT, server, REPLY_WORKER, world, &status);
            if(work_control[0] == SEND_WORK){
                subdomain.Recieve_From_Server(server, world);
                subdomain.Fullfill_Random(world,constant);
            }
        }while(work_control[0] == SEND_WORK);
    }
    //MPI_Finalize();
}
/*void PDDSparseGM::Solve_NumVR(bvp BoundValProb, std::vector<double> h_vec, std::vector<int> N_vec){
    bool done=true;
    //double start = MPI_Wtime();
    //FILE *pFile;
    //if(myid==server) Print_Problem();
    //Compute the PDDSparse Matrix
    if(myid==server){
        FILE *pFile;
        pFile = fopen("Output/Debug/times.txt","a");
        fprintf(pFile,"************SECOND ROUND***********\n");
        fclose(pFile); 
        PDDSJob auxjob;
        //Job list is fullfilled
        for(int knot_index = 0; knot_index < nNodes; knot_index++){
            auxjob.index[0] = knot_index;
            //std::cout << knot_index;
            //auxjob.N[0] = N_vec[knot_index]; auxjob.N[1] = 10000;
            auxjob.N[0] = N_vec[knot_index]; auxjob.N[1] = N_job; auxjob.h[0] = h_vec[knot_index];
            for(int jobs_per_knot = 0; jobs_per_knot < N_vec[knot_index]/N_job; jobs_per_knot ++){
                work_control[1] = 1;
                Send_Node_Loop(auxjob);
            }
        }
        Update_TimeFile("MC Simulations",server+1);
        //G and B are received from the workers
        for(int process = 0; process < server; process++){
             Receive_G_B_2();
             Receive_Metadata();
        }
        Update_TimeFile("Receiving G and B",server+1);
        Process_Metadata();
        B.clear();
        B_i.clear();
        //Compute_B_Deterministic();
        Compute_Solution_2(bvp);
        MPI_Barrier(MPI_COMM_WORLD);
        //double end = MPI_Wtime();
    } else {
        //Start and end time for the node
        //G and B storage vectors for each node
        double B_temp, knot_start, knot_end;
        PDDSJob auxjob;
        std::vector<double> G_temp;
        std::vector<int>  G_j_temp;
        //Stencil
        Stencil stencil;
        VectorFunction grad,grad_noisy;
        std::stringstream dirname,dirname_noisy;
        dirname << "Input/Patch_"<<myid;
        dirname_noisy << "Input/Patch_noisy_"<<myid;
        GMSolver solver(bvp ,parameters, h0, (unsigned int) myid +1);
        c2 = pow(fac*(1.0/nN[0]),2.0);
        G_i.clear();G_j.clear(); G.clear(); G_CT.clear(); G_var.clear();
        B.clear();B_CT.clear();B_i.clear();B_var.clear(); times.clear();
        Est_var.clear(); APL.clear();phi_trap.clear(); phiphi_trap.clear(); 
        phi_trap_VR.clear(); phiphi_trap_VR.clear();
        do{
            done = Receive_Node(auxjob);
            if(!done){
                knot_start = MPI_Wtime();
                B_temp = 0.0;
                G_j_temp.clear();
                G_temp.clear();
                stencil = Recieve_Stencil_Data_Loop();
                stencil.Compute_ipsi(bvp,c2,debug_fname);
                grad.Init(2,dirname.str());
                grad_noisy.Init(2,dirname_noisy.str());
                solver.h = auxjob.h[0];
                if(stencil.Is_Interior()){ 
                    //printf("Knot %d is interior\n",auxjob.index[0]);
                    solver.Solve(position,c2,stencil,G_j_temp,G_temp,B_temp,auxjob.N[0], auxjob.N[1],grad, grad_noisy);
                }else{
                    //printf("Knot %d is exterior\n",auxjob.index[0]);
                    solver.Solve_mix(position,c2,1.0,stencil,G_j_temp,G_temp,B_temp,auxjob.N[0], auxjob.N[1],grad, grad_noisy);
                }
                B.push_back(B_temp);
                B_CT.push_back(solver.B_CT);
                B_i.push_back(auxjob.index[0]);
                B_var.push_back(solver.var_B);
                xi.push_back(solver.sum_xi);
                phi.push_back(solver.sum_phi);
                xixi.push_back(solver.sum_xixi);
                phiphi.push_back(solver.sum_phiphi);
                phixi.push_back(solver.sum_phixi);
                phip.push_back(solver.sum_phip);
                phiphip.push_back(solver.sum_phiphip);
                phipxi.push_back(solver.sum_phipxi);
                APL.push_back(solver.APL);
                Est_var.push_back(solver.Est_var);
                RNGCalls.push_back((double)solver.N_rngcalls);
                phi_trap.push_back(solver.sum_phi_trap);
                phiphi_trap.push_back(solver.sum_phiphi_trap);
                phi_trap_VR.push_back(solver.sum_phi_trap_VR);
                phiphi_trap_VR.push_back(solver.sum_phiphi_trap_VR);
                bias_num.push_back(solver.sum_biasNum);
                bias_trap.push_back(solver.sum_biasTrap);
                for(unsigned int k = 0;k < G_temp.size(); k ++){
                    G_i.push_back(auxjob.index[0]);
                    G_j.push_back(G_j_temp[k]);
                    G.push_back(G_temp[k]);
                    G_CT.push_back(solver.G_CT[k]);
                    G_var.push_back(solver.var_G[k]);
                }
                knot_end = MPI_Wtime();
                times.push_back((knot_end-knot_start));
            }
        }while(!done);
        Send_G_B_2(server);
        Send_Metadata(server);
        //Compute_B_Deterministic();
        //double end = MPI_Wtime();
        MPI_Barrier(MPI_COMM_WORLD);
        //MPI_Comm_free(&workers);
    }
    //MPI_Finalize();
}*/
void PDDSparseGM::Solve_SemiLin_numVR(int iteration, bvp Lin_BVP){
    bool done=true;
    //double start = MPI_Wtime();
    //FILE *pFile;
    //if(myid==server) Print_Problem();
    //Compute the PDDSparse Matrix
    if(myid==server){
        FILE *pFile;
        Update_TimeFile("Iteration begins",1);
        //system("mv Output/solution.csv Output/solution_nvarred.csv");
        //system("mv Output/Debug/B.csv Output/Debug/B_nvarred.csv");
        //system("mv Output/Debug/G.csv Output/Debug/G_nvarred.csv");
        //system("mv Output/Debug/B_CT.csv Output/Debug/B_CT_nvarred.csv");
        //system("mv Output/Debug/G_CT.csv Output/Debug/G_CT_nvarred.csv");
        //system("mv Output/Debug/B_var.csv Output/Debug/B_var_nvarred.csv");
        //system("mv Output/Debug/G_var.csv Output/Debug/G_var_nvarred.csv");
        T_vec_G.clear();T_vec_Gvar.clear(); T_vec_GCT.clear(); T_vec_B.clear();
        T_vec_Bvar.clear(); T_vec_BCT.clear(); T_vec_APL.clear(); T_vec_times.clear();
        T_vec_xi.clear(); T_vec_phi.clear(); T_vec_xixi.clear(); T_vec_phiphi.clear();
        T_vec_phixi.clear(); T_vec_phip.clear(); T_vec_phiphip.clear(); T_vec_phipxi.clear();
        T_vec_Estvar.clear(); T_vec_RNGCalls.clear(); T_vec_phi_trap.clear(); T_vec_phiphi_trap.clear();
        T_vec_phi_trap_VR.clear();T_vec_phiphi_trap_VR.clear(); T_vec_bias_num.clear();
        T_vec_bias_trap.clear();
        G.clear(); G_CT.clear(); B.clear(); B_CT.clear(); G_var.clear(); B_var.clear();
        APL.clear(); times.clear(); xi.clear(); phi.clear(); xixi.clear(); phiphi.clear();
        phixi.clear(); phip.clear(); phiphip.clear(); phipxi.clear(); Est_var.clear();
        RNGCalls.clear(); PCoeff_o.clear(); PCoeff_n.clear(); phi_trap.clear(); phiphi_trap.clear();
        phi_trap_VR.clear(); phiphi_trap_VR.clear(); var_phi_trap.clear();var_phi_trap_VR.clear();
        G_j.clear(); G_i.clear(); B_i.clear(); bias_num.clear(); bias_trap.clear();
        char order[256];
        sprintf(order,"cp -r Output Output_%d",iteration);
        system(order);
        Update_TimeFile("Copying files",1);
        std::vector<int> aux_vec;
        std::vector<Eigen::Vector2d> positions;
        std::vector<int> indexes;
        aux_vec.resize(2);
        for(int Iy = 0; Iy <= 2*iN[1]-2; Iy ++){
            if(Iy%2 == 0){
                for(int Ix = 0; Ix <= iN[0]-2; Ix ++){
                    aux_vec[0] = Ix;
                    aux_vec[1] = Iy;
                    if(Iy == 0){
                        indexes.assign(interfaces[interface_map[aux_vec]].index.begin(),
                        interfaces[interface_map[aux_vec]].index.end());
                        positions.assign(interfaces[interface_map[aux_vec]].position.begin(),
                        interfaces[interface_map[aux_vec]].position.end());
                    } else {
                        indexes.assign(interfaces[interface_map[aux_vec]].index.begin()+1,
                        interfaces[interface_map[aux_vec]].index.end());
                        positions.assign(interfaces[interface_map[aux_vec]].position.begin()+1,
                        interfaces[interface_map[aux_vec]].position.end());
                    }
                    std::cout << "Sending interface [" << aux_vec[0] << ","<< aux_vec[1] << "]"<< std::endl;
                    Send_Interface(positions, indexes);
                    //Send_Stencil_Data(indexes[0]);
                    std::cout << "Interface [" << aux_vec[0] << ","<< aux_vec[1] << "] sent"<< std::endl;
                }
            } else {
                for(int Ix = 0; Ix <= iN[0]-1; Ix ++){
                    aux_vec[0] = Ix;
                    aux_vec[1] = Iy;
                    if(Ix == 0){
                        indexes.assign(interfaces[interface_map[aux_vec]].index.begin(),
                        interfaces[interface_map[aux_vec]].index.end()-1);
                        positions.assign(interfaces[interface_map[aux_vec]].position.begin(),
                        interfaces[interface_map[aux_vec]].position.end()-1);
                    } else {
                        if(Ix == iN[0]-1){
                            indexes.assign(interfaces[interface_map[aux_vec]].index.begin()+1,
                            interfaces[interface_map[aux_vec]].index.end());
                            positions.assign(interfaces[interface_map[aux_vec]].position.begin()+1,
                            interfaces[interface_map[aux_vec]].position.end());
                        }else{
                            indexes.assign(interfaces[interface_map[aux_vec]].index.begin()+1,
                            interfaces[interface_map[aux_vec]].index.end()-1);
                            positions.assign(interfaces[interface_map[aux_vec]].position.begin()+1,
                            interfaces[interface_map[aux_vec]].position.end()-1);
                        }
                    }
                    std::cout << "Sending interface [" << aux_vec[0] << ","<< aux_vec[1] << "]"<< std::endl;
                    Send_Interface(positions, indexes);
                    //Send_Stencil_Data(indexes[0]);
                    std::cout << "Interface [" << aux_vec[0] << ","<< aux_vec[1] << "] sent"<< std::endl;
                }
                
            }
        }
        Update_TimeFile("MC Simulations VR",server+1);
        //G and B are received from the workers
        #ifdef LOCAL_SERVERS
        #else
        for(int process = 0; process < server; process++){
             Receive_G_B();
             Receive_Metadata();
        }
        Update_TimeFile("Receiving G, B and Metadata",server+1);
        #endif
        //Compute_B_Deterministic();
        Compute_Solution(Lin_BVP);
        //pFile = fopen(debug_fname,"a");
        //double end = MPI_Wtime();
        Process_Metadata();
        Update_TimeFile("Processing metadata",1);
        //system("octave-cli SmoothingSplines/SmoothingSplines.m");
        //Update_TimeFile("Smoothing splines",1);
        B.clear();
        B_i.clear();
        //Compute_B_Deterministic();
        //Compute_Solution_2(bvp);
        //double end = MPI_Wtime();
        T_vec_G.clear();T_vec_Gvar.clear(); T_vec_GCT.clear(); T_vec_B.clear();
        T_vec_Bvar.clear(); T_vec_BCT.clear(); T_vec_APL.clear(); T_vec_times.clear();
        T_vec_xi.clear(); T_vec_phi.clear(); T_vec_xixi.clear(); T_vec_phiphi.clear();
        T_vec_phixi.clear(); T_vec_phip.clear(); T_vec_phiphip.clear(); T_vec_phipxi.clear();
        T_vec_Estvar.clear(); T_vec_RNGCalls.clear(); T_vec_phi_trap.clear(); T_vec_phiphi_trap.clear();
        T_vec_phi_trap_VR.clear();T_vec_phiphi_trap_VR.clear();  T_vec_bias_num.clear();
        T_vec_bias_trap.clear();
        G.clear(); G_CT.clear(); B.clear(); B_CT.clear(); G_var.clear(); B_var.clear();
        APL.clear(); times.clear(); xi.clear(); phi.clear(); xixi.clear(); phiphi.clear();
        phixi.clear(); phip.clear(); phiphip.clear(); phipxi.clear(); Est_var.clear();
        RNGCalls.clear(); PCoeff_o.clear(); PCoeff_n.clear(); phi_trap.clear(); phiphi_trap.clear();
        phi_trap_VR.clear(); phiphi_trap_VR.clear(); var_phi_trap.clear();var_phi_trap_VR.clear();
        bias_num.clear(); bias_trap.clear();G_j.clear(); G_i.clear(); B_i.clear();
    } else {
        //Start and end time for the node
        //G and B storage vectors for each node
        double B_temp, BB_temp, h_aux,distance;
        std::vector<double> G_temp, G_var_temp, x, y, x_LUT,y_LUT,u_LUT,v_LUT;
        double *array_x, *array_y, *array_u, *array_v;
        std::vector<int>  G_j_temp, indexes;
        Eigen::Vector2d X0;
        //Stencil
        Stencil stencil;
        EMFKACSolver solver;
        //spline
        gsl_spline2d *LUT_u, *LUT_v;
        gsl_interp_accel *xacc_u, *yacc_u, *xacc_v, *yacc_v;
        c2 = pow(fac*(1.0/(nN[0]-1)),2.0);
        G_i.clear();G_j.clear(); G.clear(); G_CT.clear(); G_var.clear();
        B.clear();B_CT.clear();B_i.clear();B_var.clear(); times.clear();
        G.clear(); G_CT.clear(); B.clear(); B_CT.clear(); G_var.clear(); B_var.clear();
        APL.clear(); times.clear(); xi.clear(); phi.clear(); xixi.clear(); phiphi.clear();
        phixi.clear(); phip.clear(); phiphip.clear(); phipxi.clear(); Est_var.clear();
        RNGCalls.clear(); PCoeff_o.clear(); bias_num.clear(); bias_trap.clear();
        PCoeff_n.clear(); phi_trap.clear(); phiphi_trap.clear(); Est_var_VR.clear();
        //printf("Intermediate Step\n");
        while(! Receive_Interface(x,y,indexes)){
            //stencil.Print(indexes[0]);
            std::cout << "Interface received"<< std::endl;
            for(unsigned int knot = 0; knot < x.size(); knot ++){
                //knot_start = MPI_Wtime();
                stencil = Compute_Stencil(indexes[knot]);
                stencil.Compute_ipsi(Lin_BVP,c2,debug_fname);
                for(int i = 0; i < 4; i++) stencil.global_parameters[i] = parameters[i];
                Set_LUT_FILE_Solution(indexes[knot],x_LUT,y_LUT,u_LUT);
                std::cout << "Solving knot " << indexes[knot] << std::endl;
                array_x = new double[x_LUT.size()];for(unsigned int i = 0; i < x_LUT.size(); i++){
                    array_x[i] = x_LUT[i];
                    //std::cout << array_x[i] << std::endl;
                }
                //std::cout << "y\n";
                array_y = new double[y_LUT.size()];for(unsigned int i = 0; i < y_LUT.size(); i++){
                    array_y[i] = y_LUT[i];
                    //std::cout << array_y[i] << std::endl;
                } 
                array_u = new double[u_LUT.size()];for(unsigned int i = 0; i < u_LUT.size(); i++) array_u[i] = u_LUT[i];
                LUT_u = gsl_spline2d_alloc(gsl_interp2d_bicubic, x_LUT.size(), y_LUT.size());
                xacc_u = gsl_interp_accel_alloc();
                yacc_u = gsl_interp_accel_alloc();
                gsl_spline2d_init(LUT_u, array_x, array_y, array_u, x_LUT.size(), y_LUT.size());
                //Set_LUT_FILE_Correction(indexes[knot],x_LUT,y_LUT,v_LUT);
                //array_x = new double[x_LUT.size()];for(unsigned int i = 0; i < x_LUT.size(); i++) array_x[i] = x_LUT[i];
                //array_y = new double[y_LUT.size()];for(unsigned int i = 0; i < y_LUT.size(); i++) array_y[i] = y_LUT[i];
                //array_v = new double[v_LUT.size()];for(unsigned int i = 0; i < v_LUT.size(); i++) array_v[i] = v_LUT[i];
                //LUT_v = gsl_spline2d_alloc(gsl_interp2d_bicubic, x_LUT.size(), y_LUT.size());
                //xacc_v = gsl_interp_accel_alloc();
                //yacc_v = gsl_interp_accel_alloc();
                //gsl_spline2d_init(LUT_v, array_x, array_y, array_v, x_LUT.size(), y_LUT.size());
                B_temp = 0.0;
                BB_temp = 0.0;
                G_j_temp.clear();
                G_temp.clear();
                G_var_temp.clear();
                X0(0) = x[knot]; X0(1) = y[knot];
                //printf("LUT defined in [%f %f], [%f %f]\n",array_x[0],array_y[0],array_x[x_LUT.size()-1],array_y[y_LUT.size()-1]);
                //printf("Point [%f, %f]\n",X0(0),X0(1));
                //std::cout << gsl_spline2d_eval(LUT,X0(0),X0(1),xacc,yacc) << std::endl;
                stencil.Reset();
                solver.Reset_sums();
                solver.N = 0;
                Eigen::Vector2d NP_aux, N_aux;
                distance = Lin_BVP.distance(stencil.stencil_parameters,X0,NP_aux,N_aux);
                if(distance <= -1E-04){
                    h_aux = pow(distance/((N_aux.transpose()*Lin_BVP.sigma(X0,0.0)).norm()*2*0.5826),2.0);
                    h_aux = std::min(h0,h_aux);
                    std::cout << "H_aux = " << h_aux << std::endl;
                    solver.Simulate_OMP_VR_Loop(X0,(unsigned int) N,h_aux, 0.1,
                    Lin_BVP, stencil.stencil_parameters,LUT_u,xacc_u,yacc_u);
                    solver.Reduce_Numeric(Lin_BVP,N,stencil,c2,G_temp,G_var_temp,
                    G_j_temp, B_temp, BB_temp, LUT_u, xacc_u, yacc_u);
                    B.push_back(B_temp);
                    B_i.push_back(indexes[knot]);
                    B_var.push_back(B_temp*B_temp - BB_temp);
                    for(unsigned int k = 0;k < G_temp.size(); k ++){
                        G_i.push_back(indexes[knot]);
                        G_j.push_back(G_j_temp[k]);
                        G.push_back(G_temp[k]);
                        G_var.push_back(G_var_temp[k]);
                    }
                }else{
                    B.push_back(Lin_BVP.g(X0,0.0));
                    B_i.push_back(indexes[knot]);
                    B_var.push_back(0.0);
                }
                delete array_x; delete array_y; delete array_u;
                solver.Update();
                xi.push_back(solver.xi_num);
                phi.push_back(solver.phi_num);
                xixi.push_back(solver.xixi_num);
                phiphi.push_back(solver.phiphi_num);
                phixi.push_back(solver.xiphi_num);
                phip.push_back(solver.phi_sublinear_num);
                phiphip.push_back(solver.phiphi_sublinear_num);
                phipxi.push_back(solver.xiphi_sublinear_num);
                APL.push_back(solver.APL);
                Est_var.push_back(solver.phiphi_num-solver.phi_num*solver.phi_num);
                Est_var_VR.push_back(solver.phiphi_VR_num-solver.phi_VR_num*solver.phi_VR_num);
                //std::cout << solver.phiphi_VR_num << "\t" << solver.phi_VR_num*solver.phi_VR_num << std::endl;
                RNGCalls.push_back((double)solver.RNGC);
                phi_trap.push_back(solver.phi);
                //std::cout << "Index " << indexes[knot] << "\t FKAC num sol " << solver.phi_num << "\t FKAC sol" << solver.phi << "\n";
                phiphi_trap.push_back(solver.phiphi);
                phi_trap_VR.push_back(solver.phi+solver.xi);
                phiphi_trap_VR.push_back(solver.phiphi_VR);
                bias_num.push_back(solver.phi_sublinear_num - solver.phi_num); 
                bias_trap.push_back(solver.phi_sublinear - solver.phi); 
                
            }
        }
        Send_G_B();
        Send_Metadata(server);
        //Compute_B_Deterministic();
        //double end = MPI_Wtime();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    //MPI_Finalize();
}
void PDDSparseGM::Solve_Iterative_numVR(int iteration, bvp Lin_BVP){
    bool done=true;
    //double start = MPI_Wtime();
    //FILE *pFile;
    //if(myid==server) Print_Problem();
    //Compute the PDDSparse Matrix
    if(myid==server){
        Update_TimeFile("Iteration begins",1);
        FILE *pFile;
        //system("mv Output/solution.csv Output/solution_nvarred.csv");
        //system("mv Output/Debug/B.csv Output/Debug/B_nvarred.csv");
        //system("mv Output/Debug/G.csv Output/Debug/G_nvarred.csv");
        //system("mv Output/Debug/B_CT.csv Output/Debug/B_CT_nvarred.csv");
        //system("mv Output/Debug/G_CT.csv Output/Debug/G_CT_nvarred.csv");
        //system("mv Output/Debug/B_var.csv Output/Debug/B_var_nvarred.csv");
        //system("mv Output/Debug/G_var.csv Output/Debug/G_var_nvarred.csv");
        T_vec_G.clear();T_vec_Gvar.clear(); T_vec_GCT.clear(); T_vec_B.clear();
        T_vec_Bvar.clear(); T_vec_BCT.clear(); T_vec_APL.clear(); T_vec_times.clear();
        T_vec_xi.clear(); T_vec_phi.clear(); T_vec_xixi.clear(); T_vec_phiphi.clear();
        T_vec_phixi.clear(); T_vec_phip.clear(); T_vec_phiphip.clear(); T_vec_phipxi.clear();
        T_vec_Estvar.clear(); T_vec_RNGCalls.clear(); T_vec_phi_trap.clear(); T_vec_phiphi_trap.clear();
        T_vec_phi_trap_VR.clear();T_vec_phiphi_trap_VR.clear(); T_vec_bias_num.clear();
        T_vec_bias_trap.clear();
        G.clear(); G_CT.clear(); B.clear(); B_CT.clear(); G_var.clear(); B_var.clear();
        APL.clear(); times.clear(); xi.clear(); phi.clear(); xixi.clear(); phiphi.clear();
        phixi.clear(); phip.clear(); phiphip.clear(); phipxi.clear(); Est_var.clear();
        RNGCalls.clear(); PCoeff_o.clear(); PCoeff_n.clear(); phi_trap.clear(); phiphi_trap.clear();
        phi_trap_VR.clear(); phiphi_trap_VR.clear(); var_phi_trap.clear();var_phi_trap_VR.clear();
        G_j.clear(); G_i.clear(); B_i.clear(); bias_num.clear(); bias_trap.clear();
        char order[256];
        sprintf(order,"cp -r Output Output_%d",iteration);
        Update_TimeFile("Copying files",1);
        system(order);
        std::vector<int> aux_vec;
        std::vector<Eigen::Vector2d> positions;
        std::vector<int> indexes;
        aux_vec.resize(2);
        for(int Iy = 0; Iy <= 2*iN[1]-2; Iy ++){
            if(Iy%2 == 0){
                for(int Ix = 0; Ix <= iN[0]-2; Ix ++){
                    aux_vec[0] = Ix;
                    aux_vec[1] = Iy;
                    if(Iy == 0){
                        indexes.assign(interfaces[interface_map[aux_vec]].index.begin(),
                        interfaces[interface_map[aux_vec]].index.end());
                        positions.assign(interfaces[interface_map[aux_vec]].position.begin(),
                        interfaces[interface_map[aux_vec]].position.end());
                    } else {
                        indexes.assign(interfaces[interface_map[aux_vec]].index.begin()+1,
                        interfaces[interface_map[aux_vec]].index.end());
                        positions.assign(interfaces[interface_map[aux_vec]].position.begin()+1,
                        interfaces[interface_map[aux_vec]].position.end());
                    }
                    std::cout << "Sending interface [" << aux_vec[0] << ","<< aux_vec[1] << "]"<< std::endl;
                    Send_Interface(positions, indexes);
                    //Send_Stencil_Data(indexes[0]);
                    std::cout << "Interface [" << aux_vec[0] << ","<< aux_vec[1] << "] sent"<< std::endl;
                }
            } else {
                for(int Ix = 0; Ix <= iN[0]-1; Ix ++){
                    aux_vec[0] = Ix;
                    aux_vec[1] = Iy;
                    if(Ix == 0){
                        indexes.assign(interfaces[interface_map[aux_vec]].index.begin(),
                        interfaces[interface_map[aux_vec]].index.end()-1);
                        positions.assign(interfaces[interface_map[aux_vec]].position.begin(),
                        interfaces[interface_map[aux_vec]].position.end()-1);
                    } else {
                        if(Ix == iN[0]-1){
                            indexes.assign(interfaces[interface_map[aux_vec]].index.begin()+1,
                            interfaces[interface_map[aux_vec]].index.end());
                            positions.assign(interfaces[interface_map[aux_vec]].position.begin()+1,
                            interfaces[interface_map[aux_vec]].position.end());
                        }else{
                            indexes.assign(interfaces[interface_map[aux_vec]].index.begin()+1,
                            interfaces[interface_map[aux_vec]].index.end()-1);
                            positions.assign(interfaces[interface_map[aux_vec]].position.begin()+1,
                            interfaces[interface_map[aux_vec]].position.end()-1);
                        }
                    }
                    std::cout << "Sending interface [" << aux_vec[0] << ","<< aux_vec[1] << "]"<< std::endl;
                    Send_Interface(positions, indexes);
                    //Send_Stencil_Data(indexes[0]);
                    std::cout << "Interface [" << aux_vec[0] << ","<< aux_vec[1] << "] sent"<< std::endl;
                }
                
            }
        }
        Update_TimeFile("MC Simulations",server+1);
        //G and B are received from the workers
        #ifdef LOCAL_SERVERS
        #else
        for(int process = 0; process < server; process++){
             Receive_G_B();
             Receive_Metadata();
        }
        Update_TimeFile("Receiving G and B",server+1);
        #endif
        //Compute_B_Deterministic();
        Compute_Solution(Lin_BVP);
        //pFile = fopen(debug_fname,"a");
        //double end = MPI_Wtime();
        Process_Metadata();
        Update_TimeFile("Processing metadata",1);
        //system("octave-cli SmoothingSplines/SmoothingSplines.m");
        //Update_TimeFile("Smoothing Splines",1);
        B.clear();
        B_i.clear();
        //Compute_B_Deterministic();
        //Compute_Solution_2(bvp);
        //double end = MPI_Wtime();
        T_vec_G.clear();T_vec_Gvar.clear(); T_vec_GCT.clear(); T_vec_B.clear();
        T_vec_Bvar.clear(); T_vec_BCT.clear(); T_vec_APL.clear(); T_vec_times.clear();
        T_vec_xi.clear(); T_vec_phi.clear(); T_vec_xixi.clear(); T_vec_phiphi.clear();
        T_vec_phixi.clear(); T_vec_phip.clear(); T_vec_phiphip.clear(); T_vec_phipxi.clear();
        T_vec_Estvar.clear(); T_vec_RNGCalls.clear(); T_vec_phi_trap.clear(); T_vec_phiphi_trap.clear();
        T_vec_phi_trap_VR.clear();T_vec_phiphi_trap_VR.clear();  T_vec_bias_num.clear();
        T_vec_bias_trap.clear(); T_vec_Estvar_VR.clear();
        G.clear(); G_CT.clear(); B.clear(); B_CT.clear(); G_var.clear(); B_var.clear();
        APL.clear(); times.clear(); xi.clear(); phi.clear(); xixi.clear(); phiphi.clear();
        phixi.clear(); phip.clear(); phiphip.clear(); phipxi.clear(); Est_var.clear();
        RNGCalls.clear(); PCoeff_o.clear(); PCoeff_n.clear(); phi_trap.clear(); phiphi_trap.clear();
        phi_trap_VR.clear(); phiphi_trap_VR.clear(); var_phi_trap.clear();var_phi_trap_VR.clear();
        bias_num.clear(); bias_trap.clear();G_j.clear(); G_i.clear(); B_i.clear();
    } else {
        //Start and end time for the node
        //G and B storage vectors for each node
        double B_temp, BB_temp, distance, h_aux;
        std::vector<double> G_temp, G_var_temp, x, y, x_LUT,y_LUT,u_LUT,v_LUT;
        double *array_x, *array_y, *array_u, *array_v;
        std::vector<int>  G_j_temp, indexes;
        Eigen::Vector2d X0;
        //Stencil
        Stencil stencil;
        EMFKACSolver solver;
        //spline. The v's are u0s 
        gsl_spline2d *LUT_u, *LUT_v;
        gsl_interp_accel *xacc_u, *yacc_u, *xacc_v, *yacc_v;
        c2 = pow(fac*(1.0/(nN[0]-1)),2.0);
        G_i.clear();G_j.clear(); G.clear(); G_CT.clear(); G_var.clear();
        B.clear();B_CT.clear();B_i.clear();B_var.clear(); times.clear();
        G.clear(); G_CT.clear(); B.clear(); B_CT.clear(); G_var.clear(); B_var.clear();
        APL.clear(); times.clear(); xi.clear(); phi.clear(); xixi.clear(); phiphi.clear();
        phixi.clear(); phip.clear(); phiphip.clear(); phipxi.clear(); Est_var.clear();
        RNGCalls.clear(); PCoeff_o.clear(); bias_num.clear(); bias_trap.clear();
        PCoeff_n.clear(); phi_trap.clear(); phiphi_trap.clear(); Est_var_VR.clear();
        //printf("Intermediate Step\n");
        while(! Receive_Interface(x,y,indexes)){
            //stencil.Print(indexes[0]);
            std::cout << "Interface received"<< std::endl;
            for(unsigned int knot = 0; knot < x.size(); knot ++){
                //knot_start = MPI_Wtime();
                stencil = Compute_Stencil(indexes[knot]);
                stencil.Compute_ipsi(Lin_BVP,c2,debug_fname);
                for(int i = 0; i < 4; i++) stencil.global_parameters[i] = parameters[i];
                Set_LUT_FILE_Solution(indexes[knot],x_LUT,y_LUT,u_LUT);
                std::cout << "Solving knot " << indexes[knot] << std::endl;
                array_x = new double[x_LUT.size()];for(unsigned int i = 0; i < x_LUT.size(); i++){
                    array_x[i] = x_LUT[i];
                    //std::cout << array_x[i] << std::endl;
                }
                //std::cout << "y\n";
                array_y = new double[y_LUT.size()];for(unsigned int i = 0; i < y_LUT.size(); i++){
                    array_y[i] = y_LUT[i];
                    //std::cout << array_y[i] << std::endl;
                } 
                array_u = new double[u_LUT.size()];for(unsigned int i = 0; i < u_LUT.size(); i++) array_u[i] = u_LUT[i];
                LUT_u = gsl_spline2d_alloc(gsl_interp2d_bicubic, x_LUT.size(), y_LUT.size());
                xacc_u = gsl_interp_accel_alloc();
                yacc_u = gsl_interp_accel_alloc();
                gsl_spline2d_init(LUT_u, array_x, array_y, array_u, x_LUT.size(), y_LUT.size());
                Set_LUT_FILE_Correction(indexes[knot],x_LUT,y_LUT,v_LUT);
                array_x = new double[x_LUT.size()];for(unsigned int i = 0; i < x_LUT.size(); i++) array_x[i] = x_LUT[i];
                array_y = new double[y_LUT.size()];for(unsigned int i = 0; i < y_LUT.size(); i++) array_y[i] = y_LUT[i];
                array_v = new double[v_LUT.size()];for(unsigned int i = 0; i < v_LUT.size(); i++) array_v[i] = v_LUT[i];
                LUT_v = gsl_spline2d_alloc(gsl_interp2d_bicubic, x_LUT.size(), y_LUT.size());
                xacc_v = gsl_interp_accel_alloc();
                yacc_v = gsl_interp_accel_alloc();
                gsl_spline2d_init(LUT_v, array_x, array_y, array_v, x_LUT.size(), y_LUT.size());
                B_temp = 0.0;
                BB_temp = 0.0;
                G_j_temp.clear();
                G_temp.clear();
                G_var_temp.clear();
                X0(0) = x[knot]; X0(1) = y[knot];
                //printf("LUT defined in [%f %f], [%f %f]\n",array_x[0],array_y[0],array_x[x_LUT.size()-1],array_y[y_LUT.size()-1]);
                //printf("Point [%f, %f]\n",X0(0),X0(1));
                //std::cout << gsl_spline2d_eval(LUT,X0(0),X0(1),xacc,yacc) << std::endl;
                stencil.Reset();
                solver.Reset_sums();
                solver.N = 0;
                Eigen::Vector2d NP_aux, N_aux;
                distance = Lin_BVP.distance(stencil.stencil_parameters,X0,NP_aux,N_aux);
                if(distance <= -1E-04){
                    h_aux = pow(distance/((N_aux.transpose()*Lin_BVP.sigma(X0,0.0)).norm()*2*0.5826),2.0);
                    h_aux = std::min(h0,h_aux);
                    std::cout << "H_aux = " << h_aux << std::endl;
                    solver.Simulate_OMP(X0,(unsigned int) N,h_aux, 0.1,
                    Lin_BVP, stencil.stencil_parameters,LUT_u,xacc_u,
                    yacc_u,LUT_v,xacc_v, yacc_v);
                    solver.Reduce_Numeric(Lin_BVP,N,stencil,c2,G_temp,G_var_temp,
                    G_j_temp, B_temp, BB_temp, LUT_u, xacc_u, yacc_u);
                    B.push_back(B_temp);
                    B_i.push_back(indexes[knot]);
                    B_var.push_back(B_temp*B_temp - BB_temp);
                    for(unsigned int k = 0;k < G_temp.size(); k ++){
                        G_i.push_back(indexes[knot]);
                        G_j.push_back(G_j_temp[k]);
                        G.push_back(G_temp[k]);
                        G_var.push_back(G_var_temp[k]);
                    }
                }else{
                    B.push_back(Lin_BVP.g(X0,0.0));
                    B_i.push_back(indexes[knot]);
                    B_var.push_back(0.0);
                }
                delete array_x; delete array_y; delete array_u;
                //B.push_back(B_temp);
                //B_CT.push_back(solver.B_CT);
                //B_i.push_back(auxjob.index[0]);
                //B_var.push_back(solver.var_B);
                solver.Update();
                xi.push_back(solver.xi_num);
                phi.push_back(solver.phi_num);
                xixi.push_back(solver.xixi_num);
                phiphi.push_back(solver.phiphi_num);
                phixi.push_back(solver.xiphi_num);
                phip.push_back(solver.phi_sublinear_num);
                phiphip.push_back(solver.phiphi_sublinear_num);
                phipxi.push_back(solver.xiphi_sublinear_num);
                APL.push_back(solver.APL);
                Est_var.push_back(solver.phiphi_num-solver.phi_num*solver.phi_num);
                Est_var_VR.push_back(solver.phiphi_VR_num-solver.phi_VR_num*solver.phi_VR_num);
                //std::cout << solver.phiphi_VR_num << "\t" << solver.phi_VR_num*solver.phi_VR_num << std::endl;
                RNGCalls.push_back((double)solver.RNGC);
                phi_trap.push_back(solver.phi);
                //std::cout << "Index " << indexes[knot] << "\t FKAC num sol " << solver.phi_num << "\t FKAC sol" << solver.phi << "\n";
                phiphi_trap.push_back(solver.phiphi);
                phi_trap_VR.push_back(solver.phi+solver.xi);
                phiphi_trap_VR.push_back(solver.phiphi_VR);
                bias_num.push_back(solver.phi_sublinear_num - solver.phi_num); 
                bias_trap.push_back(solver.phi_sublinear - solver.phi); 
                //knot_end = MPI_Wtime();
                //times.push_back((knot_end-knot_start));
                //knot_end = MPI_Wtime();
                
            }
        }
        Send_G_B();
        Send_Metadata(server);
        //Compute_B_Deterministic();
        //double end = MPI_Wtime();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    //MPI_Finalize();
}
/*
void PDDSparseGM::Compute_h_N(bvp BoundValProb, double eps, std::vector<double> & h_vec, std::vector<int> & N_vec){
    if(myid==server){
        system("mv Output/Debug/knot_metadata.csv Output/Debug/knot_metadata_nvarred.csv");
        std::map<direction, std::vector<std::vector <int> > > labels;
        double s_params[4],h_bias,h_dist;
        int aux_size;
        std::vector<double> knot_est_variance;
        Eigen::Vector2d X_aux, N_aux,E_P_aux;
        Eigen::SparseMatrix<double> I_sparse;
        I_sparse.resize(nNodes, nNodes);
        I_sparse.setIdentity();
        X_aux.resize(2); N_aux.resize(2); E_P_aux.resize(2);
        //max_absG_row.resize(nNodes);
        h_vec.resize(nNodes);
        N_vec.resize(nNodes);
        //var_G*u vector
        FILE *pfile;
        pfile = fopen("Output/Debug/h_and_N.csv","w");
        fprintf(pfile,"Knot,h,N\n");
        knot_est_variance.resize(nNodes);
        for(unsigned int i = 0; i < (unsigned int) nNodes; i++){
            knot_est_variance[i] = Est_var[i]*(1.0 - PCoeff_o[i]*PCoeff_o[i]);
        }
        for(int i = 0; i < nNodes; i++){
            labels = Labels_Stencil(i);
            N_vec[i] = (int)(4.0*knot_est_variance[i]/pow(eps*0.66666,2.0));
            N_vec[i] = (1+(N_vec[i]/N_job))*N_job;
            //Stencil parameters
            for(int j = 0; j < 4; j++) s_params[j] = parameters[j];
            if(labels[South].size()>0){
                //s_params[0] = interfaces[interface_map[labels[South][0]]].position[0][0];
                s_params[1] = interfaces[interface_map[labels[South][0]]].position[0][1];
            }
            if(labels[West].size()>0){
                s_params[0] = interfaces[interface_map[labels[West][0]]].position[0][0];
                //s_params[1] = interfaces[interface_map[labels[West][0]]].position[0][1];
            }
            if(labels[North].size()>0){
                aux_size = (int) interfaces[interface_map[labels[North][labels[North].size()-1]]].position.size();
                //s_params[2] = interfaces[interface_map[labels[North][labels[North].size()-1]]].position[aux_size - 1][0];
                s_params[3] = interfaces[interface_map[labels[North][labels[North].size()-1]]].position[aux_size - 1][1];
            }
            if(labels[East].size()>0){
                aux_size = (int) interfaces[interface_map[labels[East][labels[East].size()-1]]].position.size();
                s_params[2] = interfaces[interface_map[labels[East][labels[East].size()-1]]].position[aux_size - 1][0];
                //s_params[3] = interfaces[interface_map[labels[East][labels[East].size()-1]]].position[aux_size - 1][1];
            }
            X_aux = Node_Position(i);
            h_dist = pow(bvp.boundary.Dist(s_params, X_aux, E_P_aux, N_aux)/(
            (N_aux.transpose()*bvp.sigma.Value(X_aux,0.0)).norm()*2*0.5826),2.0);
            h_bias = 0.333*eps*h0/fabs(bias_num[i]);
            h_vec[i] = std::min(std::min(h_dist,h_bias),h0);
            fprintf(pfile,"%d,%e,%d\n",i,h_vec[i],N_vec[i]);
        }
        fclose(pfile);
        Update_TimeFile("Computing optimal h and N",1);
    }
}
*/
void PDDSparseGM::Update_TimeFile(std::string task,int N_processors){
    FILE *pfile;
    wclock_prev = wclock_measure;
    wclock_measure = MPI_Wtime();
    pfile = fopen("Output/Debug/times.txt","a");
    fprintf(pfile,"%s,%e,%e,%d\n",task.c_str(),wclock_measure-wclock_start,wclock_measure-wclock_prev,N_processors);
    fclose(pfile);
}
/*void PDDSparseGM::Reduce_G_B(void){
    Eigen::SparseMatrix<double> G_sparse, G_sparse_var, G_EM_sparse, 
                                B_sparse, B_sparse_var, B_EM_sparse;
    unsigned int size = (unsigned int) nNodes;
    G_sparse.resize(size, size);
    G_sparse_var.resize(size,size);
    G_EM_sparse.resize(size,size);
    G_sparse.setFromTriplets(T_vec_G.begin(),T_vec_G.end());
    G_sparse_var.setFromTriplets(T_vec_Gvar.begin(),T_vec_Gvar.end());
    G_EM_sparse.setFromTriplets(T_vec_GCT.begin(),T_vec_GCT.end());
    T_vec_G.clear(); T_vec_Gvar.clear(); T_vec_GCT.clear();
    G_i.clear(); G_j.clear(); G.clear(); G_var.clear(); 
    G_CT.clear();
    for (int k=0; k<G_sparse.outerSize(); ++k){
        for (Eigen::SparseMatrix<double>::InnerIterator it(G_sparse,k); it; ++it)
        {
            G_i.push_back(it.row());
            G_j.push_back(it.col());
            G.push_back(it.value());
            G_var.push_back(G_sparse_var.coeff(it.row(),it.col()));
            G_CT.push_back(G_EM_sparse.coeff(it.row(),it.col()));
        }
    }
    G_sparse.resize(0,0);
    G_sparse_var.resize(0,0);
    G_EM_sparse.resize(0,0);
    B_sparse.resize(size, 1);
    B_sparse_var.resize(size,1);
    B_EM_sparse.resize(size,1);
    B_sparse.setFromTriplets(T_vec_B.begin(),T_vec_B.end());
    B_sparse_var.setFromTriplets(T_vec_Bvar.begin(),T_vec_Bvar.end());
    B_EM_sparse.setFromTriplets(T_vec_BCT.begin(),T_vec_BCT.end());
    T_vec_B.clear(); T_vec_Bvar.clear(); T_vec_BCT.clear();
    B_i.clear(); B.clear(); B_var.clear(); 
    B_CT.clear();
    for (int k=0; k<B_sparse.outerSize(); ++k){
        for (Eigen::SparseMatrix<double>::InnerIterator it(B_sparse,k); it; ++it)
        {
            B_i.push_back(it.row());
            B.push_back(it.value());
            B_var.push_back(G_sparse_var.coeff(it.row(),it.col()));
            B_CT.push_back(G_EM_sparse.coeff(it.row(),it.col()));
        }
    }
    B_sparse.resize(0,0);
    B_sparse_var.resize(0,0);
    B_EM_sparse.resize(0,0);
}
*/
std::vector<double> PDDSparseGM::Read_File(char fname[256]){
    std::vector<double> output;
    char *line_buf = NULL;
    size_t line_buf_size = 0;
    int line_count = 0;
    ssize_t line_size;
    FILE *fp = fopen(fname, "r");
    if (!fp)
    {
        fprintf(stderr, "Error opening file '%s'\n", fname);
        return output;
    }
    /* Get the first line of the file. */
    line_size = getline(&line_buf, &line_buf_size, fp);
    /* Loop through until we are done with the file. */
    while (line_size > 0)
    {
        /* Increment our line count */
        line_count++;
        /* Show the line details */
        output.push_back(atof(line_buf));
        /* Get the next line */
        line_size = getline(&line_buf, &line_buf_size, fp);
    }
   /* Free the allocated line buffer */
   free(line_buf);
   line_buf = NULL;
   fclose(fp);
   return output;
}