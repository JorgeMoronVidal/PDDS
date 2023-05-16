#include "stencil.hpp"

Stencil::Stencil(void){
    //All variables of the class are emptied
    pos_north.clear(); index_north.clear();
    pos_south.clear(); index_south.clear();
    pos_east.clear(); index_east.clear();
    pos_west.clear(); index_west.clear();
}

void Stencil::Init(std::map<direction, std::vector<int> > s_index,
                   std::map<direction, std::vector<double> > s_x,
                   std::map<direction, std::vector<double> > s_y,
                   double *parameters){
    pos_north.clear(); index_north.clear(); G_north.clear();
    pos_south.clear(); index_south.clear(); G_south.clear();
    pos_east.clear(); index_east.clear(); G_east.clear();
    pos_west.clear(); index_west.clear();G_west.clear();
    ipsi_north.resize(0,0); ipsi_south.resize(0,0);
    ipsi_east.resize(0,0); ipsi_west.resize(0,0);
    std::map<int,Eigen::Vector2d> sten_map;
    Eigen::Vector2d vaux;
    vaux.resize(2);
    for(int i = 0; i < 4; i++) stencil_parameters[i] = parameters[i];
    if(s_index[North].size() > 0){
        for(int i = 0; i < (int)s_index[North].size(); i++){
            vaux[0] = s_x[North][i];
            vaux[1] = s_y[North][i];
            sten_map[s_index[North][i]] = vaux;
            index_north.push_back(s_index[North][i]);
        }
        std::vector<int>::iterator v_north = std::unique(index_north.begin(), index_north.end());
        index_north.erase(v_north, index_north.end());
        for(unsigned int i = 0; i < index_north.size(); i++) pos_north.push_back(sten_map[index_north[i]]);
        for(unsigned int i = 0; i < pos_north.size(); i++) G_north.push_back(0.0);
    }
    sten_map.clear();

    if(s_index[South].size() > 0){
        for(int i = 0; i < (int)s_index[South].size(); i++){
            vaux[0] = s_x[South][i];
            vaux[1] = s_y[South][i];
            sten_map[s_index[South][i]] = vaux;
            index_south.push_back(s_index[South][i]);
        }
        std::vector<int>::iterator v_south = std::unique(index_south.begin(), index_south.end());
        index_south.erase(v_south, index_south.end());
        for(unsigned int i = 0; i < index_south.size(); i++) pos_south.push_back(sten_map[index_south[i]]);
        for(unsigned int i = 0; i < pos_south.size(); i++) G_south.push_back(0.0);
    }
    sten_map.clear();

    if(s_index[East].size() > 0){
        for(int i = 0; i < (int)s_index[East].size(); i++){
            vaux[0] = s_x[East][i];
            vaux[1] = s_y[East][i];
            sten_map[s_index[East][i]] = vaux;
            index_east.push_back(s_index[East][i]);
        }
        std::vector<int>::iterator v_east = std::unique(index_east.begin(), index_east.end());
        index_east.erase(v_east, index_east.end());
        for(unsigned int i = 0; i < index_east.size(); i++) pos_east.push_back(sten_map[index_east[i]]);
        for(unsigned int i = 0; i < pos_east.size(); i++) G_east.push_back(0.0);
    }
    sten_map.clear();

    if(s_index[West].size() > 0){
        for(int i = 0; i < (int)s_index[West].size(); i++){
            vaux[0] = s_x[West][i];
            vaux[1] = s_y[West][i];
            sten_map[s_index[West][i]] = vaux;
            index_west.push_back(s_index[West][i]);
        }
        std::vector<int>::iterator v_west = std::unique(index_west.begin(), index_west.end());
        index_west.erase(v_west, index_west.end()); 
        for(unsigned int i = 0; i < index_west.size(); i++) pos_west.push_back(sten_map[index_west[i]]);
        for(unsigned int i = 0; i < pos_west.size(); i++) G_west.push_back(0.0);
    }
    sten_map.clear();

    counter_north = 0; counter_south = 0;
    counter_east = 0; counter_west = 0;
}
void Stencil::Reset(void){
    G_north.clear();
    GG_north.clear();
    for(unsigned int i = 0; i < pos_north.size(); i++) {
        G_north.push_back(0.0);
        GG_north.push_back(0.0);
    }
    G_south.clear();
    GG_south.clear();
    for(unsigned int i = 0; i < pos_south.size(); i++) {
        G_south.push_back(0.0);
        GG_south.push_back(0.0);
    }
    G_east.clear();
    GG_east.clear();
    for(unsigned int i = 0; i < pos_east.size(); i++) {
        G_east.push_back(0.0);
        GG_east.push_back(0.0);
    }
    G_west.clear();
    GG_west.clear();
    for(unsigned int i = 0; i < pos_west.size(); i++) {
        G_west.push_back(0.0);
        GG_west.push_back(0.0);
    }
    counter_north = 0; counter_south = 0;
    counter_east = 0; counter_west = 0;
}

Eigen::MatrixXd Stencil::Compute_ipsi(std::vector<Eigen::Vector2d> & sten_position,
                  bvp boundvalprob, double & c2, char debug_fname[256]){
    //Auxiliary vectors 
    Eigen::Vector2d vaux, sincrement;
    //Psi Matrix, its inverse and Identity are created 
    Eigen::MatrixXd Psi, iPsi, I;
    //Condition number
    double cond;
    //RNG
    std::srand( (unsigned)time( NULL ) );
    //Matrix are resized
    Psi.resize(sten_position.size(), sten_position.size());
    //And fullfilled
    for(unsigned int i = 0; i < sten_position.size(); i ++){
        for(unsigned int j = 0; j < sten_position.size(); j ++){
            Psi(i,j) = boundvalprob.RBF(sten_position[i], sten_position[j], c2);
            if(isnan(Psi(i,j))) printf("THERE IS A NAN IN PSI\n");
        }
    }
    //std::cout << Psi;
    //getchar();
    //double cond, err;
    /*Eigen::FullPivLU<Eigen::MatrixXd> lu(Psi);
    if(lu.isInvertible()){
        iPsi = lu.inverse();
    }else{
        printf("FATAL ERROR: Psi Matrix is not invertible.");
        iPsi = Psi*0.0;
    }*/
    do{
        for(unsigned int i = 0; i < sten_position.size(); i ++){
            for(unsigned int j = 0; j < sten_position.size(); j ++){
                Psi(i,j) = boundvalprob.RBF(sten_position[i], sten_position[j], c2);
                if(isnan(Psi(i,j))) printf("THERE IS A NAN IN PSI\n");
                }                   
            }
            Eigen::FullPivLU<Eigen::MatrixXd> lu(Psi);
                if(lu.isInvertible()){
                    iPsi = lu.inverse();
                    cond = Psi.norm()*iPsi.norm();
                    if(cond < 5E+9) c2 += c2* 0.5;//(1.0*std::rand()/RAND_MAX);
                    if(cond > 5E+10){
                        if(c2 < 1.0){
                            c2 = c2*c2;
                        }else{
                            c2 = sqrt(c2)-0.001;
                        }
                    }
                    std::cout << "COND " << cond << " c2 " << c2 << std::endl;
                }else{
                    //printf("FATAL ERROR: Psi Matrix is not invertible.\n");
                    iPsi = Psi*0.0;
                    c2 = 0.5*c2;
                    cond = 1E+20;
                }
            }while((cond < 5E+9) || (cond > 5E+10));
            printf("Condition number %f C2 %f\n",Psi.norm()*iPsi.norm(),c2);
    /*Eigen::BiCGSTAB<Eigen::MatrixXd> CGS;
    CGS.compute(Psi);
    iPsi = CGS.solveWithGuess(I,iPsi);
    err = CGS.error();*/
    //cond = Psi.norm()*iPsi.norm();
    I.resize(sten_position.size(), sten_position.size());
    I.setIdentity();
    //err = (Psi*iPsi-I).norm();
    //fdebug = fopen(debug_fname, "a");
    //fprintf(fdebug,"iPsi computed with cond number %f and error %f \n", cond, err);
    //printf("iPsi computed with cond number %f and error %f \n", cond, err);
    //fclose(fdebug);
    Psi.resize(0,0);
    return iPsi;
}

bool Stencil::AreSame(double a, double b)
{
    return fabs(a - b) < EPSILON;
}

void Stencil::Compute_ipsi(bvp boundvalprob, double c2, char debug_fname[256]){
	 
    if(pos_north.size() > 0){
        c2_north = c2;
		ipsi_north = Compute_ipsi(pos_north, boundvalprob, c2_north, debug_fname);
	}
	if(pos_south.size() > 0){
        c2_south = c2;
		ipsi_south = Compute_ipsi(pos_south, boundvalprob, c2_south, debug_fname);
	}
	if(pos_east.size() > 0){
        c2_east = c2;
		ipsi_east = Compute_ipsi(pos_east, boundvalprob, c2_east, debug_fname);
	}
	if(pos_west.size() > 0){
        c2_west = c2;
		ipsi_west = Compute_ipsi(pos_west, boundvalprob, c2_west, debug_fname);
	}
        std::cout << "HERE I AM \n";
}

int Stencil::G_update(Eigen::Vector2d X, double Y, bvp boundvalprob, double c2){
	double H_ij;
	if(AreSame(X(1),stencil_parameters[1])){
		//South stencil
        //std::cout << "\t south \n";
		counter_south ++;
		for(unsigned int i = 0; i < G_south.size(); i++){
            H_ij = 0.0;
			for(unsigned int j = 0; j < G_south.size(); j++){
				H_ij += ipsi_south(j,i)*boundvalprob.RBF(pos_south[j],X,c2_south);
			}
			G_south[i] += -H_ij*Y;
            GG_south[i] += pow(H_ij*Y,2.0);
		}
		return 1;
	}
	if(AreSame(X(1),stencil_parameters[3])){
		//North stencil
        //std::cout << "\t north \n";
		counter_north ++;
		for(unsigned int i = 0; i < G_north.size(); i++){
            H_ij = 0.0;
			for(unsigned int j = 0; j < G_north.size(); j++){
				H_ij += ipsi_north(j,i)*boundvalprob.RBF(pos_north[j],X,c2_north);
			}
			G_north[i] += -H_ij*Y;
            GG_north[i] += pow(H_ij*Y,2.0);
		}
		return 1;
	}
	if(AreSame(X(0),stencil_parameters[0])){
		//West stencil
        //std::cout << "\t west \n";
		counter_west ++;
		for(unsigned int i = 0; i < G_west.size(); i++){
            H_ij = 0.0;
			for(unsigned int j = 0; j < G_west.size(); j++){
				H_ij += ipsi_west(j,i)*boundvalprob.RBF(pos_west[j],X,c2_west);
			}
			G_west[i] += -H_ij*Y;
            GG_west[i] += pow(H_ij*Y,2.0);
		}
		return 1;
	}
	if(AreSame(X(0),stencil_parameters[2])){
		//East stencil
        //std::cout << "\t east \n";
		counter_east ++;
		for(unsigned int i = 0; i < G_east.size(); i++){
            H_ij = 0.0;
			for(unsigned int j = 0; j < G_east.size(); j++){
				H_ij += ipsi_east(j,i)*boundvalprob.RBF(pos_east[j],X,c2_east);
			}
			G_east[i] += -H_ij*Y;
            GG_east[i] += pow(H_ij*Y,2.0);
		}
		return 1;
	}
	std::cout << "Something went wrong computing H.\n";
    printf("[%f,%f,%f,%f] (%f,%f)\n",stencil_parameters[0],stencil_parameters[1],
    stencil_parameters[2],stencil_parameters[3],X(0),X(1));
    getchar();
	return 0;
}

void Stencil::G_return(std::vector<int> & G_j, std::vector<double> & G){
	G_j.clear();
	G.clear();
	double norm = 1.0f/(counter_north + counter_south + counter_east + counter_west);
	if(pos_north.size() > 0){
        //norm = 1.0f/(counter_north);
		for(unsigned int i = 0; i < index_north.size(); i++){
			G_j.push_back(index_north[i]);
			G.push_back(G_north[i]*norm);
		}
	}
	if(pos_south.size() > 0){
        //norm = 1.0f/(counter_south);
		for(unsigned int i = 0; i < index_south.size(); i++){
			G_j.push_back(index_south[i]);
			G.push_back(G_south[i]*norm);
		}
	}
	if(pos_east.size() > 0){
        //norm = 1.0f/(counter_east);
		for(unsigned int i = 0; i < index_east.size(); i++){
			G_j.push_back(index_east[i]);
			G.push_back(G_east[i]*norm);
		}
	}
	if(pos_west.size() > 0){
        //norm = 1.0f/(counter_west);
		for(unsigned int i = 0; i < index_west.size(); i++){
			G_j.push_back(index_west[i]);
			G.push_back(G_west[i]*norm);
		}
	}

}
void Stencil::G_return_withrep(std::vector<int> & G_j, std::vector<double> & G,
                               std::vector<double> & var_G, int N_tray){
    G_j.clear();
    G.clear();
    var_G.clear();
    double norm = 1.0/(N_tray*1.0);
    std::map<int,double> G_map, GG_map;
    if(pos_north.size() > 0){
        for(unsigned int i = 0; i < index_north.size(); i++){
            G_map[index_north[i]] = 0.0;
            GG_map[index_north[i]] = 0.0;
        }
    }
    if(pos_south.size() > 0){
        for(unsigned int i = 0; i < index_south.size(); i++){
            G_map[index_south[i]] = 0.0;
            GG_map[index_south[i]] = 0.0;
        }
    }
    if(pos_east.size() > 0){
        for(unsigned int i = 0; i < index_east.size(); i++){
            G_map[index_east[i]] = 0.0;
            GG_map[index_east[i]] = 0.0;
        }
    }
    if(pos_west.size() > 0){
        for(unsigned int i = 0; i < index_west.size(); i++){
            G_map[index_west[i]] = 0.0;
            GG_map[index_west[i]] = 0.0;
        }
    }
    if(pos_north.size() > 0){
        for(unsigned int i = 0; i < index_north.size(); i++){
            G_map[index_north[i]] =+ G_north[i]*norm;
            GG_map[index_north[i]] =+ GG_north[i]*norm;
        }
    }
    if(pos_south.size() > 0){
        for(unsigned int i = 0; i < index_south.size(); i++){
            G_map[index_south[i]] =+ G_south[i]*norm;
            GG_map[index_south[i]] =+ GG_south[i]*norm;
        }
    }
    if(pos_east.size() > 0){
        for(unsigned int i = 0; i < index_east.size(); i++){
            G_map[index_east[i]] =+ G_east[i]*norm;
            GG_map[index_east[i]] =+ GG_east[i]*norm;
        }
    }
    if(pos_west.size() > 0){
        for(unsigned int i = 0; i < index_west.size(); i++){
            G_map[index_west[i]] =+ G_west[i]*norm;
            GG_map[index_west[i]] =+ GG_west[i]*norm;
        }
    }
    for(std::map<int, double>::iterator it = G_map.begin();
        it != G_map.end();
        it ++){
        G_j.push_back(it->first);
        G.push_back(it->second);
        var_G.push_back(GG_map[it->first] - pow(it->second,2.0));
    }

}
int Stencil::G_Test_update(Eigen::Vector2d X){
    if(AreSame(X(1),stencil_parameters[1])){
        //South stencil
        //std::cout << "\t south \n";
        for(unsigned int i = 0; i < G_south.size(); i++) G_south[i] = (double)index_south[i];
            return 1;
    }
    if(AreSame(X(1),stencil_parameters[3])){
        //North stencil
        for(unsigned int i = 0; i < G_north.size(); i++) G_north[i] = (double) index_north[i];
            return 1;
    }
    if(AreSame(X(0),stencil_parameters[0])){
        //West stencil
        //std::cout << "\t west \n";
        for(unsigned int i = 0; i < G_west.size(); i++) G_west[i] = (double) index_west[i];
            return 1;
    }
    if(AreSame(X(0),stencil_parameters[2])){
        //East stencil
        for(unsigned int i = 0; i < G_east.size(); i++) G_east[i] = (double) index_east[i];
            return 1;
    }
    std::cout << "Something went wrong computing H.\n";
    printf("[%f,%f,%f,%f] (%f,%f)\n",stencil_parameters[0],stencil_parameters[1],
    stencil_parameters[2],stencil_parameters[3],X(0),X(1));
    getchar();
    return 0;
}
bool Stencil::Is_Interior(void){
    if(ipsi_north.size()* ipsi_south.size() * ipsi_east.size() * ipsi_west.size() > 0) return true;
    return false;
}
void Stencil::Projection(Eigen::Vector2d X, Eigen::Vector2d & E_P){
    /*Eigen::Vector2d ps1, ps2;
    //SW corner
    if(AreSame(E_P(0),stencil_parameters[0])  && AreSame(E_P(1),stencil_parameters[1])){
        ps1 = ps2 = X;
        ps1[0] = stencil_parameters[0];
        ps1[1] = std::max(pos_west.front()[1], X[1]);
        ps2[0] = std::max(pos_south.front()[0], X[0]);
        ps2[1] = stencil_parameters[1];
    }
    //SE corner 
    if(AreSame(E_P(0),stencil_parameters[2])  && AreSame(E_P(1),stencil_parameters[1])){
        ps1 = ps2 = X;
        ps1[0] = stencil_parameters[2];
        ps1[1] = std::max(pos_east.front()[1], X[1]);
        ps2[0] = std::min(pos_south.back()[0], X[0]);
        ps2[1] = stencil_parameters[1];
    }
    //NE corner
    if(AreSame(E_P(0),stencil_parameters[2])  && AreSame(E_P(1),stencil_parameters[3])){
        ps1 = ps2 = X;
        ps1[0] = stencil_parameters[2];
        ps1[1] = std::min(pos_east.back()[1], X[1]);
        ps2[0] = std::min(pos_north.back()[0], X[0]);
        ps2[1] = stencil_parameters[3];
    }
    //NW corner
    if(AreSame(E_P(0),stencil_parameters[0])  && AreSame(E_P(1),stencil_parameters[3])){
        ps1 = ps2 = X;
        ps1[0] = stencil_parameters[0];
        ps1[1] = std::min(pos_west.back()[1], X[1]);
        ps2[0] = std::max(pos_north.front()[0], X[0]);
        ps2[1] = stencil_parameters[1];
    }
    if(ps1.size() >0){
        FILE *pfile;
        pfile = fopen("Output/Debug/sten_project.txt","a");
        fprintf(pfile,"Stencil SW (%f,%f) NE (%f,%f)\n",stencil_parameters[0],
        stencil_parameters[1],stencil_parameters[2],stencil_parameters[3]);
        if((X-ps1).norm() > (X-ps2).norm()){
            fprintf(pfile, " X = (%f,%f) E_P = (%f,%f) ps2 = (%f,%f) \n",
            X[0], X[1], E_P[0], E_P[1], ps2[0], ps2[1]);
            E_P = ps2;
        } else {
            if((X-ps2).norm() > (X-ps1).norm()){
                fprintf(pfile, " X = (%f,%f) E_P = (%f,%f) ps1 = (%f,%f) \n",
                X[0], X[1], E_P[0], E_P[1], ps1[0], ps1[1]);
                E_P = ps1;
            }
        }
        fclose(pfile);
    }*/
    double distance = (X-E_P).norm();
        Eigen::Vector2d aux;
        aux = X;
        aux(0) = stencil_parameters[0];
        if(fabs(X(0)-stencil_parameters[0]) < distance)
        {
          E_P = aux; 
          distance = (X-E_P).norm();
        }
        aux = X;
        aux(0) = stencil_parameters[2];
        if(fabs(X(0)-stencil_parameters[2])< (X-E_P).norm())
        {
          E_P = aux; 
          distance = (X-E_P).norm();
        }
        aux = X;
        aux(1) = stencil_parameters[1];
        if(fabs(X(1)-stencil_parameters[1])< (X-E_P).norm())
        {
          E_P = aux; 
          distance = (X-E_P).norm();
        }
        aux = X;
        aux(1) = stencil_parameters[3];
        if(fabs(X(1)-stencil_parameters[3]) < (X-E_P).norm())
        {
          E_P = aux; 
          distance = (X-E_P).norm();
        }
}
void Stencil::G_Test_return(std::vector<int> & stencil_index, std::vector<double> & G){
    stencil_index.clear();
    G.clear();
    if(pos_north.size() > 0){
        //norm = 1.0f/(counter_north);
        for(unsigned int i = 0; i < index_north.size(); i++){
            stencil_index.push_back(index_north[i]);
            G.push_back(G_north[i]);
        }
    }
    if(pos_south.size() > 0){
        //norm = 1.0f/(counter_south);
        for(unsigned int i = 0; i < index_south.size(); i++){
            stencil_index.push_back(index_south[i]);
            G.push_back(G_south[i]);
        }
    }
    if(pos_east.size() > 0){
        //norm = 1.0f/(counter_east);
        for(unsigned int i = 0; i < index_east.size(); i++){
            stencil_index.push_back(index_east[i]);
            G.push_back(G_east[i]);
        }
    }
    if(pos_west.size() > 0){
        //norm = 1.0f/(counter_west);
        for(unsigned int i = 0; i < index_west.size(); i++){
            stencil_index.push_back(index_west[i]);
            G.push_back(G_west[i]);
        }
    }

}

void Stencil::Print(int node_index){
    char fname[100];
    //FILE *pf;
    //sprintf(fname,"Output/Debug/Stencils/stencil_%d.txt", node_index);
    //pf = fopen(fname, "w");
    printf("index,x,y,sten\n");
    for(unsigned int i = 0; i < index_south.size(); i++){
        printf("%d,%.4f,%.4f,south\n",index_south[i], pos_south[i][0], pos_south[i][1]);
    }
    for(unsigned int i = 0; i < index_east.size(); i++) printf("%d,%.4f,%.4f,east\n",index_east[i], pos_east[i](0), pos_east[i](1));
    for(unsigned int i = 0; i < index_north.size(); i++) printf("%d,%.4f,%.4f,north\n",index_north[i], pos_north[i](0), pos_north[i](1));
    for(unsigned int i = 0; i < index_west.size(); i++) printf("%d,%.4f,%.4f,west\n",index_west[i], pos_west[i](0), pos_west[i](1));   
    /*for(unsigned int i = 0; i < index_south.size(); i++) fprintf(pf,"%d\n",index_south[i]);
    for(unsigned int i = 0; i < index_east.size(); i++) fprintf(pf,"%d\n",index_east[i]);
    for(unsigned int i = 0; i < index_north.size(); i++) fprintf(pf,"%d\n",index_north[i]);
    for(unsigned int i = 0; i < index_west.size(); i++) fprintf(pf,"%d\n",index_west[i]);*/
    //fclose(pf);
    //sprintf(fname,"Output/Debug/Stencils/boundary_stencil_%d.txt", node_index);
    //pf = fopen(fname, "w");
    printf("x,y\n");
    printf("%.3f,%.3f\n",stencil_parameters[0], stencil_parameters[1]);
    printf("%.3f,%.3f\n",stencil_parameters[2], stencil_parameters[1]);
    printf("%.3f,%.3f\n",stencil_parameters[2], stencil_parameters[3]);
    printf("%.3f,%.3f\n",stencil_parameters[0], stencil_parameters[3]);
    printf("%.3f,%.3f\n",stencil_parameters[0], stencil_parameters[1]);
    //fclose(pf);

}

