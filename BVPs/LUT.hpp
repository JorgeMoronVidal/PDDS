#include <iostream>
#include <eigen3/Eigen/Core>
#include <vector>
#include <gsl/gsl_math.h>
#include <gsl/gsl_interp2d.h>
#include <gsl/gsl_spline2d.h>
#include <gsl/gsl_errno.h>
#include <string>
#include <fstream>
/*Variables and functions that manage with LUT in BVPs*/
/*LUT auxiliary global variables. This part only works with 2D problems*/
gsl_spline2d *spline_u,*spline_g,*spline_p,*spline_f,*spline_c,*spline_sigma_00,*spline_sigma_01,
*spline_sigma_10,*spline_sigma_11,*spline_b_0,*spline_b_1,*spline_gradient,*spline_psi,*spline_varphi,
*spline_distance;
gsl_interp_accel *xacc_u,*xacc_g,*xacc_p,*xacc_f,*xacc_c,*xacc_sigma_00,*xacc_sigma_01,*xacc_sigma_10,
*xacc_sigma_11,*xacc_b_0,*xacc_b_1,*xacc_gradient,*xacc_psi,*xacc_varphi,*xacc_distance, *yacc_u,
*yacc_g,*yacc_p,*yacc_f,*yacc_c,*yacc_sigma_00,*yacc_sigma_01,*yacc_sigma_10, *yacc_sigma_11,*yacc_b_0,
*yacc_b_1,*yacc_gradient,*yacc_psi,*yacc_varphi,*yacc_distance;
inline double LUT_u(Eigen::Vector2d position,double t){
        return gsl_spline2d_eval(spline_u, position(0), position(1), xacc_u, yacc_u);
};
inline double LUT_g(Eigen::Vector2d position,double t){
        return gsl_spline2d_eval(spline_g, position(0), position(1), xacc_g, yacc_g);
};
inline double LUT_f(Eigen::Vector2d position,double t){
        return gsl_spline2d_eval(spline_f, position(0), position(1), xacc_f, yacc_f);
};
inline double LUT_c(Eigen::Vector2d position,double t){
        return gsl_spline2d_eval(spline_c, position(0), position(1), xacc_c, yacc_c);
};
inline double LUT_psi(Eigen::Vector2d position, Eigen::Vector2d normal, double t){
        return gsl_spline2d_eval(spline_psi, position(0), position(1), xacc_psi, yacc_psi);
};
inline double LUT_varphi(Eigen::Vector2d position, Eigen::Vector2d normal, double t){
        return gsl_spline2d_eval(spline_varphi, position(0), position(1), xacc_varphi, yacc_varphi);
};
inline Eigen::Vector2d LUT_b(Eigen::Vector2d position,double t){
        return Eigen::Vector2d(
        gsl_spline2d_eval(spline_b_0, position(0), position(1), xacc_b_0, yacc_b_0)
        ,gsl_spline2d_eval(spline_b_1, position(0), position(1), xacc_b_1, yacc_b_1));
};
inline Eigen::Matrix2d LUT_sigma(Eigen::Vector2d position,double t){
         Eigen::Matrix2d m;
         m << gsl_spline2d_eval(spline_sigma_00, position(0), position(1), xacc_sigma_00, yacc_sigma_00)
        ,gsl_spline2d_eval(spline_sigma_01, position(0), position(1), xacc_sigma_01, yacc_sigma_01)
        ,gsl_spline2d_eval(spline_sigma_10, position(0), position(1), xacc_sigma_10, yacc_sigma_10)
        ,gsl_spline2d_eval(spline_sigma_11, position(0), position(1), xacc_sigma_11, yacc_sigma_11);
        return m;
};
inline Eigen::Vector2d LUT_gradient(Eigen::Vector2d position,double t){
        return Eigen::Vector2d(
        gsl_spline2d_eval_deriv_x(spline_gradient, position(0), position(1), xacc_gradient, yacc_gradient)
        ,gsl_spline2d_eval_deriv_y(spline_gradient, position(0), position(1), xacc_gradient, yacc_gradient));
};
//Initialices look up table given a set of arrays
void Init_LUT(unsigned int Nx, unsigned int Ny, double *x, double *y, double *value, 
gsl_spline2d *spline, gsl_interp_accel *xaccel, gsl_interp_accel *yaccel){
    spline = gsl_spline2d_alloc(gsl_interp2d_bicubic, Nx, Ny);
    xaccel = gsl_interp_accel_alloc();
    yaccel = gsl_interp_accel_alloc();
    gsl_spline2d_init(spline, x, y, value, Nx, Ny);
    gsl_set_error_handler_off();
}
//Initialices look up table given a folder where the LUT is stored
void Init_LUT(std::string folder, gsl_spline2d *spline, gsl_interp_accel *xaccel,
gsl_interp_accel *yaccel){
    //x and len are dinamically allocated
    double** x;
    x = new double*[2];
    unsigned int len[2];
    unsigned int  count = 0;
    std::ifstream infile;
    std::string aux,line;
    
    for(int i = 0; i < 2; i++){
        //We charge the grid coordinates
        aux = folder + "/x_" + std::to_string(i) + ".txt";
        //We check if the file actually exist
        infile.open(aux,std::ios::in);
        if(! infile){
            std::cout << aux << " couldn't be opened.\n Make sure "<< 
            "x_" + std::to_string(i) + ".txt" << " is available in " <<
            folder << std::flush;
            std::terminate();
        }
        //The file is ridden
        while (getline( infile, line )){
            count++;
        }
        len[i] = count;
        count = 0;
        x[i] = new double[len[i]];
        infile.clear();
        infile.seekg(0, std::ios::beg);
        while (getline( infile, line )){
            x[i][count] = std::stod(line);
            count++;
        }
        count = 0;
        infile.close();
    }
    //The file which stores the values of the function in the grid is open
    infile.open(folder + "/value.txt");
    if(! infile){
        std::cout << folder + "/value.txt" << " couldn't be opened.\n Make sure "<< 
        "value.txt" << " is available in " <<folder << std::flush;
        std::terminate();
    }
    std::string cent;
    std::string::iterator it;
    
    int counter[2];
    int mesh_size = 1;
    for (int i = 0; i < 2; i++){
        mesh_size *= len[i];
        counter[i] = 0;
    }
    //It's values are stored in z
    double *z;
    z = new double[mesh_size];
    //std::cout << "len[0] " << len[0] << " len[1] " << len[1] << " mesh_size "<< mesh_size << std::endl;
    while (getline( infile, line )){
        
        it = line.begin();
        do{
            while(*it == ' '){

                it++;

            }
            while( *it != ' ' && it != line.end()){

                cent += *it;
                it ++;
            }
            //std::cout << cent << std::endl;
            z[counter[1] * len[0] + counter[0]] = std::stod(cent);
            counter[0]++;
            cent.clear();
            } while(it != line.end());

        counter[1]++;//row number
        counter[0] = 0;//column number
        
    }
    //Then the spline is initialized
    infile.close();
    Init_LUT(len[0],len[1],x[1],x[2],z,spline,xaccel,yaccel);
}