#include <iostream>
#include <eigen3/Eigen/Core>
#include <vector>
#include <string>
#include <fstream>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
/*Variables and functions that manage with LUT in BVPs*/
/*LUT auxiliary global variables. This part only works with 2D problems*/
__host__ void Init_tex_LUT(unsigned int width, unsigned int height,  float* h_data, cudaTextureObject_t &texObj,cudaArray_t* cuArray){
  // Allocate CUDA array in device memory
cudaChannelFormatDesc channelDesc =
cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

cudaMallocArray(&cuArray, &channelDesc, width, height);
// Set pitch of the source (the width in memory in bytes o
const size_t spitch = width*sizeof(float);
// Copy data located at address h_data in host memory to device memory
//float h_data2[width*height];
//for(int i=0;i<width;i++){
  //for(int j=0; j<height;j++){
    //  h_data2[width*j+i]=h_data[height*i+j];
   //}
//}
cudaMemcpy2DToArray(cuArray, 0, 0, h_data, spitch, width * sizeof(float),
height, cudaMemcpyHostToDevice);
// Specify texture
struct cudaResourceDesc resDesc;
memset(&resDesc, 0, sizeof(resDesc));
resDesc.resType = cudaResourceTypeArray;
resDesc.res.array.array = cuArray;
// Specify texture object parameters
struct cudaTextureDesc texDesc;
memset(&texDesc, 0, sizeof(texDesc));
texDesc.addressMode[0] = cudaAddressModeClamp;
texDesc.addressMode[1] = cudaAddressModeClamp;
texDesc.addressMode[2] = cudaAddressModeClamp;
texDesc.filterMode = cudaFilterModeLinear;
texDesc.readMode = cudaReadModeElementType;
texDesc.normalizedCoords = 0;
// Create texture object
texObj = 0;
cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
}
//Initialices look up table given a folder where the LUT is stored
__host__ void Init_LUT(std::string folder,  float* z){
    std::ifstream infile;
    std::string aux,line;
    int len[2]={64,128};
    //The file which stores the values of the function in the grid is open
    infile.open(folder+"value.txt");
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


    }
    //Then the spline is initialized
    infile.close();
}
