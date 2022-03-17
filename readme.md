Currently being developed
To compile
-with intel compiler (C++, MPI and OpenMP)
    mpiicpc -o main -fopenmp -g -lgsl -lgslcblas main.cpp Meshes/subdomain.cpp Meshes/stencil.cpp Meshes/interface.cpp PDDAlgorithms/PDDSparseGM.cpp  -lm
- with xl compiler (C++, MPI and OpenMP)
    xlc++_r -qsmp=omp -std=gnu++11  -g -I$GSL_INCLUDE -I/m100/home/userexternal/jmoronvi -I$BOOST_INCLUDE -o  main  main.cpp  Meshes/subdomain.cpp Meshes/stencil.cpp Meshes/interface.cpp PDDAlgorithms/PDDSparseGM.cpp -L$GSL_LIB -lm -lgsl -lgslcblas


