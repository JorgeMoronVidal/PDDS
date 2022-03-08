#ifndef RECTANGLE
#define RECTANGLE
#include <math.h>
#include <eigen3/Eigen/Core>
#include <vector>
#include <iostream>
double Rectangle2D(double* params, 
            Eigen::Vector2d & position, 
            Eigen::Vector2d & exitpoint,
            Eigen::Vector2d & normal);
bool Stopping(Eigen::Vector2d position);

#endif