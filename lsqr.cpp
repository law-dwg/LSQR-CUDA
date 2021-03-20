#include "lsqr.h"
#include <iostream>

int lsqr(Matrix &A, Vector &b){
    int itn = 0;
    double beta = b.Dnrm2();
    Vector u = b*(1/beta);
    Vector v = A.transpose() * u;
    std::cout<<beta<<std::endl;
    return 0;
}