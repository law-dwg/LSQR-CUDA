#include "lsqr.h"
#include <iostream>

int lsqr(Matrix &A, Vector &b){
    int itn = 0;
    Vector u;
    Vector v;
    Vector w;
    double alpha;
    double beta = b.Dnrm2();
    if (beta>0){
        u = b*(1/beta);
        v = A.transpose() * u;
        alpha = v.Dnrm2();
        printf("does %f = ",alpha);
    };
    printf("%f?\n", alpha);
    if (alpha>0){
        v*(1/alpha);
        w=v;
    };
    
    double Arnorm = alpha * beta;
    double rhobar = alpha;
    
    double phibar = beta;
    double bnorm = beta;
    double rnorm = beta;

    double test1 = 0.0;
    double test2 = 0.0;

    double temp;
    double test3;
    double rtol;
    unsigned int istop = 0;

    double epsilon = 1e-15;
    do{
        itn++;
        Vector A_T = A.transpose();
        //3. Bidiagonialization
        u = A*v - u*alpha;
        beta = u.Dnrm2();
        u = u*(1/beta);
        v = A_T*u - v*beta;

    
    
    }while(istop==1);
    
    return 0;
}