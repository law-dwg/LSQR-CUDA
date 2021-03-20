#include "lsqr.h"
#include <iostream>
#include <math.h>

double D2Norm(double a, double b){
    const double scale = std::abs(a) + std::abs(b);
    const double zero = 0.0;

    if( scale == zero )
    {
    return zero;
    }

    const double sa = a / scale;
    const double sb = b / scale;

    return scale * sqrt( sa * sa + sb * sb );
};

int lsqr(Matrix &A, Vector &b){
    //1. Initialize
    int itn = 0;
    Vector u;
    Vector v;
    Vector w;
    Vector x(A.getColumns(),1,0);
    double alpha;
    double beta = b.Dnrm2();
    if (beta>0){
        u = b * (1/beta);
        v = A.transpose() * u;
        alpha = v.Dnrm2();
        printf("does %f = ",alpha);
    };
    printf("%f?\n", alpha);
    if (alpha>0){
        v*(1/alpha);
        w=v;
    };
    /**
    double Arnorm = alpha * beta;
    
    double bnorm = beta;
    double rnorm = beta;

    double test1 = 0.0;
    double test2 = 0.0;

    double temp;
    double test3;
    double rtol;
    **/
    
    double rhobar = alpha;
    double phibar = beta;
    double rho, phi, c, s, theta, tau, res;
    unsigned int istop = 0;
    
    Vector res_v;
    double epsilon = 1e-10;
    //2. For i=1,2,3....
    printf("2. For i=1,2,3....\n");
    do{
        itn++;
        Vector A_T = A.transpose();
        
        //3. Continue the bidiagonialization
        printf("3. Continue the bidiagonialization\n");
        u = A*v - u*alpha;
        beta = u.Dnrm2();
        if(beta>0){
            u = u * (1/beta);
            v = (A_T * u) - (v * beta);
            alpha = v.Dnrm2();
            if (alpha>0){
                v = v * (1/alpha);
            }
        }

        //4. Construct and apply next orthogonal transformation
        printf("4. Construct and apply next orthogonal transformation\n");
        double rhbar1 = rhobar;

        rho = D2Norm( rhbar1, beta );
        c = rhbar1/rho;
        s = beta/rho;
        theta = s * alpha;
        rhobar = -c * alpha;
        phi = c * phibar;
        phibar = s * phibar;
        
        tau = s * phi;

        //5. Update x,w
        printf("5. Update x,w\n");
        //w.print();
        Vector test = w * (phi/rho);
        test.print();
        x.print();
        x = x + test;
        x.print();
        w = v - (w * (theta/rho));
        //w.print();
        res_v = A*x - b;
        res_v.print();
        res = res_v.Dnrm2();
        std::cout<<"\nres: "<<res<<" iter: "<<itn<<std::endl;
        if (res < epsilon){
            istop=1;
            std::cout<<"STOPPED"<<std::endl;
        }
    }while(istop==0);
    
    return 0;
}