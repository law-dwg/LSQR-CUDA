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
    unsigned int itn=0;
    unsigned int istop = 0;
    double ctol = 0;
    double conlim = 0;
    if (conlim > 0){
        ctol = 1/conlim;
    };
    double Anorm = 0;
    double Acond = 0;
    double damp = 0;
    double dampsq = damp*damp;
    double ddnorm = 0;
    double res2 = 0;
    double xnorm = 0;
    double xxnorm = 0;
    double z = 0;
    double cs2 = -1;
    double sn2 = 0;

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
    }
    else{
        v = x;
        alpha=0;
    };

    if (alpha>0){
        v=v*(1/alpha);
        w=v;
    };
    double rhobar = alpha;
    double phibar = beta;
    double rnorm = beta;
    double r1norm = rnorm;
    double r2norm = rnorm;
    double arnorm = alpha * beta;
    double rtol;

    
    
    double rho, phi, c, s, theta, tau, res, res1;
    double res_old = 1e10;
    
    Vector res_v;
    double epsilon = 1e-16;
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
        arnorm = alpha  * std::abs(tau);
        //5. Update x,w
        printf("5. Update x,w\n");
        x = x + w * (phi/rho);
        Vector dk = w*(1/rho);
        ddnorm = dk.Dnrm2()*dk.Dnrm2();
        w = v - (w * (theta/rho));
        res_v = b - A*x;
        res = res_v.Dnrm2();

        //Test for convergence
        printf("6. Test for convergence\n");
        Acond = A.Dnrm2() * sqrt(ddnorm);
        res1 = phibar*phibar;
        double test3 = 1/ (Acond + epsilon);
        std::cout.precision(25);
        std::cout<<std::fixed<<res<<std::endl;
        if(test3<=ctol){
            istop=3;
            printf("%i\n",itn);
        };
        double test2 = arnorm / (A.Dnrm2()*res + epsilon);
        if( 1 + test2 <= 1){
            istop=5;
            printf("%i\n",itn);
        };
    }while(istop==0);
    
    return 0;
}