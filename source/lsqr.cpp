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
    //Iteration
    unsigned int itn=0;
    //Stopping critera variables
    double ddnorm = 0;
    double atol = 1e-8;
    double btol = 1e-8;
    unsigned int istop = 0;
    double ctol = 0;
    double conlim = 1e8;
    if (conlim > 0){
        ctol = 1/conlim;
    };

    double Anorm = 0;
    double Acond = 0;
    double damp = 0;
    double dampsq = damp*damp;
    double dnorm = 0;
    double dknorm = 0;
    double res2 = 0;
    double xnorm = 0;
    double xxnorm = 0;
    double z = 0;
    double cs2 = -1;
    double sn2 = 0;
    double rtol;
    double rho, phi, c, s, theta, tau, res, res1;
    double res_old = 1e10;
    Vector res_v;
    double epsilon = 1e-16;
    
    /*1. Initialize*/
    Vector u;
    Vector v;
    Vector w;
    double alpha;
    Vector x(A.getColumns(),1,0);
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

    //Norms
    double rnorm = beta;
    double r1norm = rnorm;
    double r2norm = rnorm;
    double arnorm = alpha * beta;
    
    //2. For i=1,2,3....
    printf("2. For i=1,2,3....\n");
    do{
        itn++;
        Vector A_T = A.transpose();
        
        //3. Continue the bidiagonialization
        /* Important equations for understanding.
        
        ubar_i+1 = beta_i+1 * u_i+1 = (A*v_i) - (alpha_i*u_i) 
        beta_i+1 = ||ubar_i+1||
        u_i+1 =  ubar_i+1 * (1/beta_i+1)

        vbar_i+1 = alpha_i+1 * v_i+1 = (A_t*u_i+1) - (beta_i*v_i) 
        alpha_i+1 = ||vbar_i+1||
        v_i+1 =  vbar_i+1 * (1/alpha_i+1)
        */
        printf("3. Continue the bidiagonialization\n");
        u = A*v - u*alpha; //ubar_i+1
        beta = u.Dnrm2(); //beta_i+1 = ||ubar_i+1||
        if(beta>0){
            u = u * (1/beta);//u_i+1
            v = (A_T * u) - (v * beta);//vbar_i+1
            alpha = v.Dnrm2();//alpha_i+1 
            if (alpha>0){
                v = v * (1/alpha);//v_i+1
            }
        }

        //4. Construct and apply next orthogonal transformation
        printf("4. Construct and apply next orthogonal transformation\n");

        rho = D2Norm( rhobar, beta );//rho_i
        c = rhobar/rho;//c_i
        s = beta/rho;//s_i
        theta = s * alpha;//theta_i+1
        rhobar = -c * alpha;//rhobar_i+1
        phi = c * phibar;//phi_i = c_i*phibar_i
        phibar = s * phibar;//phibar_i+1 = s_i*phibar_i
        
        //used for stopping critera
        tau = s * phi;
        arnorm = alpha  * std::abs(tau);

        //5. Update x,w
        printf("5. Update x,w\n");
        //save values for stopping criteria
        Vector dk = w*(1/rho);
        dknorm = dk.Dnrm2()*dk.Dnrm2() + ddnorm;
        /* Important equations
        x_i = x_i-1 + (phi_i/rho_i) *w_i
        w_i+1 = v_i+1 - (theta_i+1/rho_i)*w_i
        */
        x = x + w * (phi/rho);
        w = v - (w * (theta/rho));
        //residual
        res_v = b - A*x;
        res = res_v.Dnrm2(); 

        //6. Test for convergence
        printf("6. Test for convergence\n");
        /*Test 1 for convergence
        stop if ||r|| =< btol*||b|| + atol*||A||*||x||
        */
        if (res <= (btol*b.Dnrm2() + atol*A.Dnrm2()*x.Dnrm2())){
            istop=1;
        }

        /*Test 2 for convergence
        stop if ||A_T*r||/||A||*||r|| <= atol
        */
        if (arnorm / (A.Dnrm2() * res) <= atol){
            istop=2;
        }

        /*Test 3 for convergence
        stop if cond(A) => conlim
        */
        Acond = A.Dnrm2() * sqrt(ddnorm);
        if (Acond < conlim){
            istop=3;
        }
        /*
        res1 = phibar*phibar;
        double test3 = 1/ (Acond + epsilon);
        std::cout.precision(25);
        std::cout<<std::fixed<<res<<std::endl;
        
        if(res <= btol * b.Dnrm2() + atol * A.Dnrm2() * x.Dnrm2());
        
        if(test3<=ctol){
            istop=3;
            printf("%i\n",itn);
        };
        double test2 = arnorm / (A.Dnrm2()*res + epsilon);
        if( 1 + test2 <= 1){
            istop=5;
            printf("%i\n",itn);
        };*/
        printf("iteration %i\n",itn);
        printf("istop %i\n",istop);
        x.print();
    }while(istop==0);
    
    return 0;
}