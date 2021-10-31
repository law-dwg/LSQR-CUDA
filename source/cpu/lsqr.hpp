#pragma once
#include <limits.h>
#include <math.h>
#include <stdio.h>

double D2Norm(double a, double b);
template <typename Mat, typename Vec> Vec lsqr(Mat &A, Vec &b) {
  /** LSQR-alg */

  /** Input - Dense Vector - b */
  const double bnorm = b.Dnrm2();

  /** Input - Sparse Matrix - A */
  Mat A_T = A.transpose();
  double Anorm = 0; // Approximation of Frobenius norm of Abar
  double Acond = 0; // Condition number of Abar

  /** Output - Dense Vector - x */
  Vec x(A.getColumns(), 1);
  double xnorm = 0;
  double xnorm1 = 0;

  /** Optional - Damping Coeff - Damped least squares */
  bool damped = true;
  double damp = 0;
  double dampsq = damp * damp;

  /** Const */
  const double one = 1.0;
  const double negone = -1.0;
  const double zero = 0.0;

  /** Precision */
  double epsilon = 1e-16;

  /** Tolerances */
  double ctol = zero;
  double rtol = zero;
  double atol = 1e-8;  // Optional
  double btol = 1e-8;  // Optional
  double conlim = 1e8; // Optional
  if (conlim > 0) {
    ctol = 1 / conlim;
  };

  /** QR-Factorization Params */
  double dnorm = zero;
  double z = zero;
  double sn2 = zero;
  double cs2 = negone;
  double psi = zero;

  /** Stopping Criteria */
  unsigned istop = zero;
  double test1 = zero;
  double test2 = zero;
  double test3 = zero;
  unsigned itn = 0; // Iteration
  int itnlim = 2 * A.getColumns();

  /** Residuals */
  double res2 = zero;

  /*1. Initialize*/
  Vec u, v, w, dk;
  double alpha = zero;
  double beta = bnorm;
  if (beta > 0) {
    u = b * (one / beta);
    v = A_T * u;
    alpha = v.Dnrm2();
  } else {
    v = x;
    alpha = zero;
  };

  if (alpha > 0) {
    v = v * (one / alpha);
  };
  w = v;

  // Norms
  double rhobar = alpha;
  double phibar = beta;
  double rnorm = beta;
  double Arnorm = alpha * beta;
  if (Arnorm == 0) {
    printf("Exact solution is x = 0\n");
    return x;
  }

  // 2. For i=1,2,3....
  do {
    if (A.getRows() >= 6000) {
      if ((itn == (0.125) * itnlim) || (itn == 0.25 * itnlim) || (itn == (0.25 + 0.125) * itnlim) || (itn == 0.5 * itnlim) ||
          (itn == (0.5 + 0.125) * itnlim) || (itn == 0.75 * itnlim) || (itn == (0.75 + 0.125) * itnlim)) {
        printf("itn = %d\n", itn);
      }
    } else {
      if ((itn == 0.25 * itnlim) || (itn == 0.5 * itnlim) || (itn == 0.75 * itnlim)) {
        printf("itn = %d\n", itn);
      }
    }
    ++itn;

    // 3. Continue the bidiagonialization
    /* Important equations for understanding.
    ubar_i+1 = beta_i+1 * u_i+1 = (A*v_i) - (alpha_i*u_i)
    beta_i+1 = ||ubar_i+1||
    u_i+1 =  ubar_i+1 * (1/beta_i+1)
    vbar_i+1 = alpha_i+1 * v_i+1 = (A_t*u_i+1) - (beta_i*v_i)
    alpha_i+1 = ||vbar_i+1||
    v_i+1 =  vbar_i+1 * (1/alpha_i+1)
    */
    u = A * v - u * alpha; // ubar_i+1
    beta = u.Dnrm2();      // beta_i+1 = ||ubar_i+1||
    if (beta > 0) {
      u = u * (one / beta); // u_i+1
      Anorm = sqrt((Anorm * Anorm) + (alpha * alpha) + (beta * beta) + dampsq);
      v = (A_T * u) - (v * beta); // vbar_i+1
      alpha = v.Dnrm2();          // alpha_i+1
      if (alpha > 0) {
        v = v * (one / alpha); // v_i+1
      }
    }
    double rhobar1 = rhobar;
    if (damped) {
      rhobar1 = D2Norm(rhobar, damp);
      double cs1 = rhobar / rhobar1;
      double sn1 = damp / rhobar1;
      psi = sn1 * phibar;
      phibar = cs1 * phibar;
    }

    // 4. Construct and apply next orthogonal transformation
    double rho = D2Norm(rhobar1, beta); // rho_i
    double cs = rhobar1 / rho;          // c_i
    double sn = beta / rho;             // s_i
    double theta = sn * alpha;          // theta_i+1
    rhobar = -cs * alpha;               // rhobar_i+1
    double phi = cs * phibar;           // phi_i = c_i*phibar_i
    phibar = sn * phibar;               // phibar_i+1 = s_i*phibar_i
    // used for stopping critera
    double tau = sn * phi;

    // 5. Update x,w
    // save values for stopping criteria
    double t1 = phi / rho;
    double t2 = -theta / rho;
    double t3 = one / rho;
    double dknorm = zero;
    dk = w * t3;

    /* Important equations
    x_i = x_i-1 + (phi_i/rho_i) *w_i
    w_i+1 = v_i+1 - (theta_i+1/rho_i)*w_i
    */
    x = x + w * t1;
    w = v + w * t2;
    double dkdnrm2 = dk.Dnrm2();
    dknorm = dkdnrm2 * dkdnrm2 + dknorm;
    dknorm = sqrt(dknorm);
    dnorm = D2Norm(dnorm, dknorm);

    double delta = sn2 * rho;
    double gambar = -cs2 * rho;
    double rhs = phi - delta * z;
    double zbar = rhs / gambar;
    xnorm = D2Norm(xnorm1, zbar);
    double gamma = D2Norm(gambar, theta);
    cs2 = gambar / gamma;
    sn2 = theta / gamma;
    z = rhs / gamma;
    xnorm1 = D2Norm(xnorm1, z);

    // residual
    // Vec res_v = b - (A * x);
    // double res = res_v.Dnrm2();

    Acond = Anorm * dnorm;
    res2 = D2Norm(res2, psi);
    rnorm = D2Norm(res2, phibar);
    rnorm += 1e-30;
    Arnorm = alpha * std::fabs(tau);

    // 6. Test for convergence
    test1 = rnorm / bnorm;
    test2 = Arnorm / (Anorm * rnorm + epsilon);
    test3 = one / (Acond + epsilon);
    t1 = test1 / (one + Anorm * xnorm / bnorm);
    rtol = btol + atol * Anorm * xnorm / bnorm;
    t3 = one + test3;
    t2 = one + test2;
    t1 = one + t1;
    if (itn >= itnlim) {
      istop = 7;
    }
    if (t3 <= one) {
      istop = 6;
    };
    if (t2 <= one) {
      istop = 5;
    };
    if (t1 <= one) {
      istop = 4;
    };
    //  Allow for tolerances set by the user.
    if (test3 <= ctol) {
      istop = 3;
    };
    if (test2 <= atol) {
      istop = 2;
    };
    if (test1 <= rtol) {
      istop = 1;
    };
  } while (istop == 0);
  printf("ran through %d iterations \nistop=%d\n", itn, istop);
  return x;
}