#pragma once
#include "../cpu/matVec_cpu.h"
#include <stdio.h> //NULL, printf
class Vector_GPU {
    public:
        //attributes
        unsigned int h_rows, h_columns;
        unsigned int  * d_rows, * d_columns;
        double* d_mat;
        //constructors and destructor
        Vector_GPU(){};
        Vector_GPU(unsigned int r, unsigned int c, double * m):h_rows(r),h_columns(c){        
            printf("CONSTRUCTOR #1 CALLED\n");
            printf("%d x %d\n",h_rows,h_columns);
            cudaMalloc((void**)&d_rows,sizeof(unsigned int));
            cudaMalloc((void**)&d_columns,sizeof(unsigned int));
            cudaMalloc((void**)&d_mat,sizeof(double)*r*c);

            cudaMemcpy(d_rows,&r,sizeof(int),cudaMemcpyHostToDevice);
            cudaMemcpy(d_columns,&c,sizeof(int),cudaMemcpyHostToDevice);
            cudaMemcpy(d_mat,m,sizeof(double)*r*c,cudaMemcpyHostToDevice);
        };
        //Vector_GPU(unsigned int p):columns(1),rows(p){this->mat.resize(p,5.0);};
        Vector_GPU(unsigned int r, unsigned int c):h_rows(r),h_columns(c){ 
            printf("CONSTRUCTOR #2 CALLED\n");
            cudaMalloc((void**)&d_rows,sizeof(unsigned int));
            cudaMalloc((void**)&d_columns,sizeof(unsigned int));
            cudaMalloc((void**)&d_mat,sizeof(double)*r*c);

            cudaMemcpy(d_rows,&r,sizeof(unsigned int),cudaMemcpyHostToDevice);
            cudaMemcpy(d_columns,&c,sizeof(unsigned int),cudaMemcpyHostToDevice);
        };
        //Vector_GPU(unsigned int r, unsigned int c, double v){this->rows=r;this->columns=c;mat.resize(r*c,v);};
        Vector_GPU(const Vector_GPU &v):h_rows(v.h_rows),h_columns(v.h_columns){
            printf("COPY CONSTRUCTOR INVOKED\n");
            cudaMalloc((void**)&d_rows,sizeof(unsigned int));
            cudaMalloc((void**)&d_columns,sizeof(unsigned int));
            cudaMalloc((void**)&d_mat,sizeof(double)*v.h_rows*v.h_columns);

            cudaMemcpy(d_rows,&v.d_rows,sizeof(unsigned int),cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_columns,&v.d_columns,sizeof(unsigned int),cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_mat,v.d_mat,sizeof(double)*v.h_columns*v.h_rows,cudaMemcpyDeviceToDevice);
        };
        ~Vector_GPU(){
            cudaFree(d_mat);
            cudaFree(d_rows);
            cudaFree(d_columns);
            
        };
        
        //operator overloads
        Vector_GPU operator*(Vector_GPU &v);
        Vector_GPU operator*(double i);
        /*
        Vector_GPU& operator=(const Vector_GPU &v); //overwrite previous data
        
        Vector_GPU operator-(const Vector_GPU &v);
        Vector_GPU operator+(const Vector_GPU &v);
        /*double operator()(unsigned int i);
        double operator()(unsigned int r, unsigned int c);
        double operator[](unsigned int i);
        */
        //member functions
        void printmat();
        Vector_CPU matDeviceToHost();
        /*
        std::vector<double> getMat(){return this->mat;};
        void print();
        int getRows();
        int getColumns();
        double Dnrm2();
        double normalNorm();
        Vector_GPU transpose();*/
};

//create matrix class for readability / sparsity attribute
class Matrix_GPU : public Vector_GPU{
public:
    double sparsity=.70; //the number of 0-elements/non-0-elements
    Matrix_GPU(unsigned int r, unsigned int c);

};