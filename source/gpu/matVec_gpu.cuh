#pragma once
#include <stdio.h> //NULL, printf
class Vector_GPU {
    public:
        //attributes
        unsigned int* rows;
        unsigned int* columns;
        double* d_mat;
        //constructors and destructor
        Vector_GPU(){};
        Vector_GPU(unsigned int* r, unsigned int * c, double * m){
            cudaMalloc((void**)&rows,sizeof(unsigned int));
            cudaMalloc((void**)&columns,sizeof(unsigned int));
            cudaMalloc((void**)&d_mat,sizeof(double)**r**c);

            cudaMemcpy(rows,&r,sizeof(unsigned int),cudaMemcpyHostToDevice);
            cudaMemcpy(columns,&c,sizeof(unsigned int),cudaMemcpyHostToDevice);
            cudaMemcpy(d_mat,m,sizeof(double)**r**c,cudaMemcpyHostToDevice);
        };
        //Vector_GPU(unsigned int p):columns(1),rows(p){this->mat.resize(p,5.0);};
        Vector_GPU(unsigned int* r, unsigned int* c){ 
            printf("CONSTRUCTOR\n");
            cudaMalloc((void**)&rows,sizeof(unsigned int));
            cudaMalloc((void**)&columns,sizeof(unsigned int));
            cudaMalloc((void**)&d_mat,sizeof(double)*r*c);

            cudaMemcpy(rows,&r,sizeof(unsigned int),cudaMemcpyHostToDevice);
            cudaMemcpy(columns,&c,sizeof(unsigned int),cudaMemcpyHostToDevice);
        };
        //Vector_GPU(unsigned int r, unsigned int c, double v){this->rows=r;this->columns=c;mat.resize(r*c,v);};
        Vector_GPU(const Vector_GPU &v){
            printf("COPY CONSTRUCTOR INVOKED\n");
            cudaMalloc((void**)&rows,sizeof(unsigned int));
            cudaMalloc((void**)&columns,sizeof(unsigned int));
            cudaMalloc((void**)&d_mat,sizeof(double)*v.columns*v.rows);

            cudaMemcpy(rows,&v.rows,sizeof(unsigned int),cudaMemcpyDeviceToDevice);
            cudaMemcpy(columns,&v.columns,sizeof(unsigned int),cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_mat,&m.d_mat,sizeof(double)*r*c,cudaMemcpyDeviceToDevice);
        };
        ~Vector_GPU(){
            cudaFree(d_mat);
            
        };
        
        //operator overloads
        Vector_GPU operator*(Vector_GPU &v);
        Vector_GPU operator*(double i);
        Vector_GPU& operator=(const Vector_GPU &v); //overwrite previous data
        
        Vector_GPU operator-(const Vector_GPU &v);
        Vector_GPU operator+(const Vector_GPU &v);
        /*double operator()(unsigned int i);
        double operator()(unsigned int r, unsigned int c);
        double operator[](unsigned int i);
        */
        //member functions

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