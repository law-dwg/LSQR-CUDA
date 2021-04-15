#pragma once

class Vector_GPU {
    public:
        //attributes
        unsigned int rows;
        unsigned int columns;
        double * mat;
        //constructors and destructor
        Vector_GPU(){};
        Vector_GPU(unsigned int r, unsigned int c, double * m){
            cudaMalloc((void**)&mat,sizeof(double)*r*c);
            cudaMemcpy(mat,m,sizeof(double)*r*c,cudaMemcpyHostToDevice);
        };
        /*Vector_GPU(unsigned int p):columns(1),rows(p){this->mat.resize(p,5.0);};
        Vector_GPU(unsigned int r, unsigned int c):rows(r),columns(c){this->mat.resize(r*c,5);};
        Vector_GPU(unsigned int r, unsigned int c, double v){this->rows=r;this->columns=c;mat.resize(r*c,v);};
        Vector_GPU(const Vector_GPU &v):columns(v.columns),rows(v.rows),mat(v.mat){
        
        };*/
        ~Vector_GPU(){};
        /*
        //operator overloads
        Vector_GPU operator*(Vector_GPU &v);
        Vector_GPU operator*(double i);
        Vector_GPU operator-(Vector_GPU v);
        Vector_GPU operator+(Vector_GPU v);
        double operator()(unsigned int i);
        double operator()(unsigned int r, unsigned int c);
        double operator[](unsigned int i);

        //member functions
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