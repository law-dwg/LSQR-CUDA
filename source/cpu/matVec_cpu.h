#pragma once
#include <vector>
#include <iostream>
#include <iterator>
#include <algorithm>

class Vector_CPU {    
    public:
        //attributes
        unsigned int rows;
        unsigned int columns;
        std::vector<double> mat;
        double* h_mat;
        //constructors and destructor
        Vector_CPU(){};
        Vector_CPU(unsigned int p):columns(1),rows(p){this->mat.resize(p,5.0);};
        Vector_CPU(unsigned int r, unsigned int c):rows(r),columns(c){this->mat.resize(r*c,5);};
        Vector_CPU(unsigned int r, unsigned int c, double *v):rows(r),columns(c){
            mat.resize(r*c);
            mat.assign(v,v+r*c);
            h_mat = &mat[0];            
        };
        Vector_CPU(const Vector_CPU &v):columns(v.columns),rows(v.rows),mat(v.mat){};
        ~Vector_CPU(){};

        //operator overloads
        Vector_CPU operator*(Vector_CPU &v);
        Vector_CPU operator*(double i);
        Vector_CPU operator-(Vector_CPU v);
        Vector_CPU operator+(Vector_CPU v);
        double operator()(unsigned int i);
        double operator()(unsigned int r, unsigned int c);
        double operator[](unsigned int i);

        //member functions
        std::vector<double> getMat(){return this->mat;};
        void h_print();
        void print();
        int getRows();
        int getColumns();
        double Dnrm2();
        double normalNorm();
        Vector_CPU transpose();
};

//create matrix class for readability / sparsity attribute
class Matrix_CPU : public Vector_CPU{
    public:
        double sparsity=.70; //the number of 0-elements/non-0-elements
        Matrix_CPU(unsigned int r, unsigned int c);

};