#pragma once
#include <vector>
#include <iostream>

class Vector {    
    public:
        //attributes
        unsigned int rows;
        unsigned int columns;
        std::vector<double> mat;
        //constructors and destructor
        Vector(){};
        Vector(unsigned int p):columns(1),rows(p){this->mat.resize(p,5.0);};
        Vector(unsigned int r, unsigned int c):rows(r),columns(c){this->mat.resize(r*c,5);};
        Vector(unsigned int r, unsigned int c, double v){this->rows=r;this->columns=c;mat.resize(r*c,v);};
        Vector(const Vector &v):columns(v.columns),rows(v.rows),mat(v.mat){};
        ~Vector(){};

        //operator overloads
        Vector operator*(Vector &v);
        Vector operator*(double i);
        Vector operator-(Vector v);
        Vector operator+(Vector v);
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
        Vector transpose();
};

//create matrix class for readability / sparsity attribute
class Matrix : public Vector{
    public:
        double sparsity=.70; //the number of 0-elements/non-0-elements
        Matrix(unsigned int r, unsigned int c);

};