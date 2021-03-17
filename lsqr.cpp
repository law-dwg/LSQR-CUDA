#include <vector>
#include <cstdio>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "lsqr.h"
#include <math.h>
#include <algorithm>
#include <cassert>
#include <iostream>

Matrix::Matrix(unsigned int r, unsigned int c):rows(r), columns(c){
    (r!=0 && c!=0)?:throw 505; //need to change this
    this->mat.resize(this->rows*this->columns);
    int zeros = round(this->sparsity * r*c);
    int nonZeros = r*c - zeros;
    std::cout<<zeros<<std::endl;
    srand(time(0));
    
    printf("%f sparsity for %i elements leads to %f zero values which rounds to %d. That means there are %d nonzero values\n",this->sparsity,r*c,this->sparsity * r*c,zeros,nonZeros);
    printf("mat size: %i\n",mat.size());
    for (int i = 0; i < mat.size(); i++){
        if(i < nonZeros){
            mat[i]=rand()%100;
        }
        else{
            mat[i]=0;
        };
        
    }
    std::random_shuffle(mat.begin(), mat.end());
};

void Matrix::printMat(){
    printf("PRINTING MATRIX\n[");
    for (int e=0; e<(this->mat.size()); e++){
        if (e%this->rows == 0){
            (e==0)?:printf("\n ");
        };
        (e==mat.size()-1)?printf("%f",this->mat[e]):printf("%f ",this->mat[e]);
    }
    printf("]\n");
};

Matrix::~Matrix(){
};

int Matrix::getRows(){
    printf("number of rows: %i\n",this->rows);
    return this->rows;
};

int Matrix::getColumns(){
    printf("number of columns: %i\n",this->columns);
    return this->columns;
};

double Matrix::operator[](unsigned int i){
    return(mat[i]);
};

double Matrix::operator()(unsigned int i){
    return(Matrix::operator[](i));
};

double Matrix::operator()(unsigned int r, unsigned int c){
    if (r < this->rows && c < this->columns){
        return(mat[r*this->columns + c]);
    }
    else{
        
        printf("please use valid indices: (r,c) where 0=<r<%i and 0=<c<%i\n",this->rows,this->columns);
        throw 505;
        return EXIT_FAILURE;
    }
    
};

Matrix Matrix::operator*(double i){
    //probably can find a better implementation
    Matrix out(*this);
    for (int e = 0; e<out.mat.size();e++){
        out.mat[e]*=i;
    }
    return out;
};