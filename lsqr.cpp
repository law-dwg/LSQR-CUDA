#include <vector>
#include <cstdio>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "lsqr.h"
#include <math.h>

Matrix::Matrix(unsigned int r, unsigned int c){
    this->rows = r;
    this->columns = c;
    this->mat.resize(r);
    int zeros = round(this->sparsity * r*c);
    int nonZeros = r*c - zeros;
    std::cout<<zeros<<std::endl;
    printf("%f sparsity for %i elements leads to %f zero values which rounds to %d. That means there are %d nonzero values\n",this->sparsity,r*c,this->sparsity * r*c,zeros,nonZeros);
    for (int i = 0; i < mat.size(); i++){
        mat[i].resize(c,0);
    }
    //NEED TO FIGURE OUT SPARSITY
    /**for(int r=0; r<this->rows; r++){
        for(int c=0; c<this->columns; c++){
            if(rand()%nonZeros==0){
                mat[r][c]=rand()%100;
            }
        }
    }**/
};

void Matrix::printMat(){
    printf("PRINTING MATRIX\n[");
    for (int r=0;r<(this->rows);r++){
        printf("[");
        for(int c=0;c<this->columns;c++){
            (c==this->columns-1)?printf("%f",this->mat[r][c]):printf("%f ",this->mat[r][c]);
            //std::cout<<this->mat[r][c]<<" ";
        }
        printf("]");
        (r==this->rows-1)?:printf("\n");
    }
    printf("]\n");
};

Matrix::Matrix(){
    printf("DEFAULT CONSTRUCTOR\n");
};

Matrix::~Matrix(){
    printf("DECONSTRUCTOR\n");
};

int Matrix::getRows(){
    printf("number of rows: %i\n",this->rows);
    return this->rows;
};