#include <vector>
#include <cstdio>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <algorithm>
#include <cassert>

#include "matVec.h"

//constructors
Matrix::Matrix(unsigned int r, unsigned int c){
    this->rows=r; 
    this->columns=c;
    this->mat.resize(this->rows*this->columns);
    int zeros = round(this->sparsity * r*c);
    int nonZeros = r*c - zeros;
    //std::cout<<zeros<<std::endl;
    srand(time(0));
    
    //printf("%f sparsity for %i elements leads to %f zero values which rounds to %d. That means there are %d nonzero values\n",this->sparsity,r*c,this->sparsity * r*c,zeros,nonZeros);
    //printf("mat size: %i\n",mat.size());
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

//operator overloads
Vector Vector::operator*(Vector &v){
    std::vector<double> lhs = this->mat;
    std::vector<double> rhs = v.getMat();
    Vector out(this->rows);
    //std::cout<<lhs.size()<<std::endl;
    //std::cout<<rhs.size()<<std::endl;
    if(this->columns==rhs.size()){
        for(int r =0; r<this->rows; r++){
            double sum = 0; 
            for(int c=0; c<this->columns; c++){
                sum += lhs[r*this->columns + c]*rhs[c];
            }
            out.mat[r]=sum;
        }
        return out;
    }
    else{
        printf("cannot perform this multiplication\n");
        return v;
    }
};

Vector Vector::operator*(double i){
    //probably can find a better implementation
    Vector out=*this;
    for (int e = 0; e<out.mat.size();e++){
        out.mat[e]*=i;
    }
    return out;
};

Vector Vector::operator-(Vector v){
    Vector out(this->rows,this->columns);
    if(out.rows == v.rows && out.columns == v.columns){
        for(int i = 0; i<this->mat.size(); i++){
            out.mat[i] = this->mat[i] - v.mat[i];
        }
        return out;
    }
    else{
        printf("cannot perform this subtraction, vectors need to be the same size");
        return (*this);
    };
};

Vector Vector::operator+(Vector v){
    if(this->mat.size()==0){
        Vector out(v.rows,v.columns,0);
        return out;
    }
    else if(this->rows == v.rows && this->columns == v.columns){
        Vector out(this->rows,this->columns);
        for(int i = 0; i>this->mat.size(); i++){
            out.mat[i] = this->mat[i] + v.mat[i];
        }
        return out;
    }
    else{
        printf("out.mat.size() = %i ",this->mat.size());
        printf("cannot perform this addition, vectors need to be the same size");
        return (*this);
    };
};

double Vector::operator()(unsigned int i){
    return(Vector::operator[](i));
};

double Vector::operator()(unsigned int r, unsigned int c){
    if (r < this->rows && c < this->columns){
        return(mat[r*this->columns + c]);
    }
    else{
        
        printf("please use valid indices: (r,c) where 0=<r<%i and 0=<c<%i\n",this->rows,this->columns);
        //throw 505;
        return EXIT_FAILURE;
    }
    
};

double Vector::operator[](unsigned int i){
    return(this->mat[i]);
};

//member functions
void Vector::print(){
    printf("#ofRows:%i #ofCols:%i\n",this->rows,this->columns);
    printf("PRINTING MATRIX\n[");
    for (int e=0; e<(this->mat.size()); e++){
        if (e%this->columns == 0){
            (e==0)?:printf("\n ");
        };
        (e==mat.size()-1)?printf("%f",this->mat[e]):printf("%f ",this->mat[e]);
    }
    printf("]\n");
};

int Vector::getRows(){
    //printf("number of rows: %i\n",this->rows);
    return this->rows;
};

int Vector::getColumns(){
    //printf("number of columns: %i\n",this->columns);
    return this->columns;
};

Vector Vector::transpose(){
    Vector out = (*this);
    out.rows = this->columns;
    out.columns = this->rows;
    for(int r=0; r<this->rows; r++){
        for(int c=0; c<this->columns; c++){
            //printf("new index: %i, old index: %i\n",(r+c*this->rows),(c+r*this->columns));
            out.mat[r+c*this->rows]=this->mat[c+r*this->columns];
        }
    };
    return out;
};

double Vector::Dnrm2(){
    double sumScaled = 1.0;
    double magnitudeOfLargestElement=0.0;
    for(int i=0;i<this->mat.size();i++){
        if(this->mat[i]!=0){
            double value = this->mat[i];
            double absVal = std::abs(value);
            if(magnitudeOfLargestElement<absVal){
                //rescale sum to the range of the new element
                value = magnitudeOfLargestElement /absVal;
                sumScaled=sumScaled*(value*value)+1.0;
                magnitudeOfLargestElement=absVal;
            }
            else{
                //rescale the new element to the range of the snum
                value = absVal /magnitudeOfLargestElement;
                sumScaled+=value*value;
            }
        }
    }
    //printf("sumScaled: %f, magOfLargestEle: %f\n",sumScaled,magnitudeOfLargestElement);
    return magnitudeOfLargestElement*sqrt(sumScaled);
};

double Vector::normalNorm(){
    double sumScaled = 0;
    for(int i=0;i<this->mat.size();i++){
        if(this->mat[i]!=0){
            double value = this->mat[i];
            sumScaled +=  value * value;
        }
    }
    return sqrt(sumScaled);
};