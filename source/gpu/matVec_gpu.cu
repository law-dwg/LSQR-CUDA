#include <stdio.h> //NULL, printf
#include <stdlib.h> //srand, rand
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <sstream>
#include <iostream>
#include <string.h>
#include <time.h>
#include "matVec_gpu.cuh"
//nvcc -arch=sm_37
//CUDA kernels
/*void __global__ scale(double * input, double scalar, double * output){
    const unsigned int bid = blockIdx.x //1D
        + blockIdx.y * gridDim.x //2D
        + gridDim.x * gridDim.y * blockIdx.z; //3D
    const unsigned int threadsPerBlock = blockDim.x
        *blockDim.y //2D
        *blockDim.z; //3D
    const unsigned int tid = threadIdx.x //1D
        +threadIdx.y*blockDim.x //2D
        +blockDim.x*blockDim.x*threadIdx.z; //3D
    const unsigned int gid = bid * threadsPerBlock + tid;
    //printf("thread(%d,%d,%d), block(%d,%d,%d), bid=%d, gid=%d, value=%f\n",threadIdx.x,threadIdx.y,threadIdx.z,
    //    blockIdx.x,blockIdx.y,blockIdx.z,bid,gid,input[gid]);
    output[gid] = input[gid] * scalar;
}*/

void __global__ multiply(double * in1, unsigned int * rows1, unsigned int * cols1, double * in2, unsigned int * rows2, unsigned int * cols2, double * output){
    const unsigned int bid = blockIdx.x //1D
        + blockIdx.y * gridDim.x //2D
        + gridDim.x * gridDim.y * blockIdx.z; //3D
    const unsigned int threadsPerBlock = blockDim.x
        *blockDim.y //2D
        *blockDim.z; //3D
    const unsigned int tid = threadIdx.x //1D
        +threadIdx.y*blockDim.x //2D
        +blockDim.x*blockDim.x*threadIdx.z; //3D
    const unsigned int gid = bid * threadsPerBlock + tid;
    const unsigned int r = blockIdx.y * blockDim.y + threadIdx.y; // the row of M1
    const unsigned int c = blockIdx.x * blockDim.x + threadIdx.x; // the col of M2
    //printf("thread(%d,%d,%d), block(%d,%d,%d), bid=%d, gid=%d, %f * %f\n",threadIdx.x,threadIdx.y,threadIdx.z,
    //    blockIdx.x,blockIdx.y,blockIdx.z,bid,gid,in1[gid],in2[gid]);
    double sum = 0;
    //printf("gid:%i, %i %i %i %i\n",gid,*rows1, *cols1, *rows2, *cols2);
    if (*cols1 == *rows2){
        printf("row: %i \n",r);
        for (int i = 0; i < *rows1; i++){
            sum += in1[r * *cols1 + i] * in2[i* *cols2 + c];
        }
        output[r**cols2+c] = sum;
    }
    else{
        printf("MATRICIES CANNOT BE MULTIPLED, INVALID SIZES");
    }
    
    
}
/*
void __global__ assignment(double * in1, double * in2){
    const unsigned int bid = blockIdx.x //1D
        + blockIdx.y * gridDim.x //2D
        + gridDim.x * gridDim.y * blockIdx.z; //3D
    const unsigned int threadsPerBlock = blockDim.x
        *blockDim.y //2D
        *blockDim.z; //3D
    const unsigned int tid = threadIdx.x //1D
        +threadIdx.y*blockDim.x //2D
        +blockDim.x*blockDim.x*threadIdx.z; //3D
    const unsigned int gid = bid * threadsPerBlock + tid;
    printf("thread(%d,%d,%d), block(%d,%d,%d), bid=%d, gid=%d, in1=%f, in2=%f\n",threadIdx.x,threadIdx.y,threadIdx.z,
        blockIdx.x,blockIdx.y,blockIdx.z,bid,gid,in1[gid],in2[gid]);
        in1[gid] = in2[gid];
    
    
}

void __global__ subtract(double * in1, double * in2, double * out){
    const unsigned int bid = blockIdx.x //1D
        + blockIdx.y * gridDim.x //2D
        + gridDim.x * gridDim.y * blockIdx.z; //3D
    const unsigned int threadsPerBlock = blockDim.x
        *blockDim.y //2D
        *blockDim.z; //3D
    const unsigned int tid = threadIdx.x //1D
        +threadIdx.y*blockDim.x //2D
        +blockDim.x*blockDim.x*threadIdx.z; //3D
    const unsigned int gid = bid * threadsPerBlock + tid;
    printf("thread(%d,%d,%d), block(%d,%d,%d), bid=%d, gid=%d, in1=%f, in2=%f\n",threadIdx.x,threadIdx.y,threadIdx.z,
        blockIdx.x,blockIdx.y,blockIdx.z,bid,gid,in1[gid],in2[gid]);
    out[gid] = in1[gid] - in2[gid];
}

void __global__ add(double * in1, double * in2, double * out){
    const unsigned int bid = blockIdx.x //1D
        + blockIdx.y * gridDim.x //2D
        + gridDim.x * gridDim.y * blockIdx.z; //3D
    const unsigned int threadsPerBlock = blockDim.x
        *blockDim.y //2D
        *blockDim.z; //3D
    const unsigned int tid = threadIdx.x //1D
        +threadIdx.y*blockDim.x //2D
        +blockDim.x*blockDim.x*threadIdx.z; //3D
    const unsigned int gid = bid * threadsPerBlock + tid;
    printf("thread(%d,%d,%d), block(%d,%d,%d), bid=%d, gid=%d, in1=%f, in2=%f\n",threadIdx.x,threadIdx.y,threadIdx.z,
        blockIdx.x,blockIdx.y,blockIdx.z,bid,gid,in1[gid],in2[gid]);
    out[gid] = in1[gid] + in2[gid];
}*/

//Operator overloads
Vector_GPU Vector_GPU::operator*(Vector_GPU &v){
    printf("MATMULT\n");
    Vector_GPU out(this->h_rows,v.h_columns);
    dim3 grid(1,1,1);
    dim3 block(out.h_rows,out.h_columns,1);
    multiply <<<grid,block>>> (this->d_mat, this->d_rows, this->d_columns, v.d_mat,  v.d_rows, v.d_columns, out.d_mat);
    return out;
}
/*
Vector_GPU Vector_GPU::operator*(double i){
    Vector_GPU out(this->rows,this->columns);
    dim3 grid(1,1,1);
    dim3 block(this->rows * this->columns,1,1);
    scale <<<grid,block>>> (this->d_mat,i,out.d_mat);
    return out;
}

Vector_GPU& Vector_GPU::operator=(const Vector_GPU &v){
    printf("Assignment operator called\n");
    this->rows = v.rows;
    this->columns = v.columns;
    dim3 grid(1,1,1);
    dim3 block(v.rows * v.columns,1,1);
    if (columns == v.rows){
        assignment <<<grid,block>>> (this->d_mat,v.d_mat);
        
    }
    else{
        printf("ARRAYS ARE NOT THE SAME SIZE, canot perform assignment operation\n");
    }
    return *this;
}

Vector_GPU Vector_GPU::operator-(const Vector_GPU &v){
    Vector_GPU out(this->rows,this->columns);
    dim3 grid(1,1,1);
    dim3 block(v.rows * v.columns,1,1);
    if (columns == v.rows){
        subtract <<<grid,block>>> (this->d_mat,v.d_mat,out.d_mat);
        
    }
    else{
        printf("ARRAYS ARE NOT THE SAME SIZE, canot perform operation\n");
    }
    return out;   
}

Vector_GPU Vector_GPU::operator+(const Vector_GPU &v){
    Vector_GPU out(this->rows,this->columns);
    dim3 grid(1,1,1);
    dim3 block(v.rows * v.columns,1,1);
    if (columns == v.rows){
        add <<<grid,block>>> (this->d_mat,v.d_mat,out.d_mat);
        
    }
    else{
        printf("ARRAYS ARE NOT THE SAME SIZE, canot perform operation\n");
    }
    return out;   
}*/