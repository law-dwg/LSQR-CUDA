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
#include "../cpu/matVec_cpu.h"
#define TILE_DIM 4
#define BLOCK_ROWS 4
//nvcc -arch=sm_37
//CUDA kernels

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
        //printf("row: %i \n",r);
        for (int i = 0; i < *rows1; i++){
            sum += in1[r * *cols1 + i] * in2[i* *cols2 + c];
        }
        output[r**cols2+c] = sum;
    }
    else{
        printf("MATRICIES CANNOT BE MULTIPLED, INVALID SIZES");
    }
}

void __global__ scale(double * input, double * scalar, double * output){
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
    output[gid] = input[gid] * *scalar;
    printf("%f = %f * %f\n",output[gid], input[gid], *scalar);
}

void __global__ print(double * input){
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
    printf("%f\n",input[gid]);
}

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

void __global__ subtract(double * in1, double * in2, double * output){
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
    output[gid] = in1[gid] - in2[gid];
    printf("%f = %f - %f\n",output[gid], in1[gid], in2[gid]);
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
    //printf("thread(%d,%d,%d), block(%d,%d,%d), bid=%d, gid=%d, in1=%f, in2=%f\n",threadIdx.x,threadIdx.y,threadIdx.z,
    //    blockIdx.x,blockIdx.y,blockIdx.z,bid,gid,in1[gid],in2[gid]);
    out[gid] = in1[gid] + in2[gid];
    printf("%f = %f + %f\n",out[gid], in1[gid], in2[gid]);
}

void __global__ transposer(double * in1, double * output){ //need this to be named differently from transpose in cpu functions
    
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
    /*
    source found here:
    https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/
    */
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS){
        output[x*width + (y+j)] = in1[(y+j)*width + x];
    }   

    
}

//Operator overloads
Vector_GPU Vector_GPU::operator*(Vector_GPU &v){
    printf("MATMULT\n");
    Vector_GPU out(this->h_rows,v.h_columns);
    dim3 grid(1,1,1);
    dim3 block(out.h_rows,out.h_columns,1);
    multiply <<<grid,block>>> (this->d_mat, this->d_rows, this->d_columns, v.d_mat,  v.d_rows, v.d_columns, out.d_mat);
    return out;
}

Vector_GPU Vector_GPU::operator*(double h_i){
    printf("scale\n");
    Vector_GPU out(this->h_rows,this->h_columns);
    dim3 grid(1,1,1);
    dim3 block(this->h_rows,this->h_columns,1);
    double *d_i;
    cudaMalloc((void**)&d_i,sizeof(double));
    cudaMemcpy(d_i,&h_i,sizeof(double),cudaMemcpyHostToDevice);
    
    scale <<<grid,block>>> (this->d_mat,d_i,out.d_mat);
    
    return out;
}

void Vector_GPU::printmat(){
    dim3 grid(1,1,1);
    printf("PRINTING\n");
    dim3 block(this->h_rows,this->h_columns,1);
    
    print <<<grid,block>>> (this->d_mat);
};

Vector_CPU Vector_GPU::matDeviceToHost(){
    printf("matDeviceToHost\n");
    double out[this->h_columns * this->h_rows];
    unsigned int rows;
    unsigned int cols;
    cudaMemcpy(&out,d_mat,sizeof(double)*this->h_columns*this->h_rows,cudaMemcpyDeviceToHost);
    cudaMemcpy(&rows,this->d_rows,sizeof(unsigned int),cudaMemcpyDeviceToHost);
    cudaMemcpy(&cols,this->d_columns,sizeof(unsigned int),cudaMemcpyDeviceToHost);
    std::cout<<"d_rows="<<rows<<"=h_rows="<<this->h_rows<<std::endl;
    std::cout<<"d_columns="<<cols<<"=h_columns="<<this->h_columns<<std::endl;
    if(rows != this->h_rows || cols != this->h_columns){
        printf("INCONSISTENT ROWS AND COLS BETWEEN HOST AND DEVICE\n");
    }
    Vector_CPU v_cpu(this->h_rows,this->h_columns, out);
    return v_cpu;
};

Vector_GPU& Vector_GPU::operator=(const Vector_GPU &v){
    printf("Assignment operator called\n");
    this->h_rows = v.h_rows;
    this->h_columns = v.h_columns;
    cudaFree(this->d_mat);
    cudaMalloc((void**)&d_mat,sizeof(double)*v.h_rows*v.h_columns);
    //dim3 grid(1,1,1);
    //dim3 block(v.rows * v.columns,1,1);
    cudaMemcpy(this->d_rows,v.d_rows,sizeof(unsigned int),cudaMemcpyDeviceToDevice);
    cudaMemcpy(this->d_columns,v.d_columns,sizeof(unsigned int),cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_mat,v.d_mat,sizeof(double)*v.h_columns*v.h_rows,cudaMemcpyDeviceToDevice);
    //assignment <<<grid,block>>> (this->d_mat,v.d_mat);
        
    return *this;
}

Vector_GPU Vector_GPU::operator-(const Vector_GPU &v){
    printf("SUBTRACT CALLED\n");
    Vector_GPU out(this->h_rows,this->h_columns);
    dim3 grid(1,1,1);
    std::cout<<v.h_rows<<"="<<this->h_rows<<std::endl;
    dim3 block(v.h_rows * v.h_columns,1,1);
    if (this->h_rows == v.h_rows && this->h_columns == v.h_columns){
        subtract <<<grid,block>>> (this->d_mat,v.d_mat,out.d_mat);   
    }
    else{
        printf("ARRAYS ARE NOT THE SAME SIZE, canot perform operation\n");
    }
    return out;   
}

Vector_GPU Vector_GPU::operator+(const Vector_GPU &v){
    Vector_GPU out(this->h_rows,this->h_columns);
    dim3 grid(1,1,1);
    dim3 block(v.h_rows * v.h_columns,1,1);
    if (this->h_rows == v.h_rows && this->h_columns == v.h_columns){
        add <<<grid,block>>> (this->d_mat,v.d_mat,out.d_mat);
        
    }
    else{
        printf("ARRAYS ARE NOT THE SAME SIZE, canot perform operation\n");
    }
    return out;
}

int Vector_GPU::getRows(){
    printf("number of rows: %i\n",this->h_rows);
    return this->h_rows;
};

int Vector_GPU::getColumns(){
    printf("number of columns: %i\n",this->h_columns);
    return this->h_columns;
};

Vector_GPU Vector_GPU::transpose(){
    Vector_GPU out(this->h_columns,this->h_rows);
    dim3 grid(1,1,1);
    dim3 block(this->h_rows,this->h_columns,1);
    this->printmat();
    int r = this->getRows();
    int c = this->getColumns();
    int r2 = out.getRows();
    int c2 = out.getColumns();
    transposer <<<grid,block>>> (this->d_mat,out.d_mat);
    return out;
};