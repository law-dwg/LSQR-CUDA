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

void __global__ scale(double * input, double scalar,double * output){
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
    printf("thread(%d,%d,%d), block(%d,%d,%d), bid=%d, gid=%d, value=%f\n",threadIdx.x,threadIdx.y,threadIdx.z,
        blockIdx.x,blockIdx.y,blockIdx.z,bid,gid,input[gid]);
    output[gid] = input[gid] * scalar;
}

void __global__ multiply(double * in1, double * in2, double * output){
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
    printf("thread(%d,%d,%d), block(%d,%d,%d), bid=%d, gid=%d, %f * %f\n",threadIdx.x,threadIdx.y,threadIdx.z,
        blockIdx.x,blockIdx.y,blockIdx.z,bid,gid,in1[gid],in2[gid]);
    output[gid] = in1[gid] * in2[gid];


}

int main(){
    
    //host
    int array_size = 6;
    int byte_size = sizeof(double) * array_size;
    double *h_input1 = new double [array_size];
    double *h_input2 = new double [array_size];
    for (int i = 0; i < array_size; i++){
        h_input1[i]=i;
        h_input2[i]=i;
    }
    Vector_GPU h_i1(array_size,1,h_input1);
    double *h_output = new double [array_size];
    
    //device
    double *d_input1,*d_input2,*d_output;
    
    cudaMalloc((void**)&d_input1,byte_size);
    cudaMalloc((void**)&d_input2,byte_size);
    cudaMalloc((void**)&d_output,byte_size);
    cudaMemcpy(d_input1,h_input1,byte_size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_input2,h_input2,byte_size,cudaMemcpyHostToDevice);

    int nx = 2; //x thread dimension
    int ny = 3; //y thread dimension
    int nz = 1; //z thread dimension
    
    dim3 grid(1,1,1);
    dim3 block(nx/grid.x,ny/grid.y,nz/grid.z);

    scale <<<grid,block>>> (d_input1,5,d_input1);
    cudaDeviceSynchronize(); //wait for GPU to finish
    multiply <<<grid,block>>> (d_input1,d_input2,d_output);
    
    //cudaDeviceSynchronize(); //wait for GPU to finish
    cudaMemcpy(h_output,d_output,byte_size,cudaMemcpyDeviceToHost);
    
    for (int i = 0; i<6; i++){
        std::cout<<h_output[i]<<std::endl;
    }

    

    delete h_input1, h_input2, h_output, d_input1, d_input2, d_output;
    
    return 0;
}