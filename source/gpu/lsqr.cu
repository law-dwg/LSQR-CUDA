#include "matVec_gpu.cuh"
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

void __global__ print(){
    
}

int main(){
    //Check Cuda Capabale Device
    int deviceCount;
    cudaGetDeviceCount (&deviceCount);
    int device;

    if(deviceCount>0){
        for (device = 0; device < deviceCount; ++device) {
            cudaDeviceProp deviceProp;
            cudaGetDeviceProperties (&deviceProp, device);
            printf ("Device %s has compute capability %d.%d.\n", deviceProp.name, deviceProp.major, deviceProp.minor);
        }
        int array_size = 6;
        int byte_size = sizeof(double) * array_size;
        double *h_in1 = new double [array_size];
        double *h_in2 = new double [array_size];
        double * h_out = new double [array_size*array_size];
        for (int i = 0; i < array_size; i++){
            h_in1[i]=i;
            h_in2[i]=2*i;
        }
        for (int i = 0; i<array_size; i++){
            std::cout<<h_in2[i]<<std::endl;
        }
        Vector_GPU d_i1(array_size,1,h_in1);
        Vector_GPU d_i2(array_size,1,h_in2);
        unsigned int *rows = new unsigned int;
        unsigned int *columns = new unsigned int;
        cudaMemcpy(h_in1,d_i2.d_mat,sizeof(double)*array_size*1,cudaMemcpyDeviceToHost);
        cudaMemcpy(rows,d_i2.d_rows,sizeof(unsigned int),cudaMemcpyDeviceToHost);
        cudaMemcpy(columns,d_i2.d_columns,sizeof(unsigned int),cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        /*Vector_GPU d_out = d_i1 * d_i2;
        cudaMemcpy(h_out,d_out.d_mat,byte_size,cudaMemcpyDeviceToHost);*/
        for (int i = 0; i<array_size; i++){
            std::cout<<h_in1[i]<<std::endl;
        }
        printf("%d x %d\n",*rows,*columns);
        delete h_in1, h_in2, h_out, rows, columns;

    }
    else{
        printf("NO CUDA DEVICE AVAILABLE");
    }
    return 0;
}