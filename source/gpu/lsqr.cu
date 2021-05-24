#include "matVec_gpu.cuh"
//#include "../cpu/matVec_cpu.h"
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
        unsigned int rows = 6;
        unsigned int columns = 2;
        int array_size = rows*columns;
        int byte_size = sizeof(double) * array_size;
        double *h_in1 = new double [array_size];
        double *h_in2 = new double [array_size];
        double *h_in3 = new double [array_size/2];
        double * h_out = new double [array_size*array_size];
        for (int i = 0; i < array_size; i++){
            h_in1[i]=i;
            h_in2[i]=2*i;
        }
        for (int i = 0; i < array_size/2; i++){
            h_in3[i]=3*i;
        }
        /*for (int i = 0; i<array_size; i++){
            std::cout<<h_in2[i]<<std::endl;
        }*/
        Vector_GPU d_i1(rows,columns,h_in1);
        Vector_GPU d_i2(columns,rows,h_in2);
        Vector_GPU d_i3(rows,columns/2,h_in3);
        d_i3.printmat();
        
        Vector_GPU d_out = d_i1 * d_i2;
        printf("BEFORE COPY\n");
        Vector_GPU copy = d_out;
        printf("AFTER COPY\n");
        printf("BEFORE ASSIGNMENT\n");
        d_i3 = d_i1;
        printf("AFTER ASSIGNMENT\n");
        d_i3.printmat();
        cudaDeviceSynchronize();
        
        //Vector_CPU out = copy.matDeviceToHost();
        Vector_CPU h = d_i3.matDeviceToHost();
        //out.print();
        
        delete h_in1, h_in2, h_in3, h_out;

    }
    else{
        printf("NO CUDA DEVICE AVAILABLE");
    }
    return 0;
}