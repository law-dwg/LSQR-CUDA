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
    Vector_GPU d_i1(array_size,1,h_input1);
    Vector_GPU d_i2(array_size,1,h_input2);
    printf("TEST\n");
    Vector_GPU out = d_i1*3;
    printf("line 172\n");
    out = out + d_i1 * d_i2;
    //d_i1 = d_i1*out;
    cudaDeviceSynchronize(); //wait for GPU to finish
    //out = d_i2; //assignment
    printf("line 177\n");
    Vector_GPU test = out; //copy
    cudaDeviceSynchronize(); //wait for GPU to finish
    double *h_out = new double [array_size];
    cudaMemcpy(h_out,out.d_mat,byte_size,cudaMemcpyDeviceToHost);
    /*
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

    
    */
    delete h_input1, h_input2;
    
    for (int i = 0; i<array_size; i++){
        std::cout<<h_out[i]<<std::endl;
    }
    
    return 0;
}