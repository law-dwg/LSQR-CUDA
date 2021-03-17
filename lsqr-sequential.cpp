#include <iostream>
#include "lsqr.h"
#include <cassert>

int main(){
    int m,n;
    std::cout<<"how many rows in A? ";
    std::cin>>m;
    std::cout<<"how many columns in A? ";
    std::cin>>n;
    Matrix test(m,n);
    test.printMat();
    printf("value at (%i,%i) = %f\nIs the same as [%i]=%f or (%i)=%f\n",1,1, test(4,4),24,test[24],24,test(24));
    //test.printMatVec();
    //assert(m==test.getRows());
    Matrix out=test*2;
    out.printMat();
    return 0;
};