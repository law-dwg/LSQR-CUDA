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
    assert(m==test.getRows());
    return 0;
};