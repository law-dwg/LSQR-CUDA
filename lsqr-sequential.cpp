#include <iostream>
#include <cassert>
#include <vector>

#include "lsqr.h"

int main(){
    int m,n;
    std::cout<<"how many rows in A? ";
    std::cin>>m;
    std::cout<<"how many columns in A? ";
    std::cin>>n;
    Matrix ma(m,n);
    Vector vec(m);
    int b = lsqr(ma,vec);
    return 0;
};