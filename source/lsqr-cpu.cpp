#include <iostream>
#include <cassert>
#include <vector>

#include "lsqr.h"
#include "matVec.h"

int main(){
    int m,n;
    /**std::cout<<"how many rows in A? ";
    std::cin>>m;
    std::cout<<"how many columns in A? ";
    std::cin>>n;**/
    Matrix ma(3,2);
    ma.mat = {1.0, 2.0, 3.2, 4.0, 5.0, 6.1};
    ma.print();
    Vector vec(3);
    vec.mat = {1.0,200.0,-10.0};
    int b = lsqr(ma,vec);
    return 0;
};