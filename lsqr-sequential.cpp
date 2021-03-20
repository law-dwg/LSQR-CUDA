#include <iostream>
#include <cassert>
#include <vector>

#include "lsqr.h"

int main(){
    int m,n;
    /**std::cout<<"how many rows in A? ";
    std::cin>>m;
    std::cout<<"how many columns in A? ";
    std::cin>>n;**/
    Matrix ma(3,2);
    ma.mat = {1,0,1,1,0,1};
    ma.print();
    std::cout<<ma(0,0)<<" "<<ma[0]<<" "<<ma(0)<<std::endl;
    Vector vec(3);
    vec.mat = {1.0,0.0,-1.0};
    int b = lsqr(ma,vec);
    return 0;
};