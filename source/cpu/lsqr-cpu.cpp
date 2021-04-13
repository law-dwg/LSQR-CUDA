#include <iostream>
#include <cassert>
#include <vector>

#include "lsqr.h"
#include "matVec.h"

int main(){
    int m,n;
    Matrix ma(3,2);
    ma.mat = {1.0, -2.2, 3.2, 4.0, 5.0, 6.1};
    ma.print();
    Vector vec(3);
    vec.mat = {0.01,0.2,-0.2};
    int b = lsqr(ma,vec);
    return 0;
};