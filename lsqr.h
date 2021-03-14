#include <vector>

class Matrix {
    private:
        unsigned int rows;
        unsigned int columns;
        std::vector<std::vector<double>> mat;
        double sparsity=.70; //the number of 0-elements/non-0-elements
    public:
        Matrix();
        Matrix(unsigned int r, unsigned int c);
        ~Matrix();
        int getRows();
        int getColumns();
        void printMat();
};

class Vector {
    private:
        unsigned int length;
        std::vector<double> vec;
    public:
        Vector();
        Vector(unsigned int l);
        ~Vector();
};