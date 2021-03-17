#include <vector>


class Matrix {
    private:
        unsigned int rows;
        unsigned int columns;
        std::vector<double> mat;
        double sparsity=.70; //the number of 0-elements/non-0-elements
    public:
        Matrix(): Matrix(0,0) {printf("DEFAULT CONSTRUCTOR\n");};
        Matrix(unsigned int r, unsigned int c);
        Matrix(unsigned int r, unsigned int c, double v):rows(r),columns(c){mat.resize(r*c,v);};
        Matrix(const Matrix &m):rows(m.rows),columns(m.columns),mat(m.mat){};
        ~Matrix();
        int getRows();
        int getColumns();
        void printMat();
        double operator()(unsigned int i);
        double operator()(unsigned int r, unsigned int c);
        double operator[](unsigned int i);
        
        Matrix operator*(double i);
        Matrix operator*(std::vector<double> &v);
};

class Vector : public Matrix {
    private:
        unsigned int length;
        std::vector<double> vec;
    public:
        Vector();
        Vector(unsigned int l);
        ~Vector();
};