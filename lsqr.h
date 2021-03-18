#include <vector>


class Matrix {
    private:
        unsigned int rows;
        unsigned int columns;
        double sparsity=.70; //the number of 0-elements/non-0-elements
    public:
        std::vector<double> mat;
        Matrix(): Matrix(0,0) {printf("DEFAULT MATRIX CONSTRUCTOR\n");};
        Matrix(unsigned int r, unsigned int c);
        Matrix(unsigned int r, unsigned int c, double v):rows(r),columns(c){mat.resize(r*c,v);};
        Matrix(const Matrix &m):rows(m.rows),columns(m.columns),mat(m.mat){};
        Matrix(std::vector<double> v):mat(v){}
        ~Matrix();
        
        int getRows();
        int getColumns();
        std::vector<double> getMat();
        virtual void print();
        double Dnrm2();
        double normalNorm();
        double operator()(unsigned int i);
        double operator()(unsigned int r, unsigned int c);
        double operator[](unsigned int i);
        
        //template member functions have to be defined in header
        
        Matrix operator*(double i);
        template<typename T>
        T operator*(T &v){
            std::vector<double> lhs = this->mat;
            std::vector<double> rhs = v.getMat();
            T out(this->rows);
            std::cout<<rhs.size()<<std::endl;
            std::cout<<rhs.size()<<std::endl;
            if(this->rows==rhs.size()){
                for(int r =0; r<this->rows; r++){
                    double sum = 0; 
                    for(int c=0; c<this->columns; c++){
                        sum += lhs[r*this->columns + c]*rhs[c];
                    }
                    out.mat[r]=sum;
                }
                return out;
            }
            else{
                printf("cannot perform this multiplication\n");
                return v;
            }
        };

        

        
};

class Vector : public Matrix {
    private:
        unsigned int length;
    public:
        Vector(): Vector(1){printf("DEFAULT VECTOR CONSTRUCTOR");};
        Vector(unsigned int p):length(p){printf("vector constructed");mat.resize(p,5.0);};
        Vector(const Vector &v):length(v.length){this->mat=(v.mat);};
        ~Vector();

        Vector operator*(double i);
        void print();
};

int lsqr(Matrix &A, Vector &b);