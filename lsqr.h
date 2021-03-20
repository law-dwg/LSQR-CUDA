#include <vector>

class Vector {
    public:
        //attributes
        unsigned int rows;
        unsigned int columns;
        std::vector<double> mat;
        //constructors and destructor
        Vector(): Vector(1){};
        Vector(unsigned int p):columns(1),rows(p){mat.resize(p,5.0);};
        Vector(const Vector &v):columns(v.columns),rows(v.rows),mat(v.mat){};
        ~Vector(){};
        //
        std::vector<double> getMat(){return this->mat;};
        virtual void print();
        Vector operator*(double i);

        double Dnrm2();
        double normalNorm();
        double operator[](unsigned int i);
        
        Vector transpose();
};

class Matrix : public Vector {
    public: 
        double sparsity=.70; //the number of 0-elements/non-0-elements
        Matrix(): Matrix(0,0){};
        Matrix(unsigned int r, unsigned int c);
        Matrix(unsigned int r, unsigned int c, double v){this->rows=r;this->columns=c;mat.resize(r*c,v);};
        Matrix(const Matrix &m){this->mat=m.mat; this->rows=m.rows; this->columns=m.columns;};
        //Matrix(std::vector<double> v){this->mat=v;};
        ~Matrix(){};
        

        Matrix transpose();
        int getRows();
        int getColumns();
        //template member functions have to be defined in header
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
        
        double operator()(unsigned int i);
        double operator()(unsigned int r, unsigned int c);
        void print();
        Matrix operator*(double i);

};

int lsqr(Matrix &A, Vector &b);