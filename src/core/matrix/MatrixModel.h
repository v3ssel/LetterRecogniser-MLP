#ifndef _MATRIXMODEL_H
#define _MATRIXMODEL_H

#include <vector>
#include "libmatrix/matrix.h"

namespace s21 {
    class MatrixModel {
       public:
        Matrix feedForward(Matrix &input_layer);
        void backPropagation(Matrix &output_layer);

       private:
        
    };
}

#endif //_MATRIXMODEL_H
