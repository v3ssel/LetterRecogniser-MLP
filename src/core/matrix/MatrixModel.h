#ifndef _MATRIXMODEL_H
#define _MATRIXMODEL_H

#include <vector>
#include "libmatrix/matrix.h"
#include "MatrixLayer.h"

namespace s21 {
    class MatrixModel {
       public:
        MatrixModel(size_t input_layer, size_t output_layer, size_t hidden_layers, size_t neurons_in_hidden_layers);

        Matrix feedForward(Matrix &input_layer);
        void backPropagation(Matrix &output_layer);

        void activationFunction(Matrix &layer);
        double sigmoidFunction(double n);

    // private:
        std::vector<MatrixLayer> _layers;

        size_t _input_layer;
        size_t _output_layer;

    };
}

#endif //_MATRIXMODEL_H
