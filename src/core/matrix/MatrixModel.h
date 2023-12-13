#ifndef _MATRIXMODEL_H
#define _MATRIXMODEL_H

#include <vector>
#include <algorithm>
#include <numeric>

#include "libmatrix/matrix.h"
#include "../MLPModel.h"
#include "MatrixLayer.h"

namespace s21 {
    class MatrixModel : public MLPModel {
       public:
        MatrixModel(size_t input_layer, size_t output_layer, size_t hidden_layers, size_t neurons_in_hidden_layers);

        std::vector<double> feedForward(std::vector<double>& input_layer) override;
        void backPropagation() override;

        void activationFunction(Matrix &layer);
        double sigmoidFunction(double n);

        void setWeights(std::vector<double> weights);
        std::vector<double> getWeights();

        void setBiases(std::vector<double> biases);
        std::vector<double> getBiases();

    // private:
        std::vector<MatrixLayer> _layers;
    };
}

#endif //_MATRIXMODEL_H
