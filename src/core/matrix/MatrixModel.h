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

        void activationFunction(Matrix &layer);
        double sigmoidFunction(double n);

        std::vector<double> feedForward(std::vector<double>& input_layer) override;
        void backPropagation() override;
        void randomFill() override;

        std::vector<size_t> getLayersSize() override;

        void setWeights(std::vector<double> weights) override;
        std::vector<double> getWeights() override;

        void setBiases(std::vector<double> biases) override;
        std::vector<double> getBiases() override;

    // private:
        std::vector<MatrixLayer> _layers;

        size_t _input_layer_size;
        size_t _output_layer_size;
        size_t _hidden_layers;
        size_t _hidden_layers_size;
    };
}

#endif //_MATRIXMODEL_H
