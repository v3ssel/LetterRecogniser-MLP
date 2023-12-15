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
        void backPropagation(std::vector<double>& target) override;
        void randomFill() override;

        Matrix makeErrX(const Matrix &err_y, const Matrix &out_layer);
        Matrix makeErrW(const Matrix &err_x, const Matrix &in_layer);

        std::vector<size_t> getLayersSize() override;

        void setWeights(std::vector<double> weights) override;
        std::vector<double> getWeights() override;

        void setBiases(std::vector<double> biases) override;
        std::vector<double> getBiases() override;

    // private:
        std::vector<MatrixLayer> _layers;
        double _learning_rate = 0.4;
    };
}

#endif //_MATRIXMODEL_H
