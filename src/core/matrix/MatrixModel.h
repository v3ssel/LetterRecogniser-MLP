#ifndef _MATRIXMODEL_H_
#define _MATRIXMODEL_H_

#include <algorithm>
#include <numeric>

#include "../MLPModel.h"
#include "MatrixLayer.h"

namespace s21 {
class MatrixModel : public MLPModel {
   public:
    MatrixModel(size_t input_layer, size_t output_layer, size_t hidden_layers,
                size_t neurons_in_hidden_layers, double learn_rate);

    size_t getPrediction(const std::vector<double>& output_layer) override;
    std::vector<double> feedForward(
        const std::vector<double>& input_layer) override;
    void backPropagation(const std::vector<double>& target) override;
    void randomFill() override;

    void activationFunction(Matrix& layer);
    double sigmoidFunction(double n);
    double sigmoidDerivative(double n);
    Matrix applyDerivative(const Matrix& err_y, const Matrix& out_layer);

    std::vector<size_t> getLayersSize() const override;

    void setWeights(const std::vector<double>& weights) override;
    std::vector<double> getWeights() const override;

    void setBiases(const std::vector<double>& biases) override;
    std::vector<double> getBiases() const override;

    void setLearningRate(double rate) override;
    double getLearningRate() const override;

   private:
    std::vector<MatrixLayer> layers_;
    double learning_rate_;
};
}  // namespace s21

#endif  //_MATRIXMODEL_H_
