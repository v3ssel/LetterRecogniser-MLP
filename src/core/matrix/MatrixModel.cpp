#include "MatrixModel.h"

namespace s21 {
MatrixModel::MatrixModel(size_t input_layer, size_t output_layer,
                         size_t hidden_layers, size_t neurons_in_hidden_layers,
                         double learn_rate) {
    layers_.emplace_back(Matrix(input_layer, neurons_in_hidden_layers),
                         Matrix(1, neurons_in_hidden_layers), input_layer);

    for (size_t i = 0; i < hidden_layers; ++i) {
        size_t output =
            (i == hidden_layers - 1) ? output_layer : neurons_in_hidden_layers;
        layers_.emplace_back(Matrix(neurons_in_hidden_layers, output),
                             Matrix(1, output), neurons_in_hidden_layers);
    }

    layers_.push_back(MatrixLayer(output_layer));

    learning_rate_ = learn_rate;
}

size_t MatrixModel::getPrediction(const std::vector<double> &output_layer) {
    return std::distance(
        output_layer.begin(),
        std::max_element(output_layer.begin(), output_layer.end()));
}

std::vector<double> MatrixModel::feedForward(
    const std::vector<double> &input_layer) {
    layers_[0].values = Matrix(input_layer);

    for (size_t i = 0; i < layers_.size() - 1; ++i) {
        layers_[i + 1].values =
            ((layers_[i].values * layers_[i].weights) + layers_[i].bias);
        activationFunction(layers_[i + 1].values);
    }

    return layers_.back().values.ToVector();
}

void MatrixModel::backPropagation(const std::vector<double> &target) {
    Matrix err_y = layers_.back().values - Matrix(target);
    Matrix err_x = applyDerivative(err_y, layers_.back().values);
    Matrix err_w = layers_[layers_.size() - 2].values.Transpose() * err_x;

    layers_[layers_.size() - 2].weights -= (err_w * learning_rate_);
    layers_[layers_.size() - 2].bias -= (err_x * learning_rate_);

    for (int l = layers_.size() - 2; l > 0; l--) {
        err_y = (err_x * layers_[l].weights.Transpose());
        err_x = applyDerivative(err_y, layers_[l].values);
        err_w = layers_[l - 1].values.Transpose() * err_x;

        layers_[l - 1].weights -= (err_w * learning_rate_);
        layers_[l - 1].bias -= (err_x * learning_rate_);
    }
}

void MatrixModel::randomFill() {
    for (size_t i = 0; i < layers_.size(); ++i) {
        layers_[i].weights = Matrix::GenerateRandom(
            layers_[i].weights.getRows(), layers_[i].weights.getCols());
        layers_[i].bias = Matrix::GenerateRandom(layers_[i].bias.getRows(),
                                                 layers_[i].bias.getCols());
    }
}

void MatrixModel::activationFunction(Matrix &layer) {
    for (size_t i = 0; i < layer.getRows(); i++) {
        for (size_t j = 0; j < layer.getCols(); j++)
            layer(i, j) = sigmoidFunction(layer(i, j));
    }
}

double MatrixModel::sigmoidFunction(double n) {
    return 1.0l / (1.0l + std::exp(-n));
}

double MatrixModel::sigmoidDerivative(double n) { return n * (1 - n); }

Matrix MatrixModel::applyDerivative(const Matrix &err_y,
                                    const Matrix &out_layer) {
    Matrix err_x(err_y.getRows(), err_y.getCols());

    for (size_t i = 0; i < err_x.getRows(); i++) {
        for (size_t j = 0; j < err_x.getCols(); j++) {
            err_x(i, j) = sigmoidDerivative(out_layer(i, j)) * err_y(i, j);
        }
    }

    return err_x;
}

std::vector<size_t> MatrixModel::getLayersSize() const {
    std::vector<size_t> vec;

    for (auto &layer : layers_) vec.push_back(layer.size);

    return vec;
}

void MatrixModel::setWeights(const std::vector<double> &weights) {
    size_t need_weights = std::accumulate(
        layers_.begin(), layers_.end(), 0, [](int i, MatrixLayer l) {
            return i + l.weights.getCols() * l.weights.getRows();
        });
    if (weights.size() != need_weights) {
        throw std::out_of_range(
            "Number of weights is not equal to the number of weights in the "
            "model");
    }

    int w = 0;
    for (size_t layer = 0; layer < layers_.size() - 1; layer++) {
        for (size_t i = 0; i < layers_[layer].size; i++) {
            for (size_t j = 0; j < layers_[layer + 1].size; j++) {
                layers_[layer].weights(i, j) = weights[w];
                w++;
            }
        }
    }
}

std::vector<double> MatrixModel::getWeights() const {
    std::vector<double> weights;

    for (auto &layer : layers_) {
        for (size_t i = 0; i < layer.weights.getRows(); i++) {
            for (size_t j = 0; j < layer.weights.getCols(); j++) {
                weights.push_back(layer.weights(i, j));
            }
        }
    }

    return weights;
}

void MatrixModel::setBiases(const std::vector<double> &biases) {
    size_t need_biases = std::accumulate(
        layers_.begin(), layers_.end(), 0,
        [](int i, MatrixLayer l) { return i + l.bias.getCols(); });
    if (biases.size() != need_biases) {
        throw std::out_of_range(
            "Number of biases is not equal to the number of biases in the "
            "model");
    }

    int b = 0;
    for (size_t layer = 0; layer < layers_.size() - 1; layer++) {
        for (size_t j = 0; j < layers_[layer + 1].size; j++) {
            layers_[layer].bias(0, j) = biases[b];
            b++;
        }
    }
}

std::vector<double> MatrixModel::getBiases() const {
    std::vector<double> biases;

    for (auto &layer : layers_) {
        for (size_t i = 0; i < layer.bias.getRows(); i++) {
            for (size_t j = 0; j < layer.bias.getCols(); j++) {
                biases.push_back(layer.bias(i, j));
            }
        }
    }

    return biases;
}

void MatrixModel::setLearningRate(double rate) { learning_rate_ = rate; }

double MatrixModel::getLearningRate() const { return learning_rate_; }
}  // namespace s21
