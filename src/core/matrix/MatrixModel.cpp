#include "MatrixModel.h"

namespace s21 {
    MatrixModel::MatrixModel(size_t input_layer, size_t output_layer, size_t hidden_layers, size_t neurons_in_hidden_layers, double learn_rate) {
        _layers.emplace_back(Matrix(input_layer, neurons_in_hidden_layers),
                             Matrix(1, neurons_in_hidden_layers),
                             input_layer);

        for (size_t i = 0; i < hidden_layers; ++i) {
            size_t output = (i == hidden_layers - 1) ? output_layer : neurons_in_hidden_layers;
            _layers.emplace_back(Matrix(neurons_in_hidden_layers, output),
                                 Matrix(1, output),
                                 neurons_in_hidden_layers);
        }

        _layers.push_back(MatrixLayer(output_layer));

        _learning_rate = learn_rate;
    }

    size_t MatrixModel::getPrediction(const std::vector<double> &output_layer) {
        return std::distance(output_layer.begin(), std::max_element(output_layer.begin(), output_layer.end()));
    }

    std::vector<double> MatrixModel::feedForward(const std::vector<double> &input_layer) {
        _layers[0].values = Matrix(input_layer);

        for (size_t i = 0; i < _layers.size() - 1; ++i) {
            _layers[i + 1].values = ((_layers[i].values * _layers[i].weights) + _layers[i].bias);
            activationFunction(_layers[i + 1].values);
        }

        return _layers.back().values.ToVector();
    }

    void MatrixModel::backPropagation(const std::vector<double>& target) {
        Matrix err_y = _layers.back().values - Matrix(target);
        Matrix err_x = applyDerivative(err_y, _layers.back().values);
        Matrix err_w = _layers[_layers.size() - 2].values.Transpose() * err_x;
        
        _layers[_layers.size() - 2].weights -= (err_w * _learning_rate);
        _layers[_layers.size() - 2].bias -= (err_x * _learning_rate);

        for (int l = _layers.size() - 2; l > 0; l--) {
            err_y = (err_x * _layers[l].weights.Transpose());
            err_x = applyDerivative(err_y, _layers[l].values);
            err_w = _layers[l - 1].values.Transpose() * err_x;

            _layers[l - 1].weights -= (err_w * _learning_rate);
            _layers[l - 1].bias -= (err_x * _learning_rate);
        }
    }

    void MatrixModel::randomFill() {
        for (size_t i = 0; i < _layers.size(); ++i) {
            _layers[i].weights = Matrix::GenerateRandom(_layers[i].weights.getRows(), _layers[i].weights.getCols());
            _layers[i].bias = Matrix::GenerateRandom(_layers[i].bias.getRows(), _layers[i].bias.getCols());
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

    double MatrixModel::sigmoidDerivative(double n) {
        return n * (1 - n);
    }

    Matrix MatrixModel::applyDerivative(const Matrix &err_y, const Matrix &out_layer) {
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

        for (auto& layer : _layers)
            vec.push_back(layer.size);

        return vec;
    }
 
    void MatrixModel::setWeights(const std::vector<double>& weights) {
        size_t need_weights = std::accumulate(_layers.begin(), _layers.end(), 0, 
                            [](int i, MatrixLayer l) { return i + l.weights.getCols() * l.weights.getRows(); });
        if (weights.size() != need_weights) {
            throw std::out_of_range("Number of weights is not equal to the number of weights in the model");
        }
        
        int w = 0;
        for (size_t layer = 0; layer < _layers.size() - 1; layer++) {
            for (size_t i = 0; i < _layers[layer].size; i++) {
                for (size_t j = 0; j < _layers[layer + 1].size; j++) {
                    _layers[layer].weights(i, j) = weights[w];
                    w++;
                }
            }
        }
    }

    std::vector<double> MatrixModel::getWeights() const {
        std::vector<double> weights;

        for (auto &layer : _layers) {
            for (size_t i = 0; i < layer.weights.getRows(); i++) {
                for (size_t j = 0; j < layer.weights.getCols(); j++) {
                    weights.push_back(layer.weights(i, j));
                }
            }
        }

        return weights;
    }

    void MatrixModel::setBiases(const std::vector<double>& biases) {
        size_t need_biases = std::accumulate(_layers.begin(), _layers.end(), 0, [](int i, MatrixLayer l) { return i + l.bias.getCols(); });
        if (biases.size() != need_biases) {
            throw std::out_of_range("Number of biases is not equal to the number of biases in the model");
        }

        int b = 0;
        for (size_t layer = 0; layer < _layers.size() - 1; layer++) {
            for (size_t j = 0; j < _layers[layer + 1].size; j++) {
                _layers[layer].bias(0, j) = biases[b];
                b++;
            }
        }
    }

    std::vector<double> MatrixModel::getBiases() const {
        std::vector<double> biases;

        for (auto &layer : _layers) {
            for (size_t i = 0; i < layer.bias.getRows(); i++) {
                for (size_t j = 0; j < layer.bias.getCols(); j++) {
                    biases.push_back(layer.bias(i, j));
                }
            }
        }

        return biases;
    }

    void MatrixModel::setLearningRate(double rate) {
        _learning_rate = rate;
    }

    double MatrixModel::getLearningRate() const {
        return _learning_rate;
    }
} // namespace s21
