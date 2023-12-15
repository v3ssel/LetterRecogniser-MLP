#include "MatrixModel.h"

namespace s21 {
    MatrixModel::MatrixModel(size_t input_layer, size_t output_layer, size_t hidden_layers, size_t neurons_in_hidden_layers) {
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
    }

    void MatrixModel::randomFill() {
        for (size_t i = 0; i < _layers.size(); ++i) {
            _layers[i].weights = Matrix::GenerateRandom(_layers[i].weights.getRows(), _layers[i].weights.getCols());
            _layers[i].bias = Matrix::GenerateRandom(_layers[i].bias.getRows(), _layers[i].bias.getCols());
        }
    }

    std::vector<double> MatrixModel::feedForward(std::vector<double>& input_layer) {
        _layers[0].values = Matrix(input_layer);

        for (size_t i = 0; i < _layers.size() - 1; ++i) {
            _layers[i + 1].values = ((_layers[i].values * _layers[i].weights) + _layers[i].bias);
            activationFunction(_layers[i + 1].values);
            std::cout << "-------------------------------LAYER " << i + 1 << " OF " << _layers.size() << " OF NEURAL NETWORK---------------------------------";
            std::cout << "\nValues:\n";
            _layers[i].values.Print();
            std::cout << "\nWeights:\n";
            _layers[i].weights.Print();
            std::cout << "\nBias:\n";
            _layers[i].bias.Print();
        }

        return _layers.back().values.ToVector();
    }

    void MatrixModel::backPropagation(std::vector<double>& target) {
        Matrix error = _layers.back().values - Matrix(target);
        // std::cout << "--------------------ErrY--------------------" << std::endl;
        // error.Print();

        Matrix err_x = makeErrX(error, _layers.back().values);
        // std::cout << "--------------------ErrX--------------------" << std::endl;
        // err_x.Print();
        
        Matrix err_w = makeErrW(err_x, _layers[_layers.size() - 2].values);
        // std::cout << "--------------------ErrW--------------------" << std::endl;
        // err_w.Print();
        
        _layers[_layers.size() - 2].weights -= err_w * _learning_rate;
        // std::cout << "--------------------updateW--------------------" << std::endl;
        // _layers[_layers.size() - 2].weights.Print();

        for (int l = _layers.size() - 2; l > 0; l--) {
            // std::cout << "l: " << l << std::endl;
            error = (err_x * _layers[l].weights.Transpose());

            // std::cout << "-X" << std::endl;
            err_x = makeErrX(error, _layers[l].values);
            // err_x.Print();
            
            // std::cout << "--W" << std::endl;
            err_w = makeErrW(err_x, _layers[l - 1].values);
            // err_w.Print();
            
            // std::cout << "---U" << std::endl;

            _layers[l - 1].weights -= (err_w * _learning_rate);
            _layers[l - 1].bias -= (err_x * _learning_rate);
            // _layers[l - 1].weights.Print();
        }

    }

    Matrix MatrixModel::makeErrX(const Matrix &err_y, const Matrix &out_layer) {
        Matrix err_x(err_y.getRows(), err_y.getCols());
        for (size_t i = 0; i < err_x.getRows(); i++) {
            for (size_t j = 0; j < err_x.getCols(); j++) {
                err_x(i, j) = out_layer(i, j) * (1 - out_layer(i, j)) * err_y(i, j);
            }
        }
        return err_x;
    }

    Matrix MatrixModel::makeErrW(const Matrix &err_x, const Matrix &in_layer) {
        Matrix err_w(in_layer.getCols(), err_x.getCols());
        for (size_t i = 0; i < err_w.getRows(); i++) {
            for (size_t j = 0; j < err_w.getCols(); j++) {
                err_w(i, j) = in_layer(0, i) * err_x(0, j);
            }
        }
        return err_w;
    }

    std::vector<size_t> MatrixModel::getLayersSize() {
        std::vector<size_t> vec;

        for (auto& layer : _layers)
            vec.push_back(layer.size);

        return vec;
    }

    void MatrixModel::activationFunction(Matrix &layer) {
        for (auto i = 0; i < layer.getRows(); i++) {
            for (auto j = 0; j < layer.getCols(); j++)
                layer(i, j) = sigmoidFunction(layer(i, j));
        }
    }

    double MatrixModel::sigmoidFunction(double n) {
        return 1.0l / (1.0l + std::exp(-n));
    }

    void MatrixModel::setWeights(std::vector<double> weights) {
        int need_weights = std::accumulate(_layers.begin(), _layers.end(), 0, 
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

    std::vector<double> MatrixModel::getWeights() {
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

    void MatrixModel::setBiases(std::vector<double> biases) {
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

    std::vector<double> MatrixModel::getBiases() {
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

} // namespace s21
