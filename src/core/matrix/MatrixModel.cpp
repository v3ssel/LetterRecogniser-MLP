#include "MatrixModel.h"

namespace s21 {
    MatrixModel::MatrixModel(size_t input_layer, size_t output_layer, size_t hidden_layers, size_t neurons_in_hidden_layers) {
        _input_layer = input_layer;
        _output_layer = output_layer;

        _layers.emplace_back(Matrix::GenerateRandom(_input_layer, neurons_in_hidden_layers),
                             Matrix::GenerateRandom(1, neurons_in_hidden_layers),
                             _input_layer);

        for (auto i = 0; i < hidden_layers; ++i) {
            _layers.emplace_back(Matrix::GenerateRandom(neurons_in_hidden_layers, neurons_in_hidden_layers),
                                 Matrix::GenerateRandom(1, neurons_in_hidden_layers),
                                 neurons_in_hidden_layers);
        }

        _layers.push_back(MatrixLayer(_output_layer));
    }

    Matrix MatrixModel::feedForward(Matrix &input_layer) {
        _layers[0].values = input_layer;
        for (auto i = 0; i < _layers.size() - 1; ++i) {
            _layers[i + 1].values = ((_layers[i].values * _layers[i].weights) + _layers[i].bias);
            activationFunction(_layers[i + 1].values);
            std::cout << "-------------------------------LAYER " << i + 1 << " OF " << _layers.size() - 1 << " OF NEURAL NETWORK---------------------------------";
            std::cout << "\nValues:\n";
            _layers[i].values.Print();
            std::cout << "\nWeights:\n";
            _layers[i].weights.Print();
            std::cout << "\nBias:\n";
            _layers[i].bias.Print();
        }

        return _layers.back().values;
    }

    void MatrixModel::backPropagation(Matrix &output_layer) {
        
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
} // namespace s21
