#include "MultilayerPerceptron.h"
#include <iostream>

namespace s21 {
    MultilayerPerceptron::MultilayerPerceptron(std::unique_ptr<MLPModel>& m,
                                               std::unique_ptr<MLPTrainer>& t,
                                               std::unique_ptr<MLPSerializer>& s) {
        _model = std::move(m);
        _trainer = std::move(t);
        _serializer = std::move(s);
    }

    void MultilayerPerceptron::importModel(const std::string& filepath) {
        _serializer->deserialize(_model, filepath);
    }

    void MultilayerPerceptron::exportModel(const std::string& filepath) {
        _serializer->serialize(_model, filepath);
    }

    MLPTestMetrics MultilayerPerceptron::testing(const std::string& dataset_path, const size_t percent) {
        return _trainer->test(_model, dataset_path, percent);
    }

    std::vector<double> MultilayerPerceptron::learning(const bool crossvalid, const std::string& dataset_path, const size_t epochs) {
        return crossvalid
                ? _trainer->crossValidation(_model, dataset_path, epochs)
                : _trainer->train(_model, dataset_path, epochs);
    }

    char MultilayerPerceptron::prediction(const std::vector<double>& input_layer) {
        return static_cast<char>(_model->getPrediction(_model->feedForward(input_layer)) + 65);
    }

    void MultilayerPerceptron::stopTraining() {
        _trainer->stop();
    }

    void MultilayerPerceptron::setLearningRate(const double learning_rate) {
        _model->setLearningRate(learning_rate);
    }
} // namespace s21
