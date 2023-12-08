#include "MultilayerPerceptron.h"

std::vector<std::vector<double>> s21::MultilayerPerceptron::inputData() {

    return std::vector<std::vector<double>>();
}

std::vector<std::vector<double>> s21::MultilayerPerceptron::importWeights() {
    // model.setWeights();
    return std::vector<std::vector<double>>();
}

std::vector<std::vector<double>> s21::MultilayerPerceptron::learning() {
    // auto output_layer = model.feedForward(input_layer);
    // model.backPropagation(output_layer);
    return std::vector<std::vector<double>>();
}

int s21::MultilayerPerceptron::getHiddenLayers() {
    return hidden_layers;
}

void s21::MultilayerPerceptron::setHiddenLayers(int n) {
    if (n > 1 && n < 6)
        hidden_layers = n;
}

char s21::MultilayerPerceptron::prediction() {
    // auto output_layer = model.feedForward(input_layer);

    return 0;
}
