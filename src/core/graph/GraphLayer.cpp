#include "GraphLayer.h"

namespace s21 {
GraphLayer::GraphLayer(size_t size) : size_(size) {
    nodes_.resize(size);
    _input_layer = nullptr;
    _output_layer = nullptr;
}

void GraphLayer::randomize() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 1.0);

    for (auto &node : nodes_) {
        node.bias = dist(gen);
        for (auto &weight : node.weights) {
            weight = dist(gen);
        }
    }
}

std::vector<double> GraphLayer::getWeights() {
    std::vector<double> weights;

    if (_output_layer) {
        weights.reserve(_output_layer->size_ * size_);

        for (auto &node : nodes_) {
            for (auto &weight : node.weights) {
                weights.push_back(weight);
            }
        }
    }

    return weights;
}

std::vector<double> GraphLayer::getBiases() {
    std::vector<double> biases;

    if (_input_layer) {
        biases.reserve(size_);

        for (auto &node : nodes_) {
            biases.push_back(node.bias);
        }
    }

    return biases;
}

void GraphLayer::setWeights(std::vector<double>::const_iterator &begin) {
    for (auto &node : nodes_) {
        for (size_t i = 0; i < node.weights.size(); ++i) {
            node.weights[i] = *begin;
            ++begin;
        }
    }
}

void GraphLayer::setBiases(std::vector<double>::const_iterator &begin) {
    for (auto &node : nodes_) {
        node.bias = *begin;
        ++begin;
    }
}

size_t GraphLayer::getSize() { return size_; }

std::shared_ptr<GraphLayer> &GraphLayer::getInputLayer() {
    return _input_layer;
}

std::shared_ptr<GraphLayer> &GraphLayer::getOutputLayer() {
    return _output_layer;
}

void GraphLayer::setInputLayer(std::shared_ptr<GraphLayer> &input) {
    _input_layer = input;
}

void GraphLayer::setOutputLayer(std::shared_ptr<GraphLayer> &output) {
    for (auto &node : nodes_) {
        node.weights.resize(output->size_, 0.0);
    }

    _output_layer = output;
}
}  // namespace s21
