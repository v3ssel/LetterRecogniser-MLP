#include "GraphLayer.h"

namespace s21 {
GraphLayer::GraphLayer(size_t size) : _size(size) {
  _nodes.resize(size);
  _input_layer = nullptr;
  _output_layer = nullptr;
}

void GraphLayer::randomize() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<double> dist(0.0, 1.0);

  for (auto &node : _nodes) {
    node.bias = dist(gen);
    for (auto &weight : node.weights) {
      weight = dist(gen);
    }
  }
}

std::vector<double> GraphLayer::getWeights() {
  std::vector<double> weights;

  if (_output_layer) {
    weights.reserve(_output_layer->_size * _size);

    for (auto &node : _nodes) {
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
    biases.reserve(_size);

    for (auto &node : _nodes) {
      biases.push_back(node.bias);
    }
  }

  return biases;
}

void GraphLayer::setWeights(std::vector<double>::const_iterator &begin) {
  for (auto &node : _nodes) {
    for (size_t i = 0; i < node.weights.size(); ++i) {
      node.weights[i] = *begin;
      ++begin;
    }
  }
}

void GraphLayer::setBiases(std::vector<double>::const_iterator &begin) {
  for (auto &node : _nodes) {
    node.bias = *begin;
    ++begin;
  }
}

size_t GraphLayer::getSize() { return _size; }

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
  for (auto &node : _nodes) {
    node.weights.resize(output->_size, 0.0);
  }

  _output_layer = output;
}
}  // namespace s21
