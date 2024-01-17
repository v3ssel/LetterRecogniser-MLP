#include "GraphModel.h"

#include <iostream>

namespace s21 {
GraphModel::GraphModel(size_t input_layer, size_t output_layer,
                       size_t hidden_layers, size_t neurons_in_hidden_layers,
                       double learn_rate) {
  learning_rate_ = learn_rate;

  std::shared_ptr<GraphLayer> layer_ptr = nullptr;
  layers_.emplace_back(std::make_shared<GraphLayer>(input_layer));

  for (size_t i = 0; i < hidden_layers; i++) {
    layer_ptr = std::make_shared<GraphLayer>(neurons_in_hidden_layers);

    layers_.back()->setOutputLayer(layer_ptr);
    layer_ptr->setInputLayer(layers_.back());

    layers_.emplace_back(layer_ptr);
  }

  layer_ptr = std::make_shared<GraphLayer>(output_layer);

  layer_ptr->setInputLayer(layers_.back());
  layers_.back()->setOutputLayer(layer_ptr);

  layers_.emplace_back(layer_ptr);
}

size_t GraphModel::getPrediction(const std::vector<double> &output_layer) {
  return std::distance(
      output_layer.begin(),
      std::max_element(output_layer.begin(), output_layer.end()));
}

std::vector<double> GraphModel::feedForward(
    const std::vector<double> &input_layer) {
  for (size_t i = 0; i < layers_[0]->size_; ++i) {
    layers_[0]->nodes_[i].value = input_layer[i];
  }

  for (size_t i = 0; i < layers_.size() - 1; ++i) {
    summatoryFunction(layers_[i]);
    activationFunction(layers_[i]->getOutputLayer()->nodes_);
  }

  std::vector<double> result;

  for (auto &node : layers_.back()->nodes_) {
    result.push_back(node.value);
  }

  return result;
}

void GraphModel::summatoryFunction(std::shared_ptr<s21::GraphLayer> &layer) {
  for (size_t i = 0; i < layer->getOutputLayer()->getSize(); i++) {
    for (size_t j = 0; j < layer->getSize(); j++) {
      layer->getOutputLayer()->nodes_[i].value +=
          layer->nodes_[j].value * layer->nodes_[j].weights[i];
    }
    layer->getOutputLayer()->nodes_[i].value +=
        layer->getOutputLayer()->nodes_[i].bias;
  }
}

void GraphModel::activationFunction(std::vector<GraphNode> &nodes) {
  for (size_t i = 0; i < nodes.size(); i++) {
    nodes[i].value = sigmoidFunction(nodes[i].value);
  }
}

double GraphModel::sigmoidFunction(double n) {
  return 1.0l / (1.0l + std::exp(-n));
}

void GraphModel::backPropagation(const std::vector<double> &target) {
  std::vector<double> err_y;

  for (size_t i = 0; i < target.size(); i++) {
    err_y.push_back(layers_.back()->nodes_[i].value - target[i]);
  }

  std::vector<double> err_x = derivativeOfX(layers_.back(), err_y);
  std::vector<double> err_w =
      derivativeOfW(layers_.back()->getInputLayer(), err_x);

  updateWeights(layers_.back()->getInputLayer(), err_w);
  updateBias(layers_.back(), err_x);

  for (int l = layers_.size() - 2; l > 0; l--) {
    err_y = derivativeOfY(layers_[l], err_w);
    err_x = derivativeOfX(layers_[l], err_y);
    err_w = derivativeOfW(layers_[l]->getInputLayer(), err_x);

    updateWeights(layers_[l]->getInputLayer(), err_w);
    updateBias(layers_[l], err_x);
  }
}

std::vector<double> GraphModel::derivativeOfY(
    std::shared_ptr<s21::GraphLayer> &layer, std::vector<double> &err_x) {
  std::vector<double> err_y;

  for (auto &node : layer->nodes_) {
    double dy = 0;
    for (size_t i = 0; i < node.weights.size(); i++) {
      dy += node.weights[i] * err_x[i];
    }
    err_y.push_back(dy);
  }

  return err_y;
}

std::vector<double> GraphModel::derivativeOfX(
    std::shared_ptr<s21::GraphLayer> &layer, std::vector<double> &err_y) {
  std::vector<double> err_x;

  for (size_t i = 0; i < err_y.size(); i++) {
    err_x.push_back(sigmoidDerivative(layer->nodes_[i].value) * err_y[i]);
  }

  return err_x;
}

std::vector<double> GraphModel::derivativeOfW(
    std::shared_ptr<s21::GraphLayer> &layer, std::vector<double> &err_x) {
  std::vector<double> err_w;

  for (size_t i = 0; i < err_x.size(); i++) {
    for (size_t j = 0; j < layer->nodes_.size(); j++) {
      err_w.push_back(layer->nodes_[j].value * err_x[i]);
    }
  }

  return err_w;
}

double GraphModel::sigmoidDerivative(double n) { return n * (1 - n); }

void GraphModel::updateWeights(std::shared_ptr<s21::GraphLayer> &layer,
                               std::vector<double> &err_w) {
  for (auto &node : layer->nodes_) {
    for (size_t i = 0; i < node.weights.size(); i++) {
      node.weights[i] -= getLearningRate() * err_w[i];
    }
  }
}

void GraphModel::updateBias(std::shared_ptr<s21::GraphLayer> &layer,
                            std::vector<double> &err_x) {
  for (size_t i = 0; i < layer->getSize(); i++) {
    layer->nodes_[i].bias -= getLearningRate() * err_x[i];
  }
}

void GraphModel::randomFill() {
  for (auto &layer : layers_) {
    layer->randomize();
  }
}

std::vector<size_t> GraphModel::getLayersSize() const {
  std::vector<size_t> layers_size;

  for (auto &layer : layers_) {
    layers_size.push_back(layer->getSize());
  }

  return layers_size;
}

void GraphModel::setWeights(const std::vector<double> &weights) {
  size_t need_weights = std::accumulate(
      layers_.begin(), layers_.end(), 0,
      [](size_t sum, std::shared_ptr<GraphLayer> layer) -> size_t {
        if (!layer->getOutputLayer()) return sum;
        return sum + layer->getSize() * layer->getOutputLayer()->getSize();
      });

  if (need_weights != weights.size()) {
    throw std::out_of_range(
        "GraphModel::setWeights: need_weights != weights.size()");
  }

  std::vector<double>::const_iterator begin = weights.begin();

  for (size_t i = 0; i < layers_.size() - 1; i++) {
    layers_[i]->setWeights(begin);
  }
}

std::vector<double> GraphModel::getWeights() const {
  std::vector<double> weights;

  for (auto &layer : layers_) {
    auto &&layer_weights = layer->getWeights();
    weights.insert(weights.end(), layer_weights.begin(), layer_weights.end());
  }

  return weights;
}

void GraphModel::setBiases(const std::vector<double> &biases) {
  size_t need_biases =
      std::accumulate(layers_.begin() + 1, layers_.end(), 0,
                      [](int sum, std::shared_ptr<GraphLayer> &layer) -> int {
                        return sum + layer->getSize();
                      });

  if (need_biases != biases.size()) {
    throw std::out_of_range(
        "GraphModel::setBiases: need_biases != biases.size()");
  }

  std::vector<double>::const_iterator begin = biases.begin();

  for (size_t i = 1; i < layers_.size(); i++) {
    layers_[i]->setBiases(begin);
  }
}

std::vector<double> GraphModel::getBiases() const {
  std::vector<double> biases;

  for (auto &layer : layers_) {
    auto &&layer_biases = layer->getBiases();
    biases.insert(biases.end(), layer_biases.begin(), layer_biases.end());
  }

  return biases;
}

void GraphModel::setLearningRate(double rate) { learning_rate_ = rate; }

double GraphModel::getLearningRate() const { return learning_rate_; }
}  // namespace s21
