#include "GraphModel.h"
#include <iostream>

namespace s21 {
    GraphModel::GraphModel(size_t input_layer,
                           size_t output_layer,
                           size_t hidden_layers,
                           size_t neurons_in_hidden_layers,
                           double learn_rate) {
        _learning_rate = learn_rate;
                            
        std::shared_ptr<GraphLayer> layer_ptr = nullptr;
        _layers.emplace_back(std::make_shared<GraphLayer>(input_layer));
        
        for (size_t i = 0; i < hidden_layers; i++) {
            layer_ptr = std::make_shared<GraphLayer>(neurons_in_hidden_layers);

            _layers.back()->setOutputLayer(layer_ptr);
            layer_ptr->setInputLayer(_layers.back());

            _layers.emplace_back(layer_ptr);
        }

        layer_ptr = std::make_shared<GraphLayer>(output_layer);

        layer_ptr->setInputLayer(_layers.back());
        _layers.back()->setOutputLayer(layer_ptr);
        
        _layers.emplace_back(layer_ptr);
    }

    size_t GraphModel::getPrediction(const std::vector<double> &output_layer) {
        return std::distance(output_layer.begin(), std::max_element(output_layer.begin(), output_layer.end()));
    }

    std::vector<double> GraphModel::feedForward(const std::vector<double> &input_layer) {
        return std::vector<double>();
    }
    
    void GraphModel::backPropagation(const std::vector<double> &target) {
        
    }
    
    void GraphModel::randomFill() {
        for (auto &layer : _layers) {
            layer->randomize();          
        }
    }
    
    std::vector<size_t> GraphModel::getLayersSize() const {
        std::vector<size_t> layers_size;

        for (auto &layer : _layers) {
            layers_size.push_back(layer->getSize());
        }
        
        return layers_size;
    }
    
    void GraphModel::setWeights(const std::vector<double> &weights) {
        int need_weights = std::accumulate(_layers.begin(), _layers.end(), 0,
        [](int sum, std::shared_ptr<GraphLayer> layer) -> int {
            if (!layer->getOutputLayer()) return sum;
            return sum + layer->getSize() * layer->getOutputLayer()->getSize();
        });
        
        if (need_weights != weights.size()) {
            throw std::out_of_range("GraphModel::setWeights: need_weights != weights.size()");
        }

        std::vector<double>::const_iterator begin = weights.begin();

        for (size_t i = 0; i < _layers.size() - 1; i++) {
            _layers[i]->setWeights(begin);
        }
    }
    
    std::vector<double> GraphModel::getWeights() const {
        std::vector<double> weights;

        for (auto &layer : _layers) {
            auto&& layer_weights = layer->getWeights();
            weights.insert(weights.end(), layer_weights.begin(), layer_weights.end());
        }

        return weights;
    }
    
    void GraphModel::setBiases(const std::vector<double> &biases) {
        int need_biases = std::accumulate(_layers.begin() + 1, _layers.end(), 0,
        [](int sum, std::shared_ptr<GraphLayer>& layer) -> int {
            return sum + layer->getSize();
        });

        if (need_biases != biases.size()) {
            throw std::out_of_range("GraphModel::setBiases: need_biases != biases.size()");
        }

        std::vector<double>::const_iterator begin = biases.begin();

        for (size_t i = 1; i < _layers.size(); i++) {
            _layers[i]->setBiases(begin);
        }
    }
    
    std::vector<double> GraphModel::getBiases() const {
        std::vector<double> biases;

        for (auto &layer : _layers) {
            auto&& layer_biases = layer->getBiases();
            biases.insert(biases.end(), layer_biases.begin(), layer_biases.end());
        }

        return biases;
    }
    
    void GraphModel::setLearningRate(double rate) {
        _learning_rate = rate;
    }
    
    double GraphModel::getLearningRate() const {
        return _learning_rate;
    }
}
