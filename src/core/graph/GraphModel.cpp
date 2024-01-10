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
        for (size_t i = 0; i < _layers[0]->_size; ++i) {
            _layers[0]->_nodes[i].value = input_layer[i];
        }
        for (size_t i = 0; i < _layers.size() - 1; ++i) {
            summatoryFunction(_layers[i]);
            activationFunction(_layers[i]->getOutputLayer()->_nodes);
        }
        std::vector<double>result;
        for (auto &node : _layers.back()->_nodes) {
            result.push_back(node.value);
        }
        return result;
    }

    void GraphModel::summatoryFunction(std::shared_ptr<s21::GraphLayer> &layer) {
        for (auto i = 0; i < layer->getOutputLayer()->getSize(); i++) {
            for (auto j = 0; j < layer->getSize(); j++) {
                layer->getOutputLayer()->_nodes[i].value += layer->_nodes[j].value * layer->_nodes[j].weights[i];
            }
            layer->getOutputLayer()->_nodes[i].value += layer->getOutputLayer()->_nodes[i].bias;
        }
    }

    void GraphModel::activationFunction(std::vector<GraphNode> &nodes) {
        for (auto i = 0; i < nodes.size(); i++) {
            nodes[i].value = sigmoidFunction(nodes[i].value);
        }
    }

    double GraphModel::sigmoidFunction(double n) {
        return 1.0l / (1.0l + std::exp(-n));
    }
    
    void GraphModel::backPropagation(const std::vector<double> &target) {
        std::vector<double> err_y;
        for (auto i = 0; i < target.size(); i++) {
            err_y.push_back(_layers.back()->_nodes[i].value - target[i]);
        }
        // std::cout << "err_y.size: " << err_y.size() << std::endl;

        std::vector<double> err_x  = derivativeOfX(_layers.back(), err_y);
        // std::cout << "err_x.size: " << err_x.size() << std::endl;

        std::vector<double> err_w = derivativeOfW(_layers.back()->getInputLayer(), err_x);
        // std::cout << "err_w.size: " << err_w.size() << std::endl;

        updateWeights(_layers.back()->getInputLayer(), err_w);
        updateBias(_layers.back(), err_x);

        for (int l = _layers.size() - 2; l > 0; l--) {
            err_y = derivativeOfY(_layers[l], err_w);
            // std::cout << "derivativeOfY err_y.size:" << err_y.size() << std::endl;

            err_x = derivativeOfX(_layers[l], err_y);
            // std::cout << "derivativeOfX err_x.size:" << err_x.size() << std::endl;
            
            err_w = derivativeOfW(_layers[l]->getInputLayer(), err_x);
            // std::cout << "derivativeOfW err_w.size:" << err_w.size() << std::endl;

            updateWeights(_layers[l]->getInputLayer(), err_w);
            updateBias(_layers[l], err_x);
        }
    }

    std::vector<double> GraphModel::derivativeOfY(std::shared_ptr<s21::GraphLayer> &layer, std::vector<double> &err_x) {
        std::vector<double> err_y;
        for (auto &node : layer->_nodes) {
            double dy = 0;
            for (auto i = 0; i < node.weights.size(); i++) {
                dy += node.weights[i] * err_x[i];
            }
            err_y.push_back(dy);
        }
        return err_y;
    }

    std::vector<double> GraphModel::derivativeOfX(std::shared_ptr<s21::GraphLayer> &layer, std::vector<double> &err_y) {
        std::vector<double> err_x;
        for (auto i = 0; i < err_y.size(); i++) {
            err_x.push_back(sigmoidDerivative(layer->_nodes[i].value) * err_y[i]);
        }
        return err_x;
    }

    std::vector<double> GraphModel::derivativeOfW(std::shared_ptr<s21::GraphLayer> &layer, std::vector<double> &err_x) {
        std::vector<double> err_w;
        for (auto i = 0; i < err_x.size(); i++) {
            for (auto j = 0; j < layer->_nodes.size(); j++) {
                err_w.push_back(layer->_nodes[j].value * err_x[i]);
            }
        }
        return err_w;
    }

    double GraphModel::sigmoidDerivative(double n) {
        return n * (1 - n);
    }

    void GraphModel::updateWeights(std::shared_ptr<s21::GraphLayer> &layer, std::vector<double> &err_w) {
        // size_t t = 0;   //del
        for (auto &node : layer->_nodes) {
            for (auto i = 0; i < node.weights.size(); i++) {
                node.weights[i] -= getLearningRate() * err_w[i];
                // std::cout << ++t << std::endl;  //del
            }
        }
    }

    void GraphModel::updateBias(std::shared_ptr<s21::GraphLayer> &layer, std::vector<double> &err_x) {
        // size_t t = 0;   //del
        for (auto i = 0; i < layer->getSize(); i++) {
            layer->_nodes[i].bias -= getLearningRate() * err_x[i];
            // std::cout << ++t << std::endl;  //del
        }
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
