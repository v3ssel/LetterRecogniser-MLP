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
            for (auto &output_node : _layers[i]->_output_layer->_nodes) {
                summatoryFunction(_layers[i], output_node);
            }
            activationFunction(_layers[i]->_output_layer->_nodes);
        }

        return {};
    }

    void GraphModel::summatoryFunction(GraphLayer &layer, GraphNode &output_node) {
        for (auto i = 0; i < layer.getSize(); i++) {
            output_node.value += layer._nodes[i] * output_node.weights[i];
        }
        output_node.value += output_node.bias;
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
        std::vector<double> err_x;
        for (auto i = 0; i < err_y.size(); i++) {
            err_x.push_back(sigmoidDerivative(_layers.back()->_nodes[i].value) * err_y[i]);
        }
        std::vector<double> err_w;
        for (auto i = 0; i < err_x.size(); i++) {
            for (auto j = 0; j < _layers.back()->_input_layer->_nodes.size(); j++) {
                err_w.push_back(_layers.back()->_input_layer->_nodes[j].value * err_x[i]);
            }
        }

//        updateWeights(_layers.back(), err_w);
//        updateBias(_layers.back(), err_x);
//        for (auto &node : _layers.back()->_nodes) {
//            for (auto i = 0; i < node.weights.size(); i++) {
//                node.weights[i] = -= (err_w * _learning_rate);
//            }
//        }
//        for (auto &node : _layers.back()->_nodes) {
//            node.bias = -= (err_x * _learning_rate);
//        }

        for (int l = _layers.size() - 2; l > 0; l--) {
            err_y = derivativeOfY(_layers[l], err_w);
//            err_y = (err_x * _layers[l].weights.Transpose());
            err_x = derivativeOfX(_layers[l], err_y);
//            err_x = applyDerivative(err_y, _layers[l].values);
            err_w = derivativeOfW(_layers[l]->_input_layer, err_x);
//            err_w = _layers[l - 1].values.Transpose() * err_x;

//            updateWeights(_layers[l], err_w);
//            updateBias(_layers[l], err_x);
//            _layers[l - 1].weights -= (err_w * _learning_rate);
//            _layers[l - 1].bias -= (err_x * _learning_rate);
        }
    }

    void GraphModel::summatoryFunction(GraphLayer &layer, GraphNode &output_node) {
        for (auto i = 0; i < layer.getSize(); i++) {
            output_node.value += layer._nodes[i] * output_node.weights[i];
        }
        output_node.value += output_node.bias;
    }

    for (size_t i = 0; i < _layers.size() - 1; ++i) {
        for (auto &output_node : _layers[i]->_output_layer->_nodes) {
            summatoryFunction(_layers[i], output_node);
        }
        activationFunction(_layers[i]->_output_layer->_nodes);
    }

    std::vector<double> GraphModel::derivativeOfY(GraphLayer &layer, std::vector<double> &err_x) {
        std::vector<double> err_y;
        for (auto &node : layer->_nodes) {
            double dy = 0;
            for (auto i = 0; i < layer.getSize(); i++) {
                dy += err_x[i] * node.weights[i]);
            }
            err_x.push_back(dy);
        }
    }

    std::vector<double> GraphModel::derivativeOfX(GraphLayer &layer, std::vector<double> &err_y) {
        std::vector<double> err_x;
        for (auto i = 0; i < err_y.size(); i++) {
            dsigmoidDerivative(layer->_nodes[i].value) * err_y[i]);
        }
        return err_x;
    }

    std::vector<double> GraphModel::derivativeOfW(std::vector<double> &err_x) {
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

    void GraphModel::updateWeights(GraphLayer &layer, std::vector<double> &err_w) {
//        for (auto &node : layer->_nodes) {
//            for (auto i = 0; i < node.weights.size(); i++) {
//                node.weights[i] -= (err_w * _learning_rate);
//            }
//        }
    }

    void GraphModel::updateBias(GraphLayer &layer, std::vector<double> &err_x) {
//        for (auto &node : layer->_nodes) {
//            node.bias -= (err_x * _learning_rate);
//        }
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
