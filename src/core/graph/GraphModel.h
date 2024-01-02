#ifndef _GRAPHMODEL_H_
#define _GRAPHMODEL_H_

#include <numeric>
#include <algorithm>
#include "../MLPModel.h"
#include "GraphLayer.h"

namespace s21 {
    class GraphModel : public MLPModel {
       public:
        GraphModel(size_t input_layer, size_t output_layer, size_t hidden_layers, size_t neurons_in_hidden_layers, double learn_rate);

        size_t getPrediction(const std::vector<double>& output_layer) override;
        std::vector<double> feedForward(const std::vector<double>& input_layer) override;
        void backPropagation(const std::vector<double>& target) override;
        void randomFill() override;

        std::vector<size_t> getLayersSize() const override;

        void setWeights(const std::vector<double>& weights) override;
        std::vector<double> getWeights() const override;

        void setBiases(const std::vector<double>& biases) override;
        std::vector<double> getBiases() const override;

        void setLearningRate(double rate) override;
        double getLearningRate() const override;
        
    //    private:
        std::vector<std::shared_ptr<GraphLayer>> _layers;
        double _learning_rate;

    private:
        void summatoryFunction(GraphLayer &layer, GraphNode &output_node);
        void activationFunction(std::vector<GraphNode> &nodes);
        double sigmoidFunction(double n);

        double sigmoidDerivative(double n);
        void updateWeights(GraphLayer &layer, std::vector<double> &err_w);
        void updateBias(GraphLayer &layer, std::vector<double> &err_x);

    };
}

#endif // _GRAPHMODEL_H_
