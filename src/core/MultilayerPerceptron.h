#ifndef _MULTILAYERPERCEPTRON_H
#define _MULTILAYERPERCEPTRON_H

#include <vector>
#include <memory>
#include "MLPModel.h"

namespace s21 {

class MultilayerPerceptron {
private:
    int hidden_layers = 2;
    std::vector<std::vector<double>> input_layer;
    std::unique_ptr<MLPModel> model;
//    GraphModel g_model;

public:
    int getHiddenLayers();
    void setHiddenLayers(int n);
    std::vector<std::vector<double>> inputData();
    std::vector<std::vector<double>> importWeights();
    std::vector<std::vector<double>> learning();
    char prediction();
//    std::vector<std::vector<double>> learning(Graph& model);
};

}  // namespace s21

#endif //_MULTILAYERPERCEPTRON_H
