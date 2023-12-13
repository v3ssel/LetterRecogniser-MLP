#include "MultilayerPerceptron.h"

namespace s21 {
    MultilayerPerceptron::MultilayerPerceptron(std::unique_ptr<MLPModel> &m) {
        model = std::move(m);
    }

    void MultilayerPerceptron::importModel(std::string filepath) {

    }

    void MultilayerPerceptron::exportModel(std::string filepath)
    {
        std::ofstream of(filepath, std::ios_base::out);

        auto layers_info = model->getLayersSize();
        for (auto i : layers_info) {
            of << i << " ";
        }
        of << "\n";
        
        auto weights = model->getWeights();
        for (auto w : weights) {
            of << w << "\n";
        }

        auto bias = model->getBiases();
        for (auto b : bias) {
            of << b << "\n";
        }
    }

    std::vector<std::vector<double>> MultilayerPerceptron::learning() {
        // auto output_layer = model.feedForward(input_layer);
        // model.backPropagation(output_layer);
        return std::vector<std::vector<double>>();
    }


    char MultilayerPerceptron::prediction() {
        // auto output_layer = model.feedForward(input_layer);

        return 0;
    }
} // namespace s21
