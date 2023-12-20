#include "FileMLPSerializer.h"

namespace s21 {
    void s21::FileMLPSerializer::serialize(std::unique_ptr<MLPModel> &model, const std::string &filename) {
        std::ofstream of(filename, std::ios_base::out);

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
  
    void s21::FileMLPSerializer::deserialize(std::unique_ptr<MLPModel> &model, const std::string &filename) {
        std::ifstream f(filename, std::ios_base::in);

        std::string line;
        if (!std::getline(f, line)) {
            throw std::invalid_argument("Filemodel is empty");
        }

        auto [need_weights, need_bias] = getImportSize(model, line);

        std::vector<double> weights = fillImportVector(f, "weight", need_weights);
        model->setWeights(weights);

        std::vector<double> bias = fillImportVector(f, "bias", need_bias);
        model->setBiases(bias);
    }

    std::vector<double> FileMLPSerializer::fillImportVector(std::ifstream &s, const std::string &type, const size_t elements) {
        std::vector<double> vec;
        vec.reserve(elements);

        while (vec.size() < elements) {
            double num;
            if (!(s >> num)) {
                throw std::runtime_error("Cannot parse " + type + " number from file");
            }
            vec.push_back(num);
        }

        return vec;
    }
    
    std::pair<size_t, size_t> FileMLPSerializer::getImportSize(std::unique_ptr<MLPModel> &model, const std::string &line) {
        std::stringstream ss(line);
        std::vector<size_t> layers_info;

        size_t n;
        while (ss >> n) {
            layers_info.push_back(n);
            if (ss.peek() == ' ') {
                ss.ignore();
            }
        }

        if (layers_info != model->getLayersSize()) {
            throw std::invalid_argument("Filemodel does not correspond to the object model");
        }

        size_t need_weights = 0, need_bias = 0;
        for (size_t i = 1; i < layers_info.size(); i++) {
            need_weights += layers_info[i - 1] * layers_info[i];
            need_bias += layers_info[i];
        }

        return std::make_pair(need_weights, need_bias);
    }
}
