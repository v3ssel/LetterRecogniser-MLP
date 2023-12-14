#ifndef _MULTILAYERPERCEPTRON_H
#define _MULTILAYERPERCEPTRON_H

#include <algorithm>
#include <vector>
#include <memory>
#include <string>
#include <fstream>
#include <sstream>
#include "MLPModel.h"

namespace s21 {
    class MultilayerPerceptron {
       public:
        // можно сделать билдера модели, через него собираем модель и подаем сюда
        MultilayerPerceptron(std::unique_ptr<MLPModel>& model);

        void importModel(std::string filepath);
        void exportModel(std::string filepath);

        std::vector<std::vector<double>> learning();
        char prediction(std::vector<double>& input_layer);
       
    //    private:
        std::unique_ptr<MLPModel> model;

        std::vector<double> fillImportVector(std::ifstream &s, std::string type, size_t elements);
        std::pair<size_t, size_t> getImportSize(std::string &line);
    };
}  // namespace s21

#endif //_MULTILAYERPERCEPTRON_H
