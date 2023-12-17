#ifndef _MULTILAYERPERCEPTRON_H
#define _MULTILAYERPERCEPTRON_H

#include <algorithm>
#include <vector>
#include <memory>
#include <string>
#include <fstream>
#include <sstream>
#include "MLPModel.h"
#include "MLPTrainer.h"

namespace s21 {
    class MultilayerPerceptron {
       public:
        // можно сделать билдера модели, через него собираем модель и подаем сюда
        MultilayerPerceptron(std::unique_ptr<MLPModel>& model, std::unique_ptr<MLPTrainer>& trainer);

        void importModel(std::string filepath);
        void exportModel(std::string filepath);

        void testing(std::string dataset_path, size_t percent);
        void learning(std::string dataset_path, size_t epochs);
        char prediction(std::vector<double>& input_layer);
       
    //    private:
        std::unique_ptr<MLPModel> _model;
        std::unique_ptr<MLPTrainer> _trainer;

        std::vector<double> fillImportVector(std::ifstream &s, std::string type, size_t elements);
        std::pair<size_t, size_t> getImportSize(std::string &line);
    };
}  // namespace s21

#endif //_MULTILAYERPERCEPTRON_H
