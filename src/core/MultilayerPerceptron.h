#ifndef _MULTILAYERPERCEPTRON_H
#define _MULTILAYERPERCEPTRON_H

#include <algorithm>
#include <functional>
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
        MultilayerPerceptron(std::unique_ptr<MLPModel>& model,
                             std::unique_ptr<MLPTrainer>& trainer,
                             const std::function<void(double)>& view_callback);

        void importModel(const std::string& filepath);
        void exportModel(const std::string& filepath);

        void testing(const std::string& dataset_path, const size_t percent);
        void learning(const bool crossvalid, const std::string& dataset_path, const size_t epochs);
        char prediction(const std::vector<double>& input_layer);
       
    //    private:
        std::unique_ptr<MLPModel> _model;
        std::unique_ptr<MLPTrainer> _trainer;
        std::function<void(double)> _view_callback;

        std::vector<double> fillImportVector(std::ifstream &s, const std::string& type, const size_t elements);
        std::pair<size_t, size_t> getImportSize(const std::string &line);
    };
}  // namespace s21

#endif //_MULTILAYERPERCEPTRON_H
