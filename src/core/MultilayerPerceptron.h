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
#include "MLPSerializer.h"

namespace s21 {
    class MultilayerPerceptron {
       public:
        // можно сделать билдера модели, через него собираем модель и подаем сюда
        MultilayerPerceptron(std::unique_ptr<MLPModel>& model,
                             std::unique_ptr<MLPTrainer>& trainer,
                             std::unique_ptr<MLPSerializer>& serializer);

        void importModel(const std::string& filepath);
        void exportModel(const std::string& filepath);

        MLPTestMetrics testing(const std::string& dataset_path, const size_t percent);
        std::vector<double> learning(const bool crossvalid, const std::string& dataset_path, const size_t epochs);
        char prediction(const std::vector<double>& input_layer);
       
        void stopTraining();

        void setLearningRate(const double learning_rate);

       private:
        std::unique_ptr<MLPModel> _model;
        std::unique_ptr<MLPTrainer> _trainer;
        std::unique_ptr<MLPSerializer> _serializer;
    };
}  // namespace s21

#endif //_MULTILAYERPERCEPTRON_H
