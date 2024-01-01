#ifndef _MULTILAYERPERCEPTRON_H_
#define _MULTILAYERPERCEPTRON_H_

#include <vector>
#include <memory>
#include <string>

#include "MLPModel.h"
#include "MLPTrainer.h"
#include "MLPSerializer.h"
#include "ModelBuilder.h"

namespace s21 {
    class MultilayerPerceptron {
       public:
        MultilayerPerceptron(std::unique_ptr<MLPModel>& model,
                             std::unique_ptr<MLPTrainer>& trainer,
                             std::unique_ptr<MLPSerializer>& serializer);

        void importModel(const std::string& filepath);
        void exportModel(const std::string& filepath);

        MLPTestMetrics testing(const std::string& dataset_path, const size_t percent);
        std::vector<double> learning(const bool crossvalid, const std::string& dataset_path, const size_t epochs);
        char prediction(const std::vector<double>& input_layer);
       
        void stopTrainer();

        void changeModel(ModelType type, size_t hidden_layers);
        void setModel(std::unique_ptr<MLPModel>& model);
        
        void setLearningRate(const double learning_rate);
        double getLearningRate();

       private:
        std::unique_ptr<MLPModel> _model;
        std::unique_ptr<MLPTrainer> _trainer;
        std::unique_ptr<MLPSerializer> _serializer;
    };
}  // namespace s21

#endif //_MULTILAYERPERCEPTRON_H_
