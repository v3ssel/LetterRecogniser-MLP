#ifndef _CONTROLLER_H_
#define _CONTROLLER_H_

#include "../core/MultilayerPerceptron.h"
#include "../core/serializer/FileMLPSerializer.h"
#include "../core/training/EmnistMLPTrainer.h"
#include "../core/ModelBuilder.h"

namespace s21 {
    class Controller {
       public:
        static Controller& getInstance() {
            static Controller instance;
            return instance;
        }

        void makeMLP(ModelType type,
                     size_t input_layer_size,
                     size_t output_layer_size,
                     size_t hidden_layers,
                     size_t hidden_layer_size,
                     size_t learning_rate,
                     const EMNISTMLPTrainer::EpochCb &epoch_callback,
                     const EMNISTMLPTrainer::ProcessCb &process_callback);

        void changeModel(ModelType type, size_t hidden_layers);
        void setLearningRate(double learning_rate);

        void loadWeights(const std::string &path);
        void saveWeights(const std::string &path);

        char predicate(const std::vector<double> &input);

       private:
        Controller() = default;
        ~Controller() = default;
        Controller(Controller const&) = delete;

        std::unique_ptr<MultilayerPerceptron> mlp_;
    };
}

#endif // _CONTROLLER_H_
