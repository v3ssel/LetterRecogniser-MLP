#ifndef _CONTROLLER_H_
#define _CONTROLLER_H_

#include "../core/ModelBuilder.h"
#include "../core/MultilayerPerceptron.h"
#include "../core/serializer/FileMLPSerializer.h"
#include "../core/training/EmnistMLPTrainer.h"

namespace s21 {
class Controller {
   public:
    static Controller &getInstance() {
        static Controller instance;
        return instance;
    }

    void makeMLP(ModelType type, size_t input_layer_size,
                 size_t output_layer_size, size_t hidden_layers,
                 size_t hidden_layer_size, size_t learning_rate,
                 const EMNISTMLPTrainer::EpochCb &epoch_callback,
                 const EMNISTMLPTrainer::ProcessCb &process_callback);

    void changeModelTypeAndLayersSize(ModelType type, size_t hidden_layers);
    void setLearningRate(double learning_rate);

    void loadWeights(const std::string &path);
    void saveWeights(const std::string &path);
    void randomizeWeights();

    char predicate(const std::vector<double> &input);

    MLPTestMetrics startTesting(const std::string &path, const size_t percent);
    std::vector<double> startTraining(const bool cv, const std::string &path,
                                      const size_t epochs);
    void stopTrainer();

   private:
    Controller() = default;
    ~Controller() = default;
    Controller(Controller const &) = delete;

    std::unique_ptr<MultilayerPerceptron> mlp_;
};
}  // namespace s21

#endif  // _CONTROLLER_H_
