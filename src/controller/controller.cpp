#include "controller.h"

namespace s21 {
void Controller::makeMLP(ModelType type, size_t input_layer_size,
                         size_t output_layer_size, size_t hidden_layers,
                         size_t hidden_layer_size, size_t learning_rate,
                         const EMNISTMLPTrainer::EpochCb &epoch_callback,
                         const EMNISTMLPTrainer::ProcessCb &process_callback) {
  ModelBuilder builder;
  builder.setModelType(type)
      ->setInputLayerSize(input_layer_size)
      ->setOutputLayerSize(output_layer_size)
      ->setLayers(hidden_layers)
      ->setHiddenLayerSize(hidden_layer_size)
      ->setLearningRate(learning_rate);

  auto model = builder.build();
  std::unique_ptr<MLPSerializer> serializer =
      std::make_unique<FileMLPSerializer>();
  std::unique_ptr<MLPTrainer> trainer =
      std::make_unique<EMNISTMLPTrainer>(epoch_callback, process_callback);

  mlp_ = std::make_unique<MultilayerPerceptron>(model, trainer, serializer);
}

void Controller::changeModelTypeAndLayersSize(ModelType type,
                                              size_t hidden_layers) {
  mlp_->changeModelTypeAndLayersSize(type, hidden_layers);
}

void Controller::setLearningRate(double learning_rate) {
  mlp_->setLearningRate(learning_rate);
}

void Controller::loadWeights(const std::string &path) {
  mlp_->importModel(path);
}

void Controller::saveWeights(const std::string &path) {
  mlp_->exportModel(path);
}

void Controller::randomizeWeights() { mlp_->randomizeModelWeights(); }

char Controller::predicate(const std::vector<double> &input) {
  return mlp_->prediction(input);
}

MLPTestMetrics Controller::startTesting(const std::string &path,
                                        const size_t percent) {
  return mlp_->testing(path, percent);
}

std::vector<double> Controller::startTraining(const bool cv,
                                              const std::string &path,
                                              const size_t epochs) {
  return mlp_->learning(cv, path, epochs);
}

void Controller::stopTrainer() { mlp_->stopTrainer(); }
}  // namespace s21
