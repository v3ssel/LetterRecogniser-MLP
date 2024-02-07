#include "MultilayerPerceptron.h"

namespace s21 {
MultilayerPerceptron::MultilayerPerceptron(std::unique_ptr<MLPModel>& m,
                                           std::unique_ptr<MLPTrainer>& t,
                                           std::unique_ptr<MLPSerializer>& s) {
    model_ = std::move(m);
    trainer_ = std::move(t);
    serializer_ = std::move(s);
}

void MultilayerPerceptron::importModel(const std::string& filepath) {
    serializer_->deserialize(model_, filepath);
}

void MultilayerPerceptron::exportModel(const std::string& filepath) {
    serializer_->serialize(model_, filepath);
}

MLPTestMetrics MultilayerPerceptron::testing(const std::string& dataset_path,
                                             const size_t percent) {
    return trainer_->test(model_, dataset_path, percent);
}

std::vector<double> MultilayerPerceptron::learning(
    const bool crossvalid, const std::string& dataset_path,
    const size_t epochs) {
    return crossvalid ? trainer_->crossValidation(model_, dataset_path, epochs)
                      : trainer_->train(model_, dataset_path, epochs);
}

char MultilayerPerceptron::prediction(const std::vector<double>& input_layer) {
    return static_cast<char>(
        model_->getPrediction(model_->feedForward(input_layer)) + 65);
}

void MultilayerPerceptron::stopTrainer() { trainer_->stop(); }

void MultilayerPerceptron::randomizeModelWeights() { model_->randomFill(); }

void MultilayerPerceptron::changeModelTypeAndLayersSize(ModelType type,
                                                        size_t hidden_layers) {
    auto sizes = model_->getLayersSize();

    ModelBuilder builder;
    builder.setModelType(type)
        ->setInputLayerSize(sizes[0])
        ->setOutputLayerSize(sizes.back())
        ->setLayers(hidden_layers)
        ->setHiddenLayerSize(sizes[1])
        ->setLearningRate(model_->getLearningRate());

    auto new_model = builder.build();
    setModel(new_model);
}

void MultilayerPerceptron::setModel(std::unique_ptr<MLPModel>& model) {
    model_ = std::move(model);
}

std::unique_ptr<MLPModel>& MultilayerPerceptron::getModel() { return model_; }

void MultilayerPerceptron::setLearningRate(const double learning_rate) {
    model_->setLearningRate(learning_rate);
}

double MultilayerPerceptron::getLearningRate() {
    return model_->getLearningRate();
}
}  // namespace s21
