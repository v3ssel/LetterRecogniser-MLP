#include "ModelBuilder.h"

namespace s21 {
    ModelBuilder::ModelBuilder() {
        model_type_ = ModelType::Matrix;
        input_layer_size_ = 10;
        output_layer_size_ = 3;
        hidden_layer_size_ = 5;
        layers_ = 2;
        learning_rate_ = 0.1;
    }

    ModelBuilder *ModelBuilder::setModelType(ModelType model_type) {
        model_type_ = model_type;
        return this;
    }

    ModelBuilder *ModelBuilder::setInputLayerSize(size_t input_layer_size) {
        input_layer_size_ = input_layer_size;
        return this;
    }

    ModelBuilder *ModelBuilder::setOutputLayerSize(size_t output_layer_size) {
        output_layer_size_ = output_layer_size;
        return this;
    }

    ModelBuilder *ModelBuilder::setHiddenLayerSize(size_t hidden_layer_size) {
        hidden_layer_size_ = hidden_layer_size;
        return this;
    }

    ModelBuilder *ModelBuilder::setLayers(size_t layers) {
        layers_ = layers;
        return this;
    }

    ModelBuilder *ModelBuilder::setLearningRate(double learning_rate) {
        learning_rate_ = learning_rate;
        return this;
    }

    std::unique_ptr<MLPModel> ModelBuilder::build() {
        if (model_type_ == ModelType::Matrix) {
            return std::make_unique<MatrixModel>(input_layer_size_, output_layer_size_, layers_, hidden_layer_size_, learning_rate_);
        }

        return std::make_unique<GraphModel>(input_layer_size_, output_layer_size_, layers_, hidden_layer_size_, learning_rate_);
    }
}