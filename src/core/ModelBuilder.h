#ifndef _MODELBUILDER_H_
#define _MODELBUILDER_H_

#include "graph/GraphModel.h"
#include "matrix/MatrixModel.h"

namespace s21 {
    enum class ModelType {
        Graph,
        Matrix
    };

    class ModelBuilder {
       public:
        ModelBuilder();

        ModelBuilder* setModelType(ModelType model_type);
        ModelBuilder* setInputLayerSize(size_t input_layer_size);
        ModelBuilder* setOutputLayerSize(size_t output_layer_size);
        ModelBuilder* setHiddenLayerSize(size_t hidden_layer_size);
        ModelBuilder* setLayers(size_t layers);
        ModelBuilder* setLearningRate(double learning_rate);
        
        std::unique_ptr<MLPModel> build();

       private:
        ModelType model_type_;
        size_t input_layer_size_;
        size_t output_layer_size_;
        size_t hidden_layer_size_;
        size_t layers_;
        double learning_rate_;
    };
}

#endif // _MODELBUILDER_H_
