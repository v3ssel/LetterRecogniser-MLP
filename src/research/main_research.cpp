#include <iostream>
#include <chrono>

#include "../core/MultilayerPerceptron.h"
#include "../core/matrix/MatrixModel.h"
#include "../core/graph/GraphModel.h"
#include "../core/serializer/FileMLPSerializer.h"
#include "../core/training/EmnistMLPTrainer.h"

const std::string kModelPath = "C:\\Coding\\Projects\\CPP7_MLP-1\\src\\weights\\2-model-77.txt";
const std::string kDatasetPath = "C:\\Coding\\Projects\\CPP7_MLP-1\\datasets\\emnist-letters\\emnist-letters-test.csv";

void epochCb(size_t, double, double) {}
void processCb(size_t, s21::MLPTrainStages) {}

int main(int argc, char const **argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <model_type>[1 - Matrix, 2 - Graph]" << std::endl;
        return 1;
    }

    int type = 1;
    try {
        type = std::stoi(argv[1]);
        if (type != 1 && type != 2) {
            throw std::invalid_argument("<model_type>[1 - Matrix, 2 - Graph]");
        }
    } catch(std::exception& e) {
        std::cout << "Invalid model type: " << e.what() << std::endl;
        return 1;
    }

    s21::ModelBuilder builder;
    builder.setModelType(type == 1 ? s21::ModelType::Matrix : s21::ModelType::Graph)
          ->setInputLayerSize(784)
          ->setLayers(2)
          ->setHiddenLayerSize(140)
          ->setOutputLayerSize(26)
          ->setLearningRate(0.1);

    std::unique_ptr<s21::MLPModel> model = builder.build();
    std::unique_ptr<s21::MLPSerializer> serializer = std::make_unique<s21::FileMLPSerializer>();
    std::unique_ptr<s21::MLPTrainer> trainer = std::make_unique<s21::EMNISTMLPTrainer>(epochCb, processCb);
    s21::MultilayerPerceptron mlp(model, trainer, serializer);
    mlp.importModel(kModelPath);
    
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < 1000; i++) {
        mlp.testing(kDatasetPath, 10);

        if (i == 9 || i == 99 || i == 999) {
            std::cout << "Count: "
                      <<  i + 1 
                      << "| Time: "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count()
                      << "ms" << std::endl;
        }
    }

    return 0;
}
