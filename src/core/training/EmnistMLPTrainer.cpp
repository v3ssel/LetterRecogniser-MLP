#include "EmnistMLPTrainer.h"
#include <iostream>

namespace s21 {
    void EMNISTMLPTrainer::train(std::unique_ptr<MLPModel> &model, std::string &dataset_path, size_t epochs) {
        size_t accurancy = 0;
        std::unique_ptr<EMNISTDatasetReader> reader = std::make_unique<EMNISTDatasetReader>();

        for (size_t i = 0; i < epochs; i++) {
            size_t t = 0, acur = 0;
            reader->open(dataset_path);
            
            while (reader->is_open()) {
                EMNISTData data = reader->readLine();
                if (data.result == (size_t)-1) break;
                
                auto layer_size = model->getLayersSize();

                if (layer_size[0] != data.image.size()) {
                    throw std::invalid_argument("EMNISTMLPTrainer::train: This model cant be used with this dataset");
                }

                std::vector<double> expected(layer_size.back(), 0.0l);
                expected[data.result - 1] = 1.0l;

                size_t got = model->predict(data.image);
                if (got == (data.result - 1)) {
                    accurancy++;
                    acur++;
                }

                t++;
                if (t % 1000 == 0) {
                    std::cout << "Epoch " << i + 1 << " step " << t << " accurancy per thousand: " << acur << " accurancy: " << accurancy << std::endl;
                    acur = 0;
                }
                
                model->backPropagation(expected);
            }
            std::cout << "Epoch " << i + 1 << " accurancy: " << accurancy << std::endl;
        }
    }
    
    void EMNISTMLPTrainer::test(std::unique_ptr<MLPModel> &model, std::string &dataset_path, size_t percent) {
        std::unique_ptr<EMNISTDatasetReader> reader = std::make_unique<EMNISTDatasetReader>();
        reader->open(dataset_path);
    }
}
