#include "EmnistMLPTrainer.h"
#include <iostream>
#include <algorithm>

namespace s21 {
    void EMNISTMLPTrainer::train(std::unique_ptr<MLPModel> &model, std::string &dataset_path, size_t epochs) {
        size_t accurancy = 0;
        std::unique_ptr<EMNISTDatasetReader> reader = std::make_unique<EMNISTDatasetReader>();

        for (size_t i = 0; i < epochs; i++) {
            size_t t = 0, acur = 0, maxa = 0, mina = -1;
            reader->open(dataset_path);
            
            while (true) {
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
                    maxa = std::max(maxa, acur);
                    mina = std::min(mina, acur);
                    acur = 0;
                }
                
                model->backPropagation(expected);
            }
            std::cout << "Epoch " << i + 1 << " accurancy: " << accurancy << " max: " << maxa << " min: " << mina << std::endl;
            accurancy = 0;
        }
    }
    
    void EMNISTMLPTrainer::test(std::unique_ptr<MLPModel> &model, std::string &dataset_path, size_t percent) {
        std::unique_ptr<EMNISTDatasetReader> reader = std::make_unique<EMNISTDatasetReader>();
        reader->open(dataset_path);
        size_t line_count = reader->getNumberOfLines();
        std::cout << "Dataset size: " << line_count << std::endl;

        size_t test_count = static_cast<size_t>(std::ceil(line_count * percent / 100.0l));
        std::cout << "Test dataset size: " << test_count << std::endl;

        size_t accurancy = 0;
        std::vector<bool> v(26, false);

        for (size_t i = 0; i < test_count; i++) {
            size_t t = 0, acur = 0, maxa = 0, mina = -1;

            EMNISTData data = reader->readLine();
            if (data.result == (size_t)-1) break;

            size_t got = model->predict(data.image);
            if (got == (data.result - 1)) {
                v[got] = true;
                accurancy++;
            }
        }

        std::cout << "Test accurancy: " << accurancy 
                  << " percent: " << accurancy * 100 / test_count 
                  << " Get all? " << std::endl;
        for (size_t i = 0; i < 26; i++) {
            std::cout << char(i + 65) << ":" << v[i] << " ";
        }
        std::cout << std::endl;
    }
}
