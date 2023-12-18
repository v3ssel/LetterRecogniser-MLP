#include "EmnistMLPTrainer.h"
#include <iostream>
#include <algorithm>

namespace s21 {
    std::vector<double> EMNISTMLPTrainer::train(const std::unique_ptr<MLPModel> &model,
                                                const std::string &dataset_path,
                                                const size_t epochs,
                                                const std::function<void(double)>& callback) {
        
        std::unique_ptr<EMNISTDatasetReader> reader = std::make_unique<EMNISTDatasetReader>();
        size_t output_size = model->getLayersSize().back();
        std::vector<double> expected(output_size, 0.0l), errors;
        errors.reserve(epochs);

        double mse = 0;
        size_t accurancy = 0;

        try {
            for (size_t i = 0; i < epochs; i++) {
                size_t t = 0, acur = 0;
                reader->open(dataset_path);
                
                while (true) {
                    EMNISTData data = reader->readLine();
                    if (data.result == (size_t)-1) break;
                    
                    expected[data.result - 1] = 1.0l;

                    auto actual = model->feedForward(data.image);
                    mse += calculateMSE(expected, actual);

                    size_t got = model->getPrediction(actual);
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
                    expected[data.result - 1] = 0.0l;
                }
                std::cout << "Epoch " << i + 1 << " accurancy: " << accurancy << std::endl;
                accurancy = 0;

                errors.push_back(mse);
                callback(mse);
                mse = 0;
                // с выводом на экран контрольных значений ошибки для каждой эпохи
                // отчета в виде графика изменения ошибки, посчитанной на тестовой выборке, для каждой эпохи

            }
        } catch (std::exception &e) {
            throw std::runtime_error("EMNISTMLPTrainer::train: " + std::string(e.what()));
        }

        return errors;
    }

    std::vector<double> EMNISTMLPTrainer::crossValidationTrain(
                                            std::unique_ptr<MLPModel> &model,
                                            std::string &dataset_path,
                                            size_t k_groups,
                                            std::function<void(double)> callback) {

        std::unique_ptr<EMNISTDatasetReader> reader = std::make_unique<EMNISTDatasetReader>();
        size_t output_size = model->getLayersSize().back();
        std::vector<double> expected(output_size, 0.0l), errors(k_groups);
        size_t accurancy = 0, mse = 0;
        
        return std::vector<double>();
    }

    void EMNISTMLPTrainer::test(const std::unique_ptr<MLPModel> &model, const std::string &dataset_path, const size_t percent) {
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

            size_t got = model->getPrediction(data.image);
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
    
    double EMNISTMLPTrainer::calculateMSE(const std::vector<double> &expected, const std::vector<double> &actual) {
        double mse = 0;
        for (size_t i = 0; i < expected.size(); i++) {
            mse += (actual[i] - expected[i]) * (actual[i] - expected[i]);
        }

        return mse;
    }
}
