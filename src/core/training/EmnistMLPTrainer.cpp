#include "EmnistMLPTrainer.h"
#include <iostream>
#include <algorithm>

namespace s21 {
    EMNISTMLPTrainer::EMNISTMLPTrainer(const std::function<void(size_t, double, double)> &epoch_callback,
                                       const std::function<void(size_t, MLPTrainStages)> &process_callback) {
        _epoch_callback = epoch_callback;
        _process_callback = process_callback;
    }

    std::vector<double> EMNISTMLPTrainer::train(const std::unique_ptr<MLPModel> &model,
                                                const std::string &dataset_path,
                                                const size_t epochs) {
        _process_callback(0, MLPTrainStages::STARTING);
        std::vector<double> errors;
        
        try {
            errors.reserve(epochs);

            std::unique_ptr<EMNISTDatasetReader> reader = std::make_unique<EMNISTDatasetReader>();
            reader->open(dataset_path);

            size_t output_size = model->getLayersSize().back();
            size_t dataset_size = reader->getNumberOfLines();
            
            std::vector<double> expected(output_size, 0.0l);


            for (size_t i = 0; i < epochs; i++) {
                _process_callback(i + 1, MLPTrainStages::TRAINING);
                size_t accurancy = 0;
                double mse = 0;
                reader->open(dataset_path);
                
                while (true) {
                    if (_stop) {
                        _stop = false;
                        return errors;
                    }
                    
                    EMNISTData data = reader->readLine();
                    if (data.result == (size_t)-1) break;
                    
                    expected[data.result - 1] = 1.0l;
                    auto&& actual = model->feedForward(data.image);
                    mse += calculateMSE(expected, actual);

                    if (model->getPrediction(actual) == (data.result - 1)) {
                        accurancy++;
                    }
                    
                    model->backPropagation(expected);
                    expected[data.result - 1] = 0.0l;
                }

                errors.push_back(mse);
                _epoch_callback(i + 1, mse, accurancy * 100.0l / dataset_size);
            }
        } catch (std::exception &e) {
            throw std::runtime_error("EMNISTMLPTrainer::train: " + std::string(e.what()));
        }
        _process_callback(epochs, MLPTrainStages::DONE);

        return errors;
    }

    std::vector<double> EMNISTMLPTrainer::crossValidation(
                                    const std::unique_ptr<MLPModel> &model,
                                    const std::string &dataset_path,
                                    const size_t k_groups) {
        _process_callback(0, MLPTrainStages::STARTING);
        std::vector<double> errors;

        try {
            errors.reserve(k_groups);

            std::unique_ptr<EMNISTDatasetReader> reader = std::make_unique<EMNISTDatasetReader>();
            reader->open(dataset_path);

            size_t output_size = model->getLayersSize().back();
            size_t dataset_size = reader->getNumberOfLines();

            size_t group_size = dataset_size / k_groups;
            size_t group_start = 0, group_end = group_size;

            std::vector<double> expected(output_size, 0.0l);
            std::vector<EMNISTData> testingDataset;
            testingDataset.reserve(group_size);

            for (size_t k = 0; k < k_groups; k++) {
                _process_callback(k + 1, MLPTrainStages::TRAINING);
                double mse = 0;
                // size_t t = 0, acur = 0; // for debug
                reader->open(dataset_path);

                for (size_t elem = 0; elem < dataset_size; elem++) {
                    if (_stop) {
                        _stop = false;
                        return errors;
                    }

                    EMNISTData data = reader->readLine();

                    // t++;
                    if (elem >= group_start && elem < group_end && k_groups != 1) {
                        testingDataset.push_back(data);
                        continue;
                    }

                    // if (t % 1000 == 0) {
                    //     std::cout << "Group " << k + 1 << " step " << t << std::endl;
                    //     acur = 0;
                    // }

                    expected[data.result - 1] = 1.0l;
                    model->feedForward(data.image);
                    model->backPropagation(expected);
                    expected[data.result - 1] = 0.0l;
                }
                std::cout << "Group " << k + 1 << std::endl;
                // t = group_size * (k + 1);

                size_t accurancy = 0;

                _process_callback(k + 1, MLPTrainStages::TESTING);
                for (auto& elem : testingDataset) {
                    if (_stop) {
                        _stop = false;
                        return errors;
                    }
                    
                    auto&& actual = model->feedForward(elem.image);
                    mse += calculateMSE(expected, actual);

                    if ((elem.result - 1) == model->getPrediction(actual)) {
                        accurancy++;
                        // acur++;
                    }
                    
                    // t++;
                    // if (t % 1000 == 0) {
                    //     std::cout << "Group " << k + 1 << " step " << t << " test accurancy per thousand: " << acur << " accurancy: " << accurancy << std::endl;
                    //     acur = 0;
                    // }
                }
                // std::cout << "Group " << k + 1 << " test accurancy: " << accurancy << std::endl;

                group_start = group_end;
                group_end += group_size;
                testingDataset.clear();

                errors.push_back(mse);
                _epoch_callback(k + 1, mse, accurancy * 100.0l / (group_size));
            }
        } catch (std::exception &e) {
            throw std::runtime_error("EMNISTMLPTrainer::crossValidation: " + std::string(e.what()));
        }
        _process_callback(k_groups, MLPTrainStages::DONE);

        return errors;
    }

    void EMNISTMLPTrainer::test(const std::unique_ptr<MLPModel> &model, 
                                const std::string &dataset_path, 
                                const size_t percent) {
        std::unique_ptr<EMNISTDatasetReader> reader = std::make_unique<EMNISTDatasetReader>();
        reader->open(dataset_path);

        size_t line_count = reader->getNumberOfLines();
        std::cout << "Dataset size: " << line_count << std::endl;
        size_t test_count = static_cast<size_t>(std::ceil(line_count * percent / 100.0l));
        std::cout << "Test dataset size: " << test_count << std::endl;

        size_t accurancy = 0;
        std::vector<bool> v(26, false);

        for (size_t i = 0; i < test_count; i++) {
            EMNISTData data = reader->readLine();
            if (data.result == (size_t)-1) break;

            size_t got = model->getPrediction(model->feedForward(data.image));
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

    void EMNISTMLPTrainer::stop() {
        _stop = true;
    }

    double EMNISTMLPTrainer::calculateMSE(const std::vector<double> &expected, const std::vector<double> &actual) {
        double mse = 0;
        for (size_t i = 0; i < expected.size(); i++) {
            mse += (actual[i] - expected[i]) * (actual[i] - expected[i]);
        }

        return mse / expected.size();
    }
}
