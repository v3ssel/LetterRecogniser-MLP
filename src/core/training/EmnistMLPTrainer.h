#ifndef _EMNISTMLPTRAINER_H_
#define _EMNISTMLPTRAINER_H_

#include <cmath>
#include <functional>
#include "../MLPTrainer.h"
#include "EmnistDatasetReader.h"

namespace s21 {
    class EMNISTMLPTrainer : public MLPTrainer {
       public:
        EMNISTMLPTrainer(const std::function<void(size_t, double, double)> &epoch_callback,
                         const std::function<void(size_t, MLPTrainStages)> &process_callback);

        std::vector<double> train(const std::unique_ptr<MLPModel>& model, 
                                  const std::string& dataset_path,
                                  const size_t epochs) override;
                                  
        std::vector<double> crossValidation(
                                  const std::unique_ptr<MLPModel>& model,
                                  const std::string &dataset_path,
                                  const size_t k_groups) override;
        
        void test(const std::unique_ptr<MLPModel>& model, const std::string& dataset_path, const size_t percent) override;        
        
        void stop() override;

       private:
        double calculateMSE(const std::vector<double> &expected, const std::vector<double> &actual);
        bool _stop = false;
        std::function<void(size_t, double, double)> _epoch_callback;
        std::function<void(size_t, MLPTrainStages)> _process_callback;
        
    };
}

#endif // _EMNISTMLPTRAINER_H_
