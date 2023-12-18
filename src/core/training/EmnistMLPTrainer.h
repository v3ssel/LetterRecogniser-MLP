#ifndef _EMNISTMLPTRAINER_H_
#define _EMNISTMLPTRAINER_H_

#include <cmath>
#include <functional>
#include "../MLPTrainer.h"
#include "EmnistDatasetReader.h"

namespace s21 {
    class EMNISTMLPTrainer : public MLPTrainer {
       public:
        std::vector<double> train(const std::unique_ptr<MLPModel>& model, 
                                  const std::string& dataset_path,
                                  const size_t epochs,
                                  const std::function<void(double)>& callback) override;
                                  
        std::vector<double> crossValidationTrain(
                                  const std::unique_ptr<MLPModel>& model,
                                  const std::string &dataset_path,
                                  const size_t k_groups,
                                  const std::function<void(double)>& callback) override;
        void test(const std::unique_ptr<MLPModel>& model, const std::string& dataset_path, const size_t percent) override;        

       private:
        double calculateMSE(const std::vector<double> &expected, const std::vector<double> &actual);
    };
}

#endif // _EMNISTMLPTRAINER_H_
