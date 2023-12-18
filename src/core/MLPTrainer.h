#ifndef _MLPTRAINER_H_
#define _MLPTRAINER_H_

#include <memory>
#include <string>
#include "MLPModel.h"

namespace s21 {
    class MLPTrainer {
    public:
        virtual ~MLPTrainer() {}

        virtual std::vector<double> train(const std::unique_ptr<MLPModel>& model,
                                          const std::string &dataset_path, 
                                          const size_t epochs,
                                          const std::function<void(double)>& callback) = 0;
        
        virtual std::vector<double> crossValidationTrain(
                                  std::unique_ptr<MLPModel>& model,
                                  std::string &dataset_path,
                                  size_t k_groups,
                                  std::function<void(double)> callback) = 0;
        
        virtual void test(const std::unique_ptr<MLPModel>& model, const std::string &dataset_path, const size_t percent) = 0;
    };
}

#endif // _MLPTRAINER_H_
