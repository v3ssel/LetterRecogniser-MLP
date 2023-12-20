#ifndef _MLPTRAINER_H_
#define _MLPTRAINER_H_

#include <memory>
#include <string>
#include "MLPModel.h"
#include "MLPTrainStages.h"
#include "MLPTestMetrics.h"

namespace s21 {
    class MLPTrainer {
       public:
        virtual ~MLPTrainer() {}

        virtual std::vector<double> train(const std::unique_ptr<MLPModel>& model,
                                          const std::string &dataset_path, 
                                          const size_t epochs) = 0;
        
        virtual std::vector<double> crossValidation(
                                  const std::unique_ptr<MLPModel>& model,
                                  const std::string &dataset_path,
                                  const size_t k_groups) = 0;
        
        virtual MLPTestMetrics test(const std::unique_ptr<MLPModel>& model, const std::string &dataset_path, const size_t percent) = 0;

        virtual void stop() = 0;
    };
}

#endif // _MLPTRAINER_H_
