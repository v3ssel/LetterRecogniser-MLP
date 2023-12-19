#ifndef _MLPTRAINER_H_
#define _MLPTRAINER_H_

#include <memory>
#include <string>
#include "MLPModel.h"
#include "MLPTrainStages.h"

namespace s21 {
    class MLPTrainer {
       public:
        virtual ~MLPTrainer() {}

        virtual std::vector<double> train(const std::unique_ptr<MLPModel>& model,
                                          const std::string &dataset_path, 
                                          const size_t epochs,
                                          const std::function<void(size_t, double, double)>& callback,
                                          const std::function<void(size_t, MLPTrainStages)>& process_callback) = 0;
        
        virtual std::vector<double> crossValidation(
                                  const std::unique_ptr<MLPModel>& model,
                                  const std::string &dataset_path,
                                  const size_t k_groups,
                                  const std::function<void(size_t, double, double)>& callback,
                                  const std::function<void(size_t, MLPTrainStages)>& process_callback) = 0;
        
        virtual void test(const std::unique_ptr<MLPModel>& model, const std::string &dataset_path, const size_t percent) = 0;

        virtual void stop() = 0;
    };
}

#endif // _MLPTRAINER_H_
