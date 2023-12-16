#ifndef _MLPTRAINER_H_
#define _MLPTRAINER_H_

#include <memory>
#include <string>
#include "MLPModel.h"

namespace s21 {
    class MLPTrainer {
    public:
        virtual ~MLPTrainer() {}

        virtual void train(std::unique_ptr<MLPModel>& model, std::string &dataset_path, size_t epochs) = 0;
    };
}

#endif // _MLPTRAINER_H_
