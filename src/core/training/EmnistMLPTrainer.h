#ifndef _EMNISTMLPTRAINER_H_
#define _EMNISTMLPTRAINER_H_

#include <cmath>
#include "../MLPTrainer.h"
#include "EmnistDatasetReader.h"

namespace s21 {
    class EMNISTMLPTrainer : public MLPTrainer {
       public:
        void train(std::unique_ptr<MLPModel>& model, std::string &dataset_path, size_t epochs) override;
        void test(std::unique_ptr<MLPModel>& model, std::string &dataset_path, size_t percent) override;        
    };
}

#endif // _EMNISTMLPTRAINER_H_
