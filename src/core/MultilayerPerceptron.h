#ifndef _MULTILAYERPERCEPTRON_H
#define _MULTILAYERPERCEPTRON_H

#include <vector>
#include <memory>
#include <string>
#include <fstream>
#include "MLPModel.h"

namespace s21 {
    class MultilayerPerceptron {
       public:
        // можно сделать билдера модели, через него собираем модель и подаем сюда
        MultilayerPerceptron(std::unique_ptr<MLPModel>& model);

        void importModel(std::string filepath);
        void exportModel(std::string filepath);

        std::vector<std::vector<double>> learning();
        char prediction();
       
       private:
        std::unique_ptr<MLPModel> model;
    };
}  // namespace s21

#endif //_MULTILAYERPERCEPTRON_H
