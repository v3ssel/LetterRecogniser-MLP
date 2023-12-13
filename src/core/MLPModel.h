#ifndef _MLPMODEL_H_
#define _MLPMODEL_H_

#include <vector>

namespace s21 {
    class MLPModel {
       public:
        virtual std::vector<double> feedForward(std::vector<double>&) = 0;
        virtual void backPropagation() = 0;
        virtual void randomFill() = 0;

        virtual std::vector<size_t> getLayersSize() = 0;

        virtual void setWeights(std::vector<double> weights) = 0;
        virtual std::vector<double> getWeights() = 0;

        virtual void setBiases(std::vector<double> biases) = 0;
        virtual std::vector<double> getBiases() = 0;
    };
}

#endif // _MLPMODEL_H_