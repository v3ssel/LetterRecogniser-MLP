#ifndef _MLPMODEL_H_
#define _MLPMODEL_H_

#include <vector>

namespace s21 {
    class MLPModel {
       public:
        virtual std::vector<double> feedForward(std::vector<double>&) = 0;
        virtual void backPropagation() = 0;
    };
}

#endif // _MLPMODEL_H_