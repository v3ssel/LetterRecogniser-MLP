#ifndef _MLPMODEL_H_
#define _MLPMODEL_H_

#include <vector>

namespace s21 {
class MLPModel {
   public:
    virtual ~MLPModel() = default;

    virtual std::size_t getPrediction(const std::vector<double>&) = 0;
    virtual std::vector<double> feedForward(const std::vector<double>&) = 0;
    virtual void backPropagation(const std::vector<double>& target) = 0;
    virtual void randomFill() = 0;

    virtual std::vector<std::size_t> getLayersSize() const = 0;

    virtual void setWeights(const std::vector<double>& weights) = 0;
    virtual std::vector<double> getWeights() const = 0;

    virtual void setBiases(const std::vector<double>& biases) = 0;
    virtual std::vector<double> getBiases() const = 0;

    virtual void setLearningRate(double) = 0;
    virtual double getLearningRate() const = 0;
};
}  // namespace s21

#endif  // _MLPMODEL_H_