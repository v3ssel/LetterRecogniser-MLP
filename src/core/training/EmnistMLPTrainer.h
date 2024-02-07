#ifndef _EMNISTMLPTRAINER_H_
#define _EMNISTMLPTRAINER_H_

#include <cmath>
#include <functional>
#include <numeric>

#include "../MLPTrainer.h"
#include "EmnistDatasetReader.h"
#include "TFMetrics.h"

namespace s21 {
class EMNISTMLPTrainer : public MLPTrainer {
   public:
    using EpochCb = std::function<void(size_t, double, double)>;
    using ProcessCb = std::function<void(size_t, MLPTrainStages)>;

    EMNISTMLPTrainer(const EpochCb& epoch_callback,
                     const ProcessCb& process_callback);

    std::vector<double> train(const std::unique_ptr<MLPModel>& model,
                              const std::string& dataset_path,
                              const size_t epochs) override;

    std::vector<double> crossValidation(const std::unique_ptr<MLPModel>& model,
                                        const std::string& dataset_path,
                                        const size_t k_groups) override;

    MLPTestMetrics test(const std::unique_ptr<MLPModel>& model,
                        const std::string& dataset_path,
                        const size_t percent) override;

    void stop() override;

   private:
    bool stop_;
    EpochCb epoch_callback_;
    ProcessCb process_callback_;

    double calculateMSE(const std::vector<double>& expected,
                        const std::vector<double>& actual);
    void calculateMetrics(MLPTestMetrics& metrics,
                          std::vector<TFMetrics>& submetrics);
};
}  // namespace s21

#endif  // _EMNISTMLPTRAINER_H_
