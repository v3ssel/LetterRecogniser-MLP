#include "EmnistMLPTrainer.h"

namespace s21 {
EMNISTMLPTrainer::EMNISTMLPTrainer(const EpochCb &epoch_callback,
                                   const ProcessCb &process_callback) {
  _epoch_callback = epoch_callback;
  _process_callback = process_callback;
}

std::vector<double> EMNISTMLPTrainer::train(
    const std::unique_ptr<MLPModel> &model, const std::string &dataset_path,
    const size_t epochs) {
  _process_callback(0, MLPTrainStages::STARTING);
  std::vector<double> errors;

  try {
    errors.reserve(epochs);

    std::unique_ptr<EMNISTDatasetReader> reader =
        std::make_unique<EMNISTDatasetReader>();
    reader->open(dataset_path);

    size_t output_size = model->getLayersSize().back();
    size_t dataset_size = reader->getNumberOfLines();

    std::vector<double> expected(output_size, 0.0l);

    for (size_t i = 0; i < epochs; i++) {
      _process_callback(i + 1, MLPTrainStages::TRAINING);
      size_t accurancy = 0;
      double mse = 0;
      reader->open(dataset_path);

      while (true) {
        if (_stop) {
          _stop = false;
          return errors;
        }

        EMNISTData data = reader->readLine();
        if (data.result == (size_t)-1) break;

        expected[data.result - 1] = 1.0l;
        auto &&actual = model->feedForward(data.image);
        mse += calculateMSE(expected, actual);

        if (model->getPrediction(actual) == (data.result - 1)) {
          accurancy++;
        }

        model->backPropagation(expected);
        expected[data.result - 1] = 0.0l;
      }

      mse /= static_cast<double>(dataset_size);
      errors.push_back(mse);
      _epoch_callback(i + 1, mse, accurancy * 100.0l / dataset_size);
    }
  } catch (std::exception &e) {
    throw std::runtime_error("EMNISTMLPTrainer::train: " +
                             std::string(e.what()));
  }
  _process_callback(epochs, MLPTrainStages::DONE);

  return errors;
}

std::vector<double> EMNISTMLPTrainer::crossValidation(
    const std::unique_ptr<MLPModel> &model, const std::string &dataset_path,
    const size_t k_groups) {
  _process_callback(0, MLPTrainStages::STARTING);
  std::vector<double> errors;

  try {
    errors.reserve(k_groups);

    std::unique_ptr<EMNISTDatasetReader> reader =
        std::make_unique<EMNISTDatasetReader>();
    reader->open(dataset_path);

    size_t output_size = model->getLayersSize().back();
    size_t dataset_size = reader->getNumberOfLines();

    if (k_groups > dataset_size) {
      throw std::runtime_error(
          "EMNISTMLPTrainer::crossValidation: k_groups > dataset_size");
    }

    size_t group_size = dataset_size / k_groups;
    size_t group_start = 0, group_end = group_size;

    std::vector<double> expected(output_size, 0.0l);
    std::vector<EMNISTData> testingDataset;
    testingDataset.reserve(group_size);

    for (size_t k = 0; k < k_groups; k++) {
      _process_callback(k + 1, MLPTrainStages::TRAINING);
      double mse = 0;
      reader->open(dataset_path);

      for (size_t elem = 0; elem < dataset_size; elem++) {
        if (_stop) {
          _stop = false;
          return errors;
        }

        EMNISTData data = reader->readLine();

        if (elem >= group_start && elem < group_end && k_groups != 1) {
          testingDataset.push_back(data);
          continue;
        }

        expected[data.result - 1] = 1.0l;
        model->feedForward(data.image);
        model->backPropagation(expected);
        expected[data.result - 1] = 0.0l;
      }
      size_t accurancy = 0;

      _process_callback(k + 1, MLPTrainStages::TESTING);
      for (auto &elem : testingDataset) {
        if (_stop) {
          _stop = false;
          return errors;
        }

        auto &&actual = model->feedForward(elem.image);
        mse += calculateMSE(expected, actual);

        if ((elem.result - 1) == model->getPrediction(actual)) {
          accurancy++;
        }
      }

      mse /= static_cast<double>(testingDataset.size());
      group_start = group_end;
      group_end += group_size;
      testingDataset.clear();

      errors.push_back(mse);
      _epoch_callback(k + 1, mse, accurancy * 100.0l / (group_size));
    }
  } catch (std::exception &e) {
    throw std::runtime_error("EMNISTMLPTrainer::crossValidation: " +
                             std::string(e.what()));
  }
  _process_callback(k_groups, MLPTrainStages::DONE);

  return errors;
}

MLPTestMetrics EMNISTMLPTrainer::test(const std::unique_ptr<MLPModel> &model,
                                      const std::string &dataset_path,
                                      const size_t percent) {
  if (percent > 100) {
    throw std::invalid_argument("EMNISTMLPTrainer::test: percent > 100");
  }

  _process_callback(0, MLPTrainStages::STARTING);
  MLPTestMetrics metrics;

  try {
    std::unique_ptr<EMNISTDatasetReader> reader =
        std::make_unique<EMNISTDatasetReader>();
    reader->open(dataset_path);

    size_t dataset_size = reader->getNumberOfLines();
    size_t test_count =
        static_cast<size_t>(std::ceil(dataset_size * percent / 100.0l));

    std::vector<TFMetrics> submetrics(model->getLayersSize().back());
    size_t accurancy_percent = 0;

    _process_callback(0, MLPTrainStages::TESTING);
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < test_count; i++) {
      if (_stop) {
        _stop = false;
        return metrics;
      }

      EMNISTData data = reader->readLine();

      auto actual = model->feedForward(data.image);
      size_t got = model->getPrediction(actual);
      size_t expected = data.result - 1;

      for (size_t j = 0; j < actual.size(); j++) {
        if (expected == j && got == j) {
          accurancy_percent++;
          submetrics[j].tp++;
        } else if (expected == j && got != j) {
          submetrics[j].fn++;
        } else if (expected != j && got == j) {
          submetrics[j].fp++;
        } else if (expected != j && got != j) {
          submetrics[j].tn++;
        }
      }
    }
    auto end = std::chrono::high_resolution_clock::now();

    calculateMetrics(metrics, submetrics);
    metrics.accurancy_percent = accurancy_percent * 100.0l / test_count;
    metrics.testing_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  } catch (std::exception &e) {
    throw std::runtime_error("EMNISTMLPTrainer::test: " +
                             std::string(e.what()));
  }
  _process_callback(1, MLPTrainStages::DONE);

  return metrics;
}

void EMNISTMLPTrainer::stop() { _stop = true; }

double EMNISTMLPTrainer::calculateMSE(const std::vector<double> &expected,
                                      const std::vector<double> &actual) {
  double mse = 0;
  for (size_t i = 0; i < expected.size(); i++) {
    double val = actual[i] - expected[i];
    mse += val * val;
  }

  return mse;
}

void EMNISTMLPTrainer::calculateMetrics(MLPTestMetrics &metrics,
                                        std::vector<TFMetrics> &submetrics) {
  for (auto &i : submetrics) {
    if (i.tp + i.fp + i.fn + i.tn != 0)
      metrics.accurancy += static_cast<double>(i.tp + i.tn) /
                           static_cast<double>(i.tp + i.fp + i.fn + i.tn);

    double tmp_precision = 0.0l, tmp_recall = 0.0l;
    if (i.tp + i.fp != 0)
      tmp_precision =
          static_cast<double>(i.tp) / static_cast<double>(i.tp + i.fp);

    if (i.tp + i.fn != 0)
      tmp_recall = static_cast<double>(i.tp) / static_cast<double>(i.tp + i.fn);

    if (tmp_precision + tmp_recall != 0.0l)
      metrics.f_measure +=
          2.0l * ((tmp_precision * tmp_recall) / (tmp_precision + tmp_recall));

    metrics.precision += tmp_precision;
    metrics.recall += tmp_recall;
  }

  double dsize = static_cast<double>(submetrics.size());
  metrics.accurancy /= dsize;
  metrics.precision /= dsize;
  metrics.recall /= dsize;
  metrics.f_measure /= dsize;
}
}  // namespace s21
