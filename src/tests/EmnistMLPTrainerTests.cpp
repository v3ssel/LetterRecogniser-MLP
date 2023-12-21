#include <future>
#include <filesystem>
#include <gtest/gtest.h>
#include "../core/matrix/MatrixModel.h"
#include "../core/training/EmnistMLPTrainer.h"
#include "../core/serializer/FileMLPSerializer.h"

const std::string kDatasetPath = "C:\\Coding\\Projects\\CPP7_MLP-1\\src\\tests\\assets\\emnist-sample.txt";
const std::string kModelPath = "C:\\Coding\\Projects\\CPP7_MLP-1\\src\\weights\\5-model-78.txt";

TEST(EMNISTMLPTrainer, Training) {
    size_t ecb_count = 0, pcb_count = 0;
    std::vector<double> accurancy_percent;
    auto epoch_callback = [&ecb_count, &accurancy_percent](size_t, double, double accurancy) {
        ecb_count++;
        accurancy_percent.push_back(accurancy);
    };
    auto process_callback = [&pcb_count](size_t, s21::MLPTrainStages) {
        pcb_count++;
    };
    
    std::unique_ptr<s21::MLPTrainer> trainer = std::make_unique<s21::EMNISTMLPTrainer>(epoch_callback, process_callback);
    std::unique_ptr<s21::MLPModel> model = std::make_unique<s21::MatrixModel>(784, 26, 5, 140, 0.1);
    std::unique_ptr<s21::MLPSerializer> serializer = std::make_unique<s21::FileMLPSerializer>();
    serializer->deserialize(model, kModelPath);

    size_t epochs = 4;
    auto mse_metrics = trainer->train(model, kDatasetPath, epochs);
    
    EXPECT_FALSE(mse_metrics.empty());
    EXPECT_EQ(ecb_count, epochs);
    EXPECT_EQ(pcb_count, epochs + 2);
    EXPECT_TRUE(accurancy_percent.front() != accurancy_percent.back());
}

TEST(EMNISTMLPTrainer, CrossvalidationTrain) {
    size_t ecb_count = 0, pcb_count = 0;
    std::vector<double> accurancy_percent;
    auto epoch_callback = [&ecb_count, &accurancy_percent](size_t, double, double accurancy) {
        ecb_count++;
        accurancy_percent.push_back(accurancy);
    };
    auto process_callback = [&pcb_count](size_t, s21::MLPTrainStages) {
        pcb_count++;
    };
    
    std::unique_ptr<s21::MLPTrainer> trainer = std::make_unique<s21::EMNISTMLPTrainer>(epoch_callback, process_callback);
    std::unique_ptr<s21::MLPModel> model = std::make_unique<s21::MatrixModel>(784, 26, 5, 140, 0.1);
    std::unique_ptr<s21::MLPSerializer> serializer = std::make_unique<s21::FileMLPSerializer>();
    serializer->deserialize(model, kModelPath);

    size_t epochs = 4;
    auto mse_metrics = trainer->crossValidation(model, kDatasetPath, epochs);
    
    EXPECT_FALSE(mse_metrics.empty());
    EXPECT_EQ(ecb_count, epochs);
    EXPECT_EQ(pcb_count, 2 + 2 * epochs);
    EXPECT_TRUE(accurancy_percent.front() != accurancy_percent.back());
}

TEST(EMNISTMLPTrainer, Testing) {
    size_t pcb_count = 0;
    auto process_callback = [&pcb_count](size_t, s21::MLPTrainStages) {
        pcb_count++;
    };

    std::unique_ptr<s21::MLPTrainer> trainer = std::make_unique<s21::EMNISTMLPTrainer>([](size_t, double, double) {}, process_callback);
    std::unique_ptr<s21::MLPModel> model = std::make_unique<s21::MatrixModel>(784, 26, 5, 140, 0.1);
    std::unique_ptr<s21::MLPSerializer> serializer = std::make_unique<s21::FileMLPSerializer>();
    serializer->deserialize(model, kModelPath);

    s21::MLPTestMetrics metrics = trainer->test(model, kDatasetPath, 100);
    
    EXPECT_EQ(pcb_count, 3);
    EXPECT_NE(metrics.accurancy, 0.0);
    EXPECT_NE(metrics.accurancy_percent, 0.0);
    EXPECT_NE(metrics.precision, 0.0);
    EXPECT_NE(metrics.recall, 0.0);
    EXPECT_NE(metrics.f_measure, 0.0);
    EXPECT_NE(metrics.testing_time.count(), 0);
}

TEST(EMNISTMLPTrainer, TrainWithBrokenDataset) {
    std::unique_ptr<s21::MLPTrainer> trainer = std::make_unique<s21::EMNISTMLPTrainer>([](size_t, double, double) {}, [](size_t, s21::MLPTrainStages) {});
    std::unique_ptr<s21::MLPModel> model = std::make_unique<s21::MatrixModel>(784, 26, 5, 140, 0.1);

    std::filesystem::path path = kDatasetPath;
    std::string broken_dataset = path.replace_filename(path.stem().string() + "-broken" + path.extension().string()).string();

    EXPECT_THROW(trainer->train(model, broken_dataset, 1), std::runtime_error);
}

TEST(EMNISTMLPTrainer, CrossvalidationWithBrokenDataset) {
    std::unique_ptr<s21::MLPTrainer> trainer = std::make_unique<s21::EMNISTMLPTrainer>([](size_t, double, double) {}, [](size_t, s21::MLPTrainStages) {});
    std::unique_ptr<s21::MLPModel> model = std::make_unique<s21::MatrixModel>(784, 26, 5, 140, 0.1);

    std::filesystem::path path = kDatasetPath;
    std::string broken_dataset = path.replace_filename(path.stem().string() + "-broken" + path.extension().string()).string();

    EXPECT_THROW(trainer->crossValidation(model, broken_dataset, 1), std::runtime_error);
}

TEST(EMNISTMLPTrainer, CrossvalidationWithSoManyGroups) {
    std::unique_ptr<s21::MLPTrainer> trainer = std::make_unique<s21::EMNISTMLPTrainer>([](size_t, double, double) {}, [](size_t, s21::MLPTrainStages) {});
    std::unique_ptr<s21::MLPModel> model = std::make_unique<s21::MatrixModel>(784, 26, 5, 140, 0.1);

    std::filesystem::path path = kDatasetPath;
    std::string broken_dataset = path.replace_filename(path.stem().string() + "-broken" + path.extension().string()).string();

    EXPECT_THROW(trainer->crossValidation(model, broken_dataset, 1000), std::runtime_error);
}

TEST(EMNISTMLPTrainer, TestingWithBrokenDataset) {
    std::unique_ptr<s21::MLPTrainer> trainer = std::make_unique<s21::EMNISTMLPTrainer>([](size_t, double, double) {}, [](size_t, s21::MLPTrainStages) {});
    std::unique_ptr<s21::MLPModel> model = std::make_unique<s21::MatrixModel>(784, 26, 5, 140, 0.1);

    std::filesystem::path path = kDatasetPath;
    std::string broken_dataset = path.replace_filename(path.stem().string() + "-broken" + path.extension().string()).string();

    EXPECT_THROW(trainer->test(model, broken_dataset, 100), std::runtime_error);
}

TEST(EMNISTMLPTrainer, TestingWithHighPercent) {
    std::unique_ptr<s21::MLPTrainer> trainer = std::make_unique<s21::EMNISTMLPTrainer>([](size_t, double, double) {}, [](size_t, s21::MLPTrainStages) {});
    std::unique_ptr<s21::MLPModel> model = std::make_unique<s21::MatrixModel>(784, 26, 5, 140, 0.1);

    std::filesystem::path path = kDatasetPath;
    std::string broken_dataset = path.replace_filename(path.stem().string() + "-broken" + path.extension().string()).string();

    EXPECT_THROW(trainer->test(model, broken_dataset, 1000), std::invalid_argument);
}

TEST(EMNISTMLPTrainer, StopTraining) {
    std::unique_ptr<s21::MLPTrainer> trainer = std::make_unique<s21::EMNISTMLPTrainer>([](size_t, double, double) {}, [](size_t, s21::MLPTrainStages) {});
    std::unique_ptr<s21::MLPModel> model = std::make_unique<s21::MatrixModel>(784, 26, 5, 140, 0.1);
    std::unique_ptr<s21::MLPSerializer> serializer = std::make_unique<s21::FileMLPSerializer>();
    serializer->deserialize(model, kModelPath);

    std::vector<double> mse_errors;
    auto ft = std::async(std::launch::async, [&]() {
        mse_errors = trainer->train(model, kDatasetPath, 1000);
    });

    EXPECT_TRUE(ft.valid());
    trainer->stop();
    ft.get();
    EXPECT_FALSE(ft.valid());

    EXPECT_TRUE(mse_errors.empty());
}

TEST(EMNISTMLPTrainer, StopCrossvalidation) {
    std::unique_ptr<s21::MLPTrainer> trainer = std::make_unique<s21::EMNISTMLPTrainer>([](size_t, double, double) {}, [](size_t, s21::MLPTrainStages) {});
    std::unique_ptr<s21::MLPModel> model = std::make_unique<s21::MatrixModel>(784, 26, 5, 140, 0.1);
    std::unique_ptr<s21::MLPSerializer> serializer = std::make_unique<s21::FileMLPSerializer>();
    serializer->deserialize(model, kModelPath);

    std::vector<double> mse_errors;
    auto ft = std::async(std::launch::async, [&]() {
        mse_errors = trainer->crossValidation(model, kDatasetPath, 12);
    });

    EXPECT_TRUE(ft.valid());
    trainer->stop();
    ft.get();
    EXPECT_FALSE(ft.valid());

    EXPECT_TRUE(mse_errors.empty());
}

TEST(EMNISTMLPTrainer, StopTesting) {
    std::unique_ptr<s21::MLPTrainer> trainer = std::make_unique<s21::EMNISTMLPTrainer>([](size_t, double, double) {}, [](size_t, s21::MLPTrainStages) {});
    std::unique_ptr<s21::MLPModel> model = std::make_unique<s21::MatrixModel>(784, 26, 5, 140, 0.1);
    std::unique_ptr<s21::MLPSerializer> serializer = std::make_unique<s21::FileMLPSerializer>();
    serializer->deserialize(model, kModelPath);

    s21::MLPTestMetrics mse_errors;
    auto ft = std::async(std::launch::async, [&]() {
        mse_errors = trainer->test(model, kDatasetPath, 100);
    });

    EXPECT_TRUE(ft.valid());
    trainer->stop();
    ft.get();
    EXPECT_FALSE(ft.valid());

    EXPECT_EQ(mse_errors.accurancy, 0.0);
    EXPECT_EQ(mse_errors.accurancy_percent, 0.0);
    EXPECT_EQ(mse_errors.precision, 0.0);
    EXPECT_EQ(mse_errors.recall, 0.0);
    EXPECT_EQ(mse_errors.f_measure, 0.0);
}
