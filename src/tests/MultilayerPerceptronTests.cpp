#include <future>
#include <filesystem>
#include <gtest/gtest.h>

#include "../core/MultilayerPerceptron.h"
#include "../core/matrix/MatrixModel.h"
#include "../core/training/EmnistMLPTrainer.h"
#include "../core/serializer/FileMLPSerializer.h"

const std::string kDatasetPath = "C:\\Coding\\Projects\\CPP7_MLP-1\\src\\tests\\assets\\emnist-sample.txt";
const std::string kModelPath = "C:\\Coding\\Projects\\CPP7_MLP-1\\src\\tests\\assets\\model-test.txt";
const std::string kSmartModelPath = "C:\\Coding\\Projects\\CPP7_MLP-1\\src\\tests\\assets\\smartmodel.txt";

void epoch_callback(size_t, double, double) {

}

void process_callback(size_t, s21::MLPTrainStages) {

}

s21::MultilayerPerceptron getMLP() {
    std::unique_ptr<s21::MLPModel> model = std::make_unique<s21::MatrixModel>(10, 3, 2, 5, 0.3);
    model->randomFill();
    std::unique_ptr<s21::MLPTrainer> trainer = std::make_unique<s21::EMNISTMLPTrainer>(epoch_callback, process_callback);
    std::unique_ptr<s21::MLPSerializer> serializer = std::make_unique<s21::FileMLPSerializer>();

    return s21::MultilayerPerceptron(model, trainer, serializer);
}

TEST(MultilayerPerceptron, ImportExport) {
    s21::MultilayerPerceptron mlp = getMLP();
    EXPECT_NO_THROW(mlp.importModel(kModelPath));

    std::filesystem::path path(kModelPath);
    std::string export_path = path.replace_filename(path.stem().string() + "-export" + path.extension().string()).string();
    EXPECT_NO_THROW(mlp.exportModel(export_path));
    
    std::ifstream f1(kModelPath, std::ifstream::binary | std::ifstream::ate);
    std::ifstream f2(export_path, std::ifstream::binary | std::ifstream::ate);

    EXPECT_TRUE(std::equal(std::istreambuf_iterator<char>(f1.rdbuf()),
                           std::istreambuf_iterator<char>(),
                           std::istreambuf_iterator<char>(f2.rdbuf())));
}

TEST(MultilayerPerceptron, Prediction) {
    s21::MultilayerPerceptron mlp = getMLP();
    EXPECT_NO_THROW(mlp.importModel(kModelPath));

    char p = mlp.prediction({ 0, 1, 0, 2, 3, 0, 5, 2, 3, 2 });
    EXPECT_EQ(p, 'A');
}

TEST(MultilayerPerceptron, Testing) {
    std::unique_ptr<s21::MLPModel> model = std::make_unique<s21::MatrixModel>(784, 26, 5, 140, 0.1);
    s21::MultilayerPerceptron mlp = getMLP();
    mlp.setModel(model);
    EXPECT_NO_THROW(mlp.importModel(kSmartModelPath));
    
    s21::MLPTestMetrics metrics = mlp.testing(kDatasetPath, 100);    

    EXPECT_NE(metrics.accurancy, 0.0);
    EXPECT_NE(metrics.accurancy_percent, 0.0);
    EXPECT_NE(metrics.precision, 0.0);
    EXPECT_NE(metrics.recall, 0.0);
    EXPECT_NE(metrics.f_measure, 0.0);
}

TEST(MultilayerPerceptron, Learning) {
    std::unique_ptr<s21::MLPModel> model = std::make_unique<s21::MatrixModel>(784, 26, 5, 140, 0.1);
    s21::MultilayerPerceptron mlp = getMLP();
    mlp.setModel(model);
    EXPECT_NO_THROW(mlp.importModel(kSmartModelPath));

    double percent = mlp.testing(kDatasetPath, 100).accurancy_percent;
    mlp.learning(false, kDatasetPath, 20);

    EXPECT_NE(percent, mlp.testing(kDatasetPath, 100).accurancy_percent);
}

TEST(MultilayerPerceptron, StopLearning) {
    std::unique_ptr<s21::MLPModel> model = std::make_unique<s21::MatrixModel>(784, 26, 5, 140, 0.1);
    s21::MultilayerPerceptron mlp = getMLP();
    mlp.setModel(model);
    EXPECT_NO_THROW(mlp.importModel(kSmartModelPath));

    double prev_percent = mlp.testing(kDatasetPath, 100).accurancy_percent;

    auto ft = std::async(std::launch::async, [&]() {
        mlp.learning(false, kDatasetPath, 200);
    });

    EXPECT_TRUE(ft.valid());
    mlp.stopTrainer();
    ft.get();
    EXPECT_FALSE(ft.valid());

    double act_percent = mlp.testing(kDatasetPath, 100).accurancy_percent;
    EXPECT_EQ(prev_percent, act_percent);
}

TEST(MultilayerPerceptron, LearningRate) {
    s21::MultilayerPerceptron mlp = getMLP();
    double prev_lr = mlp.getLearningRate();
    mlp.setLearningRate(0.5);

    EXPECT_NE(prev_lr, mlp.getLearningRate());
}