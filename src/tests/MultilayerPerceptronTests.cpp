#include <future>
#include <filesystem>
#include <gtest/gtest.h>

#include "../core/MultilayerPerceptron.h"
#include "../core/matrix/MatrixModel.h"
#include "../core/graph/GraphModel.h"
#include "../core/training/EmnistMLPTrainer.h"
#include "../core/serializer/FileMLPSerializer.h"

const std::string kDatasetPath = "C:\\Coding\\Projects\\CPP7_MLP-1\\src\\tests\\assets\\emnist-sample.txt";
const std::string kModelPath = "C:\\Coding\\Projects\\CPP7_MLP-1\\src\\tests\\assets\\model-test.txt";
const std::string kSmartModelPath = "C:\\Coding\\Projects\\CPP7_MLP-1\\src\\tests\\assets\\smartmodel.txt";

void epoch_callback(size_t, double, double) {}
void process_callback(size_t, s21::MLPTrainStages) {}

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
                std::istreambuf_iterator<char>(), std::istreambuf_iterator<char>(f2.rdbuf())));

    f2.close();
    std::filesystem::remove(export_path);
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

TEST(MultilayerPerceptron, TwoModels) {
    s21::MultilayerPerceptron matrix_mlp = getMLP();
    s21::MultilayerPerceptron graph_mlp = getMLP();
    

    std::filesystem::path path(kModelPath);
    std::string m_model_path = path.replace_filename(path.stem().string() + "_matrix" + path.extension().string()).string();
    std::string g_model_path = path.replace_filename(path.stem().string() + "_graph" + path.extension().string()).string();

    matrix_mlp.importModel(kModelPath);
    graph_mlp.importModel(kModelPath);

    matrix_mlp.exportModel(m_model_path);
    graph_mlp.exportModel(g_model_path);

    std::ifstream f1(m_model_path, std::ifstream::binary | std::ifstream::ate);
    std::ifstream f2(g_model_path, std::ifstream::binary | std::ifstream::ate);

    EXPECT_TRUE(std::equal(std::istreambuf_iterator<char>(f1.rdbuf()),
                std::istreambuf_iterator<char>(), std::istreambuf_iterator<char>(f2.rdbuf())));

    f1.close();
    f2.close();
    std::filesystem::remove(m_model_path); 
    std::filesystem::remove(g_model_path); 
}

TEST(MultilayerPerceptron, randomizeModelWeights) {
    std::unique_ptr<s21::MLPModel> model = std::make_unique<s21::MatrixModel>(10, 3, 2, 5, 0.3);
    std::unique_ptr<s21::MLPTrainer> trainer = std::make_unique<s21::EMNISTMLPTrainer>(epoch_callback, process_callback);
    std::unique_ptr<s21::MLPSerializer> serializer = std::make_unique<s21::FileMLPSerializer>();

    auto weights = model->getWeights();
    auto bias = model->getBiases();
    
    s21::MultilayerPerceptron mlp(model, trainer, serializer);
    mlp.randomizeModelWeights();

    auto rweights = mlp.getModel()->getWeights();
    auto rbias = mlp.getModel()->getBiases();

    EXPECT_EQ(weights.size(), rweights.size());
    EXPECT_EQ(bias.size(), rbias.size());
    EXPECT_FALSE(weights == rweights);
    EXPECT_FALSE(bias == rbias);
}

TEST(MultilayerPerceptron, changeModelTypeAndLayersSize) {
    s21::MultilayerPerceptron mlp = getMLP();
    auto sizes = mlp.getModel()->getLayersSize();
    
    EXPECT_EQ(sizes.size(), 4);
    EXPECT_EQ(sizes[0], 10);
    EXPECT_EQ(sizes[1], 5);
    EXPECT_EQ(sizes[2], 5);
    EXPECT_EQ(sizes[3], 3);

    mlp.changeModelTypeAndLayersSize(s21::ModelType::Graph, 5);
    sizes = mlp.getModel()->getLayersSize();

    EXPECT_EQ(sizes.size(), 7);
    EXPECT_EQ(sizes[0], 10);
    EXPECT_EQ(sizes[1], 5);
    EXPECT_EQ(sizes[2], 5);
    EXPECT_EQ(sizes[3], 5);
    EXPECT_EQ(sizes[4], 5);
    EXPECT_EQ(sizes[5], 5);
    EXPECT_EQ(sizes[6], 3);
}
