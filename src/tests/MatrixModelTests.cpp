#include <gtest/gtest.h>
#include "../core/matrix/MatrixModel.h"


TEST(MatrixModel, LearningRate) {
    std::unique_ptr<s21::MLPModel> model = std::make_unique<s21::MatrixModel>(10, 3, 2, 5, 0.1);
    EXPECT_EQ(model->getLearningRate(), 0.1);
    model->setLearningRate(0.15);
    EXPECT_EQ(model->getLearningRate(), 0.15);
}

TEST(MatrixModel, GetLayerSizes) {
    std::unique_ptr<s21::MLPModel> model = std::make_unique<s21::MatrixModel>(10, 3, 2, 5, 0.1);

    auto layer_sizes = model->getLayersSize();

    EXPECT_EQ(layer_sizes.size(), 4);
    EXPECT_EQ(layer_sizes[0], 10);
    EXPECT_EQ(layer_sizes[1], 5);
    EXPECT_EQ(layer_sizes[2], 5);
    EXPECT_EQ(layer_sizes[3], 3);
}

TEST(MatrixModel, GetWeights) {
    std::unique_ptr<s21::MLPModel> model = std::make_unique<s21::MatrixModel>(10, 3, 2, 5, 0.1);
    model->randomFill();

    auto weights = model->getWeights();

    EXPECT_EQ(weights.size(), 90);
    EXPECT_TRUE(std::any_of(weights.begin(), weights.end(), [](double v) { return v != 0; }));
}

TEST(MatrixModel, GetBiases) {
    std::unique_ptr<s21::MLPModel> model = std::make_unique<s21::MatrixModel>(10, 3, 2, 5, 0.1);
    model->randomFill();

    auto biases = model->getBiases();

    EXPECT_EQ(biases.size(), 13);
    EXPECT_TRUE(std::any_of(biases.begin(), biases.end(), [](double v) { return v != 0; }));
}

TEST(MatrixModel, GetAndSetWeights) {
    std::unique_ptr<s21::MLPModel> model = std::make_unique<s21::MatrixModel>(10, 3, 2, 5, 0.1);
    model->randomFill();
    auto weights = model->getWeights();

    std::unique_ptr<s21::MLPModel> model2 = std::make_unique<s21::MatrixModel>(10, 3, 2, 5, 0.1);
    model2->setWeights(weights);

    auto weights2 = model2->getWeights();

    EXPECT_EQ(weights.size(), 90);
    EXPECT_EQ(weights2.size(), 90);
    EXPECT_TRUE(weights == weights2);
}

TEST(MatrixModel, GetAndSetBiases) {
    std::unique_ptr<s21::MLPModel> model = std::make_unique<s21::MatrixModel>(10, 3, 2, 5, 0.1);
    model->randomFill();
    auto biases = model->getBiases();

    std::unique_ptr<s21::MLPModel> model2 = std::make_unique<s21::MatrixModel>(10, 3, 2, 5, 0.1);
    model2->setBiases(biases);

    auto biases2 = model2->getBiases();

    EXPECT_EQ(biases.size(), 13);
    EXPECT_EQ(biases2.size(), 13);
    EXPECT_TRUE(biases == biases2);
}

TEST(MatrixModel, RandomFill) {
    std::unique_ptr<s21::MLPModel> model = std::make_unique<s21::MatrixModel>(10, 3, 2, 5, 0.1);
    auto weights = model->getWeights();
    auto biases = model->getBiases();

    EXPECT_TRUE(std::all_of(weights.begin(), weights.end(), [](double v) { return v == 0; }));
    EXPECT_TRUE(std::all_of(biases.begin(), biases.end(), [](double v) { return v == 0; }));

    model->randomFill();
    weights = model->getWeights();
    biases = model->getBiases();

    EXPECT_TRUE(std::any_of(weights.begin(), weights.end(), [](double v) { return v != 0; }));
    EXPECT_TRUE(std::any_of(biases.begin(), biases.end(), [](double v) { return v != 0; }));
}

TEST(MatrixModel, SetWeights) {
    std::unique_ptr<s21::MLPModel> model = std::make_unique<s21::MatrixModel>(10, 3, 2, 5, 0.1);
    auto weights = model->getWeights();

    EXPECT_EQ(weights.size(), 90);
    EXPECT_TRUE(std::all_of(weights.begin(), weights.end(), [](double v) { return v == 0; }));

    std::vector<double> new_weights;
    for (int i = 0; i < weights.size(); i++) new_weights.push_back(i);

    model->setWeights(new_weights);
    weights = model->getWeights();

    EXPECT_TRUE(new_weights == weights);
}

TEST(MatrixModel, SetBadWeights) {
    std::unique_ptr<s21::MLPModel> model = std::make_unique<s21::MatrixModel>(10, 3, 2, 5, 0.1);

    EXPECT_THROW(model->setWeights({1, 2, 3}), std::out_of_range);
}

TEST(MatrixModel, SetBiases) {
    std::unique_ptr<s21::MLPModel> model = std::make_unique<s21::MatrixModel>(10, 3, 2, 5, 0.1);
    auto biases = model->getBiases();
    
    EXPECT_EQ(biases.size(), 13);
    EXPECT_TRUE(std::all_of(biases.begin(), biases.end(), [](double v) { return v == 0; }));

    std::vector<double> new_biases;
    for (int i = 0; i < biases.size(); i++) new_biases.push_back(i);

    model->setBiases(new_biases);
    biases = model->getBiases();

    EXPECT_TRUE(new_biases == biases);
}

TEST(MatrixModel, SetBadBiases) {
    std::unique_ptr<s21::MLPModel> model = std::make_unique<s21::MatrixModel>(10, 3, 2, 5, 0.1);

    EXPECT_THROW(model->setBiases({1, 2, 3}), std::out_of_range);
}

TEST(MatrixModel, DefaultModel) {
    std::unique_ptr<s21::MLPModel> model = std::make_unique<s21::MatrixModel>(10, 3, 2, 5, 0.1);

    auto sizes = model->getLayersSize();
    EXPECT_EQ(sizes.size(), 4);
    EXPECT_EQ(sizes[0], 10);
    EXPECT_EQ(sizes[1], 5);
    EXPECT_EQ(sizes[2], 5);
    EXPECT_EQ(sizes[3], 3);
    
    auto weights = model->getWeights();
    EXPECT_TRUE(std::all_of(weights.begin(), weights.end(), [](double v) { return v == 0; }));
    
    auto biases = model->getBiases();
    EXPECT_TRUE(std::all_of(biases.begin(), biases.end(), [](double v) { return v == 0; }));

    EXPECT_EQ(model->getLearningRate(), 0.1);
}

TEST(MatrixModel, FeedForward) {
    std::unique_ptr<s21::MLPModel> model = std::make_unique<s21::MatrixModel>(10, 3, 2, 5, 0.1);
    model->randomFill();

    auto result = model->feedForward({ 0, 1, 0, 2, 3, 0, 5, 2, 3, 2 });
    
    EXPECT_EQ(result.size(), 3);
    EXPECT_TRUE(std::all_of(result.begin(), result.end(), [](double v) { return v != 0; }));
}

TEST(MatrixModel, GetPrediction) {
    std::unique_ptr<s21::MLPModel> model = std::make_unique<s21::MatrixModel>(10, 3, 2, 5, 0.1);
    model->randomFill();

    auto result = model->feedForward({ 0, 1, 0, 2, 3, 0, 5, 2, 3, 2 });
    size_t answer = model->getPrediction(result);

    size_t expected = 0;
    double tmp = -1111;
    for (size_t i = 0; i < result.size(); i++) {
        if (result[i] > tmp) {
            tmp = result[i];
            expected = i;
        }
    }

    EXPECT_EQ(answer, expected);
}

TEST(MatrixModel, BackPropagation) {
    std::unique_ptr<s21::MLPModel> model = std::make_unique<s21::MatrixModel>(10, 3, 2, 5, 0.1);
    model->randomFill();
    std::vector<double> test_vec = { 0, 1, 0, 2, 3, 0, 5, 2, 3, 2 };

    auto result = model->feedForward(test_vec);
    model->backPropagation({ 0.0l, 1.0l, 0.0l });
    auto new_result = model->feedForward(test_vec);

    EXPECT_FALSE(result == new_result);
}
