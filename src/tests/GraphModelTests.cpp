#include <memory>
#include <gtest/gtest.h>
#include "../core/graph/GraphModel.h"

TEST(GraphModel, RandomFill) {
    std::unique_ptr<s21::MLPModel> model = std::make_unique<s21::GraphModel>(10, 3, 2, 5, 0.1);
    auto weights = model->getWeights();
    auto biases = model->getBiases();

    EXPECT_EQ(weights.size(), 90);
    EXPECT_EQ(biases.size(), 13);
    EXPECT_TRUE(std::all_of(weights.begin(), weights.end(), [](double w) { return w == 0.0l; }));
    EXPECT_TRUE(std::all_of(biases.begin(), biases.end(), [](double w) { return w == 0.0l; }));

    model->randomFill();

    weights = model->getWeights();
    biases = model->getBiases();

    EXPECT_EQ(weights.size(), 90);
    EXPECT_EQ(biases.size(), 13);
    EXPECT_TRUE(std::any_of(weights.begin(), weights.end(), [](double w) { return w != 0.0l; }));
    EXPECT_TRUE(std::any_of(biases.begin(), biases.end(), [](double w) { return w != 0.0l; }));
}

TEST(GraphModel, GetWeights) {
    std::unique_ptr<s21::MLPModel> model = std::make_unique<s21::GraphModel>(10, 3, 2, 5, 0.1);
    model->randomFill();
    
    auto weights = model->getWeights();

    EXPECT_EQ(weights.size(), 90);
    EXPECT_TRUE(std::any_of(weights.begin(), weights.end(), [](double w) { return w != 0.0l; }));
}

TEST(GraphModel, GetBiases) {
    std::unique_ptr<s21::MLPModel> model = std::make_unique<s21::GraphModel>(10, 3, 2, 5, 0.1);
    model->randomFill();

    auto biases = model->getBiases();

    EXPECT_EQ(biases.size(), 13);
    EXPECT_TRUE(std::any_of(biases.begin(), biases.end(), [](double b) { return b != 0.0l; }));
}

TEST(GraphModel, SetWeights) {
    std::unique_ptr<s21::MLPModel> model = std::make_unique<s21::GraphModel>(10, 3, 2, 5, 0.1);

    auto weights = model->getWeights();
    EXPECT_EQ(weights.size(), 90);
    EXPECT_TRUE(std::all_of(weights.begin(), weights.end(), [](double w) { return w == 0.0l; }));

    std::vector<double> new_weights(90, 999);
    model->setWeights(new_weights);

    weights = model->getWeights();
    EXPECT_EQ(weights.size(), 90);
    EXPECT_TRUE(weights == new_weights);
    EXPECT_TRUE(std::all_of(weights.begin(), weights.end(), [](double w) { return w == 999; }));
}

TEST(GraphModel, SetBiases) {
    std::unique_ptr<s21::MLPModel> model = std::make_unique<s21::GraphModel>(10, 3, 2, 5, 0.1);

    auto biases = model->getBiases();
    EXPECT_EQ(biases.size(), 13);
    EXPECT_TRUE(std::all_of(biases.begin(), biases.end(), [](double w) { return w == 0.0l; }));

    std::vector<double> new_biases(13, 777);
    model->setBiases(new_biases);

    biases = model->getBiases();
    EXPECT_EQ(biases.size(), 13);
    EXPECT_TRUE(biases == new_biases);
    EXPECT_TRUE(std::all_of(biases.begin(), biases.end(), [](double w) { return w == 777; }));
}

TEST(GraphModel, GetAndSetWeights) {
    std::unique_ptr<s21::MLPModel> model = std::make_unique<s21::GraphModel>(10, 3, 2, 5, 0.1);
    model->randomFill();
    auto weights = model->getWeights();

    std::unique_ptr<s21::MLPModel> model2 = std::make_unique<s21::GraphModel>(10, 3, 2, 5, 0.1);
    model2->setWeights(weights);

    auto weights2 = model2->getWeights();

    EXPECT_EQ(weights.size(), 90);
    EXPECT_EQ(weights2.size(), 90);
    EXPECT_TRUE(weights == weights2);
}

TEST(GraphModel, GetAndSetBiases) {
    std::unique_ptr<s21::MLPModel> model = std::make_unique<s21::GraphModel>(10, 3, 2, 5, 0.1);
    model->randomFill();
    auto biases = model->getBiases();

    std::unique_ptr<s21::MLPModel> model2 = std::make_unique<s21::GraphModel>(10, 3, 2, 5, 0.1);
    model2->setBiases(biases);

    auto biases2 = model2->getBiases();

    EXPECT_EQ(biases.size(), 13);
    EXPECT_EQ(biases2.size(), 13);
    EXPECT_TRUE(biases == biases2);
}

TEST(GraphModel, LearningRate) {
    std::unique_ptr<s21::MLPModel> model = std::make_unique<s21::GraphModel>(10, 3, 2, 5, 0.1);
    EXPECT_EQ(model->getLearningRate(), 0.1);
    model->setLearningRate(0.2);
    EXPECT_EQ(model->getLearningRate(), 0.2);
}

TEST(GraphModel, SetBadWeights) {
    std::unique_ptr<s21::MLPModel> model = std::make_unique<s21::GraphModel>(10, 3, 2, 5, 0.1);

    EXPECT_THROW(model->setWeights({1, 2, 3}), std::out_of_range);
}

TEST(GraphModel, SetBadBiases) {
    std::unique_ptr<s21::MLPModel> model = std::make_unique<s21::GraphModel>(10, 3, 2, 5, 0.1);

    EXPECT_THROW(model->setBiases({1, 2, 3}), std::out_of_range);
}
