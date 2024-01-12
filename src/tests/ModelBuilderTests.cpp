#include <gtest/gtest.h>

#include "../core/ModelBuilder.h"

TEST(ModelBuilder, Default) {
  s21::ModelBuilder builder;
  auto model = builder.build();

  auto weights = model->getWeights();
  auto biases = model->getBiases();
  auto sizes = model->getLayersSize();

  EXPECT_EQ(sizes.size(), 4);
  EXPECT_EQ(sizes[0], 10);
  EXPECT_EQ(sizes[1], 5);
  EXPECT_EQ(sizes[2], 5);
  EXPECT_EQ(sizes[3], 3);

  EXPECT_EQ(weights.size(), 90);
  EXPECT_EQ(biases.size(), 13);
  EXPECT_TRUE(std::all_of(weights.begin(), weights.end(),
                          [](double v) { return v == 0; }));
  EXPECT_TRUE(std::all_of(biases.begin(), biases.end(),
                          [](double v) { return v == 0; }));
}

TEST(ModelBuilder, CustomModel) {
  s21::ModelBuilder builder;

  builder.setModelType(s21::ModelType::Matrix)
      ->setInputLayerSize(784)
      ->setLayers(5)
      ->setHiddenLayerSize(140)
      ->setOutputLayerSize(26)
      ->setLearningRate(0.11);

  auto model = builder.build();

  auto weights = model->getWeights();
  auto biases = model->getBiases();
  auto sizes = model->getLayersSize();

  EXPECT_EQ(sizes.size(), 7);
  EXPECT_EQ(sizes[0], 784);
  EXPECT_EQ(sizes[1], 140);
  EXPECT_EQ(sizes[2], 140);
  EXPECT_EQ(sizes[3], 140);
  EXPECT_EQ(sizes[4], 140);
  EXPECT_EQ(sizes[5], 140);
  EXPECT_EQ(sizes[6], 26);

  EXPECT_EQ(weights.size(), 191'800);
  EXPECT_EQ(biases.size(), 726);
  EXPECT_TRUE(std::all_of(weights.begin(), weights.end(),
                          [](double v) { return v == 0; }));
  EXPECT_TRUE(std::all_of(biases.begin(), biases.end(),
                          [](double v) { return v == 0; }));
}
