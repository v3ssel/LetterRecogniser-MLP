#include <filesystem>
#include <memory>
#include <gtest/gtest.h>
#include "../core/serializer/FileMLPSerializer.h"
#include "../core/matrix/MatrixModel.h"

const std::string filename = "C:\\Coding\\Projects\\CPP7_MLP-1\\src\\tests\\assets\\model-test.txt";
const std::unique_ptr<s21::MLPSerializer> serializer = std::make_unique<s21::FileMLPSerializer>();

TEST(FileMLPSerializer, SerializeMatrixModel) {
    std::unique_ptr<s21::MLPModel> rnd_model = std::make_unique<s21::MatrixModel>(10, 3, 2, 5, 0.4);
    rnd_model->randomFill();

    std::filesystem::path path(filename);
    std::string new_filename = path.replace_filename(path.stem().string() + "_serialized" + path.extension().string()).string();

    serializer->serialize(rnd_model, new_filename);
    EXPECT_TRUE(std::filesystem::exists(new_filename));

    std::unique_ptr<s21::MLPModel> des_model = std::make_unique<s21::MatrixModel>(10, 3, 2, 5, 0.4);
    serializer->deserialize(des_model, new_filename);

    auto weights_rnd = rnd_model->getWeights();
    auto weights_des = des_model->getWeights();
    
    EXPECT_EQ(weights_rnd.size(), weights_des.size());
    for (size_t i = 0; i < weights_rnd.size(); i++) {
        EXPECT_NEAR(weights_rnd[i], weights_des[i], 6);
    }
    
    auto biases_rnd = rnd_model->getBiases();
    auto biases_des = des_model->getBiases();

    EXPECT_EQ(biases_rnd.size(), biases_des.size());
    for (size_t i = 0; i < biases_rnd.size(); i++) {
        EXPECT_NEAR(biases_rnd[i], biases_des[i], 6);
    }
}

TEST(FileMLPSerializer, DeserializeMatrixModel) {
    std::unique_ptr<s21::MLPModel> model = std::make_unique<s21::MatrixModel>(10, 3, 2, 5, 0.4);

    auto weights = model->getWeights();
    auto biases = model->getBiases();

    for (size_t i = 0; i < weights.size(); i++) {
        EXPECT_EQ(weights[i], 0.0l);
    }
    for (size_t i = 0; i < biases.size(); i++) {
        EXPECT_EQ(biases[i], 0.0l);
    }
    
    serializer->deserialize(model, filename);

    weights = model->getWeights();
    biases = model->getBiases();

    for (size_t i = 0; i < weights.size(); i++) {
        EXPECT_NE(weights[i], 0.000l);
    }

    for (size_t i = 0; i < biases.size(); i++) {
        EXPECT_NE(biases[i], 0.000l);
    }
}

TEST(FileMLPSerializer, DeserializeEmptyModel) {
    std::unique_ptr<s21::MLPModel> model = std::make_unique<s21::MatrixModel>(10, 3, 2, 5, 0.4);

    EXPECT_THROW(serializer->deserialize(model, ""), std::invalid_argument);
}

TEST(FileMLPSerializer, DeserializeInvalidModel) {
    std::unique_ptr<s21::MLPModel> model = std::make_unique<s21::MatrixModel>(784, 26, 2, 140, 0.1);

    EXPECT_THROW(serializer->deserialize(model, filename), std::invalid_argument);
}

TEST(FileMLPSerializer, DeserializeInvalidFile) {
    std::filesystem::path path(filename);
    std::string broken_filename = path.replace_filename(path.stem().string() + "-broken" + path.extension().string()).string();
    std::unique_ptr<s21::MLPModel> model = std::make_unique<s21::MatrixModel>(10, 3, 2, 5, 0.4);

    EXPECT_THROW(serializer->deserialize(model, broken_filename), std::runtime_error);
}
