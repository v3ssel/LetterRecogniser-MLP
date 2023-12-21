#include <filesystem>
#include <gtest/gtest.h>
#include "../core/training/EmnistDatasetReader.h"

const std::string kDatasetPath = "C:\\Coding\\Projects\\CPP7_MLP-1\\src\\tests\\assets\\emnist-sample.txt";
auto reader = std::make_unique<s21::EMNISTDatasetReader>();

TEST(EMNISTDatasetReader, OpenInvalidFile) {
    EXPECT_THROW(reader->open(""), std::invalid_argument);
}

TEST(EMNISTDatasetReader, CountLinesClosedFile) {
    EXPECT_THROW(reader->getNumberOfLines(), std::invalid_argument);
}

TEST(EMNISTDatasetReader, ReadLineClosedFile) {
    EXPECT_THROW(reader->readLine(), std::invalid_argument);
}

TEST(EMNISTDatasetReader, ReadBrokenLine) {
    std::filesystem::path path = kDatasetPath;
    std::string broken_file = path.replace_filename(path.stem().string() + "-broken" + path.extension().string()).string();
    
    reader->open(broken_file);
    EXPECT_THROW(reader->readLine(), std::runtime_error);
}

TEST(EMNISTDatasetReader, CountLines) {
    reader->open(kDatasetPath);
    size_t lines = reader->getNumberOfLines();

    EXPECT_EQ(lines, 12);
    reader->close();
}

TEST(EMNISTDatasetReader, ReadLine) {
    reader->open(kDatasetPath);
    s21::EMNISTData data = reader->readLine();

    EXPECT_EQ(data.image.size(), 784);
    EXPECT_EQ(data.result, 23);
}

TEST(EMNISTDatasetReader, ReadWholeFile) {
    reader->open(kDatasetPath);
    size_t lines = reader->getNumberOfLines();
    
    size_t i = 0;
    for (; i < lines; i++) {
        s21::EMNISTData data = reader->readLine();

        EXPECT_EQ(data.image.size(), 784);
        EXPECT_TRUE(data.result >= 1 && data.result <= 26);
    }

    EXPECT_EQ(i, 12);
}