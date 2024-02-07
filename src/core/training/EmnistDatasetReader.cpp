#include "EmnistDatasetReader.h"

namespace s21 {
void EMNISTDatasetReader::open(const std::string &filename) {
    if (file_.is_open()) close();

    file_.open(filename);

    if (!file_.is_open()) {
        throw std::invalid_argument(
            "EMNISTDatasetReader::open: Cannot open the file \"" + filename +
            "\".");
    }
}

EMNISTData EMNISTDatasetReader::readLine() {
    if (!file_.is_open()) {
        throw std::invalid_argument(
            "EMNISTDatasetReader::readLine: File is not open.");
    }

    std::string line;
    if (!std::getline(file_, line)) {
        return EMNISTData();
    }

    std::stringstream ss(line);
    EMNISTData data;

    ss >> data.result;
    if (ss.peek() == ',') ss.ignore();

    double number;
    while (ss >> number) {
        if (ss.peek() == ',') ss.ignore();
        data.image.push_back(number / 255.0l);
    }

    if (data.image.size() != 784) {
        throw std::runtime_error(
            "EMNISTDatasetReader::readLine: EMNIST line must consist of 1 "
            "result "
            "and 784 numbers.");
    }

    return data;
}

size_t EMNISTDatasetReader::getNumberOfLines() {
    if (!file_.is_open()) {
        throw std::invalid_argument(
            "EMNISTDatasetReader::getNumberOfLines: File is not open.");
    }

    size_t lines = std::count(std::istreambuf_iterator<char>(file_),
                              std::istreambuf_iterator<char>(), '\n');

    file_.clear();
    file_.seekg(0);

    return lines;
}

void EMNISTDatasetReader::close() { file_.close(); }

bool EMNISTDatasetReader::is_open() const { return file_.is_open(); }
}  // namespace s21