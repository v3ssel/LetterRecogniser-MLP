#include "EmnistDatasetReader.h"
#include <iostream>

namespace s21 {
    void EMNISTDatasetReader::open(const std::string &filename) {
        _file.open(filename);

        if (!_file.is_open()) {
            throw std::invalid_argument("EMNISTDatasetReader::open: Cannot open the file.");
        }
    }

    EMNISTData EMNISTDatasetReader::readLine() {
        std::string line;
        if (!std::getline(_file, line)) {
            _file.close();

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

        return data;
    }

    void EMNISTDatasetReader::close() {
        _file.close();
    }
    
    bool EMNISTDatasetReader::is_open() const {
        return _file.is_open();
    }
}