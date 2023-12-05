#ifndef _MATRIXMODEL_H
#define _MATRIXMODEL_H

#include <vector>

class MatrixModel {
private:
    std::vector<std::vector<double>> weights;

public:
    MatrixModel() = default;
    MatrixModel(const MatrixModel& other) = delete;
    MatrixModel(MatrixModel&& other) = delete;
    ~MatrixModel() = default;

    bool setWeights();
    std::vector<std::vector<double>> feedForward(std::vector<std::vector<double>> &input_layer);
    void backPropagation(std::vector<std::vector<double>> &output_layer);

private:
    bool updateWeights();
};

#endif //_MATRIXMODEL_H
