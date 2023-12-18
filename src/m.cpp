#include "Windows.h"
#include <iostream>
#include <vector>
#include <memory>
#include <csignal>

#include "core/MultilayerPerceptron.h"
#include "core/matrix/MatrixModel.h"
#include "core/training/EmnistMLPTrainer.h"
#include "core/training/EmnistDatasetReader.h"

class ExView {
   public:
    void msg(double v) {
        std::cout << "ExView Got: " << v << "\n";
    }
};

int main(int argc, char const *argv[]) {
    SetConsoleOutputCP(CP_UTF8);
    std::signal(SIGSEGV, [](int s) { std::cout << "\n\nSEGA\n"; });
    // std::vector<double> v = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,8,32,37,37,37,37,37,20,7,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,22,46,114,127,127,127,127,125,77,32,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,22,46,115,172,208,245,250,250,250,250,249,206,126,8,0,0,0,0,0,0,0,0,0,0,0,0,3,36,159,207,245,252,254,254,254,255,255,254,254,245,204,34,0,0,0,0,0,0,0,0,0,0,0,7,22,95,218,244,254,254,254,254,255,255,255,255,254,250,220,50,0,0,0,0,0,0,0,0,0,1,8,90,159,232,253,254,254,251,250,250,250,250,250,254,254,254,245,114,4,0,0,0,0,0,0,0,0,20,77,218,245,254,254,254,234,145,129,127,127,127,141,222,247,254,250,129,5,0,0,0,0,0,0,0,0,34,115,243,253,254,252,250,207,64,41,38,41,40,53,176,232,254,250,141,11,1,0,0,0,0,0,0,3,84,172,252,254,251,189,144,77,11,14,38,99,76,12,84,172,252,254,220,77,20,0,0,0,0,0,0,21,170,232,254,254,220,84,40,30,19,52,170,236,213,66,36,115,245,254,249,125,37,0,0,0,0,0,0,34,204,245,254,252,177,47,52,75,22,47,207,250,231,82,23,82,233,252,250,127,37,0,0,0,0,0,2,82,233,252,254,251,143,18,19,27,26,83,233,254,246,115,8,34,204,245,250,127,37,0,0,0,0,0,4,125,249,254,254,254,221,107,51,2,37,125,249,254,250,129,10,14,143,222,249,125,37,0,0,0,0,0,5,129,250,254,254,254,233,92,32,0,37,127,250,254,252,191,129,129,191,236,233,82,21,0,0,0,0,0,9,140,250,254,255,254,222,52,11,0,39,129,250,254,254,236,218,218,236,249,220,50,9,0,0,0,0,4,32,203,254,254,255,254,217,42,13,32,101,177,252,254,254,254,254,254,254,251,170,21,2,0,0,0,0,4,32,203,254,254,254,254,229,129,117,140,212,240,254,255,254,254,254,254,248,222,79,3,0,0,0,0,0,2,21,174,252,254,254,253,217,151,148,101,174,222,254,254,254,254,250,250,236,188,38,0,0,0,0,0,0,2,21,172,252,254,247,221,92,26,27,8,53,141,250,254,254,222,141,130,152,128,16,0,0,0,0,0,0,2,20,170,252,253,207,127,10,0,0,0,37,125,249,254,250,139,13,6,16,15,1,0,0,0,0,0,0,0,9,140,250,247,159,79,3,0,0,0,32,113,243,253,243,115,4,0,1,1,0,0,0,0,0,0,0,0,4,125,237,206,47,10,0,0,0,0,7,33,158,200,158,33,0,0,0,0,0,0,0,0,0,0,0,0,2,63,111,76,7,0,0,0,0,0,0,1,20,32,20,1,0,0,0,0,0,0,0,0,0,0,0,0,0,18,32,20,1,0,0,0,0,0,0,0,2,3,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
    // };
    // int c = 0;
    // for (auto i : v) {
    //     c++;
    //     if (i >= 0 && i <= 9) std::cout << "  ";
    //     if (i >= 10 && i <= 99) std::cout << " ";
    //     std::cout << i << ",";
    //     if (c == 28) {
    //         std::cout << "\n";
    //         c = 0;
    //     }
    // }
    // std::vector<double> v2 = { 0, 1, 0, 2, 3, 0, 5, 2, 3, 2 };
    // std::vector<double> ans = { 0.0l, 1.0l, 0.0l };

    std::function<void(double)> f = std::bind(&ExView::msg, ExView(), std::placeholders::_1);

    std::unique_ptr<s21::MLPTrainer> trainer = std::make_unique<s21::EMNISTMLPTrainer>();
    std::unique_ptr<s21::MLPModel> model = std::make_unique<s21::MatrixModel>(784, 26, 2, 140, 0.15);
    // model->randomFill();
    s21::MultilayerPerceptron mlp(model, trainer, f);
    
    std::cout << "<<<<<<<-------------------------------BEFORE TRAIN-------------------------------->>>>>>>>\n";
    // mlp.exportModel("model-b.txt");
    mlp.importModel("4model-65-72.txt");
    // mlp.testing("C:\\Coding\\Projects\\CPP7_MLP-1\\datasets\\emnist-letters\\emnist-letters-test.csv", 50);

    // std::unique_ptr<s21::EMNISTDatasetReader> reader = std::make_unique<s21::EMNISTDatasetReader>();
    // reader->open("C:\\Coding\\Projects\\CPP7_MLP-1\\datasets\\emnist-letters\\em5.txt");
    // size_t i = 0;
    // while (reader->is_open()) {
    //     s21::EMNISTData data = reader->readLine();
    //     if (data.result == (size_t)-1) break;

    //     char a = mlp.prediction(data.image);
    //     std::cout << ++i << ". " << "expected: " << data.result << " and got: " << (int)a + 1 <<  " " << (char)(65 + a) << "\n"; 
    // }


    mlp.learning("C:\\Coding\\Projects\\CPP7_MLP-1\\datasets\\emnist-letters\\emnist-letters-train.csv", 1);
    // mlp.exportModel("model-a3.txt");

    // std::cout << ">>>>>>-------------------------------AFTER TRAIN---------------------------------<<<<<<\n";
    // reader = std::make_unique<s21::EMNISTDatasetReader>();
    // reader->open("C:\\Coding\\Projects\\CPP7_MLP-1\\datasets\\emnist-letters\\em5.txt");
    // i = 0;
    // while (reader->is_open()) {
    //     s21::EMNISTData data = reader->readLine();
    //     if (data.result == (size_t)-1) break;

    //     char a = mlp.prediction(data.image);
    //     std::cout << ++i << ". " << "expected: " << data.result << " and got: " << (int)a + 1 <<  " " << (char)(65 + a) << "\n"; 
    // }

    // std::unique_ptr<s21::MLPModel> model2 = std::make_unique<s21::MatrixModel>(10, 3, 2, 5, 0.4);
    // s21::MultilayerPerceptron mlp2(model2, trainer);
    // mlp2.importModel("model.txt");

    // for (int i = 0; i < 10; i++) {
    //     std::cout << "\n-------------------------------EPOCH" << i + 1 << "---------------------------------";
    //     std::cout << "\n-------------------------------LOADED MATRIX---------------------------------\n";
    //     char a = mlp2.prediction(v2);
    //     std::cout << "\nнагадали: " << "|" << (int)a << "\n";

    //     mlp2._model->backPropagation(ans);
    // }
    // std::cout << "\n-------------------------------FINAL MATRIX---------------------------------\n";
    // char a = mlp2.prediction(v2);
    // std::cout << "\nнагадали: " << (char)(a + 65) << "|" << (int)a << "\n";

    
}
