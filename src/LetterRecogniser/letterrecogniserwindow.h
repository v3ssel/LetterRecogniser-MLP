#ifndef LETTERRECOGNISERWINDOW_H
#define LETTERRECOGNISERWINDOW_H

#include <QMainWindow>
#include <QFileDialog>
#include <QtConcurrent/QtConcurrent>
#include <QFutureWatcher>
#include "../controller/controller.h"

QT_BEGIN_NAMESPACE
namespace Ui { class LetterRecogniserWindow; }
QT_END_NAMESPACE

class LetterRecogniserWindow : public QMainWindow {
    Q_OBJECT

   public:
    const size_t kInputLayerSize = 784;
    const size_t kHiddenLayerSize = 140;
    const size_t kOutputLayerSize = 26;

    LetterRecogniserWindow(QWidget *parent = nullptr);
    ~LetterRecogniserWindow();

    void StartPrediction(QImage image);

    void EpochCallback(size_t epoch, double mse, double accurancy);
    void ProcessCallback(size_t, s21::MLPTrainStages stage);

   public slots:
    void clearPaintButtonClicked();
    void loadBmpImageButtonClicked();

    void changeModelType(const QString &arg1);
    void changeLayersSize(int layers);
    void changeLearnRate(double rate);

    void loadWeightsButtonClicked();
    void saveWeightsButtonClicked();
    void randomizeWeightsButtonClicked();

    void testingButtonClicked();
    void trainingButtonClicked();

    void testingResults();
    void trainingResult();

   private:
    Ui::LetterRecogniserWindow *ui;

    QString prev_model_type_;
    int prev_layers_size_;

    QFutureWatcher<s21::MLPTestMetrics> *testing_future_watcher_;
    QFutureWatcher<std::vector<double>> *training_future_watcher_;
    bool canceled_;

    std::string exception_msg_;

    s21::ModelType chooseModelType();
    void blockButtons(bool block, bool testing);
};
#endif // LETTERRECOGNISERWINDOW_H
