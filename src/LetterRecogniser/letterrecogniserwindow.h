#ifndef LETTERRECOGNISERWINDOW_H
#define LETTERRECOGNISERWINDOW_H

#include <QMainWindow>
#include <QFileDialog>
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

    void loadWeightsButtonClicked();
    void saveWeightsButtonClicked();

private:
    Ui::LetterRecogniserWindow *ui;

    QString prev_model_type_;
    int prev_layers_size_;
    s21::ModelType chooseModelType();
};
#endif // LETTERRECOGNISERWINDOW_H
