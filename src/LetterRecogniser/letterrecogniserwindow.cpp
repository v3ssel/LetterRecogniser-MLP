#include "letterrecogniserwindow.h"
#include "./ui_letterrecogniserwindow.h"

LetterRecogniserWindow::LetterRecogniserWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::LetterRecogniserWindow) {
    ui->setupUi(this);
    this->setWindowTitle("LetterRecogniser");

    prev_model_type_ = ui->model_comboBox->currentText();
    prev_layers_size_ = ui->layers_spinBox->value();

    ui->painting_widget->setParent(this);
    QImage image(ui->painting_widget->width(), ui->painting_widget->height(), QImage::Format::Format_RGB16);
    image.fill(Qt::GlobalColor::white);
    ui->painting_widget->setImage(image);

    std::function<void(size_t, double, double)> epoch_callback =
                            std::bind(&LetterRecogniserWindow::EpochCallback, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);

    std::function<void(size_t, s21::MLPTrainStages)> process_callback =
                            std::bind(&LetterRecogniserWindow::ProcessCallback, this, std::placeholders::_1, std::placeholders::_2);

    s21::Controller::getInstance().makeMLP(chooseModelType(),
                                           kInputLayerSize,
                                           kOutputLayerSize,
                                           ui->layers_spinBox->value(),
                                           kHiddenLayerSize,
                                           ui->learnrate_doubleSpinBox->value(),
                                           epoch_callback,
                                           process_callback);

    connect(ui->clear_pushButton, &QAbstractButton::clicked, this, &LetterRecogniserWindow::clearPaintButtonClicked);
    connect(ui->load_image_pushButton, &QAbstractButton::clicked, this, &LetterRecogniserWindow::loadBmpImageButtonClicked);
    connect(ui->painting_widget, &DrawableWidget::predict, this, &LetterRecogniserWindow::StartPrediction);
    connect(ui->model_comboBox, &QComboBox::currentTextChanged, this, &LetterRecogniserWindow::changeModelType);
    connect(ui->layers_spinBox, &QSpinBox::valueChanged, this, &LetterRecogniserWindow::changeLayersSize);
    connect(ui->load_weights_pushButton, &QAbstractButton::clicked, this, &LetterRecogniserWindow::loadWeightsButtonClicked);
    connect(ui->save_weights_pushButton, &QAbstractButton::clicked, this, &LetterRecogniserWindow::saveWeightsButtonClicked);
}

LetterRecogniserWindow::~LetterRecogniserWindow() {
    delete ui;
}

void LetterRecogniserWindow::StartPrediction(QImage image) {
    std::vector<double> input;
    input.reserve(image.width() * image.height());

    for (int i = 0; i < image.width(); i++) {
        for (int j = 0; j < image.height(); j++) {
            input.push_back(qRed(image.pixel(i, j)) / 255.0);
        }
    }

//    qDebug() << "Pixel Matrix:";
//    for (int y = 0; y < 28; ++y) {
//        QString row;
//        for (int x = 0; x < 28; ++x) {
//            row += QString::number(input[y * 28 + x]) + " ";
//        }
//        qDebug() << row;
//    }

//    char answer = s21::Controller::getInstance().predicate(input);
//    ui->answer_label->setText(QString(answer));
//    qDebug() << "strttt: " << answer;
}

void LetterRecogniserWindow::EpochCallback(size_t epoch, double mse, double accurancy) {
    ui->epoch_value_label->setText(QString::number(epoch));
    ui->mse_value_label->setText(QString::number(mse));
    ui->accur_value_label->setText(QString::number(accurancy));
}

void LetterRecogniserWindow::ProcessCallback(size_t, s21::MLPTrainStages stage) {
    QString stage_str;
    switch (stage) {
        case s21::MLPTrainStages::STARTING:
            stage_str = "STARTING";
            break;
        case s21::MLPTrainStages::TRAINING:
            stage_str = "TRAINING";
            break;
        case s21::MLPTrainStages::TESTING:
            stage_str = "TESTING";
            break;
        case s21::MLPTrainStages::DONE:
            stage_str = "DONE";
            break;
    }

    ui->stage_label->setText(stage_str);
}

void LetterRecogniserWindow::clearPaintButtonClicked() {
    ui->painting_widget->clear();
}

void LetterRecogniserWindow::loadBmpImageButtonClicked() {
    QString fileName = QFileDialog::getOpenFileName(this,
                                       tr("Open File"), QDir::currentPath(), tr("BMP (*.bmp)"));
    ui->painting_widget->loadImage(fileName);
}

void LetterRecogniserWindow::changeModelType(const QString &val) {
    if (val == prev_model_type_) return;

    auto btn = QMessageBox::question(this, "Losing data", "Changing model type will lose all weights and biases.");
    if (btn == QMessageBox::StandardButton::Yes) {
        s21::Controller::getInstance().changeModel(chooseModelType(), ui->layers_spinBox->value());
        prev_model_type_ = val;
    } else {
        ui->model_comboBox->setCurrentIndex(ui->model_comboBox->findText(prev_model_type_));
    }
}

void LetterRecogniserWindow::changeLayersSize(int val) {
    if (val == prev_layers_size_) return;

    auto btn = QMessageBox::question(this, "Losing data", "Changing layers count will lose all weights and biases.");
    if (btn == QMessageBox::StandardButton::Yes) {
        s21::Controller::getInstance().changeModel(chooseModelType(), val);
        prev_layers_size_ = val;
    } else {
        ui->layers_spinBox->setValue(prev_layers_size_);
    }
}

void LetterRecogniserWindow::loadWeightsButtonClicked() {
    QString fileName = QFileDialog::getOpenFileName(this,
                                       tr("Open File"), QDir::currentPath(), tr("TXT (*.txt)"));
    try {
        s21::Controller::getInstance().loadWeights(fileName.toStdString());
    } catch (const std::exception& exp) {
        QMessageBox::critical(this, "Failed to load weights.", QString("Error message: ") + QString(exp.what()));
    }
}

void LetterRecogniserWindow::saveWeightsButtonClicked() {
    QString fileName = QFileDialog::getSaveFileName(this,
                                                    tr("Save File"), QDir::currentPath(), tr("TXT (*.txt)"));

    try {
        s21::Controller::getInstance().saveWeights(fileName.toStdString());
    } catch (const std::exception& exp) {
        QMessageBox::critical(this, "Failed to save weights.", QString("Error message: ") + QString(exp.what()));
    }
}

s21::ModelType LetterRecogniserWindow::chooseModelType() {
    return ui->model_comboBox->currentText() == "Matrix" ? s21::ModelType::Matrix : s21::ModelType::Graph;
}
