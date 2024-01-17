#include "letterrecogniserwindow.h"

#include "./ui_letterrecogniserwindow.h"

LetterRecogniserWindow::LetterRecogniserWindow(QWidget* parent)
    : QMainWindow(parent), ui_(new Ui::LetterRecogniserWindow) {
  ui_->setupUi(this);
  this->setWindowTitle("LetterRecogniser");
  ui_->trainres_graph_widget->init();

  prev_model_type_ = ui_->model_comboBox->currentText();
  prev_layers_size_ = ui_->layers_spinBox->value();

  testing_future_watcher_ = nullptr;
  training_future_watcher_ = nullptr;
  canceled_ = false;

  ui_->painting_widget->setParent(this);
  QImage image(ui_->painting_widget->width(), ui_->painting_widget->height(),
               QImage::Format::Format_RGB16);
  image.fill(Qt::GlobalColor::white);
  ui_->painting_widget->setImage(image);

  std::function<void(size_t, double, double)> epoch_callback = std::bind(
      &LetterRecogniserWindow::EpochCallback, this, std::placeholders::_1,
      std::placeholders::_2, std::placeholders::_3);

  std::function<void(size_t, s21::MLPTrainStages)> process_callback =
      std::bind(&LetterRecogniserWindow::ProcessCallback, this,
                std::placeholders::_1, std::placeholders::_2);

  s21::Controller::getInstance().makeMLP(
      chooseModelType(), kInputLayerSize, kOutputLayerSize,
      ui_->layers_spinBox->value(), kHiddenLayerSize,
      ui_->learnrate_doubleSpinBox->value(), epoch_callback, process_callback);

  connect(ui_->clear_pushButton, &QAbstractButton::clicked, this,
          &LetterRecogniserWindow::clearPaintButtonClicked);
  connect(ui_->load_image_pushButton, &QAbstractButton::clicked, this,
          &LetterRecogniserWindow::loadBmpImageButtonClicked);
  connect(ui_->painting_widget, &DrawableWidget::predict, this,
          &LetterRecogniserWindow::startPrediction);

  connect(ui_->model_comboBox, &QComboBox::currentTextChanged, this,
          &LetterRecogniserWindow::changeModelType);
  connect(ui_->layers_spinBox, &QSpinBox::valueChanged, this,
          &LetterRecogniserWindow::changeLayersSize);
  connect(ui_->learnrate_doubleSpinBox, &QDoubleSpinBox::valueChanged, this,
          &LetterRecogniserWindow::changeLearnRate);

  connect(ui_->load_weights_pushButton, &QAbstractButton::clicked, this,
          &LetterRecogniserWindow::loadWeightsButtonClicked);
  connect(ui_->save_weights_pushButton, &QAbstractButton::clicked, this,
          &LetterRecogniserWindow::saveWeightsButtonClicked);
  connect(ui_->random_weights_pushButton, &QAbstractButton::clicked, this,
          &LetterRecogniserWindow::randomizeWeightsButtonClicked);

  connect(ui_->start_testing_pushButton, &QAbstractButton::clicked, this,
          &LetterRecogniserWindow::testingButtonClicked);
  connect(ui_->start_training_pushButton, &QAbstractButton::clicked, this,
          &LetterRecogniserWindow::trainingButtonClicked);
}

LetterRecogniserWindow::~LetterRecogniserWindow() {
  if (testing_future_watcher_) {
    delete testing_future_watcher_;
  }

  if (training_future_watcher_) {
    delete training_future_watcher_;
  }

  delete ui_;
}

void LetterRecogniserWindow::startPrediction(QImage image) {
  std::vector<double> input;
  input.reserve(image.width() * image.height());

  for (int i = 0; i < image.width(); i++) {
    for (int j = 0; j < image.height(); j++) {
      input.push_back(qRed(image.pixel(i, j)) / 255.0);
    }
  }

  char answer = s21::Controller::getInstance().predicate(input);
  ui_->answer_label->setText(QString(answer));
}

void LetterRecogniserWindow::EpochCallback(size_t epoch, double mse,
                                           double accurancy) {
  ui_->epoch_value_label->setText(QString::number(epoch));
  ui_->mse_value_label->setText(QString::number(mse, 'g', 2));
  ui_->accur_value_label->setText(QString::number(accurancy, 'g', 2));
}

void LetterRecogniserWindow::ProcessCallback(size_t,
                                             s21::MLPTrainStages stage) {
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

  ui_->stage_label->setText(stage_str);
}

void LetterRecogniserWindow::clearPaintButtonClicked() {
  ui_->painting_widget->clear();
}

void LetterRecogniserWindow::loadBmpImageButtonClicked() {
  QString filename = QFileDialog::getOpenFileName(
      this, tr("Open File"), QDir::currentPath(), tr("BMP (*.bmp)"));
  if (filename.isEmpty()) return;

  ui_->painting_widget->loadImage(filename);
}

void LetterRecogniserWindow::changeModelType(const QString& val) {
  if (val == prev_model_type_) return;

  auto btn = QMessageBox::question(
      this, "Losing data",
      "Changing model type will lose all weights and biases.");
  if (btn == QMessageBox::StandardButton::Yes) {
    s21::Controller::getInstance().changeModelTypeAndLayersSize(
        chooseModelType(), ui_->layers_spinBox->value());
    prev_model_type_ = val;
  } else {
    ui_->model_comboBox->setCurrentIndex(
        ui_->model_comboBox->findText(prev_model_type_));
  }
}

void LetterRecogniserWindow::changeLayersSize(int val) {
  if (val == prev_layers_size_) return;

  auto btn = QMessageBox::question(
      this, "Losing data",
      "Changing layers count will lose all weights and biases.");
  if (btn == QMessageBox::StandardButton::Yes) {
    s21::Controller::getInstance().changeModelTypeAndLayersSize(
        chooseModelType(), val);
    prev_layers_size_ = val;
  } else {
    ui_->layers_spinBox->setValue(prev_layers_size_);
  }
}

void LetterRecogniserWindow::changeLearnRate(double rate) {
  s21::Controller::getInstance().setLearningRate(rate);
}

void LetterRecogniserWindow::loadWeightsButtonClicked() {
  QString filename = QFileDialog::getOpenFileName(
      this, tr("Open File"), QDir::currentPath(), tr("TXT (*.txt)"));
  if (filename.isEmpty()) return;

  try {
    s21::Controller::getInstance().loadWeights(filename.toStdString());
  } catch (const std::exception& exp) {
    QMessageBox::critical(this, "Failed to load weights.",
                          QString("Error message: ") + QString(exp.what()));
  }
}

void LetterRecogniserWindow::saveWeightsButtonClicked() {
  QString filename = QFileDialog::getSaveFileName(
      this, tr("Save File"), QDir::currentPath(), tr("TXT (*.txt)"));
  if (filename.isEmpty()) return;

  try {
    s21::Controller::getInstance().saveWeights(filename.toStdString());
  } catch (const std::exception& exp) {
    QMessageBox::critical(this, "Failed to save weights.",
                          QString("Error message: ") + QString(exp.what()));
  }
}

void LetterRecogniserWindow::randomizeWeightsButtonClicked() {
  s21::Controller::getInstance().randomizeWeights();
}

void LetterRecogniserWindow::testingButtonClicked() {
  if (ui_->start_testing_pushButton->text().contains("Start")) {
    QString filename =
        QFileDialog::getOpenFileName(this, tr("Open File"), QDir::currentPath(),
                                     tr("CSV (*.csv);; TXT (*.txt)"));
    if (filename.isEmpty()) return;

    size_t testing_part = ui_->testpart_doubleSpinBox->value() * 100.0;
    if (testing_part == 0) return;

    ui_->start_testing_pushButton->setText("Stop Testing");
    blockButtons(false, true);

    testing_future_watcher_ = new QFutureWatcher<s21::MLPTestMetrics>(this);
    connect(testing_future_watcher_,
            &QFutureWatcher<s21::MLPTestMetrics>::finished, this,
            &LetterRecogniserWindow::testingResults);
    connect(testing_future_watcher_,
            &QFutureWatcher<s21::MLPTestMetrics>::finished,
            testing_future_watcher_,
            &QFutureWatcher<s21::MLPTestMetrics>::deleteLater);

    QFuture<s21::MLPTestMetrics> future = QtConcurrent::run(
        [filename, testing_part, &exception_msg = exception_msg_]() {
          s21::MLPTestMetrics metrics;
          try {
            metrics = s21::Controller::getInstance().startTesting(
                filename.toStdString(), testing_part);
          } catch (const std::exception& ex) {
            exception_msg = ex.what();
            throw std::runtime_error(ex.what());
          }

          return metrics;
        });
    testing_future_watcher_->setFuture(future);

  } else {
    canceled_ = true;
    s21::Controller::getInstance().stopTrainer();

    ui_->start_testing_pushButton->setText("Start Testing");
    blockButtons(true, true);
    ProcessCallback(0, s21::MLPTrainStages::DONE);
  }
}

void LetterRecogniserWindow::trainingButtonClicked() {
  if (ui_->start_training_pushButton->text().contains("Start")) {
    QString filename =
        QFileDialog::getOpenFileName(this, tr("Open File"), QDir::currentPath(),
                                     tr("CSV (*.csv);; TXT (*.txt)"));
    if (filename.isEmpty()) return;

    ui_->start_training_pushButton->setText("Stop Training");
    blockButtons(false, false);

    bool cv = ui_->crossvalid_checkBox->isChecked();
    size_t epochs = static_cast<size_t>(ui_->epochs_count_spinBox->value());

    training_future_watcher_ = new QFutureWatcher<std::vector<double>>(this);
    connect(training_future_watcher_,
            &QFutureWatcher<std::vector<double>>::finished, this,
            &LetterRecogniserWindow::trainingResult);
    connect(training_future_watcher_,
            &QFutureWatcher<std::vector<double>>::finished,
            training_future_watcher_,
            &QFutureWatcher<std::vector<double>>::deleteLater);

    QFuture<std::vector<double>> future = QtConcurrent::run(
        [cv, filename, epochs, &exception_msg = exception_msg_]() {
          std::vector<double> res;
          try {
            res = s21::Controller::getInstance().startTraining(
                cv, filename.toStdString(), epochs);
          } catch (const std::exception& ex) {
            exception_msg = ex.what();
            throw std::runtime_error(ex.what());
          }

          return res;
        });

    training_future_watcher_->setFuture(future);

  } else {
    canceled_ = true;
    s21::Controller::getInstance().stopTrainer();

    ui_->start_training_pushButton->setText("Start Training");
    blockButtons(true, false);
    ProcessCallback(0, s21::MLPTrainStages::DONE);
  }
}

void LetterRecogniserWindow::testingResults() {
  ui_->start_testing_pushButton->setText("Start Testing");
  blockButtons(true, true);

  if (canceled_) {
    canceled_ = false;
    delete testing_future_watcher_;
    testing_future_watcher_ = nullptr;
    return;
  }

  try {
    s21::MLPTestMetrics metrics = testing_future_watcher_->result();

    ui_->avacur_value_label->setText(QString::number(metrics.accurancy, 'g', 2));
    ui_->avperc_value_label->setText(
        QString::number(metrics.accurancy_percent, 'g', 2));
    ui_->precision_value_label->setText(
        QString::number(metrics.precision, 'g', 2));
    ui_->recall_value_label->setText(QString::number(metrics.recall, 'g', 2));
    ui_->fm_value_label->setText(QString::number(metrics.f_measure, 'g', 2));
    ui_->time_value_label->setText(
        QString::number(metrics.testing_time.count() / 1000.0l, 'g', 2) + "s");
  } catch (const std::exception& ex) {
    QMessageBox::critical(
        this, "Failed to test model.",
        QString("Error message: ") + QString::fromStdString(exception_msg_));
  }

  delete testing_future_watcher_;
  testing_future_watcher_ = nullptr;
}

void LetterRecogniserWindow::trainingResult() {
  ui_->start_training_pushButton->setText("Start Training");
  blockButtons(true, false);

  if (canceled_) {
    canceled_ = false;
    delete training_future_watcher_;
    training_future_watcher_ = nullptr;
    return;
  }

  try {
    std::vector<double> mse_errors = training_future_watcher_->result();

    ui_->trainres_graph_widget->drawGraph(mse_errors);
  } catch (const std::exception& ex) {
    QMessageBox::critical(
        this, "Failed to train model.",
        QString("Error message: ") + QString::fromStdString(exception_msg_));
  }

  delete testing_future_watcher_;
  testing_future_watcher_ = nullptr;
}

s21::ModelType LetterRecogniserWindow::chooseModelType() {
  return ui_->model_comboBox->currentText() == "Matrix" ? s21::ModelType::Matrix
                                                       : s21::ModelType::Graph;
}

void LetterRecogniserWindow::blockButtons(bool unblock, bool testing) {
  if (testing) {
    ui_->start_training_pushButton->setEnabled(unblock);
  } else {
    ui_->start_testing_pushButton->setEnabled(unblock);
  }

  ui_->model_comboBox->setEnabled(unblock);
  ui_->layers_spinBox->setEnabled(unblock);
  ui_->load_weights_pushButton->setEnabled(unblock);
  ui_->save_weights_pushButton->setEnabled(unblock);
  ui_->random_weights_pushButton->setEnabled(unblock);
}
