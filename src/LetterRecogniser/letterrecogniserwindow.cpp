#include "letterrecogniserwindow.h"
#include "./ui_letterrecogniserwindow.h"

LetterRecogniserWindow::LetterRecogniserWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::LetterRecogniserWindow) {
    ui->setupUi(this);
    this->setWindowTitle("LetterRecogniser");

    QImage image(ui->painting_widget->width(), ui->painting_widget->height(), QImage::Format::Format_RGB16);
    image.fill(Qt::GlobalColor::white);
    ui->painting_widget->setImage(image);

    connect(ui->clear_pushButton, SIGNAL(clicked()), this, SLOT(clearPaintButtonClicked()));
    connect(ui->load_image_pushButton, SIGNAL(clicked()), this, SLOT(loadBmpImageButtonClicked()));
}

LetterRecogniserWindow::~LetterRecogniserWindow() {
    delete ui;
}

void LetterRecogniserWindow::clearPaintButtonClicked() {
    ui->painting_widget->clear();
}

void LetterRecogniserWindow::loadBmpImageButtonClicked() {
    QString fileName = QFileDialog::getOpenFileName(this,
                                       tr("Open File"), QDir::currentPath(), tr("BMP (*.bmp)"));
    ui->painting_widget->loadImage(fileName);
}

