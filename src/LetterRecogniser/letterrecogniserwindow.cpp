#include "letterrecogniserwindow.h"
#include "./ui_letterrecogniserwindow.h"

LetterRecogniserWindow::LetterRecogniserWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::LetterRecogniserWindow) {
    ui->setupUi(this);
}

LetterRecogniserWindow::~LetterRecogniserWindow() {
    delete ui;
}

