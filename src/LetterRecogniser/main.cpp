#include <QApplication>

#include "letterrecogniserwindow.h"

int main(int argc, char *argv[]) {
  QApplication a(argc, argv);
  LetterRecogniserWindow w;
  w.show();
  return a.exec();
}
