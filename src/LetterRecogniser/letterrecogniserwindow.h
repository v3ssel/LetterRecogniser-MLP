#ifndef LETTERRECOGNISERWINDOW_H
#define LETTERRECOGNISERWINDOW_H

#include <QMainWindow>
#include <QFileDialog>

QT_BEGIN_NAMESPACE
namespace Ui { class LetterRecogniserWindow; }
QT_END_NAMESPACE

class LetterRecogniserWindow : public QMainWindow {
    Q_OBJECT

   public:
    LetterRecogniserWindow(QWidget *parent = nullptr);
    ~LetterRecogniserWindow();

   public slots:
    void clearPaintButtonClicked();
    void loadBmpImageButtonClicked();

   private:
    Ui::LetterRecogniserWindow *ui;
};
#endif // LETTERRECOGNISERWINDOW_H
