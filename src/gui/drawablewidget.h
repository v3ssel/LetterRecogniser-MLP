#ifndef DRAWABLEWIDGET_H
#define DRAWABLEWIDGET_H

#include <QMessageBox>
#include <QPaintEvent>
#include <QPainter>
#include <QWidget>

class DrawableWidget : public QWidget {
  Q_OBJECT

 public:
  explicit DrawableWidget(QWidget* parent = nullptr);

  void mousePressEvent(QMouseEvent* mouse) override;
  void mouseMoveEvent(QMouseEvent* mouse) override;
  void mouseReleaseEvent(QMouseEvent* mouse) override;

  void paintEvent(QPaintEvent* paint) override;

  void loadImage(const QString& fileName);
  void clear();

  QImage toMNIST();
  void setImage(const QImage& new_image);

 signals:
  void predict(const QImage& image);

 private:
  bool moving_;

  QImage canvas_;
  QPoint start_point_;

  void drawLine(QPoint endPoint);
};

#endif  // DRAWABLEWIDGET_H
