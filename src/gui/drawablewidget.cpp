#include "drawablewidget.h"

DrawableWidget::DrawableWidget(QWidget *parent) : QWidget{parent} {}

void DrawableWidget::mousePressEvent(QMouseEvent *mouse) {
  if (mouse->button() == Qt::MouseButton::LeftButton) {
    moving_ = true;
    start_point_ = mouse->pos();
  }
}

void DrawableWidget::mouseMoveEvent(QMouseEvent *mouse) {
  if (moving_) {
    drawLine(mouse->pos());
  }
}

void DrawableWidget::mouseReleaseEvent(QMouseEvent *mouse) {
  if (mouse->button() == Qt::MouseButton::LeftButton) {
    moving_ = false;
  }

  emit predict(toMNIST());
}

void DrawableWidget::paintEvent(QPaintEvent *) {
  QPainter painter(this);

  painter.drawImage(canvas_.rect(), canvas_);
}

void DrawableWidget::loadImage(const QString &fileName) {
  QImage new_image;

  if (!new_image.load(fileName)) {
    QMessageBox::critical(this, "Image loading error.",
                          "Failed to load image: " + fileName + ".");
    return;
  }

  new_image =
      new_image.scaled(QSize(512, 512), Qt::AspectRatioMode::KeepAspectRatio,
                       Qt::TransformationMode::SmoothTransformation);
  setImage(new_image);
  update();

  emit predict(toMNIST());
}

void DrawableWidget::clear() {
  canvas_.fill(Qt::GlobalColor::white);
  update();
}

QImage DrawableWidget::toMNIST() {
  QImage mnist = canvas_;
  mnist.invertPixels();
  mnist = mnist.scaled(QSize(28, 28), Qt::KeepAspectRatio)
              .convertToFormat(QImage::Format::Format_Grayscale8);

  return mnist;
}

void DrawableWidget::setImage(const QImage &new_image) { canvas_ = new_image; }

void DrawableWidget::drawLine(QPoint endPoint) {
  QPainter painter(&canvas_);

  QPen pen(Qt::GlobalColor::black);
  pen.setStyle(Qt::PenStyle::SolidLine);
  pen.setWidth(15);
  pen.setCapStyle(Qt::PenCapStyle::RoundCap);
  pen.setJoinStyle(Qt::PenJoinStyle::RoundJoin);

  painter.setPen(pen);
  painter.drawLine(start_point_, endPoint);

  update();
  start_point_ = endPoint;
}
