#include "drawablewidget.h"

DrawableWidget::DrawableWidget(QWidget *parent)
    : QWidget{parent} {

}

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
}

void DrawableWidget::paintEvent(QPaintEvent *paint) {
    QPainter painter(this);

    painter.drawImage(canvas_.rect(), canvas_);
}

void DrawableWidget::loadImage(const QString& fileName) {
    QImage new_image;

    if (!new_image.load(fileName)) {
        QMessageBox::critical(this, "Image loading error.", "Failed to load image: " + fileName + ".");
        return;
    }

    qDebug() << new_image << canvas_;
    setImage(new_image);
    update();
}

void DrawableWidget::clear() {
    canvas_.fill(Qt::GlobalColor::white);
    update();
}

void DrawableWidget::setImage(const QImage &new_image) {
    canvas_ = new_image;
}

void DrawableWidget::drawLine(QPoint endPoint) {
    QPainter painter(&canvas_);

    QPen pen(Qt::GlobalColor::black);
    pen.setStyle(Qt::PenStyle::SolidLine);
    pen.setWidth(15);

    painter.setPen(pen);
    painter.drawLine(start_point_, endPoint);

    update();
    start_point_ = endPoint;
}
