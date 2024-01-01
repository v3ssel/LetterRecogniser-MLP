#ifndef DRAWABLEWIDGET_H
#define DRAWABLEWIDGET_H

#include <QWidget>
#include <QPaintEvent>
#include <QPainter>
#include <QPainter>
#include <QMessageBox>

class DrawableWidget : public QWidget {
    Q_OBJECT

   public:
    explicit DrawableWidget(QWidget *parent = nullptr);

    void mousePressEvent(QMouseEvent* mouse) override;
    void mouseMoveEvent(QMouseEvent* mouse) override;
    void mouseReleaseEvent(QMouseEvent* mouse) override;

    void paintEvent(QPaintEvent* paint) override;

    void loadImage(const QString& fileName);
    void clear();

    void setImage(const QImage &new_image);
   private:
    bool moving_;

    QImage canvas_;
    QPoint start_point_;

    void drawLine(QPoint endPoint);
};

#endif // DRAWABLEWIDGET_H
