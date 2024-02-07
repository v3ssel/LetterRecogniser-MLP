#ifndef RESULTGRAPHWIDGET_H
#define RESULTGRAPHWIDGET_H

#include <QHash>
#include <QList>
#include <QMouseEvent>
#include <QPainter>
#include <QWhatsThis>
#include <QWidget>

class ResultGraphWidget : public QWidget {
    Q_OBJECT
   public:
    explicit ResultGraphWidget(QWidget* parent = nullptr);

    void init();
    void paintEvent(QPaintEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;

    void drawGraph(std::vector<double> data);
    void createFrame(QImage& image);

   private:
    QImage canvas_;
    QHash<QPoint, double> point_to_value_;
};

#endif  // RESULTGRAPHWIDGET_H
