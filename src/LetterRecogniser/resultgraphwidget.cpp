#include "resultgraphwidget.h"
#include <QStylePainter>

ResultGraphWidget::ResultGraphWidget(QWidget *parent)
    : QWidget{parent} {
    this->setMouseTracking(true);
}

void ResultGraphWidget::init() {
    canvas_ = QImage(QSize(this->width(), this->height()), QImage::Format_ARGB32);
    canvas_.fill(Qt::TransparentMode);
    update();
}

void ResultGraphWidget::paintEvent(QPaintEvent *event) {
    QWidget::paintEvent(event);

    QPainter painter(this);
    createFrame(canvas_);
    painter.drawImage(canvas_.rect(), canvas_);
}

void ResultGraphWidget::mouseMoveEvent(QMouseEvent *event) {
    if (point_to_value.contains(event->pos())) {
        QPointF globpos = event->globalPosition();
        QWhatsThis::showText(QPoint(globpos.x(), globpos.y()), QString::number(point_to_value[event->pos()], 'g', 2));
    }
}

void ResultGraphWidget::drawGraph(std::vector<double> data) {
    if (data.empty())
        return;

    point_to_value.clear();

    QImage canvas(QSize(this->width(), this->height()), QImage::Format_ARGB32);
    canvas.fill(Qt::TransparentMode);

    QPainter painter(&canvas);

    QPen pen(Qt::GlobalColor::black);
    pen.setWidth(9);
    pen.setStyle(Qt::PenStyle::SolidLine);
    pen.setCapStyle(Qt::PenCapStyle::RoundCap);
    painter.setPen(pen);

    double min = 0.0l;
    double max = std::max(1.0, *std::max_element(data.begin(), data.end()));
    size_t distance_per_points = this->width() / (data.size() + 1);

    auto normalize = [min, max, limit = this->height()](double val) {
        return ((val - min) / (max - min)) * limit;
    };

    QList<QPoint> points;
    for (size_t x = distance_per_points, y = 0; x < this->width() && y < data.size(); x+= distance_per_points, y++) {
        QPoint point(x, this->height() - normalize(data[y]));
        points.push_back(point);
        point_to_value[point] = data[y];
    }

    QList<QLine> lines;
    for (size_t i = 0; i < points.size() - 1; i++) {
        lines.push_back(QLine(points[i], points[i + 1]));
    }

    painter.drawPoints(points);
    pen.setWidth(5);
    painter.setPen(5);
    painter.drawLines(lines);

    canvas_ = canvas;
    update();
}

void ResultGraphWidget::createFrame(QImage& image) {
    QPainter painter(&image);

    QPen pen(Qt::GlobalColor::black);
    pen.setWidth(1);
    pen.setStyle(Qt::PenStyle::SolidLine);
    pen.setCapStyle(Qt::PenCapStyle::RoundCap);
    painter.setPen(pen);

    painter.drawLine(0, 0, image.width(), 0);
    painter.drawLine(0, 0, 0, image.height());
    painter.drawLine(image.width() - 1, image.height() - 1, image.width(), 0);
    painter.drawLine(image.width() - 1, image.height() - 1, 0, image.height());
}
