#pragma once

#include <QSvgWidget>
#include <QSize>
#include <QPoint>
#include <QPointF>
class QShowEvent;
class QMouseEvent;
class QWheelEvent;

class ZoomableSvgWidget : public QSvgWidget {
	Q_OBJECT

	QSize         defaultSvgSize;
	double        scalingFactor;
	bool          mousePressed;
	QPoint        lastMousePos;

public:
	ZoomableSvgWidget(QWidget *parent);

protected: // overridables
	void showEvent(QShowEvent *event);
	void mousePressEvent(QMouseEvent* event);
	void mouseReleaseEvent(QMouseEvent* event);
	void mouseMoveEvent(QMouseEvent* event);
	void wheelEvent(QWheelEvent* event);

private: // private methods
	void fixWindowSize(QSize sz);

signals:
	void mousePressOccurred(QPointF pt);
};
