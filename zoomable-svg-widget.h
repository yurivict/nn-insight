#pragma once

#include <QSvgWidget>
#include <QSize>
#include <QPoint>
#include <QPointF>
class QString;
class QByteArray;
class QMouseEvent;
class QWheelEvent;

class ZoomableSvgWidget : public QSvgWidget {
	Q_OBJECT

	double        scalingFactor;
	bool          mousePressed;
	QPoint        lastMousePos;

public:
	ZoomableSvgWidget(QWidget *parent);

protected: // mirroring 'load' functions
	void load(const QString &file);
	void load(const QByteArray &contents);

protected: // overridables
	void mousePressEvent(QMouseEvent* event);
	void mouseReleaseEvent(QMouseEvent* event);
	void mouseMoveEvent(QMouseEvent* event);
	void wheelEvent(QWheelEvent* event);


private: // internals
	void fixWindowSize(QSize sz);

signals:
	void mousePressOccurred(QPointF pt);
};
