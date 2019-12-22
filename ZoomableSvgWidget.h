#pragma once

#include <QSvgWidget>
#include <QSize>
class QShowEvent;
class QWheelEvent;

class ZoomableSvgWidget : public QSvgWidget {
	Q_OBJECT

	QSize         defaultSvgSize;
	double        scalingFactor;
public:
	ZoomableSvgWidget(QWidget *parent);

public: // overridable
	void showEvent(QShowEvent *event);

protected:
	void wheelEvent(QWheelEvent* event);

private: // 
	void fixWindowSize(QSize sz);
};
