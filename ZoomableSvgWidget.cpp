
#include "ZoomableSvgWidget.h"
#include <QSvgRenderer>
#include <QDebug>
#include <QMouseEvent>
#include <QWheelEvent>
#include <QPointF>
#include <QApplication>
#include <QDebug>

#include "misc.h"

ZoomableSvgWidget::ZoomableSvgWidget(QWidget *parent)
: QSvgWidget(parent)
, scalingFactor(1.)
{ }

/// overridables

void ZoomableSvgWidget::showEvent(QShowEvent *event) {
	defaultSvgSize = renderer()->defaultSize();
	fixWindowSize(defaultSvgSize);
	// pass
	QSvgWidget::showEvent(event);
}

void ZoomableSvgWidget::mousePressEvent(QMouseEvent* event) {
	// emit the mousePressOccurred signal
	auto pos = event->pos();
	emit mousePressOccurred(QPointF(double(pos.x())/scalingFactor, double(pos.y())/scalingFactor));
	// pass
	QSvgWidget::mousePressEvent(event);
}

void ZoomableSvgWidget::wheelEvent(QWheelEvent* event) {
	qDebug() << "ZoomableSvgWidget::wheelEvent pos=" << event->pos() << " globalPos=" << event->globalPos() << " fromGlobal=" << mapFromGlobal(event->globalPos());
	if (QApplication::keyboardModifiers()&Qt::ControlModifier) {
		// zoom
		double angle = event->angleDelta().y();
		scalingFactor *= 1 + (angle/360*0.1);
		auto posOld = mapFromGlobal(QCursor::pos(QApplication::screens().at(0)));
		fixWindowSize(defaultSvgSize*scalingFactor);
		auto posNew = mapFromGlobal(QCursor::pos(QApplication::screens().at(0)));
		qDebug() << "posMove=" << (posNew-posOld);
		event->accept();
	} else {
		// pass
		QSvgWidget::wheelEvent(event);
	}
}

/// private methods

void ZoomableSvgWidget::fixWindowSize(QSize sz) {
	setMinimumSize(sz);
	setMaximumSize(sz);
	resize(sz);
}
