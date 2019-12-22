
#include "ZoomableSvgWidget.h"
#include <QSvgRenderer>
#include <QDebug>
#include <QWheelEvent>
#include <QApplication>

#include "misc.h"

ZoomableSvgWidget::ZoomableSvgWidget(QWidget *parent)
: QSvgWidget(parent)
, scalingFactor(1.)
{ }

void ZoomableSvgWidget::showEvent(QShowEvent *event) {
	defaultSvgSize = renderer()->defaultSize();
	fixWindowSize(defaultSvgSize);
	// pass
	QSvgWidget::showEvent(event);
}

void ZoomableSvgWidget::wheelEvent(QWheelEvent* event) {
	if (QApplication::keyboardModifiers()&Qt::ControlModifier) {
		// zoom
		double angle = event->angleDelta().y();
		scalingFactor *= 1 + (angle/360*0.1);
		fixWindowSize(defaultSvgSize*scalingFactor);
		event->accept();
	} else {
		// pass
		QSvgWidget::wheelEvent(event);
	}
}

void ZoomableSvgWidget::fixWindowSize(QSize sz) {
	setMinimumSize(sz);
	setMaximumSize(sz);
	resize(sz);
}
