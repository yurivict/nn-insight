// Copyright (C) 2022 by Yuri Victorovich. All rights reserved.

#include "svg-push-button.h"
#include "util.h"

#include <QByteArray>
#include <QPainter>
#include <QPixmap>
#include <QSize>
#include <QResizeEvent>


SvgPushButton::SvgPushButton(QByteArray svg, QWidget *parent)
: QPushButton(parent)
{
	setFlat(true);
	setIcon(QIcon(QPixmap::fromImage(Util::svgToImage(svg, QSize(26,26), QPainter::CompositionMode_SourceOver))));
}

/// overridden

void SvgPushButton::resizeEvent(QResizeEvent *event) {
	setIconSize(event->size());
	// pass
	QPushButton::resizeEvent(event);
}
