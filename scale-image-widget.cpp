// Copyright (C) 2022 by Yuri Victorovich. All rights reserved.

#include "scale-image-widget.h"

#include <QLineEdit>

#include "misc.h"

ScaleImageWidget::ScaleImageWidget(QWidget *parent)
: QWidget(parent)
, layout(this)
, widthLabel("width", this)
, widthSpinBox(this)
, heightLabel("height", this)
, heightSpinBox(this)
, unitsLabel("%", this)
, self(false)
{
	layout.addWidget(&widthLabel);
	layout.addWidget(&widthSpinBox);
	layout.addWidget(&heightLabel);
	layout.addWidget(&heightSpinBox);
	layout.addWidget(&unitsLabel);

	// alignment
	widthLabel .setAlignment(Qt::AlignRight|Qt::AlignVCenter);
	heightLabel.setAlignment(Qt::AlignRight|Qt::AlignVCenter);
	unitsLabel .setAlignment(Qt::AlignLeft|Qt::AlignVCenter);

	// size policies
	widthLabel    .setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
	widthSpinBox  .setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
	heightLabel   .setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
	heightSpinBox .setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
	unitsLabel    .setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);

	// margins
	setContentsMargins(0,0,0,0);
	layout.setContentsMargins(0,0,0,0);
	// XXX this attempt to shrink spinboxes doesn't work, and they still slightly stick outside of QLabels vertically
	widthSpinBox.findChild<QLineEdit*>()->setTextMargins(0,0,0,0);
	heightSpinBox.findChild<QLineEdit*>()->setTextMargins(0,0,0,0);

	// values
	widthSpinBox .setMinimum(1); // 1% and up
	heightSpinBox.setMinimum(1);
	widthSpinBox .setMaximum(maxValue);
	heightSpinBox.setMaximum(maxValue);

	// connect signals
	connect(&widthSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), [this](int i) {
		if (self)
			return;

		self = true;
		heightSpinBox.setValue(i);
		self = false;

		emitSignal();
	});
	connect(&heightSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), [this](int i) {
		if (self)
			return;

		self = true;
		widthSpinBox.setValue(i);
		self = false;

		emitSignal();
	});
}

/// interface

void ScaleImageWidget::setFactor(unsigned factor) {
	self = true;
	heightSpinBox.setValue(factor);
	widthSpinBox.setValue(factor);
	self = false;
}

/// internals

void ScaleImageWidget::emitSignal() {
	emit scalingFactorChanged((unsigned)widthSpinBox.value(), (unsigned)heightSpinBox.value());
}

