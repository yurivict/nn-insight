// Copyright (C) 2022 by Yuri Victorovich. All rights reserved.

#include "no-nn-is-open-widget.h"



NoNnIsOpenWidget::NoNnIsOpenWidget(QWidget *parent)
: QWidget(parent)
, layout(this)
, openNnFileButton(tr("Open a Neural Network File"), this)
, spacer(this)
{
	layout.addWidget(&openNnFileButton, 0/*row*/, 0/*col*/, 1/*rowSpan*/, 2/*columnSpan*/);
	layout.addWidget(&spacer,           1/*row*/, 0/*col*/, 1/*rowSpan*/, 2/*columnSpan*/);

	// connect signals
	connect(&openNnFileButton, SIGNAL(pressed()), this, SIGNAL(openNeuralNetworkFilePressed()));
}
