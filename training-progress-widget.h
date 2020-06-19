// Copyright (C) 2020 by Yuri Victorovich. All rights reserved.

#pragma once

#include "qcustomplot.h"

class TrainingProgressWidget : public QCustomPlot {
	Q_OBJECT

	QVector<double> xEpoch, yLoss;

public:
	TrainingProgressWidget(QWidget *parent);

public: // iface
	void setRanges(unsigned epochMax, float lossMax);
	void addDataPoint(unsigned epoch, float loss);
	void clear();
};
