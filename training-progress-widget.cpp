// Copyright (C) 2020 by Yuri Victorovich. All rights reserved.

#include "training-progress-widget.h"


TrainingProgressWidget::TrainingProgressWidget(QWidget *parent)
: QCustomPlot(parent)
{
	// set locale
	setLocale(QLocale(QLocale::English, QLocale::UnitedKingdom)); // period as decimal separator and comma as thousand separator

	{ // legend
		legend->setVisible(true);
		QFont legendFont = font();  // start out with MainWindow's font..
		legendFont.setPointSize(9); // and make a bit smaller for legend
		legend->setFont(legendFont);
		legend->setBrush(QBrush(QColor(255,255,255,230)));
		// by default, the legend is in the inset layout of the main axis rect. So this is how we access it to change legend placement:
		axisRect()->insetLayout()->setInsetAlignment(0, Qt::AlignBottom|Qt::AlignRight);
	}

	// set up the graph
	addGraph(yAxis, xAxis);
	graph(0)->setPen(QPen(QColor(255, 100, 0)));
	graph(0)->setBrush(QBrush());
	graph(0)->setLineStyle(QCPGraph::lsNone); // XXX lsLine draws lines from previous points
	graph(0)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssDisc, 5));
	graph(0)->setName("Loss vs. Epoch");

	// set labels
	xAxis->setLabel("Epoch");
	yAxis->setLabel("Loss");

	// add title layout element
	plotLayout()->insertRow(0);
	plotLayout()->addElement(0, 0, new QCPTextElement(this, "Loss vs. Training Epoch", QFont("sans", 12, QFont::Bold)));

	// make ticks on bottom axis go outward:
	xAxis->setTickLength(0, 5);
	xAxis->setSubTickLength(0, 3);
}

/// iface

void TrainingProgressWidget::setRanges(unsigned epochMax, float lossMax) {
	xAxis->setRange(0, epochMax);
	yAxis->setRange(0, lossMax);
}

void TrainingProgressWidget::addDataPoint(unsigned epoch, float loss) {
	xEpoch.push_back(epoch);
	yLoss .push_back(loss);
	graph(0)->setData(yLoss, xEpoch);
	replot();
}

void TrainingProgressWidget::clear() {
	xEpoch.clear();
	yLoss.clear();
}
