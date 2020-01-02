// Copyright (C) 2020 by Yuri Victorovich. All rights reserved.

#pragma once

#include "zoomable-svg-widget.h"
#include "plugin-interface.h"

#include <vector>

#include <QRectF>
class QMouseEvent;

class NnWidget : public ZoomableSvgWidget {
	Q_OBJECT

	const PluginInterface::Model* model;     // the model that is currently open in the widget
	struct {
		std::vector<QRectF> allOperatorBoxes;    // indexed based on OperatorId
		std::vector<QRectF> allTensorLabelBoxes; // indexed based on TensorId
		std::vector<QRectF> allInputBoxes;       // indexed based on input id
		std::vector<QRectF> allOutputBoxes;      // indexed based on output id
	} modelIndexes;

private: // types
	struct AnyObject {
		int operatorId;
		int tensorId;
		int inputIdx;
		int outputIdx;
	};

public: // constructor
	NnWidget(QWidget *parent);

public: // interface
	void open(const PluginInterface::Model *model_);
	void close();

public: // overridden
	void mousePressEvent(QMouseEvent *event) override;

signals:
	void clickedOnOperator(PluginInterface::OperatorId oid);
	void clickedOnTensorEdge(PluginInterface::TensorId tid);
	void clickedOnInput(unsigned inputIdx, PluginInterface::TensorId tid);
	void clickedOnOutput(unsigned outputIdx, PluginInterface::TensorId tid);
	void clickedOnBlankSpace();

private: // internals
	void clearIndices();
	AnyObject findObjectAtThePoint(const QPointF &pt) const;
};

