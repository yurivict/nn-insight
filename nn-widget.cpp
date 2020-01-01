
#include "nn-widget.h"
#include "svg-graphics-generator.h"

#include <QMouseEvent>
#include <QByteArray>


NnWidget::NnWidget(QWidget *parent)
: ZoomableSvgWidget(parent)
, model(nullptr)
{
}

/// interface

void NnWidget::open(const PluginInterface::Model *model_) {
	load(SvgGraphics::generateModelSvg(model_,
		{&modelIndexes.allOperatorBoxes, &modelIndexes.allTensorLabelBoxes, &modelIndexes.allInputBoxes, &modelIndexes.allOutputBoxes}));
	model = model_;
}

void NnWidget::close() {
	clearIndices();
	model = nullptr;
	load(QByteArray());
	resize(0,0);
}

/// overridden

void NnWidget::mousePressEvent(QMouseEvent *event) {
	if (model) {
		auto searchResult = findObjectAtThePoint(event->pos());
		if (searchResult.operatorId != -1)
			emit clickedOnOperator(searchResult.operatorId);
		else if (searchResult.tensorId != -1)
			emit clickedOnTensorEdge(searchResult.tensorId);
		else if (searchResult.inputIdx != -1)
			emit clickedOnInput((unsigned)searchResult.inputIdx, model->getInputs()[searchResult.inputIdx]);
		else if (searchResult.outputIdx != -1)
			emit clickedOnOutput((unsigned)searchResult.outputIdx, model->getOutputs()[searchResult.outputIdx]);
		else
			emit clickedOnBlankSpace();
	}

	// pass
	ZoomableSvgWidget::mousePressEvent(event);
}

/// internals

void NnWidget::clearIndices() {
	for (auto index : {&modelIndexes.allOperatorBoxes,&modelIndexes.allTensorLabelBoxes,&modelIndexes.allInputBoxes,&modelIndexes.allOutputBoxes})
		index->clear();
}

NnWidget::AnyObject NnWidget::findObjectAtThePoint(const QPointF &pt) const {
	// XXX ad hoc algorithm until we find some good geoindexing implementation

	// operator box?
	for (PluginInterface::OperatorId oid = 0, oide = modelIndexes.allOperatorBoxes.size(); oid < oide; oid++)
		if (modelIndexes.allOperatorBoxes[oid].contains(pt))
			return {(int)oid,-1,-1,-1};

	// tensor label?
	for (PluginInterface::TensorId tid = 0, tide = modelIndexes.allTensorLabelBoxes.size(); tid < tide; tid++)
		if (modelIndexes.allTensorLabelBoxes[tid].contains(pt))
			return {-1,(int)tid,-1,-1};

	// input box?
	for (unsigned idx = 0, idxe = modelIndexes.allInputBoxes.size(); idx < idxe; idx++)
		if (modelIndexes.allInputBoxes[idx].contains(pt))
			return {-1,-1,(int)model->getInputs()[idx],-1};

	// output box?
	for (unsigned idx = 0, idxe = modelIndexes.allOutputBoxes.size(); idx < idxe; idx++)
		if (modelIndexes.allOutputBoxes[idx].contains(pt))
			return {-1,-1,-1,(int)model->getOutputs()[idx]};

	return {-1,-1,-1,-1}; // not found
}

