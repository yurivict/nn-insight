#pragma once

#include "plugin-interface.h"

#include <vector>
#include <map>
#include <array>
#include <functional>
#include <memory>

#include <QMarginsF>
#include <QSizeF>
#include <QPoint>

namespace ModelFunctions {

typedef std::array<std::array<float, 2>, 2> Box2;  // box ((X,Y),(W,H))

void renderModelToCoordinates(const PluginInterface::Model *model,
	const QMarginsF &operatorBoxMargins,
	std::function<QSizeF(PluginInterface::OperatorId)> operatorBoxFn, // operator boxes in inches
	std::function<QSizeF(PluginInterface::TensorId)> inputBoxFn, // input boxes in inches
	std::function<QSizeF(PluginInterface::TensorId)> outputBoxFn, // output boxes in inches
	Box2 &bbox, // return: bounding box in pixels
	std::vector<Box2> &operatorBoxes, //  return: operator boxes in pixels
	std::map<PluginInterface::TensorId, ModelFunctions::Box2> &inputBoxes, //  return: input boxes in pixels
	std::map<PluginInterface::TensorId, ModelFunctions::Box2> &outputBoxes, //  return: output boxes in pixels
	std::vector<std::vector<std::vector<QPointF>>> &tensorLineCubicSplines, // cubic splines
	std::vector<std::vector<QPointF>> &tensorLabelPositions // return: tensor label positions in pixels
);

size_t computeModelFlops(const PluginInterface::Model *model);
size_t computeOperatorFlops(const PluginInterface::Model *model, PluginInterface::OperatorId operatorId);
void computeTensors(const PluginInterface::Model *model, std::vector<std::unique_ptr<float>> *tensorData);

}
