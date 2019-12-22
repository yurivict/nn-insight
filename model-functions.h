#pragma once

#include "plugin-interface.h"

#include <vector>
#include <array>
#include <functional>

#include <QMarginsF>
#include <QSizeF>
#include <QPoint>

namespace ModelFunctions {

typedef std::array<std::array<float, 2>, 2> Box2;  // box ((X,Y),(W,H))
typedef std::array<std::array<float, 2>, 4> Box4;  // arbitrary box with 4 points

void renderModelToCoordinates(const PluginInterface::Model *model,
	const QMarginsF &operatorBoxMargins,
	std::function<QSizeF(PluginInterface::OperatorId)> operatorBoxFn, // operator boxes in inches
	Box2 &bbox, // return: bounding box in pixels
	std::vector<Box4> &operatorBoxes, //  return: operator boxes in pixels
	std::vector<QPoint> &tensorLabelPositions // return: tensor label positions in pixels
);

}
