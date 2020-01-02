#pragma once

#include <QByteArray>
#include <QPointF>
#include <QColor>

#include "plugin-interface.h"

#include <array>

namespace SvgGraphics {

struct ArrowParams {
	float lineWidth;
	float headLengthOut;
	float headLengthIn;
	float headWidth;
};

QByteArray generateModelSvg(const PluginInterface::Model *model,
	const std::array<std::vector<QRectF>*,4> outIndexes // indexes: allOperatorBoxes,allTensorLabelBoxes,allInputBoxes,allOutputBoxes
);

QByteArray generateTableIcon();
QByteArray generateArrow(const QPointF &vec, QColor color, const ArrowParams arrowParams = {0.05,0.35,0.27,0.2});

}
