#pragma once

#include <QByteArray>

#include "plugin-interface.h"

#include <array>

namespace SvgGraphics {

QByteArray generateModelSvg(const PluginInterface::Model *model,
	const std::array<std::vector<QRectF>*,4> outIndexes // indexes: allOperatorBoxes,allTensorLabelBoxes,allInputBoxes,allOutputBoxes
);

}
