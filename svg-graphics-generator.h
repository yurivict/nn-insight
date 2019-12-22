#pragma once

#include <QByteArray>

#include "plugin-interface.h"

#include <tuple>

namespace SvgGraphics {

QByteArray generateModelSvg(const PluginInterface::Model *model,
	const std::tuple<std::vector<QRectF>&,std::vector<QRectF>&> outIndexes // indexes: allOperatorBoxes,allTensorLabelBoxes
);

}
