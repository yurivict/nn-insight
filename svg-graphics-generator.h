#pragma once

#include <QByteArray>

#include "plugin-interface.h"

namespace SvgGraphics {

QByteArray generateModelSvg(const PluginInterface::Model *model);

}
