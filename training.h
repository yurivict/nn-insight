// Copyright (C) 2020 by Yuri Victorovich. All rights reserved.

#pragma once

#include "plugin-interface.h"

namespace Training {

PluginInterface::Model* convertToTrainingModel(const PluginInterface::Model *model, PluginInterface::OperatorKind lossFunction); // returns ownership

}
