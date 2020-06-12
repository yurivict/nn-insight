// Copyright (C) 2020 by Yuri Victorovich. All rights reserved.

#pragma once

#include "plugin-interface.h"

#include <functional>
#include <vector>

namespace Training {

PluginInterface::Model* constructTrainingModel(const PluginInterface::Model *model, PluginInterface::OperatorKind lossFunction); // returns ownership

bool runTrainingLoop(PluginInterface::Model *model, unsigned batchSize, float trainingRate, bool *stopFlag,
	std::function<std::array<std::vector<float>,2>(bool)> getData,
	std::function<void(unsigned)> batchDone

);

}
