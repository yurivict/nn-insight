// Copyright (C) 2020 by Yuri Victorovich. All rights reserved.

#pragma once

#include "plugin-interface.h"

#include <array>
#include <functional>
#include <sstream>
#include <vector>

namespace Training {

PluginInterface::Model* constructTrainingModel(const PluginInterface::Model *model, PluginInterface::OperatorKind lossFunction); // returns ownership

std::string verifyDerivatives(const PluginInterface::Model *model, unsigned numVerifications, unsigned numDirections, float delta, std::function<std::array<std::vector<float>,2>(bool)> getData);

bool runTrainingLoop(PluginInterface::Model *model, unsigned batchSize, float trainingRate, bool *stopFlag,
	std::function<std::array<std::vector<float>,2>(bool)> getData,
	std::function<void(unsigned)> batchDone

);

}
