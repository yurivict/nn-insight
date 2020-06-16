// Copyright (C) 2020 by Yuri Victorovich. All rights reserved.

#pragma once

#include "plugin-interface.h"

#include <array>
#include <functional>
#include <map>
#include <sstream>
#include <vector>

namespace Training {

struct TrainingIO {
	std::vector<PluginInterface::TensorId>                        targetInputs;      // one per model output
	std::vector<PluginInterface::TensorId>                        lossOutputs;       // one per model output
	std::map<PluginInterface::TensorId,PluginInterface::TensorId> derivativeOutputs; // derivative -> original parameters, one per parameter tensor
};

struct OriginalIO {
	std::vector<PluginInterface::TensorId>                        inputs;
	std::vector<PluginInterface::TensorId>                        outputs;
};

PluginInterface::Model* constructTrainingModel(const PluginInterface::Model *model, PluginInterface::OperatorKind lossFunction); // returns ownership

bool getModelTrainingIO(const PluginInterface::Model *trainingModel, TrainingIO &trainingIO);
void getModelOriginalIO(const PluginInterface::Model *trainingModel, OriginalIO &originalIO);

std::string verifyDerivatives(PluginInterface::Model *trainingModel, unsigned numVerifications, unsigned numPoints, float delta, std::function<std::array<std::vector<float>,2>(bool)> getData);

bool runTrainingLoop(PluginInterface::Model *model, unsigned batchSize, float trainingRate, bool *stopFlag,
	std::function<std::array<std::vector<float>,2>(bool)> getData,
	std::function<void(unsigned)> batchDone

);

bool isTrainingLayer(const PluginInterface::Model *model, PluginInterface::TensorId tid);

}
