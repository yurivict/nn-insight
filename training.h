// Copyright (C) 2020 by Yuri Victorovich. All rights reserved.

#pragma once

#include "plugin-interface.h"

#include <array>
#include <functional>
#include <map>
#include <sstream>
#include <tuple>
#include <vector>

namespace Training {

struct TrainingIO {
	std::vector<PluginInterface::TensorId>                        targetInputs;                 // one per model output
	std::vector<PluginInterface::TensorId>                        lossOutputs;                  // one per model output
	std::map<PluginInterface::TensorId,PluginInterface::TensorId> derivativeToParameterOutputs; // derivative -> original parameters, one per parameter tensor
	std::map<PluginInterface::TensorId,PluginInterface::TensorId> parameterToDerivativeOutputs; // original parameters -> derivative, one per parameter tensor
};

struct OriginalIO {
	std::vector<PluginInterface::TensorId>                        inputs;
	std::vector<PluginInterface::TensorId>                        outputs;
};

std::tuple<PluginInterface::Model*,float> constructTrainingModel(const PluginInterface::Model *model, PluginInterface::OperatorKind lossFunction); // returns ownership

bool getModelTrainingIO(const PluginInterface::Model *trainingModel, TrainingIO &trainingIO);
void getModelOriginalIO(const PluginInterface::Model *trainingModel, OriginalIO &originalIO);

std::string verifyDerivatives(
	PluginInterface::Model *trainingModel,
	float pendingTrainingDerivativesCoefficient,
	unsigned numVerifications,
	unsigned numPoints,
	float delta,
	float tolerance, // tolerance for derivative accuracy (ex. 0.05 means 5%)
	std::function<std::array<std::vector<float>,2>(bool)> getData);

bool runTrainingLoop(PluginInterface::Model *model, unsigned batchSize, float learningRate, unsigned maxBatches, bool *stopFlag,
	std::function<std::array<std::vector<float>,2>(bool)> getData,
	std::function<void(unsigned)> batchDone

);

bool isTrainingLayer(const PluginInterface::Model *model, PluginInterface::TensorId tid);

}
