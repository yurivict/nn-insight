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
	std::vector<int/*PluginInterface::TensorId or -1*/>           parameterToTranspose;         // parameterTid->transposedParameterTid, when applicable (XXX not really IO)
};

struct OriginalIO {
	std::vector<PluginInterface::TensorId>                        inputs;
	std::vector<PluginInterface::TensorId>                        outputs;
};

enum OptimizationAlgorithm {
// see https://towardsdatascience.com/adam-latest-trends-in-deep-learning-optimization-6be9a291375c for Adam description and other discussion
	OptimizationAlgorithm_SDG_straight,      // derivatives are added to parameters
	OptimizationAlgorithm_SDG_with_inverse,  // normalized inverse derivatives are added to parameters
	OptimizationAlgorithm_Adam, // Diederik P. Kingma and Jimmy Lei Ba. Adam : A method for stochastic optimization. 2014. arXiv:1412.6980v9
	OptimizationAlgorithm_AdaGrad, // John Duchi, Elad Hazan, and Yoram Singer. Adaptive Subgradient Methods for Online Learning and Stochastic Optimization. Journal of Machine Learning Research, 12:2121–2159, 2011.
	OptimizationAlgorithm_RMSprop
};

std::tuple<PluginInterface::Model*,float> constructTrainingModel(const PluginInterface::Model *model, PluginInterface::OperatorKind lossFunction); // returns ownership

bool getModelTrainingIO(const PluginInterface::Model *trainingModel, TrainingIO &trainingIO);
void getModelOriginalIO(const PluginInterface::Model *trainingModel, OriginalIO &originalIO);
bool isTrainingNetwork(const PluginInterface::Model *model);

std::string verifyDerivatives(
	PluginInterface::Model *trainingModel,
	float pendingTrainingDerivativesCoefficient,
	unsigned numVerifications,
	unsigned numPoints,
	float delta,
	float tolerance, // tolerance for derivative accuracy (ex. 0.05 means 5%)
	std::function<std::array<std::vector<float>,2>(bool)> getData);

bool runTrainingLoop(
	PluginInterface::Model *trainingModel,
	float pendingTrainingDerivativesCoefficient,
	unsigned batchSize, float learningRate, unsigned maxBatches,
	OptimizationAlgorithm algo,
	bool *stopFlag,
	std::function<std::array<std::vector<float>,2>(bool)> getData,
	std::function<void(unsigned,float)> batchDone);

bool isTrainingLayer(const PluginInterface::Model *model, PluginInterface::TensorId tid);

}
