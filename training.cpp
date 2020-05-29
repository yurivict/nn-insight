// Copyright (C) 2020 by Yuri Victorovich. All rights reserved.

#include "in-memory-model.h"
#include "training.h"
#include "misc.h"

#include <assert.h>

#include <memory>

namespace Training {

PluginInterface::Model* convertToTrainingModel(const PluginInterface::Model *model, PluginInterface::OperatorKind lossFunction) { // returns ownership
	std::unique_ptr<InMemoryModel> training(new InMemoryModel(model));

	// add a loss function and a input for labels to each output
	for (auto modelOutput : training->getOutputs()) {

		// add the input for label
		auto inputLabel = training->addTensor(STR("label-for-" << training->getTensorName(modelOutput)), training->getTensorShape(modelOutput), training->getTensorType(modelOutput), nullptr);
		training->addInput(inputLabel);

		// add the loss function
		auto lossOutput = training->addTensor(STR("loss-for-" << training->getTensorName(modelOutput)), TensorShape{1,1}, training->getTensorType(modelOutput), nullptr);
		training->addOperator(lossFunction, {modelOutput, inputLabel}, {lossOutput}, nullptr);

		// update model outputs
		training->removeOutput(modelOutput);
		training->addOutput(lossOutput);
	}

	return training.release();
}

}
