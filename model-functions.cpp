// Copyright (C) 2020 by Yuri Victorovich. All rights reserved.

#include "model-functions.h"
#include "plugin-interface.h"
#include "nn-types.h"
#include "misc.h"

#include <assert.h>

namespace ModelFunctions {

bool isTensorComputed(const PluginInterface::Model *model, PluginInterface::TensorId tensorId) {
	return !model->getTensorHasData(tensorId) && !model->getTensorIsVariableFlag(tensorId);
}

size_t computeModelFlops(const PluginInterface::Model *model) {
	size_t flops = 0;
	for (PluginInterface::OperatorId oid = 0, oide = model->numOperators(); oid < oide; oid++)
		flops += computeOperatorFlops(model, oid);
	return flops;
}

size_t computeOperatorFlops(const PluginInterface::Model *model, PluginInterface::OperatorId operatorId) {
	std::vector<PluginInterface::TensorId> inputs, outputs;
	model->getOperatorIo(operatorId, inputs, outputs);
	switch (model->getOperatorKind(operatorId)) {
	case PluginInterface::KindConv2D: {
		auto shapeImage = model->getTensorShape(inputs[0]);
		auto shapeWeights = model->getTensorShape(inputs[1]);
		assert(shapeImage.size() == 4 && shapeImage[0] == 1);
		assert(shapeWeights.size() == 4);
		return tensorFlatSize(shapeWeights)*(shapeImage[1]*shapeImage[2]); // TODO add summations, strides, pads
	} case PluginInterface::KindFullyConnected: {
		auto shapeWeights = model->getTensorShape(inputs[1]);
		assert(shapeWeights.size() == 2);
		return tensorFlatSize(shapeWeights); // add summations, strides, pads
	} case PluginInterface::KindAdd:
	  case PluginInterface::KindRelu:
	  case PluginInterface::KindRelu6:
	  case PluginInterface::KindLeakyRelu:
	  case PluginInterface::KindTanh:
	  case PluginInterface::KindHardSwish:
	  case PluginInterface::KindSub:
	  case PluginInterface::KindMul:
	  case PluginInterface::KindDiv:
	  case PluginInterface::KindMaximum:
	  case PluginInterface::KindMinimum:
		return tensorFlatSize(model->getTensorShape(inputs[0])); // input size
	  default:
		return 0; // TODO
	}
}

size_t sizeOfModelStaticData(const PluginInterface::Model *model, unsigned &outObjectCount, size_t &outMaxStaticDataPerOperator) {
	size_t size = 0;
	outObjectCount = 0;
	outMaxStaticDataPerOperator = 0;
	for (PluginInterface::OperatorId oid = 0, oide = model->numOperators(); oid < oide; oid++) {
		auto sizeForOperator = sizeOfOperatorStaticData(model, oid, outObjectCount);
		size += sizeForOperator;
		if (sizeForOperator > outMaxStaticDataPerOperator)
			outMaxStaticDataPerOperator = sizeForOperator;
	}
	return size;
}

size_t sizeOfOperatorStaticData(const PluginInterface::Model *model, PluginInterface::OperatorId operatorId, unsigned &outObjectCount) {
	size_t size = 0;
	std::vector<PluginInterface::TensorId> inputs, outputs;
	model->getOperatorIo(operatorId, inputs, outputs);
	for (PluginInterface::TensorId tensorId : inputs)
		if (model->getTensorHasData(tensorId)) {
			size += tensorFlatSize(model->getTensorShape(tensorId))*sizeof(float); // TODO handle other types
			outObjectCount++;
		}
	return size;
}

float dataRatioOfOperator(const PluginInterface::Model *model, PluginInterface::OperatorId operatorId) {
	std::vector<PluginInterface::TensorId> inputs, outputs;
	model->getOperatorIo(operatorId, inputs, outputs);

	unsigned sizeOfInputs = 0;
	unsigned sizeOfOutputs = 0;
	for (auto i : inputs)
		if (isTensorComputed(model, i))
			sizeOfInputs += tensorFlatSize(model->getTensorShape(i));
	for (auto o : outputs)
		sizeOfOutputs += tensorFlatSize(model->getTensorShape(o));

	return float(sizeOfOutputs)/float(sizeOfInputs);
}

float dataRatioOfOperatorModelInputToIns(const PluginInterface::Model *model, PluginInterface::OperatorId operatorId) {
	std::vector<PluginInterface::TensorId> inputs, outputs;
	model->getOperatorIo(operatorId, inputs, outputs);

	unsigned sizeOfInputs = 0, cntInputs = 0;
	for (auto i : inputs)
		if (isTensorComputed(model, i)) {
			sizeOfInputs += tensorFlatSize(model->getTensorShape(i));
			cntInputs++;
		}

	assert(model->numInputs()==1);
	return float(sizeOfInputs)/cntInputs/float(tensorFlatSize(model->getTensorShape(model->getInputs()[0])));
}

float dataRatioOfOperatorModelInputToOuts(const PluginInterface::Model *model, PluginInterface::OperatorId operatorId) {
	std::vector<PluginInterface::TensorId> inputs, outputs;
	model->getOperatorIo(operatorId, inputs, outputs);

	unsigned sizeOfOutputs = 0, cntOutputs = 0;
	for (auto o : outputs) {
		sizeOfOutputs += tensorFlatSize(model->getTensorShape(o));
		cntOutputs++;
	}

	assert(model->numInputs()==1);
	return float(sizeOfOutputs)/cntOutputs/float(tensorFlatSize(model->getTensorShape(model->getInputs()[0])));
}

void computeTensors(const PluginInterface::Model *model, std::vector<std::unique_ptr<float>> *tensorData) {
}

OutputInterpretationKind guessOutputInterpretationKind(const PluginInterface::Model *model) {
	// classify based on the first output tensor shape
	auto outputTensorId = model->getOutputs()[0];
	auto outputShape = tensorStripLeadingOnes(model->getTensorShape(outputTensorId));

	switch (outputShape.size()) {
	case 1: // 1-dimensional vector with numbers, must be object classification
		switch (outputShape[0]) {
		case 1000:
			return OutputInterpretationKind_ImageNet1000; // might be wrong, but the number is the same and there aren't too many networks around
		case 1001:
			return OutputInterpretationKind_ImageNet1001;
		case 2:
			return OutputInterpretationKind_NoYes; // it could also be OutputInterpretationKind_YesNo but we can't know this
		default:
			return OutputInterpretationKind_Undefined; // we don't know from the information that we have
		}
	case 3: { // see if the shape matches the input shape
		auto inputTensorId = model->getInputs()[0];
		auto inputShape = tensorStripLeadingOnes(model->getTensorShape(inputTensorId));
		if (inputShape.size()==3 && inputShape[0]==outputShape[0] && inputShape[1]==outputShape[1])
			return OutputInterpretationKind_PixelClassification;
		return OutputInterpretationKind_Undefined; // some large shape but it doesn't match the input so we don't know
	} default:
		return OutputInterpretationKind_Undefined; // we don't know from the information that we have
	}
}

/// string-returting aggretgate versions

std::string dataRatioOfOperatorStr(const PluginInterface::Model *model, PluginInterface::OperatorId operatorId,
                                   float &outIncreaseAboveInput, float &outModelInputToOut)
{
	auto modelInputToIns = ModelFunctions::dataRatioOfOperatorModelInputToIns(model, operatorId);
	auto modelInputToOuts = ModelFunctions::dataRatioOfOperatorModelInputToOuts(model, operatorId);

	outIncreaseAboveInput = modelInputToOuts/modelInputToIns;
	outModelInputToOut = modelInputToOuts;

	return STR(ModelFunctions::dataRatioOfOperator(model, operatorId) <<
	           ", model-input-to-ins: " << modelInputToIns <<
	           ", model-input-to-outs: " << modelInputToOuts);
}

}
