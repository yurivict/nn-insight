// Copyright (C) 2020 by Yuri Victorovich. All rights reserved.

#include "model-functions.h"
#include "plugin-interface.h"
#include "nn-types.h"
#include "tensor.h"
#include "misc.h"
#include "util.h"

#include <assert.h>

namespace ModelFunctions {

bool isTensorComputed(const PluginInterface::Model *model, PluginInterface::TensorId tensorId) {
	return !model->getTensorHasData(tensorId) && !model->getTensorIsVariableFlag(tensorId);
}

std::string tensorKind(const PluginInterface::Model *model, PluginInterface::TensorId tensorId) { // TODO translations, tr() doesn't work outside of Q_OBJECT scope
	return Util::isValueIn(model->getInputs(), tensorId) ? "input"
	       : Util::isValueIn(model->getOutputs(), tensorId) ? "output"
	       : model->getTensorHasData(tensorId) ? "static tensor"
	       : model->getTensorIsVariableFlag(tensorId) ? "variable"
	       : "computed";
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
		return Tensor::flatSize(shapeWeights)*(shapeImage[1]*shapeImage[2]); // TODO add summations, strides, pads
	} case PluginInterface::KindFullyConnected: {
		auto shapeWeights = model->getTensorShape(inputs[1]);
		assert(shapeWeights.size() == 2);
		return Tensor::flatSize(shapeWeights); // add summations, strides, pads
	} case PluginInterface::KindLocalResponseNormalization: {
		return 0; // TODO
	} case PluginInterface::KindAdd:
	  case PluginInterface::KindRelu:
	  case PluginInterface::KindRelu6:
	  case PluginInterface::KindSub:
	  case PluginInterface::KindMul:
	  case PluginInterface::KindDiv:
	  case PluginInterface::KindMaximum:
	  case PluginInterface::KindMinimum:
		return Tensor::flatSize(model->getTensorShape(inputs[0])); // input size
	  case PluginInterface::KindTanh:
		return 10*Tensor::flatSize(model->getTensorShape(inputs[0])); //  tanh is expensive, maybe 10X at least
	  case PluginInterface::KindLogistic:
		return 10*Tensor::flatSize(model->getTensorShape(inputs[0])); //  logistic function is expensive, maybe 10X at least
	  case PluginInterface::KindLeakyRelu:
		return 2*Tensor::flatSize(model->getTensorShape(inputs[0])); // compare and multiply
	  case PluginInterface::KindHardSwish:
		return 5*Tensor::flatSize(model->getTensorShape(inputs[0])); // variable number of operations, 1..7, depending on value
	  case PluginInterface::KindConcatenation:
		return Tensor::flatSize(model->getTensorShape(inputs[0])); // unclear how to count flops for concatenation
	  case PluginInterface::KindArgMax:
	  case PluginInterface::KindArgMin:
		return Tensor::flatSize(model->getTensorShape(inputs[0]));
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
			size += Tensor::flatSize(model->getTensorShape(tensorId))*sizeof(float); // TODO handle other types
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
			sizeOfInputs += Tensor::flatSize(model->getTensorShape(i));
	for (auto o : outputs)
		sizeOfOutputs += Tensor::flatSize(model->getTensorShape(o));

	return float(sizeOfOutputs)/float(sizeOfInputs);
}

float dataRatioOfOperatorModelInputToIns(const PluginInterface::Model *model, PluginInterface::OperatorId operatorId) {
	std::vector<PluginInterface::TensorId> inputs, outputs;
	model->getOperatorIo(operatorId, inputs, outputs);

	unsigned sizeOfInputs = 0, cntInputs = 0;
	for (auto i : inputs)
		if (isTensorComputed(model, i)) {
			sizeOfInputs += Tensor::flatSize(model->getTensorShape(i));
			cntInputs++;
		}

	assert(model->numInputs()==1);
	return float(sizeOfInputs)/cntInputs/float(Tensor::flatSize(model->getTensorShape(model->getInputs()[0])));
}

float dataRatioOfOperatorModelInputToOuts(const PluginInterface::Model *model, PluginInterface::OperatorId operatorId) {
	std::vector<PluginInterface::TensorId> inputs, outputs;
	model->getOperatorIo(operatorId, inputs, outputs);

	unsigned sizeOfOutputs = 0, cntOutputs = 0;
	for (auto o : outputs) {
		sizeOfOutputs += Tensor::flatSize(model->getTensorShape(o));
		cntOutputs++;
	}

	assert(model->numInputs()==1);
	return float(sizeOfOutputs)/cntOutputs/float(Tensor::flatSize(model->getTensorShape(model->getInputs()[0])));
}

void computeTensors(const PluginInterface::Model *model, std::vector<std::unique_ptr<float>> *tensorData) {
}

OutputInterpretationKind guessOutputInterpretationKind(const PluginInterface::Model *model) {
	// classify based on the first output tensor shape
	auto outputTensorId = model->getOutputs()[0];
	auto outputShape = Tensor::stripLeadingOnes(model->getTensorShape(outputTensorId));

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
		auto inputShape = Tensor::stripLeadingOnes(model->getTensorShape(inputTensorId));
		if (inputShape.size()==3 && inputShape[0]==outputShape[0] && inputShape[1]==outputShape[1])
			return OutputInterpretationKind_PixelClassification;
		return OutputInterpretationKind_Undefined; // some large shape but it doesn't match the input so we don't know
	} default:
		return OutputInterpretationKind_Undefined; // we don't know from the information that we have
	}
}

std::string getOperatorExtraInfoString(const PluginInterface::Model *model, PluginInterface::OperatorId operatorId) {
	typedef PluginInterface PI;
	switch (model->getOperatorKind(operatorId)) {
	  case PI::KindConv2D:
	  case PI::KindDepthwiseConv2D: {
		std::vector<PI::TensorId> inputs, outputs;
		model->getOperatorIo(operatorId, inputs, outputs);
		assert(inputs.size()==3);
		auto filterShape = model->getTensorShape(inputs[1]);
		assert(filterShape.size()==4);
		return Util::stringToSubscript(STR(filterShape[1] << "x" << filterShape[2]));
	} case PI::KindMaxPool:
	  case PI::KindAveragePool: {
		int filterWidth=0, filterHeight=0;
		std::unique_ptr<PI::OperatorOptionsList> opts(model->getOperatorOptions(operatorId));
		for (auto &o : *opts)
			if (o.name == PI::OperatorOption_FILTER_WIDTH)
				filterWidth = o.value.as<int>();
			else if (o.name == PI::OperatorOption_FILTER_HEIGHT)
				filterHeight = o.value.as<int>();
		return Util::stringToSubscript(STR(filterWidth << "x" << filterHeight));
	} default:
		return ""; // no extra info
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
