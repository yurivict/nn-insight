// Copyright (C) 2020 by Yuri Victorovich. All rights reserved.

#include "misc.h"
#include "model-functions.h"
#include "nn-types.h"
#include "plugin-interface.h"
#include "tensor.h"
#include "util.h"

#include <assert.h>
#include <half.hpp>

#include <limits>
#include <memory>
#include <string>
#include <vector>

/// templetized helpers

template<typename T>
void BufferQuantizer(const TensorShape &shape, T *data, unsigned quantizationSegments) {
	// size
	auto flatSize = Tensor::flatSize(shape);
	if (flatSize <= 2)
		return; // can't quantize two or fewer numbers

	// find min/max
	T min = std::numeric_limits<T>::max();
	T max = std::numeric_limits<T>::lowest();
	T *de = data+flatSize;
	for (T *d = data; d < de; d++) {
		if (*d < min)
			min = *d;
		if (*d > max)
			max = *d;
	}
	if (min == max)
		return;

	// delta
	T delta = (max - min)/(T)quantizationSegments;
	if (delta == 0)
		return;

	// quantize
	auto one = [&min,&delta](T &val) {
		unsigned slot = (val-min)/delta;
		val = min + slot*delta;
	};
	for (T *d = data; d < de; d++)
		one(*d);
}

/// implementations

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
	  case PluginInterface::KindRSqrt:
		return 25*Tensor::flatSize(model->getTensorShape(inputs[0])); // it's very expensive to compute RSqrt
	  case PluginInterface::KindConcatenation:
		return Tensor::flatSize(model->getTensorShape(inputs[0])); // unclear how to count flops for concatenation
	  case PluginInterface::KindArgMax:
	  case PluginInterface::KindArgMin:
		return Tensor::flatSize(model->getTensorShape(inputs[0]));
	  case PluginInterface::KindSquaredDifference:
		return Tensor::flatSize(model->getTensorShape(inputs[0]))*2;
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

	// XXX the below is incorrect for unbalanced operators (data use isn't equal between branches)
	return float(sizeOfInputs)/float(Tensor::flatSize(model->getTensorShape(model->getInputs()[0])));
}

float dataRatioOfOperatorModelInputToOuts(const PluginInterface::Model *model, PluginInterface::OperatorId operatorId) {
	std::vector<PluginInterface::TensorId> inputs, outputs;
	model->getOperatorIo(operatorId, inputs, outputs);

	unsigned sizeOfOutputs = 0, cntOutputs = 0;
	for (auto o : outputs) {
		sizeOfOutputs += Tensor::flatSize(model->getTensorShape(o));
		cntOutputs++;
	}

	//assert(model->numInputs()==1);
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
	} case PI::KindFullyConnected: {
		std::vector<PI::TensorId> inputs, outputs;
		model->getOperatorIo(operatorId, inputs, outputs);
		auto filterShape = model->getTensorShape(inputs[1]);
		assert(filterShape.size()==2);
		return Util::stringToSubscript(STR(filterShape[0] << "x" << filterShape[1]));
	} case PI::KindMaxPool:
	  case PI::KindAveragePool: {
		int filterWidth=0, filterHeight=0;
		std::unique_ptr<PI::OperatorOptionsList> opts(model->getOperatorOptions(operatorId));
		assert(opts); // Pool operstors have to have options
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

void indexOperatorsByTensors(const PluginInterface::Model *model, std::vector<int/*PluginInterface::OperatorId or -1*/> &tensorProducers, std::vector<std::vector<PluginInterface::OperatorId>> &tensorConsumers) {
	const int NoOperator = -1;
	tensorProducers.resize(model->numTensors());
	tensorConsumers.resize(model->numTensors());
	{
		for (auto &p : tensorProducers)
			p = NoOperator;
		for (PluginInterface::OperatorId oid = 0, oide = (PluginInterface::OperatorId)model->numOperators(); oid < oide; oid++) {
			std::vector<PluginInterface::TensorId> oinputs, ooutputs;
			model->getOperatorIo(oid, oinputs, ooutputs);
			for (auto o : ooutputs)
				tensorProducers[o] = oid;
			for (auto i : oinputs)
				tensorConsumers[i].push_back(oid);
		}
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

void quantize(PluginInterface::Model *model, bool quantizeWeights, unsigned weightsQuantizationSegments, bool quantizeBiases, unsigned biasesQuantizationSegments) {
	std::vector<bool> doneTensors; // this is needed because some tensors can be shared
	doneTensors.resize(model->numTensors());

	auto quantizeTensor = [](const TensorShape &shape, PluginInterface::DataType type, void *data, unsigned quantizationSegments) {
		assert(data != nullptr);
		switch (type) {
		case PluginInterface::DataType_Float16:
			BufferQuantizer<half_float::half>(shape, (half_float::half*)data, quantizationSegments);
			break;
		case PluginInterface::DataType_Float32:
			BufferQuantizer<float>(shape, (float*)data, quantizationSegments);
			break;
		case PluginInterface::DataType_Float64:
			BufferQuantizer<double>(shape, (double*)data, quantizationSegments);
			break;
		case PluginInterface::DataType_Int8:
			BufferQuantizer<int32_t>(shape, (int32_t*)data, quantizationSegments);
			break;
		case PluginInterface::DataType_UInt8:
			BufferQuantizer<uint32_t>(shape, (uint32_t*)data, quantizationSegments);
			break;
		case PluginInterface::DataType_Int16:
			BufferQuantizer<int16_t>(shape, (int16_t*)data, quantizationSegments);
			break;
		case PluginInterface::DataType_Int32:
			BufferQuantizer<int32_t>(shape, (int32_t*)data, quantizationSegments);
			break;
		case PluginInterface::DataType_Int64:
			BufferQuantizer<int64_t>(shape, (int64_t*)data, quantizationSegments);
			break;
		}
	};
	for (unsigned o = 0, oe = model->numOperators(); o < oe; o++)
		switch (model->getOperatorKind(o)) {
		case PluginInterface::KindConv2D:
		case PluginInterface::KindFullyConnected: {
			std::vector<PluginInterface::TensorId> inputs, outputs;
			model->getOperatorIo(o, inputs, outputs);
			assert(inputs.size() == 3);
			auto wtid = inputs[1], btid = inputs[2];
			if (quantizeWeights && !doneTensors[wtid]) {
				quantizeTensor(model->getTensorShape(wtid), model->getTensorType(wtid), model->getTensorDataWr(wtid), weightsQuantizationSegments);
				doneTensors[wtid] = true;
			}
			if (quantizeBiases && !doneTensors[btid]) {
				quantizeTensor(model->getTensorShape(btid), model->getTensorType(btid), model->getTensorDataWr(btid), biasesQuantizationSegments);
				doneTensors[btid] = true;
			}
			break;
		} default:
			; // do nothing
		}
}

void iterateThroughParameters(const PluginInterface::Model *model, std::function<void(PluginInterface::OperatorId,unsigned,PluginInterface::TensorId)> cb) {
	for (PluginInterface::OperatorId oid = 0, oide = model->numOperators(); oid < oide; oid++)
		switch (model->getOperatorKind(oid)) { // look into all operators that contain parameters
		case PluginInterface::KindFullyConnected: {
			std::vector<PluginInterface::TensorId> inputs, outputs;
			model->getOperatorIo(oid, inputs, outputs);
			assert(inputs.size()==3 && outputs.size()==1);
			cb(oid, 1, inputs[1]); // weights
			cb(oid, 2, inputs[2]); // bias
			break;
		} default:
			; // do nothing, operator doesn't contain parameters
		}
}

}
