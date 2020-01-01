
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

/// string-returting aggretgate versions

std::string dataRatioOfOperatorStr(const PluginInterface::Model *model, PluginInterface::OperatorId operatorId,
                                   float &outIncreaseOboveInput, float &outModelInputToOut)
{
	auto modelInputToIns = ModelFunctions::dataRatioOfOperatorModelInputToIns(model, operatorId);
	auto modelInputToOuts = ModelFunctions::dataRatioOfOperatorModelInputToOuts(model, operatorId);

	if (modelInputToOuts>1 && modelInputToIns<modelInputToOuts) // if increases the data rate above the model input rate
		outIncreaseOboveInput = modelInputToOuts/modelInputToIns;
	else
		outIncreaseOboveInput = 0;
	outModelInputToOut = modelInputToOuts;

	return STR(ModelFunctions::dataRatioOfOperator(model, operatorId) <<
	           ", model-input-to-ins: " << modelInputToIns <<
	           ", model-input-to-outs: " << modelInputToOuts);
}

}
