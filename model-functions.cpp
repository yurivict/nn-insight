
#include "model-functions.h"
#include "plugin-interface.h"

#include <assert.h>

namespace ModelFunctions {

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

void computeTensors(const PluginInterface::Model *model, std::vector<std::unique_ptr<float>> *tensorData) {
}

}
