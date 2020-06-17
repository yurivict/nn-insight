// Copyright (C) 2020 by Yuri Victorovich. All rights reserved.

#include "misc.h"
#include "model-validator.h"
#include "tensor.h"


namespace ModelValidator {

bool validate(const PluginInterface::Model *model, std::ostream &os) {
	bool ok = true;
	auto addError = [&](const std::string &msg) {
		os << "ERROR " << msg << std::endl;
		ok = false;
	};
	// check integrity of operator shapes
	for (PluginInterface::OperatorId oid = 0, oide = model->numOperators(); oid < oide; oid++) {

		std::vector<PluginInterface::TensorId> inputs, outputs;
		model->getOperatorIo(oid, inputs, outputs);

		switch (model->getOperatorKind(oid)) {
		case PluginInterface::KindFullyConnected: {
			if (inputs.size() != 3)
				addError(STR("Operator#" << oid << " (" << model->getOperatorKind(oid) << "): should have 3 inputs but has " << inputs.size() << " inputs"));
			if (outputs.size() != 1)
				addError(STR("Operator#" << oid << " (" << model->getOperatorKind(oid) << "): should have 1 output but has " << outputs.size() << " outputs"));
			if (inputs.size() == 3) {
				auto const inputShape   = model->getTensorShape(inputs[0]);
				auto const weightsShape = model->getTensorShape(inputs[1]);
				auto const biasShape    = model->getTensorShape(inputs[2]);
				auto const outputShape  = model->getTensorShape(outputs[0]);

				if (inputShape.size()<2 || inputShape[0]!=1)
					addError(STR("Operator#" << oid << " (" << model->getOperatorKind(oid) << "): has wrong input shape: " << inputShape << ", expected [1,N1{,...}]"));
				if (inputShape.size()!=2 || inputShape[0]!=1)
					addError(STR("Operator#" << oid << " (" << model->getOperatorKind(oid) << "): has wrong output shape: " << outputShape << ", expected [1,N]"));
				if (weightsShape.size()!=2 || weightsShape[0]!=Tensor::flatSize(outputShape) || weightsShape[1]!=Tensor::flatSize(inputShape))
					addError(STR("Operator#" << oid << " (" << model->getOperatorKind(oid) << "): has wrong weights shape: "
					             << weightsShape << ", expected [" << Tensor::flatSize(outputShape) << "," << Tensor::flatSize(inputShape) << "]"));
				if (biasShape.size()!=1 || biasShape[0]!=Tensor::flatSize(outputShape))
					addError(STR("Operator#" << oid << " (" << model->getOperatorKind(oid) << "): has wrong bias shape: "
					             << biasShape << ", expected [" << Tensor::flatSize(outputShape) << "]"));
			}
			break;
		} default:
			; // TODO
		}
	}

	return ok;
}

}
