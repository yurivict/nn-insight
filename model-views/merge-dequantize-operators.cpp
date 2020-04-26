// Copyright (C) 2020 by Yuri Victorovich. All rights reserved.

#include "merge-dequantize-operators.h"

#include "../misc.h"


namespace ModelViews {

typedef PluginInterface PI;

MergeDequantizeOperators::MergeDequantizeOperators(const PluginInterface::Model *original_)
: original(original_)
{
	// find all dequantize operators and their tensor outputs
	unsigned numDequantizeOperators = 0;
	tensorIsDequantizeInput .resize(original->numTensors());
	tensorIsDequantizeOutput.resize(original->numTensors());
	for (PI::OperatorId oid = 0, oide = original->numOperators(); oid < oide; oid++)
		if (original->getOperatorKind(oid) == PI::KindDequantize) {
			std::vector<PI::TensorId> inputs, outputs;
			original->getOperatorIo(oid, inputs, outputs);
			//PRINT("oid=" << oid << " is Dequantize: inputs.size=" << inputs.size() << " outputs.size=" << outputs.size())
			numDequantizeOperators++;
			if (inputs.size() != 1 || outputs.size() != 1)
				FAIL("MergeDequantizeOperators: Dequantize operator should have 1 input and 1 output,"
				     " found a Dequantize operator with " << inputs.size() << " input(s) and " << outputs.size() << " output(s)")
			if (!original->getTensorHasData(inputs[0]) || original->getTensorHasData(outputs[0]))
				FAIL("MergeDequantizeOperators: Dequantize operator tensor types aren't consistent with Dequantize definition")
			tensorIsDequantizeInput[inputs[0]] = true;
			tensorIsDequantizeOutput[outputs[0]] = true;
		} else
			operatorMap.push_back(oid);
	// print a notice to the user
	PRINT("MergeDequantizeOperators: merged " << numDequantizeOperators << " operators out of a total of " << original->numOperators() << " operators in a model")
}

unsigned MergeDequantizeOperators::numInputs() const {
	return original->numInputs();
}

std::vector<PI::TensorId> MergeDequantizeOperators::getInputs() const {
	return original->getInputs();
}

unsigned MergeDequantizeOperators::numOutputs() const {
	return original->numOutputs();
}

std::vector<PI::TensorId> MergeDequantizeOperators::getOutputs() const {
	return original->getOutputs();
}

unsigned MergeDequantizeOperators::numOperators() const {
	return operatorMap.size();
}

void MergeDequantizeOperators::getOperatorIo(unsigned operatorIdx, std::vector<PI::TensorId> &inputs, std::vector<PI::TensorId> &outputs) const {
	return original->getOperatorIo(operatorMap[operatorIdx], inputs, outputs);
}

PI::OperatorKind MergeDequantizeOperators::getOperatorKind(unsigned operatorIdx) const {
	return original->getOperatorKind(operatorMap[operatorIdx]);
}

PI::OperatorOptionsList* MergeDequantizeOperators::getOperatorOptions(unsigned operatorIdx) const {
	return original->getOperatorOptions(operatorMap[operatorIdx]);
}

unsigned MergeDequantizeOperators::numTensors() const {
	return original->numTensors();
}

TensorShape MergeDequantizeOperators::getTensorShape(PI::TensorId tensorId) const {
	assert(!tensorIsDequantizeInput[tensorId]); // dequantize input can't be queried
	return original->getTensorShape(tensorId);
}

PI::DataType MergeDequantizeOperators::getTensorType(PI::TensorId tensorId) const {
	assert(!tensorIsDequantizeInput[tensorId]); // dequantize input can't be queried
	return original->getTensorType(tensorId);
}

std::string MergeDequantizeOperators::getTensorName(PI::TensorId tensorId) const {
	assert(!tensorIsDequantizeInput[tensorId]); // dequantize input can't be queried
	return original->getTensorName(tensorId);
}

bool MergeDequantizeOperators::getTensorHasData(PI::TensorId tensorId) const {
	assert(!tensorIsDequantizeInput[tensorId]); // dequantize input can't be queried
	if (tensorIsDequantizeOutput[tensorId]) {
		assert(!original->getTensorHasData(tensorId));
		return true; // Dequantize output is presented as static data
	} else
		return original->getTensorHasData(tensorId);
}

const void* MergeDequantizeOperators::getTensorData(PI::TensorId tensorId) const {
	assert(!tensorIsDequantizeInput[tensorId]); // dequantize input can't be queried
	return original->getTensorData(tensorId);
}

const float* MergeDequantizeOperators::getTensorDataF32(PI::TensorId tensorId) const {
	assert(!tensorIsDequantizeInput[tensorId]); // dequantize input can't be queried
	return original->getTensorDataF32(tensorId);
}

bool MergeDequantizeOperators::getTensorIsVariableFlag(PI::TensorId tensorId) const {
	assert(!tensorIsDequantizeInput[tensorId]); // dequantize input can't be queried
	return original->getTensorIsVariableFlag(tensorId);
}

} // ModelViews
