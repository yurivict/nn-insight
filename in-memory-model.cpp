// Copyright (C) 2022 by Yuri Victorovich. All rights reserved.

#include "in-memory-model.h"

#include <algorithm>

typedef PluginInterface PI;

void InMemoryModel::addInput(PI::TensorId tid) {
	inputs.push_back(tid);
}

void InMemoryModel::removeInput(PI::TensorId tid) {
	auto elt = std::find(inputs.begin(), inputs.end(), tid);
	assert(elt != inputs.end());
	inputs.erase(elt);
}

void InMemoryModel::addOutput(PI::TensorId tid) {
	outputs.push_back(tid);
}

void InMemoryModel::removeOutput(PI::TensorId tid) {
	auto elt = std::find(outputs.begin(), outputs.end(), tid);
	assert(elt != outputs.end());
	outputs.erase(elt);
}

PI::TensorId InMemoryModel::addTensor(const std::string &name, TensorShape shape, PI::DataType type, uint8_t *staticTensorData) { // staticTensorData ownership is passed
	PI::TensorId tid = tensors.size();

	tensors.push_back(TensorInfo{name, shape, type, std::shared_ptr<uint8_t>(staticTensorData)});

	return tid;
}

void InMemoryModel::setTensorName(PI::TensorId tid, const std::string &name_) {
	tensors[tid].name = name_;
}

void InMemoryModel::addOperator(PluginInterface::OperatorKind kind, std::vector<PI::TensorId> inputs, std::vector<PI::TensorId> outputs, PI::OperatorOptionsList *options) { // consumes options
	operators.push_back(OperatorInfo{kind, inputs, outputs, std::unique_ptr<PI::OperatorOptionsList>(options)});
}
