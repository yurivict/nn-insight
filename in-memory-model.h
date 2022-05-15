// Copyright (C) 2022 by Yuri Victorovich. All rights reserved.

#include "plugin-interface.h"
#include "misc.h"

#include <memory>
#include <string>
#include <vector>

class InMemoryModel : public PluginInterface::Model {

	typedef PluginInterface PI;

	struct TensorInfo {
		std::string                                name;
		TensorShape                                shape;
		PI::DataType                               type;
		std::shared_ptr<uint8_t>                   staticTensorData;
	};
	struct OperatorInfo {
		PI::OperatorKind                           kind;
		std::vector<PI::TensorId>                  inputs;
		std::vector<PI::TensorId>                  outputs;
		std::unique_ptr<PI::OperatorOptionsList>   options;
	};

	std::vector<PI::TensorId>           inputs;
	std::vector<PI::TensorId>           outputs;
	std::vector<TensorInfo>             tensors;
	std::vector<OperatorInfo>           operators;

public:
	InMemoryModel(const PI::Model *other)
	: inputs(other->getInputs())
	, outputs(other->getOutputs())
	{
		// copy tensors
		auto copyTensorData = [](const TensorShape &shape, PI::DataType type, const void *data) -> uint8_t* {
			auto flatSize = Tensor::flatSize(shape);
			switch (type) {
			case PI::DataType_Int32: {
				std::unique_ptr<int32_t> copy;
				copy.reset(new int32_t[flatSize]);
				std::memcpy(copy.get(), data, flatSize*sizeof(int32_t));
				return (uint8_t*)copy.release();
			} case PI::DataType_Float32: {
				std::unique_ptr<float> copy;
				copy.reset(new float[flatSize]);
				std::memcpy(copy.get(), data, flatSize*sizeof(float));
				return (uint8_t*)copy.release();
			} default: {
				FAIL("unsupported tensor type " << type)
			}}
		};
		tensors.reserve(other->numTensors());
		for (PI::TensorId t = 0, te=other->numTensors(); t < te; t++)
			tensors.push_back({
				other->getTensorName(t),
				other->getTensorShape(t),
				other->getTensorType(t),
				other->getTensorHasData(t) ?
					std::shared_ptr<uint8_t>(copyTensorData(other->getTensorShape(t), other->getTensorType(t), other->getTensorData(t)))
					:
					std::shared_ptr<uint8_t>()
			});

		// copy operators
		operators.reserve(other->numOperators());
		for (PI::OperatorId o = 0, oe=other->numOperators(); o < oe; o++) {
			operators.push_back({});
			auto &orec = operators[o];
			orec.kind = other->getOperatorKind(o);
			other->getOperatorIo(o, orec.inputs, orec.outputs);
			orec.options.reset(other->getOperatorOptions(o));
		}
	}

public: // interface implementation
	unsigned numInputs() const override {
		return inputs.size();
	}
	std::vector<PI::TensorId> getInputs() const override {
		return inputs;
	}
	unsigned numOutputs() const override {
		return outputs.size();
	}
	std::vector<PI::TensorId> getOutputs() const override {
		return outputs;
	}
	unsigned numOperators() const override {
		return operators.size();
	}
	void getOperatorIo(PI::OperatorId operatorId, std::vector<PI::TensorId> &inputs, std::vector<PI::TensorId> &outputs) const override {
		auto &o = operators[operatorId];
		inputs = o.inputs;
		outputs = o.outputs;
	}
	PI::OperatorKind getOperatorKind(PI::OperatorId operatorId) const override {
		return operators[operatorId].kind;
	}
	PI::OperatorOptionsList* getOperatorOptions(PI::OperatorId operatorId) const override {
		return operators[operatorId].options ? new PI::OperatorOptionsList(*operators[operatorId].options) : nullptr;
	}
	unsigned numTensors() const override {
		return tensors.size();
	}
	TensorShape getTensorShape(PI::TensorId tensorId) const override {
		return tensors[tensorId].shape;
	}
	PI::DataType getTensorType(PI::TensorId tensorId) const override {
		return tensors[tensorId].type;
	}
	std::string getTensorName(PI::TensorId tensorId) const override {
		return tensors[tensorId].name;
	}
	bool getTensorHasData(PI::TensorId tensorId) const override {
		return (bool)tensors[tensorId].staticTensorData;
	}
	const void* getTensorData(PI::TensorId tensorId) const override {
		return tensors[tensorId].staticTensorData.get();
	}
	void* getTensorDataWr(PI::TensorId tensorId) const override {
		return tensors[tensorId].staticTensorData.get();
	}
	const float* getTensorDataF32(PI::TensorId tensorId) const override {
		assert(tensors[tensorId].type == PI::DataType_Float32);
		return (float*)tensors[tensorId].staticTensorData.get();
	}
	bool getTensorIsVariableFlag(PI::TensorId tensorId) const override {
		return false; // TODO?
	}

public: // iface for changing the model
	void addInput(PI::TensorId tid);
	void removeInput(PI::TensorId tid);
	void addOutput(PI::TensorId tid);
	void removeOutput(PI::TensorId tid);
	PI::TensorId addTensor(const std::string &name, TensorShape shape, PI::DataType type, uint8_t *staticTensorData);
	void setTensorName(PI::TensorId tid, const std::string &name);
	void addOperator(PluginInterface::OperatorKind kind, std::vector<PI::TensorId> inputs, std::vector<PI::TensorId> outputs, PI::OperatorOptionsList *options); // consumes options
};
