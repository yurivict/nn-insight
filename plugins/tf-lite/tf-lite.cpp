// Copyright (C) 2020 by Yuri Victorovich. All rights reserved.

#include "../../plugin-interface.h"
#include "../../misc.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <string>

#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <assert.h>
#include <errno.h>

#include "tflite_schema_generated.h" // a complete definition of TF Lite format based on the .fbs file from the TD Lite source tree

namespace Helpers {
	template<class C1, class C2>
	static void convertContainers(const C1 &src, C2 &dst) {
		for (auto c : src)
			dst.push_back(c);
	}
	const std::vector<int32_t> convertFlatbuffersIntListToStl(const flatbuffers::Vector<int> *lst) {
		std::vector<int32_t> v;
		convertContainers(*lst, v);
		return v;
	}

	PluginInterface::PaddingType convertPaddingType(tflite::Padding padding) {
		switch (padding) {
		case tflite::Padding_SAME:                           return PluginInterface::PaddingType_SAME;
		case tflite::Padding_VALID:                          return PluginInterface::PaddingType_VALID;
		}
	}

	PluginInterface::ActivationFunction convertActivationFunction(tflite::ActivationFunctionType atype) {
		switch (atype) {
		case tflite::ActivationFunctionType_NONE:            return PluginInterface::ActivationFunction_NONE;
		case tflite::ActivationFunctionType_RELU:            return PluginInterface::ActivationFunction_RELU;
		case tflite::ActivationFunctionType_RELU_N1_TO_1:    return PluginInterface::ActivationFunction_RELU_N1_TO_1;
		case tflite::ActivationFunctionType_RELU6:           return PluginInterface::ActivationFunction_RELU6;
		case tflite::ActivationFunctionType_TANH:            return PluginInterface::ActivationFunction_TANH;
		case tflite::ActivationFunctionType_SIGN_BIT:        return PluginInterface::ActivationFunction_SIGN_BIT;
		}
	}

	static PluginInterface::OperatorKind opcodeToOperatorKind(tflite::BuiltinOperator opcode) {
		switch (opcode) {
#define CASE(hisName,myName) case tflite::BuiltinOperator_##hisName: return PluginInterface::Kind##myName;
#include "operator-list.cpp" // generated from schema.fbs
		default:
			FAIL("unknown opcode=" << opcode)
#undef CASE
		}
	}
	static PluginInterface::OperatorOptionsList* convertOperatorOptions(const tflite::Operator *o, tflite::BuiltinOperator opcode) {

		std::unique_ptr<PluginInterface::OperatorOptionsList> ourOpts(new PluginInterface::OperatorOptionsList);

		switch (opcode) {
#include "operator-options.cpp" // generated from schema.fbs
		default:
			// no nothing: no options for this operator
			ourOpts.reset(nullptr); // no options - no array
		}

		return ourOpts.release();
	}
}

class TfLitePlugin : public PluginInterface {

	class Model : public PluginInterface::Model {
		const TfLitePlugin     *plugin;
		const tflite::SubGraph *subgraph; // the subgraoph that this model represents

		public:
			Model(const TfLitePlugin *plugin_, const tflite::SubGraph *subgraph_)
			: plugin(plugin_)
			, subgraph(subgraph_)
			{ }

		public: // interface
			unsigned numInputs() const override {
				return subgraph->inputs()->size();
			}
			std::vector<TensorId> getInputs() const override {
				std::vector<TensorId> idxs;
				Helpers::convertContainers(*subgraph->inputs(), idxs);
				return idxs;
			}
			unsigned numOutputs() const override {
				return subgraph->outputs()->size();
			}
			std::vector<TensorId> getOutputs() const override {
				std::vector<TensorId> idxs;
				Helpers::convertContainers(*subgraph->outputs(), idxs);
				return idxs;
			}
			unsigned numOperators() const override {
				return subgraph->operators()->size();
			}
			void getOperatorIo(OperatorId operatorId, std::vector<TensorId> &inputs, std::vector<TensorId> &outputs) const override {
				auto o = subgraph->operators()->Get(operatorId);
				Helpers::convertContainers(*o->inputs(), inputs);
				Helpers::convertContainers(*o->outputs(), outputs);
			}
			OperatorKind getOperatorKind(OperatorId operatorId) const override {
				auto opcode_index = subgraph->operators()->Get(operatorId)->opcode_index();
				assert(opcode_index < plugin->model->operator_codes()->size());
				return Helpers::opcodeToOperatorKind(plugin->model->operator_codes()->Get(opcode_index)->builtin_code());
			}
			PluginInterface::OperatorOptionsList* getOperatorOptions(OperatorId operatorId) const override {
				auto o = subgraph->operators()->Get(operatorId);
				assert(o->opcode_index() < plugin->model->operator_codes()->size());
				return Helpers::convertOperatorOptions(o, plugin->model->operator_codes()->Get(o->opcode_index())->builtin_code());
			}
			unsigned numTensors() const override {
				return subgraph->tensors()->size();
			}
			TensorShape getTensorShape(TensorId tensorId) const override {
				std::vector<unsigned> shape;
				assert(tensorId < subgraph->tensors()->size());
				if (subgraph->tensors()->Get(tensorId)->shape() != nullptr)
					Helpers::convertContainers(*subgraph->tensors()->Get(tensorId)->shape(), shape);
				else
					{ } // leave the shape empty: it must be a scalar in such case
				return shape;
			}
			DataType getTensorType(TensorId tensorId) const override {
				switch (subgraph->tensors()->Get(tensorId)->type()) {
				case tflite::TensorType_FLOAT16: return DataType_Float16;
				case tflite::TensorType_FLOAT32: return DataType_Float32;
				case tflite::TensorType_INT8:    return DataType_Int8;
				case tflite::TensorType_UINT8:   return DataType_UInt8;
				case tflite::TensorType_INT16:   return DataType_Int16;
				case tflite::TensorType_INT32:   return DataType_Int32;
				case tflite::TensorType_INT64:   return DataType_Int64;
				default:
					FAIL("unknown TfLite tensor type code=" << subgraph->tensors()->Get(tensorId)->type())
				}
			}
			std::string getTensorName(TensorId tensorId) const override {
				return subgraph->tensors()->Get(tensorId)->name()->c_str();
			}
			bool getTensorHasData(TensorId tensorId) const override {
				auto buffer = subgraph->tensors()->Get(tensorId)->buffer();
				if (buffer < plugin->model->buffers()->size()) {
					auto data = plugin->model->buffers()->Get(buffer)->data();
					return data != nullptr && data->size() > 0;
				} else {
					return false;
				}
			}
			const void* getTensorData(TensorId tensorId) const override {
				auto buffer = subgraph->tensors()->Get(tensorId)->buffer();
				assert(buffer < plugin->model->buffers()->size());
				auto data = plugin->model->buffers()->Get(buffer)->data();
				assert(data!=nullptr && data->size()!=0);
				return data->Data();
			}
			void* getTensorDataWr(TensorId tensorId) const override {
				return nullptr; // until we implement in-place writability it isn't writable
			}
			const float* getTensorDataF32(TensorId tensorId) const override {
				assert(subgraph->tensors()->Get(tensorId)->type() == tflite::TensorType_FLOAT32);
				return static_cast<const float*>(Model::getTensorData(tensorId));
			}
			bool getTensorIsVariableFlag(TensorId tensorId) const override {
				return subgraph->tensors()->Get(tensorId)->is_variable();
			}
	};

	std::string                           modelFileName;
	int                                   fd;              // handle of the open file
	size_t                                fileSize;        // file size
	void*                                 mmappedPtr;
	const tflite::Model*                  model;
	std::string                           err;             // error message in case the error occurs

public:
	TfLitePlugin()
	: fd(-1)
	, fileSize(0)
	, mmappedPtr(nullptr)
	, model(nullptr)
	{
	}
	~TfLitePlugin() {
		if (mmappedPtr)
			closeFileReleaseMemory();
	}


public: // interface implementation

	uint32_t capabilities() const override {
		return 0; // no capability flags to report
	}

	std::string filePath() const override {
		return modelFileName;
	}

	std::string modelDescription() const override {
		return model->description()->c_str();
	}

	virtual bool open(const std::string &modelFileName_) override {
		// open the file
		fd = ::open(modelFileName_.c_str(), O_RDONLY); // since TF Lite models aren't officially writable we open it in RO mode
		if (fd == -1) {
			PRINT_ERR("failed to open the tflite file '" << modelFileName_ << "': " << strerror(errno))
			return false;
		}

		// find its size
		struct stat sb;
		if (::fstat(fd, &sb) == -1) {
			auto err = STR("failed to find the tflite file '" << modelFileName_ << "' length: " << strerror(errno));
			if (::close(fd) == -1)
				err += STR("; failed to close the tflite file '" << modelFileName_ << "': " << strerror(errno));
			PRINT_ERR(err)
			return false;
		}
		fileSize = sb.st_size;

		// mmap the file for efficient access
		void *m = ::mmap(0/*addr*/, sb.st_size, PROT_READ, MAP_SHARED/*flags*/, fd, 0/*offset*/);
		if (m == MAP_FAILED) {
			auto err = STR("failed to mmap the tflite file '" << modelFileName_ << "': " << strerror(errno));
			if (::close(fd) == -1)
				err += STR("; failed to close the tflite file '" << modelFileName_ << "': " << strerror(errno));
			PRINT_ERR(err)
			return false;
		}
		mmappedPtr = m;

		// view the memory as a model
		model = tflite::GetModel(mmappedPtr);

		// check if we can take this model
		if (model->subgraphs()->size() != 1) {
			PRINT_ERR("we only support TF Lite models with subgraph count of 1, the model '" << modelFileName_ << "' has " << model->subgraphs()->size() << " subgraphs")
			closeFileReleaseMemory();
			return false;
		}

		// return
		modelFileName = modelFileName_;
		return true;
	}

	std::string errorMessage() const override {
		return "some strange error occurred"; // no error
	}
	size_t numModels() const override {
		return 1; // .tflite file always contains only one model
	}
	const Model* getModel(unsigned index) const override {
		// checks
		if (index != 0) {
			std::cerr << "ERROR only index=1 is available for TF Lite models" << std::endl;
			return nullptr;
		}

		return new Model(this, model->subgraphs()->Get(0)); // returns the object ownership
	}
	void write(const PluginInterface::Model *model, const std::string &fileName) const override {
		PRINT("TfLite plugin doesn't support model writing yet")
	}

private:
	void closeFileReleaseMemory() {
		// delete the memory object
		modelFileName.clear();
		model = nullptr;

		// unmap
		if (::munmap(mmappedPtr, fileSize) == -1)
			PRINT_ERR("failed to unmmap the tflite file '" << modelFileName << "': " << strerror(errno))
		mmappedPtr = nullptr;
		fileSize = 0;

		// close the file
		if (::close(fd) == -1)
			PRINT_ERR("failed to close the tflite file '" << modelFileName << "': " << strerror(errno))
		fd = -1;
	}
};


//
// exported function
//

extern "C" {

PluginInterface* createPluginInterface() {
	return new TfLitePlugin;
}

};

