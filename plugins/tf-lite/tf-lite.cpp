

#include "../../plugin-interface.h"
#include "../../misc.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <memory>

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

	static PluginInterface::OperatorKind opcodeToOperatorKind(tflite::BuiltinOperator opcode) {
		switch (opcode) {
#define CASE(hisName,myName) case tflite::BuiltinOperator_##hisName: return PluginInterface::Kind##myName;
		CASE(ADD, Add) CASE(AVERAGE_POOL_2D, AveragePool) CASE(CONCATENATION, Concatenation) CASE(CONV_2D, Conv2D) CASE(DEPTHWISE_CONV_2D, DepthwiseConv2D) CASE(DEPTH_TO_SPACE, Unknown) CASE(DEQUANTIZE, Unknown) CASE(EMBEDDING_LOOKUP, Unknown) CASE(FLOOR, Unknown) CASE(FULLY_CONNECTED, FullyConnected)
		CASE(HASHTABLE_LOOKUP, Unknown) CASE(L2_NORMALIZATION, Unknown) CASE(L2_POOL_2D, Unknown) CASE(LOCAL_RESPONSE_NORMALIZATION, Unknown) CASE(LOGISTIC, Unknown) CASE(LSH_PROJECTION, Unknown) CASE(LSTM, Unknown) CASE(MAX_POOL_2D, MaxPool) CASE(MUL, Mul) CASE(RELU, Relu)
		CASE(RELU_N1_TO_1, Unknown) CASE(RELU6, Relu6) CASE(RESHAPE, Reshape) CASE(RESIZE_BILINEAR, Unknown) CASE(RNN, Unknown) CASE(SOFTMAX, Softmax) CASE(SPACE_TO_DEPTH, Unknown) CASE(SVDF, Unknown) CASE(TANH, Tanh) CASE(CONCAT_EMBEDDINGS, Unknown)
		CASE(SKIP_GRAM, Unknown) CASE(CALL, Unknown) CASE(CUSTOM, Unknown) CASE(EMBEDDING_LOOKUP_SPARSE, Unknown) CASE(PAD, Pad) CASE(UNIDIRECTIONAL_SEQUENCE_RNN, Unknown) CASE(GATHER, Unknown) CASE(BATCH_TO_SPACE_ND, Unknown) CASE(SPACE_TO_BATCH_ND, Unknown) CASE(TRANSPOSE, Transpose)
		CASE(MEAN, Mean) CASE(SUB, Sub) CASE(DIV, Div) CASE(SQUEEZE, Unknown) CASE(UNIDIRECTIONAL_SEQUENCE_LSTM, Unknown) CASE(STRIDED_SLICE, StridedSlice) CASE(BIDIRECTIONAL_SEQUENCE_RNN, Unknown) CASE(EXP, Unknown) CASE(TOPK_V2, Unknown) CASE(SPLIT, Unknown)
		CASE(LOG_SOFTMAX, Unknown) CASE(DELEGATE, Unknown) CASE(BIDIRECTIONAL_SEQUENCE_LSTM, Unknown) CASE(CAST, Unknown) CASE(PRELU, Unknown) CASE(MAXIMUM, Maximum) CASE(ARG_MAX, Unknown) CASE(MINIMUM, Minimum) CASE(LESS, Unknown) CASE(NEG, Unknown)
		CASE(PADV2, Unknown) CASE(GREATER, Unknown) CASE(GREATER_EQUAL, Unknown) CASE(LESS_EQUAL, Unknown) CASE(SELECT, Unknown) CASE(SLICE, Unknown) CASE(SIN, Unknown) CASE(TRANSPOSE_CONV, Unknown) CASE(SPARSE_TO_DENSE, Unknown) CASE(TILE, Unknown)
		CASE(EXPAND_DIMS, Unknown) CASE(EQUAL, Unknown) CASE(NOT_EQUAL, Unknown) CASE(LOG, Unknown) CASE(SUM, Unknown) CASE(SQRT, Unknown) CASE(RSQRT, Unknown) CASE(SHAPE, Unknown) CASE(POW, Unknown) CASE(ARG_MIN, Unknown)
		CASE(FAKE_QUANT, Unknown) CASE(REDUCE_PROD, Unknown) CASE(REDUCE_MAX, Unknown) CASE(PACK, Unknown) CASE(LOGICAL_OR, Unknown) CASE(ONE_HOT, Unknown) CASE(LOGICAL_AND, Unknown) CASE(LOGICAL_NOT, Unknown) CASE(UNPACK, Unknown) CASE(REDUCE_MIN, Unknown)
		CASE(FLOOR_DIV, Unknown) CASE(REDUCE_ANY, Unknown) CASE(SQUARE, Unknown) CASE(ZEROS_LIKE, Unknown) CASE(FILL, Unknown) CASE(FLOOR_MOD, Unknown) CASE(RANGE, Unknown) CASE(RESIZE_NEAREST_NEIGHBOR, Unknown) CASE(LEAKY_RELU, LeakyRelu) CASE(SQUARED_DIFFERENCE, Unknown)
		CASE(MIRROR_PAD, Unknown) CASE(ABS, Unknown) CASE(SPLIT_V, Unknown) CASE(UNIQUE, Unknown) CASE(CEIL, Unknown) CASE(REVERSE_V2, Unknown) CASE(ADD_N, Unknown) CASE(GATHER_ND, Unknown) CASE(COS, Unknown) CASE(WHERE, Unknown)
		CASE(RANK, Unknown) CASE(ELU, Unknown) CASE(REVERSE_SEQUENCE, Unknown) CASE(MATRIX_DIAG, Unknown) CASE(QUANTIZE, Unknown) CASE(MATRIX_SET_DIAG, Unknown) CASE(ROUND, Unknown) CASE(HARD_SWISH, Unknown) CASE(IF, Unknown) CASE(WHILE, Unknown)
		CASE(NON_MAX_SUPPRESSION_V4, Unknown) CASE(NON_MAX_SUPPRESSION_V5, Unknown)
		default:
			FAIL("unknown opcode=" << opcode)
#undef CASE
		}
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
			unsigned numTensors() const override {
				return subgraph->tensors()->size();
			}
			TensorShape getTensorShape(TensorId tensorId) const override {
				std::vector<unsigned> shape;
				Helpers::convertContainers(*subgraph->tensors()->Get(tensorId)->shape(), shape);
				return shape;
			}
			std::string getTensorName(TensorId tensorId) const override {
				return subgraph->tensors()->Get(tensorId)->name()->c_str();
			}
			bool getTensorIsVariableFlag(TensorId tensorId) const override {
				return subgraph->tensors()->Get(tensorId)->is_variable();
			}
			bool getTensorHasData(TensorId tensorId) const override {
				auto buffer = subgraph->tensors()->Get(tensorId)->buffer();
				assert(buffer < plugin->model->buffers()->size());
				return plugin->model->buffers()->Get(buffer)->data() != nullptr;
			}
	};

	std::string                           modelFileName;
	int                                   fd;              // handle of the open file
	size_t                                fileSize;        // file size
	void*                                 mmappedPtr;
	const tflite::Model*                  model;
	std::string                           err;             // error message in case the error occurs
	std::unique_ptr<Model>                modelObj;        // the model that we own

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

	std::string filePath() const override {
		return modelFileName;
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
		modelObj.reset(new Model(this, model->subgraphs()->Get(0)));
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
		if (modelObj.get() == nullptr) {
			std::cerr << "ERROR 'open' hasn't been called" << std::endl;
			return nullptr;
		}

		return modelObj.get();
	}

private:
	void closeFileReleaseMemory() {
		// delete the memory object
		modelFileName.clear();
		modelObj.reset(nullptr);
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

