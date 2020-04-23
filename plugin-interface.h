// Copyright (C) 2020 by Yuri Victorovich. All rights reserved.

#pragma once

//
// PluginInterface is the interface of all plugins, common for all NN model types
//

#include <string>
#include <vector>
#include <ostream>
#include <functional>

#include "nn-types.h"
#include "tensor.h"

class PluginInterface {

public:
	virtual ~PluginInterface() { }

	// capability flags
	enum Capabilities {Capability_CanWrite=0x00000001};

	// types related to the plugin interface
	typedef unsigned TensorId;
	typedef unsigned OperatorId;
	enum OperatorKind { // all distinct operator kinds should be listed here
		// XXX each value here has to be mirrored in plugin-interface.cpp (CASE)
		KindConv2D,
		KindDepthwiseConv2D,
		KindPad,
		KindFullyConnected,
		KindLocalResponseNormalization,
		KindMaxPool,
		KindAveragePool,
		// activation functions
		KindRelu,
		KindRelu6,
		KindLeakyRelu,
		KindTanh,
		KindLogistic,
		KindHardSwish,
		//
		KindAdd,
		KindSub,
		KindMul,
		KindDiv,
		KindMaximum,
		KindMinimum,
		KindTranspose,
		KindReshape,
		KindSoftmax,
		KindConcatenation,
		KindStridedSlice,
		KindMean,
		// Misc
		KindArgMax,
		KindArgMin,
		KindSquaredDifference,
		//
		KindResizeBilinear,
		//
		KindUnknown
	};

	enum PaddingType {
		PaddingType_SAME,    // pad with zeros where data isn't available, result has the same shape
		PaddingType_VALID    // no padding, iterate only when all data is available for the extent of the kernel, result has a smaller shape
	};

	enum ActivationFunction {
		ActivationFunction_NONE,
		ActivationFunction_RELU,
		ActivationFunction_RELU_N1_TO_1,
		ActivationFunction_RELU6,
		ActivationFunction_TANH,
		ActivationFunction_SIGN_BIT
	};

	// OperatorOptionName represents a unique option value with a specific meaning each available only for some operators
	enum OperatorOptionName {
		OperatorOption_UNKNOWN,
		// XXX for now options are a list of all options that occur as fiels for operator options in schema.fbs
		// corresponds to https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/schema/schema.fbs 58719c1 Dec 17, 2019
		// Regenerate from schema.fbs: 'flatc --jsonschema schema.fb', then prepend with 'var fbsSchema = ' and append the following:
#if 0
if (false) {
        Object.keys(fbsSchema.definitions).forEach(function(name) {
                if (name.length>7 && name.substring(name.length-7)=="Options" && name!="tflite_BuiltinOptions") {
                        print("struct: "+name);
                        Object.keys(fbsSchema.definitions[name].properties).forEach(function(p) {
                                print("... opt: "+p);
                        });
                }
        });
}
#endif
		OperatorOption_ALIGN_CORNERS,
		OperatorOption_ALPHA,
		OperatorOption_AXIS,
		OperatorOption_BATCH_DIM,
		OperatorOption_BEGIN_MASK,
		OperatorOption_BETA,
		OperatorOption_BIAS,
		OperatorOption_BLOCK_SIZE,
		OperatorOption_BODY_SUBGRAPH_INDEX,
		OperatorOption_CELL_CLIP,
		OperatorOption_COMBINER,
		OperatorOption_COND_SUBGRAPH_INDEX,
		OperatorOption_DEPTH_MULTIPLIER,
		OperatorOption_DILATION_H_FACTOR,
		OperatorOption_DILATION_W_FACTOR,
		OperatorOption_ELLIPSIS_MASK,
		OperatorOption_ELSE_SUBGRAPH_INDEX,
		OperatorOption_EMBEDDING_DIM_PER_CHANNEL,
		OperatorOption_END_MASK,
		OperatorOption_FILTER_HEIGHT,
		OperatorOption_FILTER_WIDTH,
		OperatorOption_FUSED_ACTIVATION_FUNCTION,
		OperatorOption_IDX_OUT_TYPE,
		OperatorOption_IN_DATA_TYPE,
		OperatorOption_INCLUDE_ALL_NGRAMS,
		OperatorOption_KEEP_DIMS,
		OperatorOption_KEEP_NUM_DIMS,
		OperatorOption_KERNEL_TYPE,
		OperatorOption_MAX,
		OperatorOption_MAX_SKIP_SIZE,
		OperatorOption_MERGE_OUTPUTS,
		OperatorOption_MIN,
		OperatorOption_MODE,
		OperatorOption_NARROW_RANGE,
		OperatorOption_NEW_AXIS_MASK,
		OperatorOption_NEW_HEIGHT,
		OperatorOption_NEW_SHAPE,
		OperatorOption_NEW_WIDTH,
		OperatorOption_NGRAM_SIZE,
		OperatorOption_NUM,
		OperatorOption_NUM_BITS,
		OperatorOption_NUM_CHANNELS,
		OperatorOption_NUM_COLUMNS_PER_CHANNEL,
		OperatorOption_NUM_SPLITS,
		OperatorOption_OUT_DATA_TYPE,
		OperatorOption_OUT_TYPE,
		OperatorOption_OUTPUT_TYPE,
		OperatorOption_PADDING,
		OperatorOption_PROJ_CLIP,
		OperatorOption_RADIUS,
		OperatorOption_RANK,
		OperatorOption_SEQ_DIM,
		OperatorOption_SHRINK_AXIS_MASK,
		OperatorOption_SQUEEZE_DIMS,
		OperatorOption_STRIDE_H,
		OperatorOption_STRIDE_W,
		OperatorOption_SUBGRAPH,
		OperatorOption_THEN_SUBGRAPH_INDEX,
		OperatorOption_TIME_MAJOR,
		OperatorOption_TYPE,
		OperatorOption_VALIDATE_INDICES,
		OperatorOption_VALUES_COUNT,
		OperatorOption_WEIGHTS_FORMAT
	};

	enum OperatorOptionType {
		OperatorOption_TypeBool,
		OperatorOption_TypeFloat,
		OperatorOption_TypeInt,
		OperatorOption_TypeUInt,
		OperatorOption_TypeIntArray,
		OperatorOption_TypePaddingType,
		OperatorOption_TypeActivationFunction
	};
	struct OperatorOptionValue {
		OperatorOptionType type;
		bool     b;
		float    f;
		int32_t  i;
		uint32_t u;
		PaddingType        paddingType;
		ActivationFunction activationFunction;
		std::vector<int32_t>  ii;
		OperatorOptionValue(bool b_)                                : type(OperatorOption_TypeBool), b(b_) { }
		OperatorOptionValue(float f_)                               : type(OperatorOption_TypeFloat), f(f_) { }
		OperatorOptionValue(int32_t i_)                             : type(OperatorOption_TypeInt), i(i_) { }
		OperatorOptionValue(uint32_t u_)                            : type(OperatorOption_TypeUInt), u(u_) { }
		OperatorOptionValue(const std::vector<int32_t> &ii_)        : type(OperatorOption_TypeIntArray), ii(ii_) { }
		OperatorOptionValue(PaddingType paddingType_) : type(OperatorOption_TypePaddingType), paddingType(paddingType_) { }
		OperatorOptionValue(ActivationFunction activationFunction_) : type(OperatorOption_TypeActivationFunction), activationFunction(activationFunction_) { }

		// templetized getter
		template<typename T> T as() const; // not implemented by default
	};

	// OperatorOption is what is returned by the plugin for individual operators
	struct OperatorOption { // represents a "variable": type name = value;
		OperatorOptionName  name;   // like a variable name, name is assigned a fixed meaning across operators
		OperatorOptionValue value;  // like a variable type and value
	};

	typedef std::vector<OperatorOption> OperatorOptionsList;

	friend std::ostream& operator<<(std::ostream &os, OperatorKind okind);
	friend std::ostream& operator<<(std::ostream &os, PaddingType paddingType);
	friend std::ostream& operator<<(std::ostream &os, ActivationFunction afunc);
	friend std::ostream& operator<<(std::ostream &os, OperatorOptionName optName);
	friend std::ostream& operator<<(std::ostream &os, OperatorOptionType optType);
	friend std::ostream& operator<<(std::ostream &os, const OperatorOptionValue &optValue);

	// inner-classes
	class Model { // Model represents one of potentially many models contained in the file
	protected:
		virtual ~Model() { } // has to be inlined for plugins to contain it too // not to be called by users, hence 'protected'
	public: // interface
		virtual unsigned                numInputs() const = 0;                                                          // how many inputs does this model have
		virtual std::vector<TensorId>   getInputs() const = 0;                                                          // input indexes
		virtual unsigned                numOutputs() const = 0;                                                         // how many outputs does this model have
		virtual std::vector<TensorId>   getOutputs() const = 0;                                                         // output indexes
		virtual unsigned                numOperators() const = 0;                                                       // how many operators does this model have
		virtual void                    getOperatorIo(unsigned operatorIdx, std::vector<TensorId> &inputs, std::vector<TensorId> &outputs) const = 0;
		virtual OperatorKind            getOperatorKind(unsigned operatorIdx) const = 0;
		virtual OperatorOptionsList*    getOperatorOptions(unsigned operatorIdx) const = 0;
		virtual unsigned                numTensors() const = 0;                                                         // number of tensors in this model
		virtual TensorShape             getTensorShape(TensorId tensorId) const = 0;
		virtual std::string             getTensorName(TensorId tensorId) const = 0;
		virtual bool                    getTensorHasData(TensorId tensorId) const = 0;                                  // tensors that are fixed have buffers
		virtual const float*            getTensorData(TensorId tensorId) const = 0;                                     // can only be called when getTensorHasData()=true
		virtual bool                    getTensorIsVariableFlag(TensorId tensorId) const = 0;                           // some tensors are variables that can be altered

	public: // convenience functions
		bool isTensorComputed(TensorId tensorId) const;
	};

	// plugin interface
	virtual uint32_t capabilities() const = 0;
	virtual std::string filePath() const = 0;                                        // returns back the file name that it was opened from
	virtual std::string modelDescription() const = 0;                                // description of the model from the NN file
	virtual bool open(const std::string &filePath_) = 0;                             // open the model (can only be done once per object, close is implicit on destruction for simplicity)
	virtual std::string errorMessage() const = 0;                                    // returns the error of the last operation if it has failed
	virtual size_t numModels() const = 0;                                            // how many models does this file contain
	virtual const Model* getModel(unsigned index) const = 0;                         // access to one model, the Model object is owned by the plugin
	virtual void write(const Model *model, const std::string &fileName) const = 0;  // write the model to disk
};

// gcc-9 needs explicit template specializations to be outside of class scope
template<> inline bool PluginInterface::OperatorOptionValue::as() const {return b;}
template<> inline float PluginInterface::OperatorOptionValue::as() const {return f;}
template<> inline int32_t PluginInterface::OperatorOptionValue::as() const {return i;}
template<> inline PluginInterface::PaddingType PluginInterface::OperatorOptionValue::as() const {return paddingType;}
template<> inline PluginInterface::ActivationFunction PluginInterface::OperatorOptionValue::as() const {return activationFunction;}

