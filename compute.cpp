
#include "compute.h"
#include "plugin-interface.h"
#include "nn-types.h"
#include "nn-operators.h"
#include "image.h"
#include "misc.h"
#include "util.h"

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <cmath>

#include <assert.h>

#if defined(DEBUG)
#define PRINT_OPTS(opts...) PRINT(opts)
#else
#define PRINT_OPTS(opts...)
#endif

namespace Compute {

typedef PluginInterface PI;

//
// local helpers
//

class OperatorOptions {
public:
	template<PI::OperatorOptionName Option, PI::OperatorOptionType OType, typename CType>
	static bool GetOption1(const PI::OperatorOptionsList &opts, CType *val1) {
		for (auto &o : opts)
			if (o.name == Option) {
				assert(o.value.type == OType);
				*val1 = o.value.as<CType>();
				return true; // found
			}
		return false; // not found
	}
};

//
// exported functions
//

bool compute(
	const PI::Model *model,
	std::shared_ptr<float> &inputTensor, const TensorShape &inputShape,
	std::unique_ptr<std::vector<std::shared_ptr<const float>>> &tensorData,
	std::function<void(const std::string&)> cbWarningMessage,
	std::function<void(PI::TensorId)> cbTensorComputed)
{

	/// allocate tensors array

	if (!tensorData) {
		tensorData.reset(new std::vector<std::shared_ptr<const float>>);
		tensorData->resize(model->numTensors());
	}

	/// find the model's input

	auto modelInputs = model->getInputs();
	if (modelInputs.size() != 1) {
		cbWarningMessage(STR("We currently only support models with a single input, the current model has " << modelInputs.size() << " inputs"));
		return false;
	}

	/// resize the source image

	if (!(*tensorData.get())[modelInputs[0]]) { // tensor not ready
		assert(inputShape.size()==3);
		TensorShape requiredShape = model->getTensorShape(modelInputs[0]);

		// adjust the required shape to the form [H,W,C]
		if (requiredShape.size() == 4) { // assume [B,H,W,C]
			if (requiredShape[0] != 1) {
				cbWarningMessage(STR("Model's required shape " << requiredShape << " has 4 elements but doesn't begin with B=1,"
				                     " don't know how to adjust the image for it"));
				return false;
			}
			requiredShape = tensorGetLastDims(requiredShape, 3);
		} else if (requiredShape.size() == 3) {
			if (requiredShape[0] == 1) { // assume [B=1,H,W], remove B and add C=1 for monochrome image
				requiredShape = tensorGetLastDims(requiredShape, 2);
				requiredShape.push_back(1);
			} else { // see if the shape is image-like
				if (requiredShape[2]!=1 && requiredShape[2]!=3) { // expect C=1 or C=3, otherwise we can't handle it
					cbWarningMessage(STR("Model's required shape " << requiredShape << " has 3 elements but has C=1 or C=3,"
					                     " it doesn't look like it describes an image,"
					                     " don't know how to adjust the image for it"));
					return false;
				}
			}
		} else {
			cbWarningMessage(STR("Model's required shape " << requiredShape << " isn't standard, don't know how to adjust the image for it"));
			return false;
		}

		// now we have requiredShape=[H,W,C], resize the image
		auto &sharedPtrInput = (*tensorData.get())[modelInputs[0]];
		if (inputShape != requiredShape)
			sharedPtrInput.reset(Image::resizeImage(inputTensor.get(), inputShape, requiredShape));
		else
			sharedPtrInput = inputTensor;
		// notify the caller
		cbTensorComputed(modelInputs[0]);
	}

	/// compute operators

	for (PI::OperatorId oid = 0, oide = (PI::OperatorId)model->numOperators(); oid<oide; oid++) {
		// get operator's inputs/outputs
		std::vector<PI::TensorId> inputs, outputs;
		model->getOperatorIo(oid, inputs, outputs);

		// get operator options from the model
		std::unique_ptr<PI::OperatorOptionsList> opts(model->getOperatorOptions(oid));

		// helpers
		auto translatePadding = [](const TensorShape &filterShape, PI::PaddingType paddingType, WidthHeight wh) {
			return filterShape[wh==WIDTH ? 2:1]/2;
		};
		auto applyActivationFunction = [](size_t size, float *data, PI::ActivationFunction activationFunction) {
			auto applyRELU = [](float &val) {
				if (val < 0)
					val = 0;
			};
			auto applyRELU_N1_TO_1 = [](float &val) {
				if (val < -1)
					val = -1;
				else if (val > 1)
					val = 1;
			};
			auto applyRELU6 = [](float &val) {
				if (val < 0)
					val = 0;
				else if (val > 6)
					val = 6;
			};
			auto applyTANH = [](float &val) {
				val = std::tanh(val);
			};
			auto applySIGN_BIT = [](float &val) {
				val = std::signbit(val) ? 1 : 0;
			};
			switch (activationFunction) {
			case PI::ActivationFunction_RELU:
				for (auto e = data+size; data<e; data++)
					applyRELU(*data);
				return;
			case PI::ActivationFunction_RELU_N1_TO_1:
				for (auto e = data+size; data<e; data++)
					applyRELU_N1_TO_1(*data);
				return;
			case PI::ActivationFunction_RELU6:
				for (auto e = data+size; data<e; data++)
					applyRELU6(*data);
				return;
			case PI::ActivationFunction_TANH:
				for (auto e = data+size; data<e; data++)
					applyTANH(*data);
				return;
			case PI::ActivationFunction_SIGN_BIT:
				for (auto e = data+size; data<e; data++)
					applySIGN_BIT(*data);
				return;
			case PI::ActivationFunction_NONE:
				return;
			}
		};

		// by operator kind
		auto operatorKind = model->getOperatorKind(oid);
		switch (operatorKind) {
		case PI::KindConv2D: {
			assert(inputs.size()==3 && outputs.size()==1);
			assert(opts); // need to have options present
			assert((*tensorData)[inputs[0]]); // need to have the input data present

			// operator options required to run this operator
			int strideWidth=0, strideHeight=0;
			int dilationWidth=0, dilationHeight=0;
			PI::PaddingType paddingType;
			PI::ActivationFunction activationFunction;

			// parse the operator options supplied by the model into the above variables
			unsigned numParsed =
				OperatorOptions::GetOption1<PI::OperatorOption_STRIDE_W,            PI::OperatorOption_TypeInt,int>(*opts, &strideWidth)
				+ OperatorOptions::GetOption1<PI::OperatorOption_STRIDE_H,          PI::OperatorOption_TypeInt,int>(*opts, &strideHeight)
				+ OperatorOptions::GetOption1<PI::OperatorOption_DILATION_W_FACTOR, PI::OperatorOption_TypeInt,int>(*opts, &dilationWidth)
				+ OperatorOptions::GetOption1<PI::OperatorOption_DILATION_H_FACTOR, PI::OperatorOption_TypeInt,int>(*opts, &dilationHeight)
				+ OperatorOptions::GetOption1<PI::OperatorOption_PADDING, PI::OperatorOption_TypePaddingType,PI::PaddingType>(*opts, &paddingType)
				+ OperatorOptions::GetOption1<PI::OperatorOption_FUSED_ACTIVATION_FUNCTION,
					PI::OperatorOption_TypeActivationFunction,PI::ActivationFunction>(*opts, &activationFunction);
			assert(numParsed==6); // need to have 6 options
			assert(numParsed==opts->size()); // all options are parsed
			UNUSED(numParsed)

			PRINT_OPTS("KindConv2D: have " << opts->size() << " options:"
			           " strideWidth=" << strideWidth <<
			           " strideHeight=" << strideHeight <<
			           " dilationWidth=" << dilationWidth <<
			           " strideHeight=" << strideHeight <<
			           " paddingType=" << paddingType <<
			           " activationFunction=" << activationFunction
			)

			// tensors
			auto filterShape = model->getTensorShape(inputs[1]);
			auto outputShape = model->getTensorShape(outputs[0]);
			auto outputShapeSize = tensorFlatSize(outputShape);

			// create output data
			std::unique_ptr<float> outputData(new float[outputShapeSize]);

			// compute
			NnOperators::Conv2D(
				model->getTensorShape(inputs[0]), (*tensorData)[inputs[0]].get(), // input
				filterShape, model->getTensorData(inputs[1]), // filter
				model->getTensorShape(inputs[2]), model->getTensorData(inputs[2]), // bias
				outputShape, outputData.get(), // output
				translatePadding(filterShape,paddingType,WIDTH), translatePadding(filterShape,paddingType,HEIGHT),
				strideWidth, strideHeight,
				dilationWidth, dilationHeight
			);

			// activation function
			applyActivationFunction(outputShapeSize, outputData.get(), activationFunction);

			// save the data
			(*tensorData)[outputs[0]].reset(outputData.release());

			// notify the caller
			cbTensorComputed(outputs[0]);

			break;
		} case PI::KindDepthwiseConv2D: {
			assert(inputs.size()==3 && outputs.size()==1);
			assert(opts); // need to have options present
			assert((*tensorData)[inputs[0]]); // need to have the input data present

			// operator options required to run this operator
			int depthMultiplier=0;
			int strideWidth=0, strideHeight=0;
			int dilationWidth=0, dilationHeight=0;
			PI::PaddingType paddingType;
			PI::ActivationFunction activationFunction;

			// parse the operator options supplied by the model into the above variables
			unsigned numParsed =
				OperatorOptions::GetOption1<PI::OperatorOption_DEPTH_MULTIPLIER,    PI::OperatorOption_TypeInt,int>(*opts, &depthMultiplier)
				+ OperatorOptions::GetOption1<PI::OperatorOption_STRIDE_W,          PI::OperatorOption_TypeInt,int>(*opts, &strideWidth)
				+ OperatorOptions::GetOption1<PI::OperatorOption_STRIDE_H,          PI::OperatorOption_TypeInt,int>(*opts, &strideHeight)
				+ OperatorOptions::GetOption1<PI::OperatorOption_DILATION_W_FACTOR, PI::OperatorOption_TypeInt,int>(*opts, &dilationWidth)
				+ OperatorOptions::GetOption1<PI::OperatorOption_DILATION_H_FACTOR, PI::OperatorOption_TypeInt,int>(*opts, &dilationHeight)
				+ OperatorOptions::GetOption1<PI::OperatorOption_PADDING, PI::OperatorOption_TypePaddingType,PI::PaddingType>(*opts, &paddingType)
				+ OperatorOptions::GetOption1<PI::OperatorOption_FUSED_ACTIVATION_FUNCTION,
					PI::OperatorOption_TypeActivationFunction,PI::ActivationFunction>(*opts, &activationFunction);
			assert(numParsed==7); // need to have 7 options
			assert(numParsed==opts->size()); // all options are parsed
			UNUSED(numParsed)

			PRINT_OPTS("KindDepthwiseConv2D: have " << opts->size() << " options:"
			           " depthMultiplier=" << depthMultiplier <<
			           " strideWidth=" << strideWidth <<
			           " strideHeight=" << strideHeight <<
			           " dilationWidth=" << dilationWidth <<
			           " strideHeight=" << strideHeight <<
			           " paddingType=" << paddingType <<
			           " activationFunction=" << activationFunction
			)

			// tensors
			auto filterShape = model->getTensorShape(inputs[1]);
			auto outputShape = model->getTensorShape(outputs[0]);
			auto outputShapeSize = tensorFlatSize(outputShape);

			// create output data
			std::unique_ptr<float> outputData(new float[outputShapeSize]);

			// compute
			NnOperators::DepthwiseConv2D(
				model->getTensorShape(inputs[0]), (*tensorData)[inputs[0]].get(), // input
				filterShape, model->getTensorData(inputs[1]), // filter
				model->getTensorShape(inputs[2]), model->getTensorData(inputs[2]), // bias
				outputShape, outputData.get(), // output
				translatePadding(filterShape,paddingType,WIDTH), translatePadding(filterShape,paddingType,HEIGHT),
				strideWidth, strideHeight,
				dilationWidth, dilationHeight,
				depthMultiplier
			);

			// activation function
			applyActivationFunction(outputShapeSize, outputData.get(), activationFunction);

			// save the data
			(*tensorData)[outputs[0]].reset(outputData.release());

			// notify the caller
			cbTensorComputed(outputs[0]);

			break;
		} case PI::KindMaxPool:
		  case PI::KindAveragePool: {
			assert(inputs.size()==1 && outputs.size()==1);
			assert(opts); // need to have options present
			assert((*tensorData)[inputs[0]]); // need to have the input data present

			// operator options required to run this operator
			int strideWidth=0, strideHeight=0;
			int filterWidth=0, filterHeight=0;
			PI::PaddingType paddingType;
			PI::ActivationFunction activationFunction;

			// parse the operator options supplied by the model into the above variables
			unsigned numParsed =
				OperatorOptions::GetOption1<PI::OperatorOption_STRIDE_W,            PI::OperatorOption_TypeInt,int>(*opts, &strideWidth)
				+ OperatorOptions::GetOption1<PI::OperatorOption_STRIDE_H,          PI::OperatorOption_TypeInt,int>(*opts, &strideHeight)
				+ OperatorOptions::GetOption1<PI::OperatorOption_FILTER_WIDTH,      PI::OperatorOption_TypeInt,int>(*opts, &filterWidth)
				+ OperatorOptions::GetOption1<PI::OperatorOption_FILTER_HEIGHT,     PI::OperatorOption_TypeInt,int>(*opts, &filterHeight)
				+ OperatorOptions::GetOption1<PI::OperatorOption_PADDING, PI::OperatorOption_TypePaddingType,PI::PaddingType>(*opts, &paddingType)
				+ OperatorOptions::GetOption1<PI::OperatorOption_FUSED_ACTIVATION_FUNCTION,
					PI::OperatorOption_TypeActivationFunction,PI::ActivationFunction>(*opts, &activationFunction);
			assert(numParsed==6); // need to have 6 options
			assert(numParsed==opts->size()); // all options are parsed
			UNUSED(numParsed)

			PRINT_OPTS(operatorKind << ": have " << opts->size() << " options:"
			           " strideHeight=" << strideHeight <<
			           " strideHeight=" << strideHeight <<
			           " filterHeight=" << filterHeight <<
			           " filterHeight=" << filterHeight <<
			           " paddingType=" << paddingType <<
			           " activationFunction=" << activationFunction
			)

			// tensors
			auto outputShape = model->getTensorShape(outputs[0]);
			auto outputShapeSize = tensorFlatSize(outputShape);

			// create output data
			std::unique_ptr<float> outputData(new float[outputShapeSize]);

			// compute
			(operatorKind==PI::KindMaxPool ? NnOperators::MaxPool : NnOperators::AveragePool)(
				model->getTensorShape(inputs[0]), (*tensorData)[inputs[0]].get(), // input
				outputShape, outputData.get(), // output
				filterWidth/2, filterHeight/2,
				strideWidth, strideHeight,
				filterWidth, filterHeight
			);

			// activation function
			applyActivationFunction(outputShapeSize, outputData.get(), activationFunction);

			// save the data
			(*tensorData)[outputs[0]].reset(outputData.release());

			// notify the caller
			cbTensorComputed(outputs[0]);

			break;
		} case PI::KindReshape: {
			assert((inputs.size()==1 || inputs.size()==2) && outputs.size()==1); // XXX now sure why the 'new_shape' is in both input[1] and 'new_shape' option
			assert(opts); // need to have options present, but we ignore them for now ...
			assert((*tensorData)[inputs[0]]); // need to have the input data present
			assert(tensorFlatSize(model->getTensorShape(outputs[0])) == tensorFlatSize(model->getTensorShape(inputs[0])));

			PRINT_OPTS("Reshape: have " << opts->size() << " options, but we ignored them for now")

			// just share the data array
			(*tensorData)[outputs[0]] = (*tensorData)[inputs[0]];

			// notify the caller
			cbTensorComputed(outputs[0]);

			break;
		} case PI::KindSoftmax: {
			assert(inputs.size()==1 && outputs.size()==1);
			assert(opts); // need to have options present
			assert((*tensorData)[inputs[0]]); // need to have the input data present

			// operator options required to run this operator
			float beta=0;

			unsigned numParsed =
				OperatorOptions::GetOption1<PI::OperatorOption_BETA,    PI::OperatorOption_TypeFloat,float>(*opts, &beta);
			assert(numParsed==1); // need to have 1 options
			assert(numParsed==opts->size()); // all options are parsed
			UNUSED(numParsed)

			PRINT_OPTS("Softmax: have " << opts->size() << " options:"
			           " beta=" <<  beta)

			// create output data
			std::unique_ptr<float> outputData(new float[tensorFlatSize(model->getTensorShape(outputs[0]))]);

			// compute
			NnOperators::Softmax(
				model->getTensorShape(inputs[0]), (*tensorData)[inputs[0]].get(), // input
				model->getTensorShape(outputs[0]), outputData.get(), // output
				beta
			);

			// save the data
			(*tensorData)[outputs[0]].reset(outputData.release());

			// notify the caller
			cbTensorComputed(outputs[0]);

			break;
		} default: {
			cbWarningMessage(STR("Computation didn't succeed: operator #" << (oid+1) << ": " << operatorKind << " isn't yet implemented"));
			return false; // failed to compute the model to the end
		}}
	}

	return true; // successfully computed the model to the end
}

}
