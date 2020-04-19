// Copyright (C) 2020 by Yuri Victorovich. All rights reserved.

#include "compute.h"
#include "plugin-interface.h"
#include "nn-types.h"
#include "tensor.h"
#include "nn-operators.h"
#include "image.h"
#include "misc.h"
#include "util.h"

#include <string>
#include <vector>
#include <array>
#include <memory>
#include <functional>
#include <cmath>
#include <cstring>

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
	std::array<unsigned,4> imageRegion,
	std::tuple<InputNormalizationRange,InputNormalizationColorOrder> inputNormalization,
	std::shared_ptr<float> &inputTensor, const TensorShape &inputShape,
	std::unique_ptr<std::vector<std::shared_ptr<const float>>> &tensorData,
	std::function<void(const std::string&)> cbWarningMessage,
	std::function<void(PI::TensorId)> cbTensorComputed)
{
	assert(inputShape.size()==3);

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

	// input tensor is either reused, or reallocated when alterations are needed
	auto &sharedPtrInput = (*tensorData.get())[modelInputs[0]];
	sharedPtrInput = inputTensor; // initially assign with inputShape, but replace later with a newly allocated one if any transformations are performed
	float *inputAllocated = nullptr; // keep track if new allocations
	TensorShape myInputShape = inputShape;

	/// extract the region if required

	if (imageRegion[0]!=0 || imageRegion[1]!=0 || imageRegion[2]+1!=myInputShape[1] || imageRegion[3]+1!=myInputShape[0]) {
		sharedPtrInput.reset((inputAllocated = Image::regionOfImage(sharedPtrInput.get(), myInputShape, imageRegion)));
		myInputShape = {imageRegion[3]-imageRegion[1]+1, imageRegion[2]-imageRegion[0]+1, myInputShape[2]};
	}

	/// resize the source image

	{
		TensorShape requiredShape = model->getTensorShape(modelInputs[0]);

		// adjust the required shape to the form [H,W,C]
		if (requiredShape.size() == 4) { // assume [B,H,W,C]
			if (requiredShape[0] != 1) {
				cbWarningMessage(STR("Model's required shape " << requiredShape << " has 4 elements but doesn't begin with B=1,"
				                     " don't know how to adjust the image for it"));
				return false;
			}
			requiredShape = Tensor::getLastDims(requiredShape, 3);
		} else if (requiredShape.size() == 3) {
			if (requiredShape[0] == 1) { // assume [B=1,H,W], remove B and add C=1 for monochrome image
				requiredShape = Tensor::getLastDims(requiredShape, 2);
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

		// now we have requiredShape=[H,W,C], resize the image if needed
		if (myInputShape != requiredShape)
			sharedPtrInput.reset((inputAllocated = Image::resizeImage(sharedPtrInput.get(), myInputShape, requiredShape)));
	}

	/// normalize input

	if (inputNormalization != InputNormalization{InputNormalizationRange_0_255,InputNormalizationColorOrder_RGB}) { // 0..255/RGB is how images are imported from files
		auto inputTensorShape = model->getTensorShape(modelInputs[0]);
		auto inputTensorSize = Tensor::flatSize(inputTensorShape);

		const float *src = sharedPtrInput.get();
		if (!inputAllocated) // need to allocate because we change the data, otherwise use the allocated above one
			sharedPtrInput.reset((inputAllocated = new float[inputTensorSize]));

		// helpers
		auto normalizeRange = [](const float *src, float *dst, size_t sz, float min, float max) {
			float m = (max-min)/256.; // XXX or 255.?
			for (auto srce = src+sz; src<srce; )
				*dst++ = min + (*src++)*m;
		};
		auto normalizeSub = [](const float *src, float *dst, size_t sz, const std::vector<float> &sub) {
			unsigned i = 0;
			for (auto srce = src+sz; src<srce; ) {
				*dst++ = *src++ - sub[i];
				if (++i == sub.size())
					i = 0;
			}
		};
		auto reorderArrays = [](const float *src, float *dst, size_t sz, const std::vector<unsigned> &permutation) {
			float tmp[permutation.size()];
			for (auto srce = src+sz; src<srce; src+=permutation.size()) {
				float *ptmp = tmp;
				for (auto idx : permutation)
					*ptmp++ = src[idx];
				for (auto t : tmp)
					*dst++ = t;
			}
		};

		// normalize value range
		switch (std::get<0>(inputNormalization)) {
		case InputNormalizationRange_0_1:
			normalizeRange(src, inputAllocated, inputTensorSize, 0, 1);
			src = inputAllocated;
			break;
		case InputNormalizationRange_0_255:
			break; // already at 0..255
		case InputNormalizationRange_0_128:
			normalizeRange(src, inputAllocated, inputTensorSize, 0, 128);
			src = inputAllocated;
			break;
		case InputNormalizationRange_0_64:
			normalizeRange(src, inputAllocated, inputTensorSize, 0, 64);
			src = inputAllocated;
			break;
		case InputNormalizationRange_0_32:
			normalizeRange(src, inputAllocated, inputTensorSize, 0, 32);
			src = inputAllocated;
			break;
		case InputNormalizationRange_0_16:
			normalizeRange(src, inputAllocated, inputTensorSize, 0, 16);
			src = inputAllocated;
			break;
		case InputNormalizationRange_0_8:
			normalizeRange(src, inputAllocated, inputTensorSize, 0, 8);
			src = inputAllocated;
			break;
		case InputNormalizationRange_M1_P1:
			normalizeRange(src, inputAllocated, inputTensorSize, -1, 1);
			src = inputAllocated;
			break;
		case InputNormalizationRange_M05_P05:
			normalizeRange(src, inputAllocated, inputTensorSize, -0.5, 0.5);
			src = inputAllocated;
			break;
		case InputNormalizationRange_14_34:
			normalizeRange(src, inputAllocated, inputTensorSize, 0.25, 0.75);
			src = inputAllocated;
			break;
		case InputNormalizationRange_ImageNet:
			assert(*inputTensorShape.rbegin()==3);
			normalizeSub(src, inputAllocated, inputTensorSize, {123.68, 116.78, 103.94});
			src = inputAllocated;
			break;
		}

		// normalize color order
		switch (std::get<1>(inputNormalization)) {
		case InputNormalizationColorOrder_RGB:
			break; // already RGB
		case InputNormalizationColorOrder_BGR:
			reorderArrays(src, inputAllocated, inputTensorSize, {2,1,0});
			break;
		}
	}

	// notify the caller that the input tensor has been computed

	cbTensorComputed(modelInputs[0]);

	/// compute operators

	for (PI::OperatorId oid = 0, oide = (PI::OperatorId)model->numOperators(); oid<oide; oid++) {
		// get operator's inputs/outputs
		std::vector<PI::TensorId> inputs, outputs;
		model->getOperatorIo(oid, inputs, outputs);

		// get operator options from the model
		std::unique_ptr<PI::OperatorOptionsList> opts(model->getOperatorOptions(oid));

		// helpers
		auto translatePadding = [](unsigned stride, unsigned dilationRate,
		                           WidthHeight wh, const TensorShape &inputShape, const TensorShape &filterShape, const TensorShape &outputShape) {
			//return filterShape[wh==WIDTH ? 2:1]/2;
			unsigned shapeIdx = wh==WIDTH ? 2:1;
			return std::get<0>(computePaddingValues(stride, dilationRate, inputShape[shapeIdx], filterShape[shapeIdx], outputShape[shapeIdx]));
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
		auto doArgMxx = [&](float v0, std::function<bool(float,float)> cmp) {
			assert(inputs.size()==1);
			assert(outputs.size()==1);
			assert(opts); // need to have options present // TODO check the output_type operator option

			auto inputShape = model->getTensorShape(inputs[0]);
			assert(Tensor::flatSize(model->getTensorShape(outputs[0])) == 1);

			// create output data
			std::unique_ptr<float> outputData(new float[1]); // always return one number

			// compute
			auto input = (*tensorData)[inputs[0]].get();
			int idx = -1;
			for (unsigned i = 0, ie = Tensor::flatSize(inputShape); i < ie; i++) {
				auto v = *input++;
				if (cmp(v, v0)) {
					idx = i;
					v0 = v;
				}
			}
			outputData.get()[0] = idx;

			// save the data
			(*tensorData)[outputs[0]].reset(outputData.release());

			// notify the caller
			cbTensorComputed(outputs[0]);
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
			PI::ActivationFunction activationFunction = PluginInterface::ActivationFunction_NONE;

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
			auto inputShape  = model->getTensorShape(inputs[0]);
			auto filterShape = model->getTensorShape(inputs[1]);
			auto outputShape = model->getTensorShape(outputs[0]);
			auto outputShapeSize = Tensor::flatSize(outputShape);

			// create output data
			std::unique_ptr<float> outputData(new float[outputShapeSize]);

			// compute
			NnOperators::Conv2D(
				inputShape, (*tensorData)[inputs[0]].get(), // input
				filterShape, model->getTensorData(inputs[1]), // filter
				model->getTensorShape(inputs[2]), model->getTensorData(inputs[2]), // bias
				outputShape, outputData.get(), // output
				translatePadding(strideWidth,  dilationWidth,  WIDTH,  inputShape, filterShape, outputShape),
				translatePadding(strideHeight, dilationHeight, HEIGHT, inputShape, filterShape, outputShape),
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
			PI::ActivationFunction activationFunction = PluginInterface::ActivationFunction_NONE;

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
			auto inputShape  = model->getTensorShape(inputs[0]);
			auto filterShape = model->getTensorShape(inputs[1]);
			auto outputShape = model->getTensorShape(outputs[0]);
			auto outputShapeSize = Tensor::flatSize(outputShape);

			// create output data
			std::unique_ptr<float> outputData(new float[outputShapeSize]);

			// compute
			NnOperators::DepthwiseConv2D(
				inputShape, (*tensorData)[inputs[0]].get(), // input
				filterShape, model->getTensorData(inputs[1]), // filter
				model->getTensorShape(inputs[2]), model->getTensorData(inputs[2]), // bias
				outputShape, outputData.get(), // output
				translatePadding(strideWidth,  dilationWidth,  WIDTH,  inputShape, filterShape, outputShape),
				translatePadding(strideHeight, dilationHeight, HEIGHT, inputShape, filterShape, outputShape),
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
		} case PI::KindFullyConnected: {
			assert(inputs.size()==3 && outputs.size()==1);
			assert(opts); // need to have options present
			assert((*tensorData)[inputs[0]]); // need to have the input data present

			// operator options required to run this operator
			bool keepNumDims = false;
			int  weightsFormat = 0;
			PI::ActivationFunction activationFunction = PluginInterface::ActivationFunction_NONE;

			// parse the operator options supplied by the model into the above variables
			unsigned numParsed =
				OperatorOptions::GetOption1<PI::OperatorOption_KEEP_NUM_DIMS,    PI::OperatorOption_TypeBool,bool>(*opts, &keepNumDims)
				+ OperatorOptions::GetOption1<PI::OperatorOption_WEIGHTS_FORMAT, PI::OperatorOption_TypeInt, int> (*opts, &weightsFormat)
				+ OperatorOptions::GetOption1<PI::OperatorOption_FUSED_ACTIVATION_FUNCTION,
					PI::OperatorOption_TypeActivationFunction,PI::ActivationFunction>(*opts, &activationFunction);
			assert(numParsed==3); // need to have 3 options
			assert(numParsed==opts->size()); // all options are parsed
			UNUSED(numParsed)

			if (weightsFormat != 0) {
				cbWarningMessage(STR("Computation didn't succeed: operator #" << (oid+1) << ": " << operatorKind << " option weights_format isn't zero"));
				return false; // failed to compute the model to the end
			}

			PRINT_OPTS("FullyConnected: have " << opts->size() << " options:"
			           " keepNumDims=" << keepNumDims <<
			           " weightsFormat=" << weightsFormat <<
			           " activationFunction=" << activationFunction
			)

			// tensors
			auto inputShape  = model->getTensorShape(inputs[0]);
			auto filterShape = model->getTensorShape(inputs[1]);
			auto outputShape = model->getTensorShape(outputs[0]);
			auto outputShapeSize = Tensor::flatSize(outputShape);

			// create output data
			std::unique_ptr<float> outputData(new float[outputShapeSize]);

			// compute
			NnOperators::FullyConnected(
				inputShape, (*tensorData)[inputs[0]].get(), // input
				filterShape, model->getTensorData(inputs[1]), // filter
				model->getTensorShape(inputs[2]), model->getTensorData(inputs[2]), // bias
				outputShape, outputData.get() // output
			);

			// activation function
			applyActivationFunction(outputShapeSize, outputData.get(), activationFunction);

			// save the data
			(*tensorData)[outputs[0]].reset(outputData.release());

			// notify the caller
			cbTensorComputed(outputs[0]);

			break;
		} case PI::KindLocalResponseNormalization: {
			assert(inputs.size()==1 && outputs.size()==1);
			assert(opts); // need to have options present
			assert((*tensorData)[inputs[0]]); // need to have the input data present

			// operator options required to run this operator
			int radius = 0;
			float alpha = 0, beta = 0, bias = 0;

			// parse the operator options supplied by the model into the above variables
			unsigned numParsed =
				OperatorOptions::GetOption1<PI::OperatorOption_RADIUS,    PI::OperatorOption_TypeInt,int>(*opts, &radius)
				+ OperatorOptions::GetOption1<PI::OperatorOption_ALPHA,   PI::OperatorOption_TypeFloat,float> (*opts, &alpha)
				+ OperatorOptions::GetOption1<PI::OperatorOption_BETA,    PI::OperatorOption_TypeFloat,float> (*opts, &beta)
				+ OperatorOptions::GetOption1<PI::OperatorOption_BIAS,    PI::OperatorOption_TypeFloat,float> (*opts, &bias);
			assert(numParsed==4); // need to have 4 options
			assert(numParsed==opts->size()); // all options are parsed
			UNUSED(numParsed)

			PRINT_OPTS("LocalResponseNormalization: have " << opts->size() << " options:"
			           " radius=" << radius <<
			           " alpha=" << alpha <<
			           " beta=" << beta <<
			           " bias=" << bias
			)

			// create output data
			std::unique_ptr<float> outputData(new float[Tensor::flatSize(model->getTensorShape(outputs[0]))]);

			// compute
			NnOperators::LocalResponseNormalization(
				model->getTensorShape(inputs[0]), (*tensorData)[inputs[0]].get(), // input
				model->getTensorShape(outputs[0]), outputData.get(), // output
				radius, alpha, beta, bias
			);

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
			PI::ActivationFunction activationFunction = PluginInterface::ActivationFunction_NONE;

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
			           " filterWidth=" << filterWidth <<
			           " filterHeight=" << filterHeight <<
			           " paddingType=" << paddingType <<
			           " activationFunction=" << activationFunction
			)

			// tensors
			auto inputShape  = model->getTensorShape(inputs[0]);
			TensorShape filterShape = {0,(unsigned)filterHeight,(unsigned)filterWidth,0};
			auto outputShape = model->getTensorShape(outputs[0]);
			auto outputShapeSize = Tensor::flatSize(outputShape);

			// create output data
			std::unique_ptr<float> outputData(new float[outputShapeSize]);

			// compute
			(operatorKind==PI::KindMaxPool ? NnOperators::MaxPool : NnOperators::AveragePool)(
				inputShape, (*tensorData)[inputs[0]].get(), // input
				outputShape, outputData.get(), // output
				translatePadding(strideWidth,  1/*dilationWidth*/,  WIDTH,  inputShape, filterShape, outputShape),
				translatePadding(strideHeight, 1/*dilationHeight*/, HEIGHT, inputShape, filterShape, outputShape),
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
		} case PI::KindTanh: {
			assert(inputs.size()==1 && outputs.size()==1);
			assert(!opts || opts->empty()); // tanh has no options
			assert((*tensorData)[inputs[0]]); // need to have the input data present

			PRINT_OPTS("Tanh: activation function")

			// tensors
			auto inputShape = model->getTensorShape(inputs[0]);
			auto outputShape = model->getTensorShape(outputs[0]);
			auto inputShapeSize = Tensor::flatSize(inputShape);
			assert(inputShape==outputShape);
			UNUSED(outputShape)

			// create output data
			std::unique_ptr<float> outputData(new float[inputShapeSize]);

			// compute
			auto input = (*tensorData)[inputs[0]].get();
			auto output = outputData.get();
			for (auto inpute = input+inputShapeSize; input<inpute; input++, output++)
				*output = std::tanh(*input);

			// save the data
			(*tensorData)[outputs[0]].reset(outputData.release());

			// notify the caller
			cbTensorComputed(outputs[0]);

			break;
		} case PI::KindLogistic: {
			assert(inputs.size()==1 && outputs.size()==1);
			assert(!opts || opts->empty()); // tanh has no options
			assert((*tensorData)[inputs[0]]); // need to have the input data present

			PRINT_OPTS("Logistic: activation function")

			// tensors
			auto inputShape = model->getTensorShape(inputs[0]);
			auto outputShape = model->getTensorShape(outputs[0]);
			auto inputShapeSize = Tensor::flatSize(inputShape);
			assert(inputShape==outputShape);
			UNUSED(outputShape)

			// create output data
			std::unique_ptr<float> outputData(new float[inputShapeSize]);

			// compute
			auto input = (*tensorData)[inputs[0]].get();
			auto output = outputData.get();
			for (auto inpute = input+inputShapeSize; input<inpute; input++, output++)
				*output = 1./(1. + std::exp(*input));

			// save the data
			(*tensorData)[outputs[0]].reset(outputData.release());

			// notify the caller
			cbTensorComputed(outputs[0]);

			break;
		} case PI::KindReshape: {
			assert((inputs.size()==1 || inputs.size()==2) && outputs.size()==1); // XXX now sure why the 'new_shape' is in both input[1] and 'new_shape' option
			assert(opts); // need to have options present, but we ignore them for now ...
			assert((*tensorData)[inputs[0]]); // need to have the input data present
			assert(Tensor::flatSize(model->getTensorShape(outputs[0])) == Tensor::flatSize(model->getTensorShape(inputs[0])));

			PRINT_OPTS("Reshape: have " << opts->size() << " options, but we ignored them for now")

			// just share the data array
			(*tensorData)[outputs[0]] = (*tensorData)[inputs[0]];

			// notify the caller
			cbTensorComputed(outputs[0]);

			break;
		} case PI::KindHardSwish: {
			assert(inputs.size()==1 && outputs.size()==1);
			assert(!opts || opts->empty()); // h-swish has no options
			assert((*tensorData)[inputs[0]]); // need to have the input data present

			PRINT_OPTS("HardSwish: activation function")

			// tensors
			auto inputShape = model->getTensorShape(inputs[0]);
			auto outputShape = model->getTensorShape(outputs[0]);
			auto inputShapeSize = Tensor::flatSize(inputShape);
			assert(inputShape==outputShape);
			UNUSED(outputShape)

			// create output data
			std::unique_ptr<float> outputData(new float[inputShapeSize]);

			// compute
			auto input = (*tensorData)[inputs[0]].get();
			auto output = outputData.get();
			auto hardSwish = [](float x) {
				// defined in the "Searching for MobileNet3" paper (https://arxiv.org/pdf/1905.02244.pdf)
				// h-swish(x) = x*(ReLU6(x+3)/6)
				if (x>=3)
					return x;
				else if (x<=-3)
					return (float)0;
				else
					return x*(x+3)/6;
			};
			for (auto inpute = input+inputShapeSize; input<inpute; input++, output++)
				*output = hardSwish(*input);

			// save the data
			(*tensorData)[outputs[0]].reset(outputData.release());

			// notify the caller
			cbTensorComputed(outputs[0]);

			break;
		} case PI::KindAdd:
		  case PI::KindMul: {
			assert(inputs.size()==2 && outputs.size()==1);
			assert(opts); // need to have options present
			assert((*tensorData)[inputs[0]]); // need to have the input data present
			assert(model->getTensorShape(inputs[0]) == model->getTensorShape(outputs[0])); // produces the same shape as consumes

			// operator options required to run this operator
			PI::ActivationFunction activationFunction = PluginInterface::ActivationFunction_NONE;

			unsigned numParsed =
				OperatorOptions::GetOption1<PI::OperatorOption_FUSED_ACTIVATION_FUNCTION,
					PI::OperatorOption_TypeActivationFunction,PI::ActivationFunction>(*opts, &activationFunction);
			assert(numParsed==1); // need to have 1 options
			assert(numParsed==opts->size()); // all options are parsed
			UNUSED(numParsed)

			PRINT_OPTS(operatorKind << ": have " << opts->size() << " options:"
			           " activationFunction=" << activationFunction)

			// tensors
			auto input1Shape = model->getTensorShape(inputs[0]);
			auto input2Shape = model->getTensorShape(inputs[1]);
			auto outputShape = model->getTensorShape(outputs[0]);
			auto input1ShapeSize = Tensor::flatSize(input1Shape);

			// create output data
			std::unique_ptr<float> outputData(new float[input1ShapeSize]);

			// compute
			if (input1Shape==input2Shape) { // 2 streams added to each other
				(operatorKind==PI::KindAdd ? NnOperators::Add : NnOperators::Mul)(
					model->getTensorShape(inputs[0]), (*tensorData)[inputs[0]].get(), // input1
					model->getTensorShape(inputs[1]), (*tensorData)[inputs[1]].get(), // input2
					outputShape, outputData.get() // output
				);
			} else if (input2Shape.size()==1 && input2Shape[0]==1) { // operation with a constant from the model
				auto input = (*tensorData)[inputs[0]].get();
				auto output = outputData.get();
				const float *inpute = input+input1ShapeSize;
				auto Const = model->getTensorData(inputs[1])[0];
				if (operatorKind==PI::KindAdd)
					for (; input<inpute; input++, output++)
						*output = (*input) + Const;
				else
					for (; input<inpute; input++, output++)
						*output = (*input) * Const;
			} else if (Tensor::isSubset(input1Shape, input2Shape)) { // operation with a smaller computed vector (computed)
				auto input1 = (*tensorData)[inputs[0]].get();
				auto input2 = (*tensorData)[inputs[1]].get();
				auto output = outputData.get();
				auto input1e = input1+input1ShapeSize;
				auto input2b = input2;
				auto input2e = input2+Tensor::flatSize(input2Shape);
				if (operatorKind==PI::KindAdd)
					for (; input1<input1e; input1++, output++) {
						*output = (*input1) + (*input2);
						if (++input2 >= input2e)
							input2 = input2b;
					}
				else
					for (; input1<input1e; input1++, output++) {
						*output = (*input1) * (*input2);
						if (++input2 >= input2e)
							input2 = input2b;
					}
			} else {
				cbWarningMessage(STR("Computation didn't succeed: operator #" << (oid+1) <<
				                     ": " << operatorKind << " isn't yet implemented for shapes " << input1Shape << " and " << input2Shape));
				return false; // failed to compute the model to the end
			}

			// activation function
			applyActivationFunction(input1ShapeSize, outputData.get(), activationFunction);

			// save the data
			(*tensorData)[outputs[0]].reset(outputData.release());

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
			std::unique_ptr<float> outputData(new float[Tensor::flatSize(model->getTensorShape(outputs[0]))]);

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
		} case PI::KindConcatenation: {
			assert(outputs.size()==1);
			assert(opts); // need to have options present

			// operator options required to run this operator
			int axis = 0;
			PI::ActivationFunction activationFunction = PluginInterface::ActivationFunction_NONE;

			// parse the operator options supplied by the model into the above variables
			unsigned numParsed =
				OperatorOptions::GetOption1<PI::OperatorOption_AXIS,            PI::OperatorOption_TypeInt,int>(*opts, &axis)
				+ OperatorOptions::GetOption1<PI::OperatorOption_FUSED_ACTIVATION_FUNCTION,
					PI::OperatorOption_TypeActivationFunction,PI::ActivationFunction>(*opts, &activationFunction);
			assert(numParsed==2); // need to have 2 options
			assert(numParsed==opts->size()); // all options are parsed
			UNUSED(numParsed)

			// input buffers and sizes array
			std::tuple<const float*,unsigned> ins[inputs.size()];
			for (unsigned i = 0, ie = inputs.size(); i<ie; i++) {
				auto inputTensorId = inputs[i];
				auto inputShape = model->getTensorShape(inputTensorId);
				ins[i] = {(*tensorData)[inputTensorId].get(), Tensor::flatSize(Tensor::getLastDims(inputShape, inputShape.size()-axis))};
			}

			// tensors
			auto outputShape = model->getTensorShape(outputs[0]);
			auto outputShapeSize = Tensor::flatSize(outputShape);

			// create output data
			std::unique_ptr<float> outputData(new float[outputShapeSize]);

			// compute
			for (auto out = outputData.get(), oute = out+outputShapeSize; out<oute; )
				for (auto &in : ins) {
					auto &inBuf = std::get<0>(in);
					auto sz = std::get<1>(in);
					std::memcpy(out, inBuf, sz*sizeof(float));
					inBuf += sz;
					out += sz;
				}

			// activation function
			applyActivationFunction(outputShapeSize, outputData.get(), activationFunction);

			// save the data
			(*tensorData)[outputs[0]].reset(outputData.release());

			// notify the caller
			cbTensorComputed(outputs[0]);

			break;
		} case PI::KindArgMax: {
			doArgMxx(std::numeric_limits<float>::lowest(), [](float f1,float f2) {return f1>f2;});
			break;
		} case PI::KindArgMin: {
			doArgMxx(std::numeric_limits<float>::max(), [](float f1,float f2) {return f1<f2;});
			break;
		} case PI::KindResizeBilinear: {
			assert(inputs.size()==1 && outputs.size()==1);
			assert(opts); // need to have options present
			assert((*tensorData)[inputs[0]]); // need to have the input data present

			// operator options required to run this operator
			bool alignCorners = false;

			unsigned numParsed =
				OperatorOptions::GetOption1<PI::OperatorOption_ALIGN_CORNERS, PI::OperatorOption_TypeFloat,bool>(*opts, &alignCorners);
			assert(numParsed==1); // need to have 1 options
			assert(numParsed==opts->size()); // all options are parsed
			UNUSED(numParsed)

			PRINT_OPTS("ResizeBilinear: have " << opts->size() << " options:"
			           " alignCorners=" << alignCorners)

			// create output data
			std::unique_ptr<float> outputData(new float[Tensor::flatSize(model->getTensorShape(outputs[0]))]);

			// compute
			NnOperators::ResizeBilinear(
				model->getTensorShape(inputs[0]), (*tensorData)[inputs[0]].get(), // input
				model->getTensorShape(outputs[0]), outputData.get(), // output
				alignCorners
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
