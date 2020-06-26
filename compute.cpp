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

static float computeLossMeanSquareError(const float *src1, const float *src2, unsigned sz) {
	float sum = 0;
	auto sq = [](float x) {return x*x;};
	for (auto e = src1+sz; src1 < e;)
		sum += sq((*src1++) - (*src2++));
	return sum;
}

// helper for operators Concatenate and Split
template<typename OneFloat, typename ManyFloat>
void CopyTensorSlices(
	const PI::Model *model
	, PI::TensorId one
	, const std::vector<PI::TensorId> &many
	, OneFloat *oneTensorData
	, std::shared_ptr<ManyFloat> *manyTensorData
	, int axis
	, std::function<void(OneFloat* &one, ManyFloat* &split, unsigned num)> fnCopy)
{
	// compute inside and outside tensor sizes
	TensorShape oneShape = model->getTensorShape(one);
	unsigned outsideTensorSize = Tensor::sizeBetweenDims(oneShape, 0, axis-1);
	unsigned insideTensorSize  = Tensor::sizeBetweenDims(oneShape, axis+1, oneShape.size()-1);

	// create output data
	unsigned manySize = many.size();
	ManyFloat* manyDataPtr[manySize];
	unsigned outputSliceSize[manySize];
	for (unsigned o = 0; o < manySize; o++) {
		TensorShape manyShape = model->getTensorShape(many[o]);
		outputSliceSize[o] = manyShape[axis]*insideTensorSize;
		manyDataPtr[o] = manyTensorData[o].get();
	}

	OneFloat *oneDataPtr0 = oneTensorData, *oneDataPtr = oneDataPtr0;
	for (unsigned io = 0; io < outsideTensorSize; io++)
		for (unsigned o = 0; o < manySize; o++)
			fnCopy(oneDataPtr, manyDataPtr[o], outputSliceSize[o]);
	assert(oneDataPtr == oneDataPtr0+Tensor::flatSize(oneShape));
}

//
// exported functions
//

bool buildComputeInputs(
	const PI::Model *model,
	std::array<unsigned,4> imageRegion,
	std::tuple<InputNormalizationRange,InputNormalizationColorOrder> inputNormalization,
	std::shared_ptr<float> &inputTensor, const TensorShape &inputShape,
	std::map<PI::TensorId, std::shared_ptr<const float>> &inputs, // output the set of inputs
	std::function<void(PI::TensorId)> cbTensorComputed,
	std::function<void(const std::string&)> cbWarningMessage)
{
	assert(inputShape.size()==3);

	/// find the model's input

	auto modelInputs = model->getInputs();

	// input tensor is either reused, or reallocated when alterations are needed
	auto convertInputImage = [&](PI::TensorId tensorId, TensorShape requiredShape, std::shared_ptr<const float> &inputImage) {
		inputImage = inputTensor; // initially assign with inputShape, but replace later with a newly allocated one if any transformations are performed
		float *inputAllocated = nullptr; // keep track of new allocations
		TensorShape myInputShape = inputShape;

		/// extract the region if required

		if (imageRegion[0]!=0 || imageRegion[1]!=0 || imageRegion[2]+1!=myInputShape[1] || imageRegion[3]+1!=myInputShape[0]) {
			inputImage.reset((inputAllocated = Image::regionOfImage(inputImage.get(), myInputShape, imageRegion)));
			myInputShape = {imageRegion[3]-imageRegion[1]+1, imageRegion[2]-imageRegion[0]+1, myInputShape[2]};
		}

		/// resize the source image

		{
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
				inputImage.reset((inputAllocated = Image::resizeImage(inputImage.get(), myInputShape, requiredShape)));
		}

		/// normalize input

		if (inputNormalization != InputNormalization{InputNormalizationRange_0_255,InputNormalizationColorOrder_RGB}) { // 0..255/RGB is how images are imported from files
			auto inputTensorSize = Tensor::flatSize(requiredShape);

			const float *src = inputImage.get();
			if (!inputAllocated) // need to allocate because we change the data, otherwise use the allocated above one
				inputImage.reset((inputAllocated = new float[inputTensorSize]));

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
				assert(*requiredShape.rbegin()==3);
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

		return true;
	};
	auto convertInputFromJsonFile = [](PI::TensorId tensorId, const TensorShape &requiredShape, std::shared_ptr<const float> &inputTensor) {
		std::shared_ptr<const float> foundTensor;
		if (Tensor::readTensorDataAsJson(CSTR("tensor#" << tensorId << ".json"), requiredShape, foundTensor)) { // match the name with one in main-window.cpp
			inputTensor = foundTensor;
			return true;
		}

		return false; // failed to read the tensor with the requested shape
	};

	/// convert inputs

	bool imageImported = false;
	for (auto tensorId : modelInputs) {
		const auto &shape = model->getTensorShape(tensorId);

		// first, try the file
		if (convertInputFromJsonFile(tensorId, shape, inputs[tensorId])) {
			cbTensorComputed(tensorId); // notify the caller that the input tensor has been computed
			continue; // imported
		}
		// second, try the supplied image
		if (!imageImported && convertInputImage(tensorId, shape, inputs[tensorId])) {
			cbTensorComputed(tensorId); // notify the caller that the input tensor has been computed
			imageImported = true;
			continue; // imported
		}

		// failed to find data for the input tensor
		cbWarningMessage(STR("couldn't find input data for the tensor#" << tensorId << " with shape=" << shape));
		return false;
	}


	return true;
}

void fillInputs(
	std::map<PI::TensorId, std::shared_ptr<const float>> &inputs,
	std::unique_ptr<std::vector<std::shared_ptr<const float>>> &tensorData)
{
	for (auto it : inputs)
		(*tensorData)[it.first] = it.second;
		
}

bool compute(
	const PI::Model *model,
	std::unique_ptr<std::vector<std::shared_ptr<const float>>> &tensorData,
	std::function<void(PI::TensorId)> cbTensorComputed,
	std::function<void(const std::string&)> cbWarningMessage)
{
	/// compute operators

	for (PI::OperatorId oid = 0, oide = (PI::OperatorId)model->numOperators(); oid<oide; oid++) {
		// get operator's inputs/outputs
		std::vector<PI::TensorId> inputs, outputs;
		model->getOperatorIo(oid, inputs, outputs);

		// get operator options from the model
		std::unique_ptr<PI::OperatorOptionsList> opts(model->getOperatorOptions(oid));

		// helpers
		auto getTensorDataDynamicOrStatic = [model,&tensorData](PI::TensorId tensorId) -> const float* {
			auto &dynamic = (*tensorData)[tensorId];
			assert(dynamic || model->getTensorHasData(tensorId)); // at least one of dynamic and static should be available
			assert(!(dynamic && model->getTensorHasData(tensorId))); // both dynamic and static can't be available
			return dynamic ? dynamic.get() : model->getTensorDataF32(tensorId);
		};
		auto translatePadding = [](unsigned stride, unsigned dilationRate,
		                           WidthHeight wh, const TensorShape &inputShape, const TensorShape &filterShape, const TensorShape &outputShape) {
			//return filterShape[wh==WIDTH ? 2:1]/2;
			unsigned shapeIdx = wh==WIDTH ? 2:1;
			return std::get<0>(computePaddingValues(stride, dilationRate, inputShape[shapeIdx], filterShape[shapeIdx], outputShape[shapeIdx]));
		};
		auto computeSingleOperator = [&](float(*fn)(float f)) {
			assert(inputs.size()==1 && outputs.size()==1);
			assert(!opts || opts->empty()); // h-swish has no options
			assert((*tensorData)[inputs[0]]); // need to have the input data present

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
				*output = fn(*input);

			// save the data
			(*tensorData)[outputs[0]].reset(outputData.release());

			// notify the caller
			cbTensorComputed(outputs[0]);
		};
		auto computeDualOperator = [](
			const float *input1, const TensorShape &input1Shape,
			const float *input2, const TensorShape &input2Shape,
			float *output, const TensorShape &outputShape,
			float(*fn)(float i1, float i2)
		) {
			// by type of inputs
			if (input1Shape==input2Shape) { // Large vs. Large
				const float *input1e = input1+Tensor::flatSize(input1Shape);
				// input2 can only be dynamic here
				for (; input1<input1e; )
					*output++ = fn(*input1++, *input2++);
				return true;
			} else if (input1Shape.size()==0 || (input1Shape.size()==1 && input1Shape[0]==1)) { //  Const vs. Large
				auto input2ShapeSize = Tensor::flatSize(input2Shape);
				const float *input2e = input2+input2ShapeSize;
				auto Const = input1[0];
				for (; input2<input2e; )
					*output++ = fn(Const,*input2++);
				return true;
			} else if (input2Shape.size()==01 || (input2Shape.size()==1 && input2Shape[0]==1)) { // Large vs. Const
				const float *input1e = input1+Tensor::flatSize(input1Shape);
				auto Const = input2[0];
				for (; input1<input1e; )
					*output++ = fn(*input1++, Const);
				return true;
			} else if (Tensor::isSubset(input1Shape, input2Shape)) { // Large vs. Small
				auto input1e = input1+Tensor::flatSize(input1Shape);
				auto input2b = input2;
				auto input2e = input2+Tensor::flatSize(input2Shape);
				for (; input1<input1e; input1++, output++) {
					*output = fn(*input1, *input2);
					if (++input2 >= input2e)
						input2 = input2b;
				}
				return true;
			} else {
				return false;
			}
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
			PI::ActivationFunction activationFunction = PI::ActivationFunction_NONE;

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
				filterShape, model->getTensorDataF32(inputs[1]), // filter - assume that it is always a static tensor
				model->getTensorShape(inputs[2]), model->getTensorDataF32(inputs[2]), // bias - assume that it is always a static tensor
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
			PI::ActivationFunction activationFunction = PI::ActivationFunction_NONE;

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
				filterShape, model->getTensorDataF32(inputs[1]), // filter
				model->getTensorShape(inputs[2]), model->getTensorDataF32(inputs[2]), // bias
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
		} case PI::KindPad: {
			// tensors
			auto inputDataShape = model->getTensorShape(inputs[0]);
			auto inputPaddingsShape = model->getTensorShape(inputs[1]);
			auto outputShape = model->getTensorShape(outputs[0]);

			// check that shapes are consistent
			assert(inputDataShape.size() <= 4); // TfLite has max=4 hardcoded in PadParams
			assert(inputPaddingsShape.size()==2 && inputPaddingsShape[0]==inputDataShape.size() && inputPaddingsShape[1]==2);

			// inputs
			assert(model->getTensorType(inputs[1]) == PI::DataType_Int32);
			auto paddings = static_cast<const std::array<int32_t,2>*>(model->getTensorData(inputs[1]));

			// create output data
			std::unique_ptr<float> outputData(new float[Tensor::flatSize(outputShape)]);

			// compute
			NnOperators::Pad(
				paddings,
				inputDataShape, (*tensorData)[inputs[0]].get(), // input
				outputShape, outputData.get() // output
			);

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
			PI::ActivationFunction activationFunction = PI::ActivationFunction_NONE;

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
			auto biasShape   = model->getTensorShape(inputs[2]);
			auto outputShape = model->getTensorShape(outputs[0]);
			auto outputShapeSize = Tensor::flatSize(outputShape);

			// create output data
			std::unique_ptr<float> outputData(new float[outputShapeSize]);

			// compute
			NnOperators::FullyConnected(
				inputShape, (*tensorData)[inputs[0]].get(), // input
				filterShape, model->getTensorDataF32(inputs[1]), // filter
				biasShape, biasShape.size()==1 ? model->getTensorDataF32(inputs[2]) : nullptr, // bias
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
			PI::ActivationFunction activationFunction = PI::ActivationFunction_NONE;

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
		} case PI::KindRSqrt: {
			computeSingleOperator([](float f) -> float {return 1./std::sqrt(f);});
			break;
		} case PI::KindAdd:
		  case PI::KindSub:
		  case PI::KindMul: {
			assert(inputs.size()==2 && outputs.size()==1);
			assert(opts); // need to have options present
			assert(model->getTensorShape(inputs[0])==model->getTensorShape(outputs[0]) || model->getTensorShape(inputs[1])==model->getTensorShape(outputs[0])); // produces the same shape as consumes TODO should be in the model validation stage

			// operator options required to run this operator
			PI::ActivationFunction activationFunction = PI::ActivationFunction_NONE;

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
			auto input2ShapeSize = Tensor::flatSize(input2Shape);
			auto maxInputShapeSize = std::max(input1ShapeSize,input2ShapeSize);

			// create output data
			std::unique_ptr<float> outputData(new float[maxInputShapeSize]);

			// compute
			bool succ = operatorKind==PI::KindAdd ?
				computeDualOperator( // KindAdd
					getTensorDataDynamicOrStatic(inputs[0]), model->getTensorShape(inputs[0]),
					getTensorDataDynamicOrStatic(inputs[1]), model->getTensorShape(inputs[1]),
					outputData.get(), outputShape,
					[](float f1, float f2) {return f1+f2;})
				: operatorKind==PI::KindSub ?
				computeDualOperator( // KindSub
					getTensorDataDynamicOrStatic(inputs[0]), model->getTensorShape(inputs[0]),
					getTensorDataDynamicOrStatic(inputs[1]), model->getTensorShape(inputs[1]),
					outputData.get(), outputShape,
					[](float f1, float f2) {return f1-f2;})
				:
				computeDualOperator( // KindMul
					getTensorDataDynamicOrStatic(inputs[0]), model->getTensorShape(inputs[0]),
					getTensorDataDynamicOrStatic(inputs[1]), model->getTensorShape(inputs[1]),
					outputData.get(), outputShape,
					[](float f1, float f2) {return f1*f2;});
			if (!succ) {
				cbWarningMessage(STR("Computation didn't succeed: operator #" << (oid+1) <<
				                     ": " << operatorKind << " isn't yet implemented for shapes " << input1Shape << " and " << input2Shape));
				return false; // failed to compute the model to the end
			}

			// activation function
			applyActivationFunction(maxInputShapeSize, outputData.get(), activationFunction);

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
			PI::ActivationFunction activationFunction = PI::ActivationFunction_NONE;

			// parse the operator options supplied by the model into the above variables
			unsigned numParsed =
				OperatorOptions::GetOption1<PI::OperatorOption_AXIS, PI::OperatorOption_TypeInt,int>(*opts, &axis)
				+ OperatorOptions::GetOption1<PI::OperatorOption_FUSED_ACTIVATION_FUNCTION,
					PI::OperatorOption_TypeActivationFunction,PI::ActivationFunction>(*opts, &activationFunction);
			assert(numParsed==2); // need to have 2 options
			assert(numParsed==opts->size()); // all options are parsed
			UNUSED(numParsed)

			// input tensors
			std::shared_ptr<const float> inputTensorData[inputs.size()];
			for (unsigned o = 0, oe = sizeof(inputTensorData)/sizeof(inputTensorData[0]); o < oe; o++)
				inputTensorData[o] = (*tensorData)[inputs[o]];

			// input buffers and sizes array
			std::tuple<const float*,unsigned> ins[inputs.size()];
			for (unsigned i = 0, ie = inputs.size(); i<ie; i++) {
				auto inputTensorId = inputs[i];
				auto inputShape = model->getTensorShape(inputTensorId);
				ins[i] = {(*tensorData)[inputTensorId].get(), Tensor::flatSize(Tensor::getLastDims(inputShape, inputShape.size()-axis))};
			}

			// create output data
			auto outputShapeSize = Tensor::flatSize(model->getTensorShape(outputs[0]));
			std::unique_ptr<float> outputData(new float[outputShapeSize]);

			// compute
			CopyTensorSlices<float,const float>(model, outputs[0], inputs, outputData.get(), inputTensorData, axis,
				[](float* &one, const float* &split, unsigned num) {
					std::memcpy(one, split, num*sizeof(float));
					one += num;
					split += num;
				}
			);

			// activation function
			applyActivationFunction(outputShapeSize, outputData.get(), activationFunction);

			// save the data
			(*tensorData)[outputs[0]].reset(outputData.release());

			// notify the caller
			cbTensorComputed(outputs[0]);

			break;
		} case PI::KindSplit: {
			assert(inputs.size()==2);
			assert(opts); // need to have options present

			// operator options required to run this operator
			int num_splits = 0;

			// parse the operator options supplied by the model into the above variables
			unsigned numParsed = OperatorOptions::GetOption1<PI::OperatorOption_NUM_SPLITS, PI::OperatorOption_TypeInt,int>(*opts, &num_splits);
			PRINT("numParsed=" << numParsed)
			assert(numParsed==1); // need to have 1 option
			assert(numParsed==opts->size()); // all options are parsed
			UNUSED(numParsed)

			// checks
			assert(num_splits == outputs.size()); // runtime check should be in the model verifier

			// argument1 has the axis index
			assert(model->getTensorShape(inputs[0]) == TensorShape({1}));
			const int axis = model->getTensorDataF32(inputs[0])[0];

			// create output data
			std::shared_ptr<float> outputTensorData[outputs.size()];
			for (unsigned o = 0, oe = sizeof(outputTensorData)/sizeof(outputTensorData[0]); o < oe; o++)
				outputTensorData[o].reset(new float[Tensor::flatSize(model->getTensorShape(outputs[o]))]);

			// compute
			CopyTensorSlices<const float,float>(model, inputs[1], outputs, (*tensorData)[inputs[1]].get(), outputTensorData, axis,
				[](const float* &one, float* &split, unsigned num) {
					std::memcpy(split, one, num*sizeof(float));
					one += num;
					split += num;
				}
			);

			// save the data and notify the caller
			for (unsigned o = 0, oe = outputs.size(); o < oe; o++) {
				(*tensorData)[outputs[o]] = outputTensorData[o];
				cbTensorComputed(outputs[o]);
			}

			break;
		} case PI::KindMean: {
			assert(inputs.size()==2);
			assert(outputs.size()==1);
			assert(model->getTensorType(inputs[1]) == PI::DataType_Int32);
			assert(opts); // need to have options present

			// tensors
			auto outputShape = model->getTensorShape(outputs[0]);
			auto outputShapeSize = Tensor::flatSize(outputShape);

			// create output data
			std::unique_ptr<float> outputData(new float[outputShapeSize]);

			// compute
			NnOperators::Mean(
				model->getTensorShape(inputs[0]), (*tensorData)[inputs[0]].get(), // input
				outputShape, outputData.get(), // output
				static_cast<const int32_t*>(model->getTensorData(inputs[1])), Tensor::flatSize(model->getTensorShape(inputs[1]))
			);

			// save the data
			(*tensorData)[outputs[0]].reset(outputData.release());

			// notify the caller
			cbTensorComputed(outputs[0]);

			break;
		} case PI::KindRelu: {
			assert(inputs.size()==1 && outputs.size()==1);
			assert(!opts); // need to have options present
			assert(model->getTensorShape(inputs[0]) == model->getTensorShape(outputs[0])); // produces the same shape as consumes TODO should be in the model validation stage

			auto sz = Tensor::flatSize(model->getTensorShape(inputs[0]));

			// create output data
			std::unique_ptr<float> outputData(new float[sz]);

			// compute
			auto input = (*tensorData)[inputs[0]].get();
			auto output = outputData.get();
			auto computeRelu = [](float x) -> float {
				if (x >= 0)
					return x;
				else
					return 0;
			};
			for (auto inpute = input+sz; input<inpute; input++, output++)
				*output = computeRelu(*input);

			// save the data
			(*tensorData)[outputs[0]].reset(outputData.release());

			// notify the caller
			cbTensorComputed(outputs[0]);

			break;
		} case PI::KindSign: {
			assert(inputs.size()==1 && outputs.size()==1);
			assert(!opts); // need to have options present
			assert(model->getTensorShape(inputs[0]) == model->getTensorShape(outputs[0])); // produces the same shape as consumes TODO should be in the model validation stage

			auto sz = Tensor::flatSize(model->getTensorShape(inputs[0]));

			// create output data
			std::unique_ptr<float> outputData(new float[sz]);

			// compute
			auto input = (*tensorData)[inputs[0]].get();
			auto output = outputData.get();
			auto computeSign = [](float x) -> float {
				if (x > 0)
					return +1;
				if (x < 0)
					return -1;
				return 0;
			};
			for (auto inpute = input+sz; input<inpute; input++, output++)
				*output = computeSign(*input);

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
		} case PI::KindSquaredDifference: {
			assert(inputs.size()==2 && outputs.size()==1);
			assert(opts); // need to have options present
			assert((*tensorData)[inputs[0]]); // need to have the input data present
			assert(model->getTensorShape(inputs[0]) == model->getTensorShape(outputs[0])); // produces the same shape as consumes TODO should be in the model validation stage

			assert(opts->size() == 0); // all options are parsed

			PRINT_OPTS(operatorKind << ": have " << opts->size() << " options")

			// tensors
			auto input1Shape = model->getTensorShape(inputs[0]);
			auto input2Shape = model->getTensorShape(inputs[1]);
			auto outputShape = model->getTensorShape(outputs[0]);
			auto input1ShapeSize = Tensor::flatSize(input1Shape);

			// create output data
			std::unique_ptr<float> outputData(new float[input1ShapeSize]);

			// compute
			if (!computeDualOperator(
					(*tensorData)[inputs[0]].get(), model->getTensorShape(inputs[0]),
					getTensorDataDynamicOrStatic(inputs[1]), model->getTensorShape(inputs[1]),
					outputData.get(), outputShape,
					[](float f1, float f2) {return (f1-f2)*(f1-f2);}))
			{
				cbWarningMessage(STR("Computation didn't succeed: operator #" << (oid+1) <<
				                     ": " << operatorKind << " isn't yet implemented for shapes " << input1Shape << " and " << input2Shape));
				return false; // failed to compute the model to the end
			}

			// save the data
			(*tensorData)[outputs[0]].reset(outputData.release());

			// notify the caller
			cbTensorComputed(outputs[0]);

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
		} case PI::KindResizeNearestNeighbor: {
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
			NnOperators::ResizeNearestNeighbor(
				model->getTensorShape(inputs[0]), (*tensorData)[inputs[0]].get(), // input
				model->getTensorShape(outputs[0]), outputData.get(), // output
				alignCorners
			);

			// save the data
			(*tensorData)[outputs[0]].reset(outputData.release());

			// notify the caller
			cbTensorComputed(outputs[0]);

			break;
		} case PI::KindOuterProduct: {
			assert(inputs.size()==2 && outputs.size()==1);
			assert(!opts); // no options are defined for OuterProduct

			auto input1Shape = model->getTensorShape(inputs[0]);
			auto input2Shape = model->getTensorShape(inputs[1]);
			assert(input1Shape.size()==2 && input1Shape[0]==1 && input2Shape.size()==2 && input2Shape[0]==1);

			// create output data
			std::unique_ptr<float> outputData(new float[input1Shape[1]*input2Shape[1]]);

			// compute
			auto computeOuterProduct = [](const float *left, unsigned Nleft, const float *right, unsigned Nright, float *result) {
				auto righte = right + Nright;
				for (auto lefte = left+Nleft; left < lefte; left++)
					for (auto r = right; r < righte; r++)
						*result++ = *left * *r;
			};
			computeOuterProduct(
				getTensorDataDynamicOrStatic(inputs[0]), input1Shape[1],
				getTensorDataDynamicOrStatic(inputs[1]), input2Shape[1],
				outputData.get()
			);

			// save the data
			(*tensorData)[outputs[0]].reset(outputData.release());

			// notify the caller
			cbTensorComputed(outputs[0]);

			break;
		} case PI::KindLossMeanSquareError: {
			assert(inputs.size()==2 && outputs.size()==1 && Tensor::flatSize(model->getTensorShape(outputs[0]))==1);
			assert(model->getTensorShape(inputs[0]) == model->getTensorShape(inputs[1]));

			// create output data
			std::unique_ptr<float> outputData(new float[1]);

			// compute
			outputData.get()[0] = computeLossMeanSquareError(
				(*tensorData)[inputs[0]].get(),
				(*tensorData)[inputs[1]].get(),
				Tensor::flatSize(model->getTensorShape(inputs[0]))
			);

			// save the data
			(*tensorData)[outputs[0]].reset(outputData.release());

			// notify the caller
			cbTensorComputed(outputs[0]);

			break;
		} case PI::KindLossMeanAbsoluteError: {
			assert(inputs.size()==2 && outputs.size()==1 && Tensor::flatSize(model->getTensorShape(outputs[0]))==1);
			assert(model->getTensorShape(inputs[0]) == model->getTensorShape(inputs[1]));

			auto sz = Tensor::flatSize(model->getTensorShape(inputs[0]));

			// create output data
			std::unique_ptr<float> outputData(new float[1]);

			// compute
			if (sz==1) // a simplified computation in a 1D case
				outputData.get()[0] = std::abs((*tensorData)[inputs[0]].get()[0] - (*tensorData)[inputs[1]].get()[0]);
			else
				outputData.get()[0] = std::sqrt(computeLossMeanSquareError(
					(*tensorData)[inputs[0]].get(),
					(*tensorData)[inputs[1]].get(),
					sz
				));

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
