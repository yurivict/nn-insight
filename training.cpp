// Copyright (C) 2020 by Yuri Victorovich. All rights reserved.

#include "compute.h"
#include "in-memory-model.h"
#include "training.h"
#include "misc.h"
#include "model-functions.h"
#include "rng.h"
#include "util.h"

#include <assert.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace Training {

typedef PluginInterface PI;

PI::Model* constructTrainingModel(const PI::Model *model, PI::OperatorKind lossFunction) { // returns ownership
	// index model
	std::vector<int/*PluginInterface::OperatorId or -1*/> tensorProducers;
	std::vector<std::vector<PluginInterface::OperatorId>> tensorConsumers;
	ModelFunctions::indexOperatorsByTensors(model, tensorProducers, tensorConsumers);

	// local objects
	std::unique_ptr<InMemoryModel> training(new InMemoryModel(model));
	std::map<std::string,unsigned> opNos;

	// frozen layers (they will be sklipped during model construction process)
	std::vector<bool> frozenLayers; // bool[TensorId]
	frozenLayers.resize(model->numTensors());
	for (auto i : model->getInputs())
		frozenLayers[i] = true;

	// all derivative tensors
	std::vector<int/*PI::TensorId or -1*/> derivatives; // tensorId -> derivativeTensorId
	derivatives.resize(model->numTensors());
	std::fill(derivatives.begin(), derivatives.end(), -1);

	//float pendingDerivativeLossCoefficient = 1.;

	// helpers
	auto tname = [](const std::string &name) {
		return STR("training-" << name);
	};
	auto CreateDefaultOperatorOptions = [](PI::OperatorKind kind) -> PI::OperatorOptionsList* { // returns ownership
		switch (kind) {
		case PI::KindAdd:
		case PI::KindSub:
		case PI::KindMul:
		case PI::KindDiv:
			return new PI::OperatorOptionsList({{PI::OperatorOption_FUSED_ACTIVATION_FUNCTION, PI::OperatorOptionValue(PI::ActivationFunction_NONE)}});
		case PI::KindFullyConnected:
			return new PI::OperatorOptionsList({
				{PI::OperatorOption_FUSED_ACTIVATION_FUNCTION, PI::OperatorOptionValue(PI::ActivationFunction_NONE)},
				{PI::OperatorOption_WEIGHTS_FORMAT, PI::OperatorOptionValue(0)}, // XXX wrong, need special type for WeightsFormat
				{PI::OperatorOption_KEEP_NUM_DIMS, PI::OperatorOptionValue(true)}
			});
		default:
			return nullptr;
		}
	};
	auto StaticFloat32 = [&training,&opNos,tname](float value) -> PI::TensorId {
		std::unique_ptr<uint8_t> vals(new uint8_t[sizeof(value)]);
		*(float*)vals.get() = value;
		return training->addTensor(tname(STR("static-no" << ++opNos["static"])), TensorShape(), PI::DataType_Float32, vals.release());
	};
	auto StaticFloat32Tensor = [&training,&opNos,tname](const TensorShape &shape, float *values) -> PI::TensorId { // consumes ownership of values
		return training->addTensor(tname(STR("static-no" << ++opNos["static"])), shape, PI::DataType_Float32, (uint8_t*)values);
	};
	auto Dual = [&training,&opNos,tname,CreateDefaultOperatorOptions](PI::OperatorKind kind, PI::TensorId arg1, PI::TensorId arg2) { // dual symmetric
		auto otid = training->addTensor(tname(STR(kind << "-no" << ++opNos[STR(kind)])), training->getTensorShape(arg1), training->getTensorType(arg1), nullptr);
		training->addOperator(kind, {arg1,arg2}, {otid}, CreateDefaultOperatorOptions(kind));
		return otid;
	};
	auto DualWithTypeShape = [&training,&opNos,tname,CreateDefaultOperatorOptions](PI::OperatorKind kind, PI::TensorId arg1, PI::TensorId arg2, TensorShape shape, PI::DataType type) {
		auto otid = training->addTensor(tname(STR(kind << "-no" << ++opNos[STR(kind)])), shape, type, nullptr);
		training->addOperator(kind, {arg1,arg2}, {otid}, CreateDefaultOperatorOptions(kind));
		return otid;
	};
	auto TripleWithTypeShape = [&training,&opNos,tname,CreateDefaultOperatorOptions](PI::OperatorKind kind,
		PI::TensorId arg1, PI::TensorId arg2, PI::TensorId arg3, TensorShape shape, PI::DataType type)
	{
		auto otid = training->addTensor(tname(STR(kind << "-no" << ++opNos[STR(kind)])), shape, type, nullptr);
		training->addOperator(kind, {arg1,arg2,arg3}, {otid}, CreateDefaultOperatorOptions(kind));
		return otid;
	};
	auto GetOperatorSingleInput = [](const PI::Model *model, PI::OperatorId oid) -> PI::TensorId {
		std::vector<PI::TensorId> inputs;
		std::vector<PI::TensorId> outputs;
		model->getOperatorIo(oid, inputs, outputs);
		assert(inputs.size()==1 && outputs.size()==1);
		return inputs[0];
	};
	/*
	auto GetOperatorTwoInputs = [](const PI::Model *model, PI::OperatorId oid) -> std::array<PI::TensorId,2> {
		std::vector<PI::TensorId> inputs;
		std::vector<PI::TensorId> outputs;
		model->getOperatorIo(oid, inputs, outputs);
		assert(inputs.size()==2 && outputs.size()==1);
		return {inputs[0],inputs[1]};
	};
	*/
	auto GetOperatorThreeInputs = [](const PI::Model *model, PI::OperatorId oid) -> std::array<PI::TensorId,3> {
		std::vector<PI::TensorId> inputs;
		std::vector<PI::TensorId> outputs;
		model->getOperatorIo(oid, inputs, outputs);
		assert(inputs.size()==3 && outputs.size()==1);
		return {inputs[0],inputs[1],inputs[2]};
	};

	// add a loss function and an input for labels to each output
	struct OutputInfo {
		std::string    name;
		TensorShape    shape;
		PI::DataType   type;
		PI::TensorId   lossInput;
		PI::TensorId   lossTarget;
		PI::TensorId   lossOutput;
		PI::TensorId   derivativeOfLossToInput;
	};
	std::vector<OutputInfo> outputInfo;
	assert(!training->getOutputs().empty());
	for (auto modelOutput : training->getOutputs()) {

		// collect all info into the OutputInfo
		OutputInfo oinfo({training->getTensorName(modelOutput), training->getTensorShape(modelOutput), training->getTensorType(modelOutput), modelOutput, 0, 0});

		// add the input for target
		oinfo.lossTarget = training->addTensor(tname(STR("target-of-tensor-" << modelOutput)), oinfo.shape, oinfo.type, nullptr);
		training->addInput(oinfo.lossTarget);

		// add the loss function
		oinfo.lossOutput = training->addTensor(tname(STR("loss-of-tensor-" << modelOutput)), TensorShape{1,1}, training->getTensorType(modelOutput), nullptr);
		training->addOperator(lossFunction, {modelOutput, oinfo.lossTarget}, {oinfo.lossOutput}, nullptr/*loss function has no options*/);

		// update model outputs
		training->addOutput(oinfo.lossOutput);
		//training->removeOutput(modelOutput);

		// save loss outputs/inputs
		outputInfo.push_back(oinfo);
	}

	assert(outputInfo.size()==1); // TODO losses for multiple-outputs need to be added, TODO need an Add operator

	// add the loss derivative
	switch (lossFunction) {
	case PI::KindLossCrossEntropy: { // CAVEAT coefficient -1/N isn't included
		assert(outputInfo[0].type == PI::DataType_Float32);
		assert(outputInfo[0].shape.size()==2 && outputInfo[0].shape[0]==1); // has to be of the form {1,N}
		// ∂CE(O,L)/∂x = -1/N Σᵢ(Lᵢ/Oᵢ - (1-Lᵢ)/(1-Oᵢ))
		auto onesTid = training->addTensor(tname(STR("ones-vector-for-" << outputInfo[0].name)), outputInfo[0].shape, outputInfo[0].type, (uint8_t*)Util::arrayOfOnes<float>(outputInfo[0].shape[1]));

		auto derivativeOfLoss = Dual(
			PI::KindMul,
			Dual(
				PI::KindSub,
				Dual(
					PI::KindDiv,
					outputInfo[0].lossInput/*=O*/,
					outputInfo[0].lossTarget/*=L*/
				),
				Dual(
					PI::KindDiv,
					Dual(
						PI::KindSub,
						onesTid,
						outputInfo[0].lossInput/*=O*/
					),
					Dual(
						PI::KindSub,
						onesTid,
						outputInfo[0].lossTarget/*=L*/
					)
				)
			),
			outputInfo[0].lossOutput
		);
		derivatives[outputInfo[0].lossInput] = derivativeOfLoss;
		outputInfo[0].derivativeOfLossToInput = derivativeOfLoss;
		training->addOutput(derivativeOfLoss);

		break;
	} case PI::KindLossMeanSquareError: { // CAVEAT coefficient 2/N isn't included
		// ∂L2(O,L)/∂x = 2/N Σᵢ(Oᵢ-Lᵢ)
		auto derivativeOfLoss = outputInfo[0].lossInput; /*=O=derivativeOfLoss*/
		//pendingDerivativeLossCoefficient = 2./;

		outputInfo[0].derivativeOfLossToInput = derivativeOfLoss;
		//training->addOutput(outputInfo[0].lossInput/*=O=derivativeOfLoss*/);

		derivatives[outputInfo[0].lossInput] = derivativeOfLoss;

		break;
	} default:
		assert(false); // no other loss functions are supported yet
	}

	// build the TODO set
	std::set<PI::TensorId> tensorsToDo; // output tensors that need to be back-traversed
	for (auto &oi : outputInfo)
		tensorsToDo.insert(oi.lossInput);

	// back-traverse
	while (!tensorsToDo.empty()) {
		// pick one tensor
		auto it = tensorsToDo.begin();
		auto tid  = *it;
		tensorsToDo.erase(it);
		PRINT(">>> iteration tid=" << tid)

		// find the operator that produced it
		auto oid = tensorProducers[tid];
		if (oid == -1)
			continue; // model input, not operator

		// by operator kind
		PRINT("operator kind=" << model->getOperatorKind(oid))
		switch (model->getOperatorKind(oid)) {
		case PI::KindTanh: {
			assert(derivatives[tid] != -1);
			// find input tensor
			PI::TensorId inputTid = GetOperatorSingleInput(model, oid);
			if (!frozenLayers[inputTid]) { // ∂tanh(x)/∂x = 1-tanh(x)^2
				auto derivativeOverTanhInput = Dual(
					PI::KindMul,
					derivatives[tid],
					DualWithTypeShape(
						PI::KindSub,
						StaticFloat32(1),
						Dual(
							PI::KindMul,
							tid,
							tid
						),
						training->getTensorShape(tid),
						training->getTensorType(tid)
					)
				);
				if (derivatives[inputTid] == -1) {
					derivatives[inputTid] = derivativeOverTanhInput;
					tensorsToDo.insert(inputTid);
				} else
					derivatives[inputTid] = Dual(PI::KindAdd, derivatives[inputTid], derivativeOverTanhInput);
			}
			break;
		} case PI::KindFullyConnected: {
			// find input tensors
			auto inputTids = GetOperatorThreeInputs(model, oid);
			// FC(x,W,B) = Wx+B
			if (!frozenLayers[inputTids[0]]) { // ∂FC(x,W,B)/∂x = W', ∂Loss/∂x = W'⋅∂Loss/∂FC(x,W,B)
				auto weightsShape = model->getTensorShape(inputTids[1]);
				assert(weightsShape.size() == 2);
				auto weightsData = model->getTensorData(inputTids[1]);
				assert(weightsData);
				auto derivativeOverFcInput = TripleWithTypeShape(
					PI::KindFullyConnected,
					derivatives[tid],
					StaticFloat32Tensor({weightsShape[1],weightsShape[0]}, Tensor::transposeMatrixIndices1and2of2(weightsShape, (float*)weightsData)),
					StaticFloat32(0),
					{1,weightsShape[1]},
					training->getTensorType(tid)
				);
				PRINT("derivative of input " << inputTids[0] << " is " << derivativeOverFcInput)
				assert(derivatives[inputTids[0]] == -1);
				derivatives[inputTids[0]] = derivativeOverFcInput;
				tensorsToDo.insert(inputTids[0]);
			}
			if (!frozenLayers[inputTids[1]]){ // ∂FC(x,W,B)/∂W = x, ∂Loss/∂W = ∂Loss/∂FC(x,W,B)⊗x
				auto weightsShape = model->getTensorShape(inputTids[1]);
				assert(weightsShape.size() == 2);
				auto weightsData = model->getTensorData(inputTids[1]);
				assert(weightsData);
				auto derivativeOverFcWeights = DualWithTypeShape(
					PI::KindOuterProduct,
					derivatives[tid],
					inputTids[0],
					{1,(unsigned)Tensor::flatSize(training->getTensorShape(tid)),(unsigned)Tensor::flatSize(training->getTensorShape(inputTids[0]))},
					training->getTensorType(tid)
				);
				PRINT("derivative of weights " << inputTids[1] << " is " << derivativeOverFcWeights)
				assert(derivatives[inputTids[1]] == -1);
				derivatives[inputTids[1]] = derivativeOverFcWeights;
				training->addOutput(derivativeOverFcWeights);
				training->setTensorName(derivativeOverFcWeights, tname(STR("derivative-of-tensor-" << inputTids[1])));
			}
			if (!frozenLayers[inputTids[2]]) { // ∂FC(x,W,B)/∂B = 1, ∂Loss/∂B = ∂Loss/∂FC(x,W,B)
				auto derivativeOverFcBias = derivatives[tid];
				PRINT("derivative of bias " << inputTids[2] << " is " << derivativeOverFcBias)
				assert(derivatives[inputTids[2]] == -1);
				derivatives[inputTids[2]] = derivativeOverFcBias;
				training->addOutput(derivativeOverFcBias);
				training->setTensorName(derivativeOverFcBias, tname(STR("derivative-of-tensor-" << inputTids[2])));
			}
			break;
		} default: {
			PRINT("Training: operator not yet supported: " << model->getOperatorKind(oid))
		}}
		PRINT("<<< iteration tid=" << tid)
	}

	//
	PRINT("training model is ready")
	return training.release();
}

bool getModelTrainingIO(const PI::Model *trainingModel, TrainingIO &trainingInputsOutputs) {
	auto tname = [](const char *what, PluginInterface::TensorId tid) {
		return STR("training-" << what << "-of-tensor-" << tid);
	};

	// index tensors by name
	std::map<std::string,PI::TensorId> nameToTensor;
	for (PI::TensorId tid = 0, tide = trainingModel->numTensors(); tid<tide; tid++)
		nameToTensor[trainingModel->getTensorName(tid)] = tid;

	// targetInputs and lossOutputs
	for (auto o : trainingModel->getOutputs())
		if (!isTrainingLayer(trainingModel, o)) {
			// targetInputs
			auto i = nameToTensor.find(tname("target", o));
			if (i != nameToTensor.end())
				trainingInputsOutputs.targetInputs.push_back(i->second);
			else
				return false; // training-target-* input isn't found, maybe not a training model?
			// lossOutputs
			i = nameToTensor.find(tname("loss", o));
			if (i != nameToTensor.end())
				trainingInputsOutputs.lossOutputs.push_back(i->second);
			else
				return false; // training-loss-* output isn't found, maybe not a training model?
		}

	// derivativeOutputs
	auto addTensorDerivative = [&nameToTensor,&trainingInputsOutputs,&tname](PI::TensorId tid) {
		auto i = nameToTensor.find(tname("derivative", tid));
		if (i != nameToTensor.end()) {
			trainingInputsOutputs.derivativeOutputs[i->second] = tid;
			return true;
		} else
			return false; // training-derivative-* output for bias isn't found, maybe not a training model?
	};
	bool succ = true;
	ModelFunctions::iterateThroughParameters(trainingModel, [trainingModel,addTensorDerivative,&succ](PluginInterface::OperatorId oid, unsigned anum, PluginInterface::TensorId tid) {
		if (!isTrainingLayer(trainingModel, tid)) {
			if (!addTensorDerivative(tid))
				succ = false; // training-derivative-* output for a parameter isn't found, maybe not a training model?
		}
	});

	return succ; // found all training inputs/outputs
}

void getModelOriginalIO(const PI::Model *trainingModel, OriginalIO &originalIO) {
	for (auto i : trainingModel->getInputs())
		if (!isTrainingLayer(trainingModel, i))
			originalIO.inputs.push_back(i);
	for (auto o : trainingModel->getOutputs())
		if (!isTrainingLayer(trainingModel, o))
			originalIO.outputs.push_back(o);
}

std::string verifyDerivatives(PluginInterface::Model *trainingModel, unsigned numVerifications, unsigned numPoints, float delta, std::function<std::array<std::vector<float>,2>(bool)> getData) {

	// get TrainingIO
	TrainingIO trainingIO;
	if (!getModelTrainingIO(trainingModel, trainingIO))
		return "ERROR Not a training model!";

	// get OriginalIO
	OriginalIO originalIO;
	getModelOriginalIO(trainingModel, originalIO);

	std::ostringstream ss;
	for (unsigned v = 1; v <= numVerifications; v++) {
		auto sample = getData(false/*validation*/);

		// generate the set of derivative deltas that we will test
		std::map<PluginInterface::TensorId, std::set<std::vector<unsigned>>> tensorTestPoints; // TensorId -> array of points
		ModelFunctions::iterateThroughParameters(trainingModel, [&](PluginInterface::OperatorId oid, unsigned anum, PluginInterface::TensorId tid) {
			auto shape = trainingModel->getTensorShape(tid);
			auto numPts = std::min(numPoints,(unsigned)Tensor::flatSize(shape));
			auto &pts = tensorTestPoints[tid];
			while (pts.size() < numPts)
				pts.insert(Tensor::generateRandomPoint(shape));
		});

		// tensor data
		std::unique_ptr<std::vector<std::shared_ptr<const float>>> tensorData(new std::vector<std::shared_ptr<const float>>);
		tensorData->resize(trainingModel->numTensors());

		auto assignTensor = [&tensorData](PI::TensorId tid, const float *data, unsigned size) {
			auto tensor = new float[size];
			tensorData->data()[tid].reset(tensor);
			std::copy(data, data+size, tensor);
		};

		// assign input
		assert(originalIO.inputs.size()==1); // only support single input models for now
		assignTensor(originalIO.inputs[0], sample[0].data(), sample[0].size());

		// assign target
		assert(trainingIO.targetInputs.size()==1); // only support single output models for now
		assignTensor(trainingIO.targetInputs[0], sample[1].data(), sample[1].size());

		// compute the loss for the center point
		assert(trainingIO.lossOutputs.size()==1); // only support single-output models for now
		Compute::compute(trainingModel, tensorData, [](PI::TensorId) {}, [](const std::string&) {});
		auto loss = (*tensorData)[trainingIO.lossOutputs[0]].get()[0];

		auto testOnePoint = [&](PI::TensorId tid, const std::vector<unsigned> &pt) {
			// get weight value
			assert(trainingModel->getTensorHasData(tid));
			auto &weightValue = ((float*)trainingModel->getTensorDataWr(tid))[Tensor::offset(trainingModel->getTensorShape(tid), pt)];
			// alter the weight
			float prevValue = weightValue;
			weightValue += delta;
			// compute loss
			Compute::compute(trainingModel, tensorData, [](PI::TensorId) {}, [](const std::string&) {});
			auto lossPlus = (*tensorData)[trainingIO.lossOutputs[0]].get()[0];
			// bring the weight value back
			weightValue = prevValue;
			// report
			return STR("loss=" << lossPlus << " Δloss=" << (lossPlus-loss));
		};

		ss << "Verification #" << v << ": args=" << sample[0] << " target=" << sample[1] << " loss=" << loss << std::endl;
		for (auto &oneTensor : tensorTestPoints) {
			ss << "  - Tensor #" << oneTensor.first << std::endl;
			for (auto &pt : oneTensor.second)
				ss << "    - pt=" << pt << ": " << testOnePoint(oneTensor.first, pt) << std::endl;
		}
	}
	// TODO
	return ss.str();
}

bool runTrainingLoop(PluginInterface::Model *model, unsigned batchSize, float trainingRate, bool *stopFlag,
	std::function<std::array<std::vector<float>,2>(bool)> getData,
	std::function<void(unsigned)> batchDone
) {
	// TODO
	return true;
}

bool isTrainingLayer(const PluginInterface::Model *model, PluginInterface::TensorId tid) {
	auto beginsWith = [](const std::string &str,const char *small) {
		return str.size()>std::strlen(small) && std::strncmp(str.c_str(), small, std::strlen(small))==0;
	};
	return beginsWith(model->getTensorName(tid), "training-");
}

}
