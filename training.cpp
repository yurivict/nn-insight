// Copyright (C) 2020 by Yuri Victorovich. All rights reserved.

#include "in-memory-model.h"
#include "training.h"
#include "misc.h"
#include "util.h"

#include <assert.h>

#include <memory>
#include <string>

namespace Training {

typedef PluginInterface PI;

PI::Model* convertToTrainingModel(const PI::Model *model, PI::OperatorKind lossFunction) { // returns ownership
	std::unique_ptr<InMemoryModel> training(new InMemoryModel(model));
	std::map<std::string,unsigned> opNos;

	// helpers
	auto tname = [](const std::string &name) {
		return STR("training-" << name);
	};
	auto Dual = [&training,&opNos,tname](PI::OperatorKind kind, PI::TensorId arg1, PI::TensorId arg2) -> PI::TensorId {
		auto otid = training->addTensor(tname(STR(kind << "-no" << ++opNos[STR(kind)])), training->getTensorShape(arg1), training->getTensorType(arg1), nullptr);
		training->addOperator(kind, {arg1,arg2}, {otid}, nullptr/*options*/);
		return otid;
	};

	// add a loss function and an input for labels to each output
	struct OutputInfo {
		std::string    name;
		TensorShape    shape;
		PI::DataType   type;
		PI::TensorId   lossInput;
		PI::TensorId   lossLabel;
		PI::TensorId   lossOutput;
	};
	std::vector<OutputInfo> outputInfo;
	assert(!training->getOutputs().empty());
	for (auto modelOutput : training->getOutputs()) {

		// collect all info into the OutputInfo
		OutputInfo oinfo({training->getTensorName(modelOutput), training->getTensorShape(modelOutput), training->getTensorType(modelOutput), modelOutput, 0, 0});

		// add the input for label
		oinfo.lossLabel = training->addTensor(tname(STR("label-for-" << oinfo.name)), oinfo.shape, oinfo.type, nullptr);
		training->addInput(oinfo.lossLabel);

		// add the loss function
		oinfo.lossOutput = training->addTensor(tname(STR("loss-for-" << training->getTensorName(modelOutput))), TensorShape{1,1}, training->getTensorType(modelOutput), nullptr);
		training->addOperator(lossFunction, {modelOutput, oinfo.lossLabel}, {oinfo.lossOutput}, nullptr/*options*/);

		// update model outputs
		training->removeOutput(modelOutput);

		// save loss outputs/inputs
		outputInfo.push_back(oinfo);
	}

	assert(outputInfo.size()==1); // TODO losses for multiple-outputs need to be added, TODO need an Add operator

	// add the loss derivative
	switch (lossFunction) {
	case PI::KindLossCrossEntropy: {
		assert(outputInfo[0].type == PI::DataType_Float32);
		assert(outputInfo[0].shape.size()==2 && outputInfo[0].shape[0]==1); // has to be of the form {1,N}
		// ∂CE(O,L)/∂x = = -1/N Σᵢ(Lᵢ/Oᵢ - (1-Lᵢ)/(1-Oᵢ))
		auto onesTid = training->addTensor(tname(STR("ones-vector-for-" << outputInfo[0].name)), outputInfo[0].shape, outputInfo[0].type, (uint8_t*)Util::arrayOfOnes<float>(outputInfo[0].shape[1]));

		auto derivativeOfLoss = Dual(
			PI::KindMul,
			Dual(
				PI::KindSub,
				Dual(
					PI::KindDiv,
					outputInfo[0].lossInput/*=O*/,
					outputInfo[0].lossLabel/*=L*/
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
						outputInfo[0].lossLabel/*=L*/
					)
				)
			),
			outputInfo[0].lossOutput
		);
		training->addOutput(derivativeOfLoss);

		break;
	} default:
		assert(false); // no other loss functions are supported yet
	}

	return training.release();
}

}
