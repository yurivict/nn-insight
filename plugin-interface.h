#pragma once

//
// PluginInterface is the interface of all plugins, common for all NN model types
//

#include <string>
#include <vector>
#include <ostream>
#include <functional>

#include "nn-types.h"

class PluginInterface {

  //std::function<void()> onDestroy;

public:
  virtual ~PluginInterface() {
    //onDestroy();
  }

  // types related to the plugin interface
	typedef unsigned TensorId;
	typedef unsigned OperatorId;
  enum OperatorKind { // all distinct operator kinds should be listed here
  	// XXX each value here has to be mirrored in plugin-interface.cpp (CASE)
  	KindConv2D,
  	KindDepthwiseConv2D,
  	KindPad,
  	KindFullyConnected,
  	KindMaxPool,
  	KindAveragePool,
  	KindAdd,
  	KindRelu,
	KindRelu6,
	KindLeakyRelu,
	KindTanh,
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
	KindUnknown
  };

	friend std::ostream& operator<<(std::ostream &os, OperatorKind okind);

  // inner-classes
  class Model { // Model represents one of potentially many models contained in the file
  protected:
	virtual ~Model() { } // has to be inlined for plugins to contain it too
  public: // interface
	virtual unsigned                numInputs() const = 0;                                                          // how many inputs does this model have
	virtual std::vector<TensorId>   getInputs() const = 0;                                                          // input indexes
	virtual unsigned                numOutputs() const = 0;                                                         // how many outputs does this model have
	virtual std::vector<TensorId>   getOutputs() const = 0;                                                         // output indexes
	virtual unsigned                numOperators() const = 0;                                                       // how many operators does this model have
	virtual void                    getOperatorIo(unsigned operatorIdx, std::vector<TensorId> &inputs, std::vector<TensorId> &outputs) const = 0;
	virtual OperatorKind            getOperatorKind(unsigned operatorIdx) const = 0;
	virtual unsigned                numTensors() const = 0;                                                         // number of tensors in this model
	virtual TensorShape             getTensorShape(TensorId tensorId) const = 0;
	virtual std::string             getTensorName(TensorId tensorId) const = 0;
	virtual bool                    getTensorHasData(TensorId tensorId) const = 0;                                  // tensors that are fixed have buffers
	virtual bool                    getTensorIsVariableFlag(TensorId tensorId) const = 0;                           // some tensors are variables that can be altered
  };

  // custom interface
  inline void setOnDestroy(std::function<void()> onDestroy_) {
    //onDestroy = onDestroy_;
  }

  // plugin interface
  virtual std::string filePath() const = 0;                // returns back the file name that it was opened from
  virtual bool open(const std::string &filePath_) = 0;     // open the model (can only be done once per object, close is implicit on destruction for simplicity)
  virtual std::string errorMessage() const = 0;            // returns the error of the last operation if it has failed
  virtual size_t numModels() const = 0;                    // how many models does this file contain
  virtual const Model* getModel(unsigned index) const = 0; // access to one model, the Model object is owned by the plugin
};
