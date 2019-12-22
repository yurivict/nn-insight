

#include "plugin-interface.h"

std::ostream& operator<<(std::ostream &os, PluginInterface::OperatorKind okind) {
#define CASE(kind) case PluginInterface::Kind##kind: os << #kind; break;
	switch (okind) {
  	CASE(Conv2D) CASE(DepthwiseConv2D) CASE(Pad) CASE(FullyConnected) CASE(MaxPool) CASE(AveragePool) CASE(Add) CASE(Relu) CASE(Relu6) CASE(LeakyRelu)
	CASE(Tanh) CASE(Sub) CASE(Mul) CASE(Div) CASE(Maximum) CASE(Minimum) CASE(Transpose) CASE(Reshape) CASE(Softmax) CASE(Concatenation)
	CASE(StridedSlice) CASE(Mean)
	CASE(Unknown)
	}
#undef CASE
	return os;
}
