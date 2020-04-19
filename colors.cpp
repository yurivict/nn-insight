// Copyright (C) 2020 by Yuri Victorovich. All rights reserved.

#include "colors.h"

namespace Colors {

QColor getOperatorColor(PluginInterface::OperatorKind okind) {
	switch (okind) {
	case PluginInterface::KindConv2D:
		return Qt::blue;
	case PluginInterface::KindDepthwiseConv2D:
		return QColor(100,100,255);
	case PluginInterface::KindPad:
		return QColor(212,170,11);
	case PluginInterface::KindFullyConnected:
		return QColor(50,50,150);
	case PluginInterface::KindLocalResponseNormalization:
		return QColor(32,216,39);
	case PluginInterface::KindMaxPool:
	case PluginInterface::KindAveragePool:
		return QColor(50,150,50);
	case PluginInterface::KindRelu:
	case PluginInterface::KindRelu6:
	case PluginInterface::KindLeakyRelu:
	case PluginInterface::KindTanh:
	case PluginInterface::KindLogistic:
	case PluginInterface::KindHardSwish:
		return QColor(120,14,30); // activation functions
	case PluginInterface::KindAdd:
	case PluginInterface::KindSub:
	case PluginInterface::KindMul:
	case PluginInterface::KindDiv:
		return QColor(255,71,181); // arithmetic functions
	case PluginInterface::KindReshape:
		return QColor(160,170,32);
	case PluginInterface::KindSoftmax:
		return QColor(170,32,170);
	case PluginInterface::KindConcatenation:
		return QColor(32,170,170);
	case PluginInterface::KindMean:
		return QColor(130,11,212);
	case PluginInterface::KindArgMax:
	case PluginInterface::KindArgMin:
		return QColor(128,128,0); // olive
	case PluginInterface::KindResizeBilinear:
		return QColor(163,62,241);
	default: // unknown?
		return Qt::red;
	}
}

}
