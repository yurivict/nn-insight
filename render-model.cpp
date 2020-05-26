// Copyright (C) 2020 by Yuri Victorovich. All rights reserved.

#include "model-functions.h"

#include <string>
#include <vector>
#include <map>
#include <set>
#include <sstream>

#include <stdio.h>
#include <errno.h>
#include <assert.h>

#include "model-functions.h"
#include "plugin-interface.h"
#include "graphviz-cgraph.h"
#include "misc.h"
#include "util.h"

#include <QSizeF>

// helper classes
class ConvStrToFloat {
public:
	static float conv(const std::string &str) {
		return std::stof(str);
	}
};

namespace ModelFunctions {

void renderModelToCoordinates(const PluginInterface::Model *model,
	const QMarginsF &operatorBoxMargins,
	std::function<QSizeF(PluginInterface::OperatorId)> operatorBoxFn, // operator boxes in inches
	std::function<QSizeF(PluginInterface::TensorId)> inputBoxFn, // input boxes in inches
	std::function<QSizeF(PluginInterface::TensorId)> outputBoxFn, // output boxes in inches
	Box2 &bbox,
	std::vector<Box2> &operatorBoxes,
	std::map<PluginInterface::TensorId, Box2> &inputBoxes,
	std::map<PluginInterface::TensorId, Box2> &outputBoxes,
	std::vector<std::vector<std::vector<std::array<float,2>>>> &tensorLineCubicSplines,
	std::vector<std::vector<std::array<float,2>>> &tensorLabelPositions
) {

	// map tensors to operators
	std::vector<int/*PluginInterface::OperatorId or -1*/> tensorProducers;
	const int NoOperator = -1;
	tensorProducers.resize(model->numTensors());
	std::vector<std::vector<PluginInterface::OperatorId>> tensorConsumers;
	tensorConsumers.resize(model->numTensors());
	{
		for (auto &p : tensorProducers)
			p = NoOperator;
		for (PluginInterface::OperatorId oid = 0, oide = (PluginInterface::OperatorId)model->numOperators(); oid < oide; oid++) {
			std::vector<PluginInterface::TensorId> oinputs, ooutputs;
			model->getOperatorIo(oid, oinputs, ooutputs);
			for (auto o : ooutputs)
				tensorProducers[o] = oid;
			for (auto i : oinputs)
				tensorConsumers[i].push_back(oid);
		}
	}

	/// build the graphviz graph

	// create the graph
	Graphviz_CGraph graph("NnModel", Util::getScreenDPI());
	graph.setGraphOrdering(true/*orderingIn*/, false/*orderingOut*/); // preserve edge order as they are consumed as inputs.
	                                                                  // ideally we need to have both "in" and "out" ordering
	                                                                  // see https://gitlab.com/graphviz/graphviz/issues/1645
	graph.setDefaultNodeShape("box");
	graph.setDefaultNodeSize(0,0);

	unsigned edgeNo = 1; // needed for edge names, without names multiple edges between same nodes are shown as the same edge

	// add operators as nodes
	std::vector<Graphviz_CGraph::Node> operatorNodes;
	for (PluginInterface::OperatorId oid = 0, oide = model->numOperators(); oid < oide; oid++) {
		auto node = graph.addNode(CSTR("Op_" << oid)); // operator
		auto szBox = operatorBoxFn(oid);
		graph.setNodeSize(node,
			operatorBoxMargins.left() + szBox.width() + operatorBoxMargins.right(),
			operatorBoxMargins.top() + szBox.height() + operatorBoxMargins.bottom()
		);
		operatorNodes.push_back(node);
	}

	// add tensors as edges
	std::vector<std::vector<Graphviz_CGraph::Node>> tensorEdges;
	tensorEdges.resize(model->numTensors());
	for (PluginInterface::TensorId tid = 0, tide = model->numTensors(); tid < tide; tid++)
		if (tensorProducers[tid] != NoOperator) // operator->operator
			for (auto oidConsumer : tensorConsumers[tid]) { // tensorConsumers can be empty (a corner case of a dangling operator output in a broken model)
				auto edge = graph.addEdge(operatorNodes[tensorProducers[tid]], operatorNodes[oidConsumer], CSTR("edge#" << edgeNo++)/*name(key)*/); // operator->operator
				graph.setEdgeLabel(edge, CSTR(model->getTensorShape(tid)));
				tensorEdges[tid].push_back(edge);
			}

	// add inputs
	std::map<PluginInterface::TensorId, Graphviz_CGraph::Node> inputNodes;
	for (auto i : model->getInputs()) {
		auto node = graph.addNode(CSTR("In_" << i)); // input
		auto szBox = inputBoxFn(i);
		graph.setNodeSize(node,
			operatorBoxMargins.left() + szBox.width() + operatorBoxMargins.right(),
			operatorBoxMargins.top() + szBox.height() + operatorBoxMargins.bottom()
		);
		inputNodes[i] = node;
		// edges
		for (auto oidConsumer : tensorConsumers[i]) {
			auto edge = graph.addEdge(node, operatorNodes[oidConsumer], CSTR("edge#" << edgeNo++)/*name(key)*/); // input->operator
			graph.setEdgeLabel(edge, CSTR(model->getTensorShape(i)));
			tensorEdges[i].push_back(edge);
		}
	}

	// add outputs
	std::map<PluginInterface::TensorId, Graphviz_CGraph::Node> outputNodes;
	for (auto o : model->getOutputs()) {
		auto node = graph.addNode(CSTR("Out_" << o)); // output
		auto szBox = outputBoxFn(o);
		graph.setNodeSize(node,
			operatorBoxMargins.left() + szBox.width() + operatorBoxMargins.right(),
			operatorBoxMargins.top() + szBox.height() + operatorBoxMargins.bottom()
		);
		outputNodes[o] = node;
		// edge
		Graphviz_CGraph::Edge edge;
		if (tensorProducers[o] != NoOperator) // operator->output
			edge = graph.addEdge(operatorNodes[tensorProducers[o]], node, CSTR("edge#" << edgeNo++)/*name(key)*/); // operator->output
		else { // input->output (the corner case of the output connected directly to the input)
			assert(inputNodes.find(o) != inputNodes.end());
			edge = graph.addEdge(inputNodes[o], node, CSTR("edge#" << edgeNo++)/*name(key)*/); // input->output
		}
		graph.setEdgeLabel(edge, CSTR(model->getTensorShape(o)));
		tensorEdges[o].push_back(edge);
	}

	/// render the graph
	graph.render();

	// bbox
	bbox = graph.getBBox();

	// operators (nodes)
	for (auto node : operatorNodes)
		operatorBoxes.push_back(Box2{{graph.getNodePos(node), graph.getNodeSize(node)}});
	// inputs (nodes)
	for (auto it : inputNodes)
		inputBoxes[it.first] = Box2{{graph.getNodePos(it.second), graph.getNodeSize(it.second)}};
	// outputs (nodes)
	for (auto it : outputNodes)
		outputBoxes[it.first] = Box2{{graph.getNodePos(it.second), graph.getNodeSize(it.second)}};

	// tensors (edges)
	tensorLabelPositions.resize(tensorEdges.size());
	tensorLineCubicSplines.resize(tensorEdges.size());
	for (PluginInterface::TensorId tid = 0, tide = tensorEdges.size(); tid<tide; tid++)
		for (auto edge : tensorEdges[tid]) {
			// splines
			tensorLineCubicSplines[tid].push_back(graph.getEdgeSplines(edge));
			// label position
			tensorLabelPositions[tid].push_back(graph.getEdgeLabelPosition(edge));
		}
}

}
