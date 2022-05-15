// Copyright (C) 2022 by Yuri Victorovich. All rights reserved.

#pragma once

#include <array>
#include <vector>

//
// Graphviz_CGraph is a wrapper interface to GraphViz
// All sizes, supplied and retrieved, are in user inches. CAVEAT: GraphViz always uses 72dpi internally, regardless of what value user supplies to it.
// see https://gitlab.com/graphviz/graphviz/issues/1649. We asjust the values accordingly.
//

class Graphviz_CGraph {

	void *_g; // opaque reference to Agnode_t
	void *symNodeShape;
	void *symNodeWidth;
	void *symNodeHeight;
	void *symEdgeLabel;

	float userDPI;

public: // types
	typedef void* Node; // opaque pointer for the user
	typedef void* Edge; // opaque pointer for the user

public: // contructor/destructor
	Graphviz_CGraph(const char *graphName, float userDPI_);
	~Graphviz_CGraph();

public: // iface
	// adding objects
	Node addNode(const char *name);
	Edge addEdge(Node node1, Node node2, const char *name);
	void setDefaultNodeShape(const char *shapeValue);
	void setDefaultNodeSize(float width, float height);
	void setGraphOrdering(bool orderingIn, bool orderingOut);
	void setGraphPad(float padX, float padY);
	void setGraphMargin(float marginX, float marginY);
	void setNodeShape(Node node, const char *shapeValue);
	void setNodeSize(Node node, float width, float height);
	void setEdgeLabel(Edge edge, const char *label);

	// rendering
	void render();

	// getting information
	std::array<std::array<float, 2>, 2> getBBox() const;
	std::array<float,2> getNodePos(Node node) const;
	std::array<float,2> getNodeSize(Node node) const;
	std::array<float,2> getEdgePath(Node node) const;
	std::vector<std::array<float,2>> getEdgeSplines(Edge edge) const;
	std::array<float,2> getEdgeLabelPosition(Edge edge) const;

	// (DEBUG)
	void writeDotToStdio() const;

private: // internals
	static float assumedDPI();
	float userInchesToInches(float inches) const;
	float inchesToUserInches(float inches) const;
	float pixelsToUserInches(float pixels) const;
	std::array<float,2> pixelsToUserInches(const std::array<float,2> pixels) const;
};
