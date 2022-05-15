// Copyright (C) 2022 by Yuri Victorovich. All rights reserved.

#include "graphviz-cgraph.h"
#include "misc.h"
#include "util.h"

#include <gvc.h>
#include <gvplugin.h>
#include <stdio.h>
#include <assert.h>

#include <array>
#include <chrono>
#include <vector>
#include <string>

#define S(str) ((char*)str)
#define GRAPH            ((Agraph_t*)this->_g)
#define NODE(opaqueNode) ((Agnode_t*)opaqueNode)
#define EDGE(opaqueEdge) ((Agedge_t*)opaqueEdge)
#define SYM(opaqueSym)   ((Agsym_t*)opaqueSym)

extern gvplugin_library_t gvplugin_dot_layout_LTX_library;

lt_symlist_t lt_preloaded_symbols[] = {
	{"gvplugin_dot_layout_LTX_library", (void*)(&gvplugin_dot_layout_LTX_library)},
	{0, 0}
};

/// static initializer

static GVC_t *gvc = gvContextPlugins(lt_preloaded_symbols, 1/*demand_loading*/); // XXX CAVEAT no way to destroy this object during static destructors

/// static helpers

static std::array<float,2> parseTwoFloats(const char *str) {
	assert(str);
	float p[2];
	auto n = ::sscanf(str, "%f,%f", &p[0], &p[1]);
	assert(n == 2);
	UNUSED(n)

	return {p[0],p[1]};
}


/// Graphviz_CGraph

Graphviz_CGraph::Graphviz_CGraph(const char *graphName, float userDPI_)
: symNodeShape(nullptr)
, symNodeWidth(nullptr)
, symNodeHeight(nullptr)
, symEdgeLabel(nullptr)
, userDPI(userDPI_)
{
	_g = agopen(S(graphName), Agdirected, NULL);

	// set the dpi value of 72 because that's what GraphViz assumes anyway
	agattr(GRAPH, AGRAPH, S("dpi"), S(CSTR(assumedDPI())));
}

Graphviz_CGraph::~Graphviz_CGraph() {
	gvFreeLayout(gvc, GRAPH);
	agclose(GRAPH); // the memory is lost by GraphViz, see https://gitlab.com/graphviz/graphviz/issues/1651
}

/// interface implementation

Graphviz_CGraph::Node Graphviz_CGraph::addNode(const char *name) {
	return (Node)agnode(GRAPH, S(name), TRUE/*create*/);
}

Graphviz_CGraph::Edge Graphviz_CGraph::addEdge(Node node1, Node node2, const char *name) {
	return (Edge)agedge(GRAPH, NODE(node1), NODE(node2), S(name), TRUE/*create*/);
}

void Graphviz_CGraph::setDefaultNodeShape(const char *shapeValue) {
	assert(!symNodeShape);
	symNodeShape = agattr(GRAPH, AGNODE, S("shape"), S(shapeValue));
}

void Graphviz_CGraph::setDefaultNodeSize(float width, float height) {
	assert(!symNodeWidth && !symNodeHeight);
	symNodeWidth = agattr(GRAPH, AGNODE, S("width"), S(CSTR(userInchesToInches(width))));
	symNodeHeight = agattr(GRAPH, AGNODE, S("height"), S(CSTR(userInchesToInches(height))));
}

void Graphviz_CGraph::setGraphOrdering(bool orderingIn, bool orderingOut) {
	if (orderingIn || orderingOut) {
		assert(!(orderingIn && orderingOut)); // graphviz doesn't support this yet: see https://gitlab.com/graphviz/graphviz/issues/1645
		agattr(GRAPH, AGRAPH, S("ordering"), orderingIn ? S("in") : S("out"));
	}
}

void Graphviz_CGraph::setGraphPad(float padX, float padY) {
	agattr(GRAPH, AGRAPH, S("pad"), S(CSTR(padX << "," << padY)));
}

void Graphviz_CGraph::setGraphMargin(float marginX, float marginY) {
	agattr(GRAPH, AGRAPH, S("margin"), S(CSTR(marginX << "," << marginY)));
}

void Graphviz_CGraph::setNodeShape(Node node, const char *shapeValue) {
	assert(symNodeShape); // need to call setDefaultNodeShape() first
	agxset(NODE(node), SYM(symNodeShape), S("box"));
}

void Graphviz_CGraph::setNodeSize(Node node, float width, float height) {
	assert(symNodeWidth && symNodeHeight); // need to call setDefaultNodeSize() first
	agxset(NODE(node), SYM(symNodeWidth), S(CSTR(userInchesToInches(width))));
	agxset(NODE(node), SYM(symNodeHeight), S(CSTR(userInchesToInches(height))));
}

void Graphviz_CGraph::setEdgeLabel(Edge edge, const char *label) {
	if (!symEdgeLabel)
		symEdgeLabel = agattr(GRAPH, AGEDGE, S("label"), S(""));
	agxset(EDGE(edge), SYM(symEdgeLabel), S(label));
}

void Graphviz_CGraph::render() {
	// time begin
	auto tmStart = std::chrono::high_resolution_clock::now();

	int err = gvLayout(gvc, GRAPH, S("dot"));
	if (err)
		FAIL("GraphViz failed to render the graph")

	attach_attrs(GRAPH);

	{ // time end
		auto tmStop = std::chrono::high_resolution_clock::now();
		PRINT("graph was rendered in " << std::chrono::duration_cast<std::chrono::milliseconds>(tmStop - tmStart).count() << " milliseconds")
	}
}

// getting information

std::array<std::array<float, 2>, 2> Graphviz_CGraph::getBBox() const {
	auto str = agget(GRAPH, S("bb"));
	assert(str);

	float p[4];
	auto n = ::sscanf(str, "%f,%f,%f,%f", &p[0], &p[1], &p[2], &p[3]);
	assert(n == 4);
	UNUSED(n)

	return {{
		{{pixelsToUserInches(p[0]), pixelsToUserInches(p[1])}},
		{{pixelsToUserInches(p[2]), pixelsToUserInches(p[3])}}
	}};
}

std::array<float,2> Graphviz_CGraph::getNodePos(Graphviz_CGraph::Node node) const {
	return pixelsToUserInches(parseTwoFloats(agget(NODE(node), S("pos"))));
}

std::array<float,2> Graphviz_CGraph::getNodeSize(Graphviz_CGraph::Node node) const {
	auto strW = agget(NODE(node), S("width"));
	auto strH = agget(NODE(node), S("height"));
	return {inchesToUserInches(std::stof(strW)), inchesToUserInches(std::stof(strH))};
}

std::vector<std::array<float,2>> Graphviz_CGraph::getEdgeSplines(Edge edge) const {
	auto splines = agget(EDGE(edge), S("pos"));
	assert(splines);

	std::vector<std::string> segments;
	Util::splitString(splines, segments, ' ');

	std::vector<std::array<float,2>> pts;
	pts.push_back({-1,-1}); // startp
	pts.push_back({-1,-1}); // endp

	for (auto &s : segments)
		switch (s[0]) {
		case 's': // startp
			assert(s.size()>=5 && s[1]==',');
			pts[0] = pixelsToUserInches(parseTwoFloats(s.c_str()+2));
			break;
		case 'e': // endp
			assert(s.size()>=5 && s[1]==',');
			pts[1] = pixelsToUserInches(parseTwoFloats(s.c_str()+2));
			break;
		default: // a spline point
			pts.push_back(pixelsToUserInches(parseTwoFloats(s.c_str())));
		}
	assert((pts.size()-2)%3==1); // n = 1 (mod 3) for splines

	return pts; // {startp,endp, 1+2*n points defining the spline}, startp,endp are optional, {-1,-1} means it isn't set
}

std::array<float,2> Graphviz_CGraph::getEdgeLabelPosition(Edge edge) const { // CAVEAT there's no way to specify boxes for edge labels, we just have to use positions
	return pixelsToUserInches(parseTwoFloats(agget(EDGE(edge), S("lp"))));
}

// DEBUG

void Graphviz_CGraph::writeDotToStdio() const {
	agwrite(GRAPH, stdout);
}

/// internals

float Graphviz_CGraph::assumedDPI() {
	return 72; // GraphViz always uses 72dpi internally, see https://gitlab.com/graphviz/graphviz/issues/1649
}

float Graphviz_CGraph::userInchesToInches(float inches) const {
	return inches*userDPI/assumedDPI();
}

float Graphviz_CGraph::inchesToUserInches(float inches) const {
	return inches*assumedDPI()/userDPI;
}

float Graphviz_CGraph::pixelsToUserInches(float pixels) const {
	return pixels/userDPI;
}

std::array<float,2> Graphviz_CGraph::pixelsToUserInches(const std::array<float,2> pixels) const {
	return {{
		pixelsToUserInches(pixels[0]),
		pixelsToUserInches(pixels[1])
	}};
}
