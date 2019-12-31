
#include "graphviz-cgraph.h"
#include "misc.h"
#include "util.h"

#include <gvc.h>
#include <gvplugin.h>
#include <stdio.h>
#include <assert.h>

#include <array>
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
	float p[2];
	auto n = ::sscanf(str, "%f,%f", &p[0], &p[1]);
	assert(n == 2);
	UNUSED(n)

	return {p[0],p[1]};
}


/// Graphviz_CGraph

Graphviz_CGraph::Graphviz_CGraph(const char *graphName)
: symNodeShape(nullptr)
, symNodeWidth(nullptr)
, symNodeHeight(nullptr)
, symEdgeLabel(nullptr)
, dpi(600) // default value
{
	_g = agopen(S(graphName), Agdirected, NULL);

}

Graphviz_CGraph::~Graphviz_CGraph() {
	agclose(GRAPH);
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
	symNodeWidth = agattr(GRAPH, AGNODE, S("width"), S(CSTR(pixelsToInches(width))));
	symNodeHeight = agattr(GRAPH, AGNODE, S("height"), S(CSTR(pixelsToInches(height))));
}

void Graphviz_CGraph::setGraphDpi(unsigned dpi_) {
	agattr(GRAPH, AGRAPH, S("dpi"), S(CSTR(dpi_)));
	dpi = dpi_;
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
	agxset(NODE(node), SYM(symNodeWidth), S(CSTR(width)));
	agxset(NODE(node), SYM(symNodeHeight), S(CSTR(height)));
}

void Graphviz_CGraph::setEdgeLabel(Edge edge, const char *label) {
	if (!symEdgeLabel)
		symEdgeLabel = agattr(GRAPH, AGEDGE, S("label"), S(""));
	agxset(EDGE(edge), SYM(symEdgeLabel), S(label));
}

void Graphviz_CGraph::render() {
	int err = gvLayout(gvc, GRAPH, S("dot"));
	if (err)
		FAIL("GraphViz failed to render the graph")

	attach_attrs(GRAPH);
}

// getting information

std::array<std::array<float, 2>, 2> Graphviz_CGraph::getBBox() const {
	auto str = agget(GRAPH, S("bb"));
	assert(str);

	float p[4];
	auto n = ::sscanf(str, "%f,%f,%f,%f", &p[0], &p[1], &p[2], &p[3]);
	assert(n == 4);
	UNUSED(n)

	return {{{{p[0],p[1]}},{{p[2],p[3]}}}};
}

std::array<float,2> Graphviz_CGraph::getNodePos(Graphviz_CGraph::Node node) const {
	auto str = agget(NODE(node), S("pos"));
	assert(str);

	return parseTwoFloats(str);
}

std::array<float,2> Graphviz_CGraph::getNodeSize(Graphviz_CGraph::Node node) const {
	auto strW = agget(NODE(node), S("width"));
	auto strH = agget(NODE(node), S("height"));
	return {inchesToPixels(std::stof(strW)), inchesToPixels(std::stof(strH))};
}

std::string Graphviz_CGraph::getEdgeSplines(Edge edge) const {
	auto str = agget(EDGE(edge), S("pos"));
	assert(str);
	return str; // TODO parse splines here instead of by the caller
}

std::array<float,2> Graphviz_CGraph::getEdgeLabelPosition(Edge edge) const {
	auto str = agget(EDGE(edge), S("lp"));
	assert(str);

	return parseTwoFloats(str);
}

// DEBUG

void Graphviz_CGraph::writeDotToStdio() const {
	agwrite(GRAPH, stdout);
}

// internals

float Graphviz_CGraph::pixelsToInches(float pixels) const {
	return pixels/72;//dpi; // graphviz uses 72 regardless of the actually supplied dpi, see https://gitlab.com/graphviz/graphviz/issues/1649
}

float Graphviz_CGraph::inchesToPixels(float inches) const {
	return inches*72;//dpi;
}
