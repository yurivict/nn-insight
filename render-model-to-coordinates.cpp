
#include "model-functions.h"

#include <string>
#include <vector>
#include <set>
#include <sstream>

#include <nlohmann/json.hpp>

#include <stdio.h>

#include "model-functions.h"
#include "plugin-interface.h"
#include "misc.h"
#include "util.h"

#include <QSizeF>
#include <QPoint>

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
	Box2 &bbox,
	std::vector<Box4> &operatorBoxes,
	std::vector<QPoint> &tensorLabelPositions
) {

	// map tensors to operators
	std::vector<int> tensorProducers;
	tensorProducers.resize(model->numTensors());
	std::vector<std::set<PluginInterface::OperatorId>> tensorConsumers;
	tensorConsumers.resize(model->numTensors());
	{
		for (auto &p : tensorProducers)
			p = -1;
		for (PluginInterface::OperatorId oid = 0, oide = (PluginInterface::OperatorId)model->numOperators(); oid < oide; oid++) {
			std::vector<PluginInterface::TensorId> oinputs, ooutputs;
			model->getOperatorIo(oid, oinputs, ooutputs);
			for (auto o : ooutputs)
				tensorProducers[o] = oid;
			for (auto i : oinputs)
				tensorConsumers[i].insert(oid);
		}
	}

	// helpers
	auto toDOT = [&](const PluginInterface::Model *model, std::ostream &os) {
		os << "digraph D {" << std::endl;
		os << "  graph [ dpi = " << Util::getScreenDPI() << " ]; " << std::endl;
		for (PluginInterface::OperatorId oid = 0, oide = model->numOperators(); oid < oide; oid++) {
			auto szBox = operatorBoxFn(oid);
			os << "Op_" << oid << " [shape=box"
			   << ",width=" << (operatorBoxMargins.left() + szBox.width() + operatorBoxMargins.right())
			   << ",height=" << (operatorBoxMargins.top() + szBox.height() + operatorBoxMargins.bottom()) <<  "];"
			   << std::endl;
		}
		for (PluginInterface::TensorId tid = 0, tide = model->numTensors(); tid < tide; tid++)
			if (tensorProducers[tid] != -1 && !tensorConsumers[tid].empty())
				for (auto oidConsumer : tensorConsumers[tid])
					os << "Op_" << tensorProducers[tid] << " -> Op_" << oidConsumer << " ["
					   << "id=" << tid
					   << ";label=\"" << STR(model->getTensorShape(tid)) << "\""
					   << "];" << std::endl;
		os << "}" << std::endl;
	};
	auto renderDotAsJson = [](const std::string &dot) {
		// open the pipe
		FILE *pipe = ::popen("dot -Tjson", "r+");
		if (pipe == nullptr)
			FAIL("unable to start the 'dot' process to render the model as a graph")

		// write into the pipe
		if (::fwrite(dot.c_str(), 1, dot.size(), pipe) != dot.size())
			FAIL("failed to write the graph to 'dot' process")
		::fflush(pipe);

		// read from the pipe
		std::ostringstream ss;
		char buf[1025];
		size_t sz;
		while (true) {
			sz = ::fread(buf, 1, sizeof(buf)-1, pipe);
			if (sz > 0) {
				buf[sz] = 0;
				ss << buf;
			} else
				break;
		}

		// close the pipe
		if (::pclose(pipe) == -1)
			FAIL("failed to close the pipe to the 'dot' process")

		return ss.str();
	};

	// print graph's dot
	std::ostringstream ssDot;
	toDOT(model, ssDot);
	ssDot << "\n@\n"; // EOF signal for DOT

	//PRINT("dot: " << ssDot.str())
	auto graphJson = renderDotAsJson(ssDot.str());
	//PRINT("json: " << graphJson)

	// parse it as json
	using json = nlohmann::json;
	json j = json::parse(graphJson);

	auto floatsToArray1x2 = [](auto &boxFloats, auto &box) {
		auto it = boxFloats.begin();
		for (auto &a1 : box)
			for (auto &a2 : a1)
				a2 = *it++;
	};
	auto floatsToArray2x2 = [](auto &src, auto &dst) {
		for (unsigned i1 = 0, i1e = src.size(); i1 < i1e; i1++) {
			auto &src1 = src[i1];
			auto &dst1 = dst[i1];
			for (unsigned i2 = 0, i2e = src1.size(); i2 < i2e; i2++)
				dst1[i2] = src1[i2];
		}
	};

	// bbox
	std::vector<float> boxFloats;
	Util::splitString<std::vector<float>, ConvStrToFloat>(j["bb"].get<std::string>(), boxFloats, ',');
	if (boxFloats.size() != 4)
		FAIL("The 'bb' array in JSON printed by the dot utility is expected to have 4 elements, but has " << boxFloats.size() << " elements")
	floatsToArray1x2(boxFloats, bbox);

	// objects
	operatorBoxes.resize(model->numOperators());
	for (auto obj : j["objects"]) {
		auto name = obj["name"].get<std::string>();
		if (name.size() < 4 || (name[0]!='O' || name[1]!='p' || name[2]!='_'))
			FAIL("got returned the object name that we don't recognize: " << name)
		auto opId = std::stoi(name.substr(3));
		if (opId >= operatorBoxes.size())
			FAIL("dot returned an invalid name " << name)

		for (auto d : obj["_draw_"])
			if (d["op"] == "p")
				floatsToArray2x2(d["points"], operatorBoxes[opId]);
	}

	// edges
	tensorLabelPositions.resize(model->numTensors());
}

}
