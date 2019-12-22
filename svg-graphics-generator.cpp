

#include "plugin-interface.h"
#include <QSvgGenerator>
#include <QPainter>
#include <QPainterPath>
#include <QByteArray>
#include <QBuffer>
#include <QRectF>
#include <QMarginsF>
#include <QFontMetrics>

#include <vector>

#include "model-functions.h"
#include "misc.h"
#include "util.h"
#include "fonts.h"

namespace SvgGraphics {

class SvgGenerator : public QSvgGenerator {
public:
	QByteArray ba;
	QBuffer    buff;

	SvgGenerator(unsigned width, unsigned height, const char *title)
	: buff(&ba)
	{
		setOutputDevice(&buff);
		setResolution(Util::getScreenDPI());
		setViewBox(QRect(0, 0, width, height));
		setTitle(title);
		//setDescription("");
	}
};

QByteArray generateModelSvg(const PluginInterface::Model *model) {
	// options
	qreal  operatorBoxRadius          = 5;
	qreal  operatorBoxBorderWidth     = 2;
	QColor clrOperatorBorder          = Qt::black;
	QColor clrOperatorTitleText       = Qt::white;
	QColor clrOperatorTitleBackground = Qt::blue;
	QColor clrTensorLabel             = Qt::black;

/*
	auto inputs = model->getInputs();
	auto outputs = model->getOutputs();
	// for now only support single-input single-output models, FIXME later
	assert(inputs.size() == 1);
	assert(outputs.size() == 1);
*/
	// find all tail operators with no ins leading to operators or no outs consumed by other operators
/*
	std::vector<PluginInterface::OperatorId> tailInOps, tailOutOps;
	iterateThroughOperators([&](PluginInterface::OperatorId oid) {
		std::vector<PluginInterface::TensorId> oinputs, ooutputs;
		model->getOperatorInfo(oid, oinputs, ooutputs);
		// any oinputs in outputs of any other operators?
		bool anyInsInOthersOuts = false;
		bool anyOutsInOthersIns = false;
		for (auto i : oinputs)
			if(tensorAssociations[i].size() > 1)
				anyInsInOthersOuts = true;
		for (auto o : ooutputs)
			if(tensorAssociations[o].size() > 1)
				anyOutsInOthersIns = true;
		if (!anyInsInOthersOuts)
			tailInOps.push_back(oid);
		if (!anyOutsInOthersIns)
			tailOutOps.push_back(oid);
	});
	for (auto o : tailInOps)
		PRINT("generateModelSvg: inOp: " << o)
	for (auto o : tailOutOps)
		PRINT("generateModelSvg: outOp: " << o)
*/

	//for (auto it = j.begin(); it != j.end(); ++it)
	//   std::cout << "key: " << it.key() << ", value:" << it.value() << '\n';

	//PRINT("json: bb=" << j["bb"].get<std::string>())

	// render the model to the coordinates
	ModelFunctions::Box2 bbox;
	std::vector<ModelFunctions::Box4> operatorBoxes;
	std::vector<QPoint> tensorLabelPositions;
	{
		QFontMetrics fm(Fonts::fontOperatorTitle);
		qreal dpi = (qreal)Util::getScreenDPI();
		ModelFunctions::renderModelToCoordinates(model,
			QMarginsF(0.2, 0.1, 0.22, 0.1), // operator box margins in inches
			[model,&fm,dpi](PluginInterface::OperatorId oid) {
				auto szPixels = fm.size(Qt::TextSingleLine, S2Q(STR(model->getOperatorKind(oid))));
				return QSizeF(qreal(szPixels.width())/dpi, qreal(szPixels.height())/dpi);
			},
			bbox,
			operatorBoxes,
			tensorLabelPositions
		);
	}

	// generate the SVG file

	SvgGenerator generator(bbox[1][0]/*width*/, bbox[1][1]/*height*/, "NN Model");

	QPainter painter;
	//painter.setRenderHint(QPainter::Antialiasing);
	painter.begin(&generator);

	auto dotYToQtY = [&bbox](float Y) {
		return bbox[1][1]-Y;
	};
	auto dotBoxToQtBox = [&](const ModelFunctions::Box4 &box4) {
		assert(box4[0][0]==box4[3][0]);
		assert(box4[0][1]==box4[1][1]);
		assert(box4[1][0]==box4[2][0]);
		assert(box4[2][1]==box4[3][1]);
		return ModelFunctions::Box2({{
			{box4[1][0], dotYToQtY(box4[0][1])}, // coord
			{box4[0][0]-box4[1][0], box4[1][1]-box4[2][1]}
		}});
	};
	auto drawOperator = [&](QPainter &painter,
	                        const ModelFunctions::Box2 &box,
	                        const std::string &title,
	                        const std::vector<std::string> &inputDescriptions,
	                        const std::string &fusedDescription)
	{
		QRectF bbox(box[0][0], box[0][1], box[1][0], box[1][1]);
		// border
		QPainterPath path;
		path.addRoundedRect(bbox, operatorBoxRadius, operatorBoxRadius);
		QPen pen(clrOperatorBorder, operatorBoxBorderWidth);
		painter.setPen(pen);
		painter.fillPath(path, clrOperatorTitleBackground);
		painter.drawPath(path);
		// texts
		painter.setPen(clrOperatorTitleText);
		painter.setFont(Fonts::fontOperatorTitle);
		painter.drawText(bbox, Qt::AlignCenter, S2Q(title));
	};
	auto drawTensorLabel = [&](QPainter &painter,
	                           const QPoint &pt,
	                           const std::string &label,
	                           const QFontMetrics &fm)
	{
		// ASSUME that painter.setPen and painter.setFont have been called by the caller
		auto textSize = fm.size(Qt::TextSingleLine, S2Q(label));
		QPoint textSizeHalf(textSize.width()/2, textSize.height()/2);
		painter.drawText(QRect(QPoint(pt - textSizeHalf), QPoint(pt + textSizeHalf)), Qt::AlignCenter, S2Q(label));
	};

	{ // draw operator boxes
		PluginInterface::OperatorId oid = 0;
		for (auto &dotBox : operatorBoxes)
			drawOperator(painter, dotBoxToQtBox(dotBox), STR(model->getOperatorKind(oid++)), {}, "");
	}

	{ // draw tensor labels
		painter.setPen(clrTensorLabel);
		painter.setFont(Fonts::fontTensorLabel);
		QFontMetrics fm(Fonts::fontTensorLabel);
		PluginInterface::TensorId tid = 0;
		for (auto &tensorLabelPosition : tensorLabelPositions)
			if (tensorLabelPosition != QPoint()) // not all tensors are between operators
				drawTensorLabel(painter, tensorLabelPosition, STR(model->getTensorShape(tid++)), fm);
	}

	painter.end();

	return generator.ba;
}

} // SvgGraphics
