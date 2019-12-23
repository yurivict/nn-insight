

#include "plugin-interface.h"
#include <QSvgGenerator>
#include <QPainter>
#include <QPainterPath>
#include <QByteArray>
#include <QBuffer>
#include <QMarginsF>
#include <QFontMetrics>
#include <QRectF>
#include <QRect>
#include <QPointF>

#include <vector>

#include <assert.h>

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

QByteArray generateModelSvg(const PluginInterface::Model *model, const std::tuple<std::vector<QRectF>&,std::vector<QRectF>&> outIndexes) {
	// options
	qreal  operatorBoxRadius          = 5;
	qreal  operatorBoxBorderWidth     = 2;
	QColor clrOperatorBorder          = Qt::black;
	QColor clrOperatorTitleText       = Qt::white;
	QColor clrOperatorTitleBackground = Qt::blue;
	QColor clrTensorLine              = Qt::black;
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
	std::vector<std::vector<QPointF>> tensorLineCubicSplines;
	std::vector<QPointF> tensorLabelPositions;
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
			tensorLineCubicSplines,
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
	auto dotQPointToQtQPoint = [&](const QPointF &pt) {
		return QPointF(pt.x(), dotYToQtY(pt.y()));
	};
	auto dotVectorQPointToQtVectorQPoint = [&](const std::vector<QPointF> &pts) {
		std::vector<QPointF> res;
		for (auto &p : pts)
			res.push_back(dotQPointToQtQPoint(p));
		return res;
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
	auto boxToQRectF = [](const ModelFunctions::Box2 &box) {
		return QRectF(box[0][0], box[0][1], box[1][0], box[1][1]);
	};
	auto drawOperator = [&](QPainter &painter,
	                        const QRectF &bbox,
	                        const std::string &title,
	                        const std::vector<std::string> &inputDescriptions,
	                        const std::string &fusedDescription)
	{
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
	auto drawTensorSplines = [&](QPainter &painter,
	                             const std::vector<QPointF> &splines)
	{
		assert(splines.size()%3 == 2);

		QPainterPath path;
		path.moveTo(splines[0]);
		for (unsigned i = 1; i+2 < splines.size(); i += 3)
			path.cubicTo(splines[i+0], splines[i+1], splines[i+2]);
		path.lineTo(splines[splines.size()-1]); // TODO draw the arrow here
		painter.drawPath(path);
	};
	auto drawTensorLabel = [&](QPainter &painter,
	                           const QPointF &pt,
	                           const std::string &label,
	                           const QFontMetrics &fm,
				   QRectF &outTextRect
				   )
	{
		// ASSUME that painter.setPen and painter.setFont have been called by the caller
		auto textSize = fm.size(Qt::TextSingleLine, S2Q(label));
		QPointF textSizeHalf(textSize.width()/2, textSize.height()/2);
		painter.drawText(
			outTextRect = QRectF(QPointF(pt - textSizeHalf), QPointF(pt + textSizeHalf)),
			Qt::AlignCenter,
			S2Q(label)
		);
	};

	// resize indexes
	std::get<0>(outIndexes).resize(model->numOperators());
	std::get<1>(outIndexes).resize(model->numTensors());

	{ // draw operator boxes
		PluginInterface::OperatorId oid = 0;
		for (auto &dotBox : operatorBoxes) {
			// draw and add to the index
			drawOperator(painter,
				std::get<0>(outIndexes)[oid] = boxToQRectF(dotBoxToQtBox(dotBox)),
				STR(model->getOperatorKind(oid)),
				{},
				""
			);
			oid++;
		}
	}

	{ // draw tensor splines
		painter.setPen(clrTensorLine);
		for (auto &tensorLineCubicSpline : tensorLineCubicSplines)
			if (!tensorLineCubicSpline.empty()) // not all tensors are between operators
				drawTensorSplines(painter, dotVectorQPointToQtVectorQPoint(tensorLineCubicSpline));
	}

	{ // draw tensor labels
		painter.setPen(clrTensorLabel);
		painter.setFont(Fonts::fontTensorLabel);
		QFontMetrics fm(Fonts::fontTensorLabel);
		PluginInterface::TensorId tid = 0;
		for (auto &tensorLabelPosition : tensorLabelPositions) {
			if (tensorLabelPosition != QPointF()) { // not all tensors are between operators
				QRectF outTextRect;
				drawTensorLabel(painter,
					dotQPointToQtQPoint(tensorLabelPosition),
					STR(model->getTensorShape(tid)),
					fm,
					outTextRect
				);
				// add to the index
				std::get<1>(outIndexes)[tid] = outTextRect;
			}
			tid++;
		}
	}

	painter.end();

	return generator.ba;
}

} // SvgGraphics
