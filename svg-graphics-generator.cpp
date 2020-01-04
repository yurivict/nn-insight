// Copyright (C) 2020 by Yuri Victorovich. All rights reserved.

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
#include <QBrush>

#include <vector>
#include <array>
#include <map>
#include <cmath>

#include <assert.h>

#include "svg-graphics-generator.h"
#include "plugin-interface.h"
#include "model-functions.h"
#include "misc.h"
#include "util.h"
#include "fonts.h"
#include "constant-values.h"
#include "colors.h"

namespace SvgGraphics {

class SvgGenerator : public QSvgGenerator {
public:
	QByteArray ba;
	QBuffer    buff;

	SvgGenerator(unsigned width, unsigned height, unsigned marginPixelsX, unsigned marginPixelsY, const char *title)
	: buff(&ba)
	{
		setOutputDevice(&buff);
		setResolution(Util::getScreenDPI());
		setViewBox(QRect(-marginPixelsX, -marginPixelsY, width+2*marginPixelsX, height+2*marginPixelsY));
		setTitle(title);
		//setDescription("");
	}
};

QByteArray generateModelSvg(const PluginInterface::Model *model, const std::array<std::vector<QRectF>*,4> outIndexes) {
	// options
	qreal  operatorBoxRadius          = 5;
	qreal  operatorBoxBorderWidth     = 2;
	QColor clrOperatorBorder          = Qt::black;
	QColor clrOperatorTitleText       = Qt::white;
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
	std::vector<ModelFunctions::Box2> operatorBoxes;
	std::map<PluginInterface::TensorId, ModelFunctions::Box2> inputBoxes;
	std::map<PluginInterface::TensorId, ModelFunctions::Box2> outputBoxes;
	std::vector<std::vector<std::vector<QPointF>>> tensorLineCubicSplines;
	std::vector<std::vector<QPointF>> tensorLabelPositions;
	{
		QFontMetrics fm(Fonts::fontOperatorTitle);
		qreal dpi = (qreal)Util::getScreenDPI();
		ModelFunctions::renderModelToCoordinates(model,
			QMarginsF(0.2, 0.1, 0.22, 0.1), // operator box margins in inches
			[model,&fm,dpi](PluginInterface::OperatorId oid) {
				auto szPixels = fm.size(Qt::TextSingleLine, S2Q(STR(model->getOperatorKind(oid))));
				return QSizeF(qreal(szPixels.width())/dpi, qreal(szPixels.height())/dpi);
			},
			[&fm,dpi](PluginInterface::TensorId tid) {
				auto szPixels = fm.size(Qt::TextSingleLine, S2Q(STR("Input#" << tid)));
				return QSizeF(qreal(szPixels.width())/dpi, qreal(szPixels.height())/dpi);
			},
			[&fm,dpi](PluginInterface::TensorId tid) {
				auto szPixels = fm.size(Qt::TextSingleLine, S2Q(STR("Output#" << tid)));
				return QSizeF(qreal(szPixels.width())/dpi, qreal(szPixels.height())/dpi);
			},
			bbox,
			operatorBoxes,
			inputBoxes,
			outputBoxes,
			tensorLineCubicSplines,
			tensorLabelPositions
		);
	}

	// generate the SVG file

	float marginPixelsX = ConstantValues::nnDisplayMarginInchesX*Util::getScreenDPI();
	float marginPixelsY = ConstantValues::nnDisplayMarginInchesY*Util::getScreenDPI();

	SvgGenerator generator(bbox[1][0]/*width*/, bbox[1][1]/*height*/, (unsigned)marginPixelsX, (unsigned)marginPixelsY, "NN Model");

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
	auto dotBoxToQtBox = [&](const ModelFunctions::Box2 &box2) {
		auto center = box2[0];
		auto size = box2[1];
		return ModelFunctions::Box2{{
			{center[0]-size[0]/2, dotYToQtY(center[1]+size[1]/2)}, // coord
			{size[0], size[1]}
		}};
	};
	auto boxToQRectF = [](const ModelFunctions::Box2 &box) {
		return QRectF(box[0][0], box[0][1], box[1][0], box[1][1]);
	};
	auto drawBox = [&](QPainter &painter,
	                        const QRectF &bbox,
	                        const std::string &title,
				QColor clrBackground,
	                        const std::vector<std::string> &inputDescriptions,
	                        const std::string &fusedDescription)
	{
		// border
		QPainterPath path;
		path.addRoundedRect(bbox, operatorBoxRadius, operatorBoxRadius);
		QPen pen(clrOperatorBorder, operatorBoxBorderWidth);
		painter.setPen(pen);
		painter.fillPath(path, clrBackground);
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
	outIndexes[0]->resize(model->numOperators());
	outIndexes[1]->resize(model->numTensors());
	// outIndexes[2] with inputs and outIndexes[3] with outputs will be push_back to

	{ // draw operator boxes
		PluginInterface::OperatorId oid = 0;
		for (auto &dotBox : operatorBoxes) {
			drawBox(painter,
				(*outIndexes[0])[oid] = boxToQRectF(dotBoxToQtBox(dotBox)),
				STR(model->getOperatorKind(oid)),
				Colors::getOperatorColor(model->getOperatorKind(oid)),
				{},
				""
			);
			oid++;
		}
	}

	// draw input boxes
	for (auto it : inputBoxes) {
		auto box = boxToQRectF(dotBoxToQtBox(it.second));
		outIndexes[2]->push_back(box);
		drawBox(painter,
			box,
			STR("Input#" << it.first/*tid*/),
			Qt::gray,
			{},
			""
		);
	}

	// draw output boxes
	for (auto it : outputBoxes) {
		auto box = boxToQRectF(dotBoxToQtBox(it.second));
		outIndexes[3]->push_back(box);
		drawBox(painter,
			box,
			STR("Output#" << it.first/*tid*/),
			Qt::gray,
			{},
			""
		);
	}

	{ // draw tensor splines
		painter.setPen(clrTensorLine);
		for (auto &tensorLineCubicSplineList : tensorLineCubicSplines)
			for (auto &tensorLineCubicSpline : tensorLineCubicSplineList) // not all tensors are between operators, so the spline list here can be empty
				drawTensorSplines(painter, dotVectorQPointToQtVectorQPoint(tensorLineCubicSpline));
	}

	{ // draw tensor labels
		painter.setPen(clrTensorLabel);
		painter.setFont(Fonts::fontTensorLabel);
		QFontMetrics fm(Fonts::fontTensorLabel);
		PluginInterface::TensorId tid = 0;
		for (auto &tensorLabelPositionsList : tensorLabelPositions) {
			for (auto &tensorLabelPosition : tensorLabelPositionsList) { // not all tensors are between operators, so the label list here can be empty
				QRectF outTextRect;
				drawTensorLabel(painter,
					dotQPointToQtQPoint(tensorLabelPosition),
					STR(model->getTensorShape(tid)),
					fm,
					outTextRect
				);
				// add to the index
				(*outIndexes[1])[tid] = outTextRect;
			}
			tid++;
		}
	}

	painter.end();

	// shift indexes by margins
	for (auto &outIndex : outIndexes)
		for (auto &rect : *outIndex)
			rect.adjust(marginPixelsX,marginPixelsY, marginPixelsX,marginPixelsY);

	return generator.ba;
}

QByteArray generateNnAppIcon() {
	SvgGenerator generator(128/*width*/, 128/*height*/, 0/*marginPixelsX*/, 0/*marginPixelsY*/, "Table Icon");

	// set of connected circles, all coordinates are w/in 0..1 range in x and y
	float radius = 0.11;
	float margin = 0.03;
	float MR = margin+radius;
	struct {
		QPointF pt;
		QColor  color;
	} circles[] = {
		{{MR, 0.3},   {100,153,237}},
		{{MR, 0.7},   {246,219,64}},
		{{0.5, MR},   {131,96,244}},
		{{0.5, 0.45}, {219,83,91}},
		{{0.5, 1-MR}, {113,210,223}},
		{{1-MR, 0.5}, {246,219,64}}
	};
	struct {
		unsigned idx1;
		unsigned idx2;
	} lines[] = {
		{0,2},
		{0,3},
		{0,4},
		{1,2},
		{1,4},
		{2,5},
		{3,4},
		{3,5},
		{4,5}
	};

	// paint
	QPainter painter;
	painter.begin(&generator);
	float coef = 128.;
	painter.setPen(QPen(Qt::black, 4));
	for (auto &c : circles) {
		painter.setBrush(c.color);
		painter.drawEllipse(c.pt*coef, radius*coef, radius*coef);
	}
	for (auto &l : lines) {
		auto pt1 = circles[l.idx1].pt;
		auto pt2 = circles[l.idx2].pt;
		QPointF unit = (pt2-pt1);
		unit /= std::sqrt(unit.x()*unit.x()+unit.y()*unit.y());
		pt1 += unit*radius;
		pt2 -= unit*radius;
		painter.drawLine(pt1*coef, pt2*coef);
	}
	painter.end();

	return generator.ba;
}

QByteArray generateTableIcon() {
	SvgGenerator generator(20/*width*/, 20/*height*/, 0/*marginPixelsX*/, 0/*marginPixelsY*/, "Table Icon");

	QPainter painter;
	//painter.setRenderHint(QPainter::Antialiasing);
	painter.begin(&generator);
	painter.setPen(Qt::black);
	float marginX = 2, marginY = 4;
	float l = 0  + marginX;
	float t = 0  + marginY;
	float r = 20 - marginX;
	float b = 20 - marginY;
	painter.drawRect(QRectF(l,t, (r-l),(b-t)));
	for (unsigned i=1; i<4; i++) {
		float y = t + float(i)*(b-t+1)/4;
		painter.drawLine(l,y, r,y);
	}
	for (unsigned i=1; i<3; i++) {
		float x = l + float(i)*(r-l+1)/3;
		painter.drawLine(x,t+(b-t)/4, x,b);
	}
	painter.end();

	return generator.ba;
}

QByteArray generateArrow(const QPointF &vec, QColor color, const ArrowParams arrowParams) {
	typedef qreal C;

	// determine the size: it is always a square
	unsigned dx = vec.x() > 0 ? vec.x() : -vec.x();
	unsigned dy = vec.y() > 0 ? vec.y() : -vec.y();
	unsigned SZ = dx > dy ? dx : dy;

	auto rotate = [](const QPointF pt, C sinAngle, C cosAngle) {
		return QPointF(pt.x()*cosAngle - pt.y()*sinAngle,
		               pt.x()*sinAngle + pt.y()*cosAngle);
	};

	SvgGenerator generator(SZ, SZ, 0/*marginPixelsX*/, 0/*marginPixelsY*/, "Arrow");

	// params
	const ArrowParams &p = arrowParams;

	// scale factor and angle
	C vecLen = std::sqrt(vec.x()*vec.x() + vec.y()*vec.y());
	C cosAngle = vec.x()/vecLen;
	C sinAngle = vec.y()/vecLen;

	// draw the arrow from 0,0 to 1,0
	std::vector<QPointF> pts;
	auto add = [&pts](qreal x, qreal y) {
		pts.push_back(QPointF(x, y));
	};
	add(0,                 -p.lineWidth/2);
	add(0,                 +p.lineWidth/2);
	add(1-p.headLengthIn,  +p.lineWidth/2);
	add(1-p.headLengthOut, +p.headWidth/2);
	add(1,                 0);
	add(1-p.headLengthOut, -p.headWidth/2);
	add(1-p.headLengthIn,  -p.lineWidth/2);
	add(0,                 -p.lineWidth/2);

	for (auto &pt : pts) {
		// move arrow to x in -0.5 -> +0.5
		pt += QPointF(-0.5, 0.);
		// rotate and center
		pt = rotate(pt, sinAngle, cosAngle);
		// scale to sz dimensions
		pt.rx() *= vecLen;
		pt.ry() *= vecLen;
		// shift to the location require (center of the box)
		pt.rx() += SZ/2;
		pt.ry() += SZ/2;
	}

	// pts -> QPainterPath
	QPainterPath path;
	int idx = 0;
	for (auto &pt : pts) {
		if (idx++ == 0)
			path.moveTo(pt);
		else
			path.lineTo(pt);
	}

	QPainter painter;
	painter.begin(&generator);
	painter.fillPath(path, QBrush(color));
	//painter.setPen(QPen(Qt::blue));
	//painter.drawRect(1,1, SZ-2,SZ-2);
	painter.end();

	return generator.ba;
}


} // SvgGraphics
