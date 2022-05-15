// Copyright (C) 2022 by Yuri Victorovich. All rights reserved.

#include "image-grid-widget.h"

#include <QFontMetrics>

#include <algorithm>

#include <assert.h>

#include "util.h"

// NN Insight sometimes shows garbage instead of BW images due to this big: "QLabel::setPixmap fails and QLabel displays garbage", see https://bugreports.qt.io/browse/QTBUG-81516

ImageGridWidget::ImageGridWidget(QWidget *parent)
: QWidget(parent)
, layout(this)
{
}

void ImageGridWidget::setSizesAndData(size_t width, size_t height, size_t lastRow, std::function<ImageGridWidget::ImageData(unsigned x, unsigned y)> cbGetImage) {
#if defined(WITH_ASSERTS)
	unsigned imgWidth = 0;
#endif
	unsigned imgHeight = 0;

	std::unique_ptr<QFontMetrics> labelFontMetrics;

	imageWidgets.resize(height);
	for (unsigned row = 0; row < height; row++) {
		auto rowWidth = (row+1)<height ? width : lastRow;
		imageWidgets[row].resize(rowWidth);
		for (unsigned col = 0; col < rowWidth; col++) {
			auto img = cbGetImage(col, row);
			auto &imgLabelString = std::get<0>(img);
			auto &imgImage       = std::get<2>(img);
			if (row==0 && col==0) {
#if defined(WITH_ASSERTS)
				imgWidth = imgImage.width();
#endif
				imgHeight = imgImage.height();
			} else { // all images are assumed to have the same size, because they are slices of the same tensor
				assert(imgImage.width() == imgWidth);
				assert(imgImage.height() == imgHeight);
			}
			auto &widgets = imageWidgets[row][col];
			auto &labelWidget = std::get<0>(widgets);
			auto &imageWidget = std::get<1>(widgets);

			// create and set up widgets
			labelWidget.reset(new QLabel(imgLabelString, this));
			imageWidget.reset(new QLabel(this));
			imageWidget.get()->setPixmap(QPixmap::fromImage(imgImage));
			if (!labelFontMetrics)
				labelFontMetrics.reset(new QFontMetrics(labelWidget->font()));

			{ // set widths to be the same
				auto maxWidth = std::max(labelFontMetrics->size(0/*flags*/, labelWidget->text()).width(), imgImage.width());
				labelWidget->setMinimumWidth(maxWidth);
				imageWidget->setMinimumWidth(maxWidth);
			}

			// set policies, widget margins, and tooltips
			for (auto l : {&labelWidget, &imageWidget}) {
				l->get()->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
				l->get()->setContentsMargins(0,0,0,0);
				l->get()->setToolTip(QString(tr("One slice of the activation layer tensor %1")).arg(imgLabelString));
			}
			labelWidget->setAlignment(Qt::AlignCenter|Qt::AlignBottom); // bottom - closer to the image
			imageWidget->setAlignment(Qt::AlignCenter); // in case it is smaller than the label
			//labelWidget->setStyleSheet("QLabel { background-color : red; color : blue; }");

			// add widgets to the layout
			layout.addWidget(labelWidget.get(), 2*row+0, col);
			layout.addWidget(imageWidget.get(), 2*row+1, col);
		}
	}

	// find total label height
	unsigned totalLabelHeight = 0;
	for (auto &widgetRow : imageWidgets)
		totalLabelHeight += std::get<0>(widgetRow[0])->height();

	// find total image cell widths
	unsigned totalImageCellWidths = 0;
	{
		std::vector<unsigned> maxWidths;
		maxWidths.resize(width);
		for (auto &widgetRow : imageWidgets) {
			unsigned col = 0;
			for (auto &widgetCell : widgetRow) {
				maxWidths[col] = std::max(maxWidths[col], (unsigned)std::max(std::get<0>(widgetCell)->width(), std::get<1>(widgetCell)->width()));
				col++;
			}
		}
		for (auto w : maxWidths)
			totalImageCellWidths += w;
	}

	// set spacing
	unsigned hSpacing = Util::getScreenDPI()*horizontalSpacing;
	layout.setHorizontalSpacing(hSpacing);
	layout.setVerticalSpacing  (0); // the layout has no spacing, instead spacing is provided by the spacing widgets

	// TODO add margins?
	layout.setContentsMargins(0,0,0,0);

	// set size
	resize(
		totalImageCellWidths + hSpacing*(width-1),
		totalLabelHeight + imgHeight*height
	);
}
