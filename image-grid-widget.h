// Copyright (C) 2020 by Yuri Victorovich. All rights reserved.

#pragma once

#include <QWidget>
#include <QGridLayout>
#include <QImage>
#include <QLabel>
#include <QString>

#include <functional>
#include <memory>
#include <tuple>
#include <vector>

class ImageGridWidget : public QWidget {
	Q_OBJECT

// parameters
	const float       horizontalSpacing  = 0.1; // in inches

// fields
	QGridLayout       layout;
	std::vector<std::vector<std::tuple<std::unique_ptr<QLabel>,std::unique_ptr<QLabel>>>> imageWidgets; // by row, by col: {label, image}

public: // types
	typedef std::tuple<QString,std::unique_ptr<uchar>,QImage> ImageData;

public: // constructor
	ImageGridWidget(QWidget *parent);

public: // interface
	void setSizesAndData(size_t width, size_t height, size_t lastRow, std::function<ImageData(unsigned x, unsigned y)> cbGetImage);
};
