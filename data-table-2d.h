// Copyright (C) 2022 by Yuri Victorovich. All rights reserved.

#pragma once

#include <QWidget>
#include <QLabel>
#include <QTableView>
#include <QComboBox>
#include <QCheckBox>
#include <QStackedWidget>
#include <QScrollArea>
#include <QSpinBox>
#include <QAbstractTableModel>
#include <QVBoxLayout>
#include <QHBoxLayout>

#include "image-grid-widget.h"
#include "tensor.h"

#include <vector>
#include <memory>

// non-template base so that non-templated class can be used
class DataTable2DBase : public QWidget {
public:
	DataTable2DBase(QWidget *parent);
	virtual ~DataTable2DBase();
public: // interface
	virtual void dataChanged(const void *data) = 0; // notify the widget that the data changed (only the data itself, not the array shape)
};

template<typename T>
class DataTable2D : public DataTable2DBase {
	TensorShape      shape;
	const T*         data;
	unsigned         dimVertical;
	unsigned         dimHorizontal;
	bool             self;   // to prevent signals from programmatically changed values

	QVBoxLayout                          layout;
	QWidget                              headerWidget;
	QHBoxLayout                            headerLayout;
	QLabel                                 shapeLabel;
	QWidget                                shapeDimensionsWidget;
	QHBoxLayout                              shapeDimensionsLayout;
	std::vector<std::unique_ptr<QComboBox>>  shapeDimensionsComboBoxes;
	QLabel                                 dataRangeLabel;
	QLabel                                 colorSchemaLabel;
	QComboBox                              colorSchemaComboBox;
	QWidget                              header1Widget; // second line
	QHBoxLayout                            header1Layout;
	QWidget                                filler1Widget;
	QLabel                                 scaleBwImageLabel;
	QSpinBox                               scaleBwImageSpinBox;
	QCheckBox                              viewDataAsBwImageCheckBox;
	QStackedWidget                       dataViewStackWidget; // to be able to stack various views of the same data
	QTableView                             tableView;
	std::unique_ptr<QAbstractTableModel>     tableModel;
	QScrollArea                            imageViewScrollArea;
	ImageGridWidget                          imageView;
	bool                                     imageViewInitialized;

public: // constructor
	DataTable2D(const TensorShape &shape_, const T *data_, QWidget *parent);

public: // interface
	void dataChanged(const void *data) override; // notify the widget that the data changed (only the data itself, not the array shape)

private: // internals
	void dataChangedTyped(const T *data_); // notify the widget that the data changed (only the data itself, not the array shape)
	std::vector<unsigned> mkIdxs() const;
	void updateBwImageView(bool initialUpdate);
	void setShowImageViewMode(bool showImageView);
};
