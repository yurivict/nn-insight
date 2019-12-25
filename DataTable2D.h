#pragma once

#include <QWidget>
#include <QLabel>
#include <QTableView>
#include <QComboBox>
#include <QCheckBox>
#include <QStackedWidget>
#include <QScrollArea>
#include <QAbstractTableModel>
#include <QVBoxLayout>
#include <QHBoxLayout>

#include "nn-types.h"

#include <vector>
#include <memory>

class DataTable2D : public QWidget {
	Q_OBJECT

	TensorShape      shape;
	const float*     data;
	unsigned         dimVertical;
	unsigned         dimHorizontal;

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
	QCheckBox                              viewDataAsBwImageCheckBox;
	QStackedWidget                       dataViewStackWidget; // to be able to stack various views of the same data
	QTableView                             tableView;
	std::unique_ptr<QAbstractTableModel>     tableModel;
	QScrollArea                            imageViewScrollArea;
	QLabel                                   imageView;

public:
	DataTable2D(const TensorShape &shape_, const float *data_, QWidget *parent);

private: // internals
	std::vector<unsigned> mkIdxs() const;
};
