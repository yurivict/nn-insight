#pragma once

#include <QWidget>
#include <QLabel>
#include <QTableView>
#include <QComboBox>
#include <QAbstractTableModel>
#include <QVBoxLayout>
#include <QHBoxLayout>

#include "nn-types.h"

#include <memory>

class DataTable2D : public QWidget {
	Q_OBJECT

	TensorShape      shape;
	const float*     data;

	QVBoxLayout                          layout;
	QWidget                              headerWidget;
	QHBoxLayout                            headerLayout;
	QLabel                                 shapeLabel;
	QLabel                                 dataRangeLabel;
	QLabel                                 colorSchemaLabel;
	QComboBox                              colorSchemaComboBox;
	QTableView                           tableView;
	std::unique_ptr<QAbstractTableModel>   tableModel;

public:
	DataTable2D(const TensorShape &shape_, const float *data_, QWidget *parent);
};
