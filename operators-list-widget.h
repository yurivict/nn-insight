#pragma once

#include <QTableView>
#include <QAbstractTableModel>

#include <memory>

#include "plugin-interface.h"

class OperatorsListWidget : public QTableView {
	Q_OBJECT

	std::unique_ptr<QAbstractTableModel>    tableModel;

public: // constructor
	OperatorsListWidget(QWidget *parent);

public: // interface
	void setNnModel(const PluginInterface::Model *model);
	void clearNnModel();
};
