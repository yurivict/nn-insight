#pragma once

#include <QTableView>
#include <QAbstractTableModel>

#include <memory>

#include "plugin-interface.h"

class OperatorsListWidget : public QTableView {
	Q_OBJECT

	std::unique_ptr<QAbstractTableModel>    tableModel;
	bool                                    self;        // to prevent signals from programmatically changed values

public: // constructor
	OperatorsListWidget(QWidget *parent);

private: // overridden
	void selectionChanged(const QItemSelection &selected, const QItemSelection &deselected) override;

public: // interface
	void setNnModel(const PluginInterface::Model *model);
	void clearNnModel();
	void selectOperator(PluginInterface::OperatorId operatorId);

signals:
	void operatorSelected(PluginInterface::OperatorId operatorId);
};
