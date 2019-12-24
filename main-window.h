#pragma once

#include <QMainWindow>
#include <QSplitter>
#include <QScrollArea>
#include <QStackedWidget>
#include <QGroupBox>
#include "ZoomableSvgWidget.h"
#include "DataTable2D.h"
#include <QLabel>
#include <QVBoxLayout>
#include <QGridLayout>
#include <QStatusBar>
#include <QRectF>
class QEvent;
#if defined(USE_PERFTOOLS)
#include <QTimer>
#endif

#include "plugin-manager.h"
#include "plugin-interface.h"

#include <vector>
#include <memory>

class MainWindow : public QMainWindow {
	Q_OBJECT

public:
	MainWindow();
	~MainWindow();

	bool loadModelFile(const QString &filePath);

private: // fields
	// widgets
	QSplitter                        mainSplitter;
	QScrollArea                        svgScrollArea;
	ZoomableSvgWidget                  svgWidget;
	QWidget                            rhsWidget;
	QVBoxLayout                          rhsLayout;
	QStackedWidget                       detailsStack;
	QGroupBox                              noDetails;       // page#0
	QGroupBox                              operatorDetails; // page#1
	QGridLayout                              operatorDetailsLayout;
	QLabel                                   operatorTypeLabel;
	QLabel                                   operatorTypeValue;
	QLabel                                   operatorInputsLabel;
	QLabel                                   operatorOutputsLabel;
	QLabel                                   operatorComplexityLabel;
	QLabel                                   operatorComplexityValue;
	QGroupBox                              tensorDetails;   // page#2
	QLabel                               blankRhsLabel; // leftover label
	std::unique_ptr<DataTable2D>         dataTable;

	QStatusBar                     statusBar;
#if defined(USE_PERFTOOLS)
	QLabel                         memoryUseLabel;
	QTimer                         memoryUseTimer;
#endif

	const PluginManager::Plugin*     plugin;    // plugin in use for the model
	std::unique_ptr<PluginInterface> pluginInterface; // the file is opened through this handle
	const PluginInterface::Model*    model;     // the model from the file that is currently open

	struct {
		std::vector<QRectF> allOperatorBoxes; // indexed based on OperatorId
		std::vector<QRectF> allTensorLabelBoxes; // indexed based on TensorId
	} modelIndexes;
	std::vector<std::unique_ptr<QWidget>>   tempDetailWidgets;

private: // types
	struct AnyObject {
		int operatorId;
		int tensorId;
	};

private: // private methods
	AnyObject findObjectAtThePoint(const QPointF &pt);
	void showOperatorDetails(PluginInterface::OperatorId operatorId);
	void showTensorDetails(PluginInterface::TensorId tensorId);
	void removeTableIfAny();
};

