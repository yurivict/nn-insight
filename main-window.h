#pragma once

#include <QMainWindow>
#include <QSplitter>
#include <QScrollArea>
#include "ZoomableSvgWidget.h"
#include <QLabel>
#include <QVBoxLayout>
class QEvent;

#include "plugin-manager.h"
#include "plugin-interface.h"



class MainWindow : public QMainWindow {
	Q_OBJECT

public:
	MainWindow();
	~MainWindow();

	bool loadModelFile(const QString &filePath);

private:
	// widgets
	QSplitter                        mainSplitter;
	QScrollArea                        svgScrollArea;
	ZoomableSvgWidget                  svgWidget;
	QWidget                            rhsWidget;
	QVBoxLayout                          rhsLayout;
	QLabel                               blankRhsLabel;

	const PluginManager::Plugin*     plugin;    // plugin in use for the model
	std::unique_ptr<PluginInterface> pluginInterface; // the file is opened through this handle
	const PluginInterface::Model*    model;     // the model from the file that is currently open
};

