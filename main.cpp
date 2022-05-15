// Copyright (C) 2022 by Yuri Victorovich. All rights reserved.

#include "main-window.h"

#include "plugin-interface.h"
#include "plugin-manager.h"
#include "misc.h"

#include "svg-graphics-generator.h"

#include <QApplication>
#include <QSettings>

#include <memory>


QSettings appSettings("NN Insight"); // a global settings object allowing the app to have persistent settings

int main(int argc, char **argv) {

	if (argc != 2)
		FAIL("Usage: nn-insight {network.tflite}")

	QApplication app(argc, argv);

	std::unique_ptr<MainWindow> mainWindow(new MainWindow);
	mainWindow->loadModelFile(argv[1]);
	mainWindow->show();
	mainWindow.release(); // MainWindow objects are self-destroyed on close

	return app.exec();
}

