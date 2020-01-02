// Copyright (C) 2020 by Yuri Victorovich. All rights reserved.

#include "main-window.h"

#include "plugin-interface.h"
#include "plugin-manager.h"
#include "misc.h"

#include "svg-graphics-generator.h"

#include <QApplication>

#include <QFile>

int main(int argc, char **argv) {

	if (argc != 2)
		FAIL("Usage: nn-insight {network.tflite}")

	QApplication app(argc, argv);

	MainWindow mainWindow;
	mainWindow.loadModelFile(argv[1]);
	mainWindow.show();

	return app.exec();
}

