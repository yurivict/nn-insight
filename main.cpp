
#include "main-window.h"

#include "plugin-interface.h"
#include "plugin-manager.h"
#include "misc.h"

#include "svg-graphics-generator.h"

#include <QApplication>

#include <QFile>

int main(int argc, char **argv) {

	QApplication app(argc, argv);
/*
	// XXX test plugins
	// load the plugin
	auto plugin = PluginManager::loadPlugin("tf-lite");
	auto pluginLibInterface = PluginManager::getInterface(plugin);
	PRINT("loaded plugin/got its library interface")
	//
	auto pluginInterface = pluginLibInterface();
	PRINT("instantiated the plugin: " << pluginInterface)
	//
	//auto succ = pluginInterface->open("/home/yuri/nn-models/android_graph-yolo.tflite");
	auto succ = pluginInterface->open("/home/yuri/nn-models/android_graph-yolo.tflite");
	PRINT("loaded the model, success=" << succ)
	//
	auto model = pluginInterface->getModel(0);
	PRINT("model: numIns=" << model->numInputs())
	PRINT("model: numOuts=" << model->numOutputs())
	PRINT("model: numOperators=" << model->numOperators())
	PRINT("model: numTensors=" << model->numTensors())
	auto svg = SvgGraphics::generateModelSvg(model);
	QFile file("my.svg");
	file.open(QIODevice::WriteOnly);
	file.write(svg);
	file.close();
	//
	delete pluginInterface;
	PRINT("unloaded the model")
	//
	PluginManager::unloadPlugin(plugin);
	exit(1);
	//
*/

	MainWindow mainWindow;
	mainWindow.loadModelFile(argv[1]);
	mainWindow.show();

	return app.exec();
}

