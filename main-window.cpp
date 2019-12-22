
#include "main-window.h"
#include "nn-model-viewer.h"
#include "plugin-interface.h"
#include "plugin-manager.h"

#include "svg-graphics-generator.h"
#include "util.h"
#include "misc.h"

#include <QEvent>
#include <QWheelEvent>
#include <QDebug>
#include <QSvgRenderer>

#include <memory>


MainWindow::MainWindow()
: mainSplitter(this)
,   svgScrollArea(&mainSplitter)
,     svgWidget(&mainSplitter)
,   rhsWidget(&mainSplitter)
,      rhsLayout(&rhsWidget)
,      blankRhsLabel("Select some operator", &rhsWidget)
, plugin(nullptr)
{
	setCentralWidget(&mainSplitter);
	mainSplitter.addWidget(&svgScrollArea);
	  svgScrollArea.setWidget(&svgWidget);
	mainSplitter.addWidget(&rhsWidget);

	rhsLayout.addWidget(&blankRhsLabel);

	svgScrollArea.setWidgetResizable(true);
	svgScrollArea.setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
	svgScrollArea.setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOn);

	svgScrollArea.setSizeAdjustPolicy(QAbstractScrollArea::AdjustToContents);

	qDebug() << "SVG: defaultSize()=" << svgWidget.renderer()->defaultSize();
}

MainWindow::~MainWindow() {
	if (plugin)
		PluginManager::unloadPlugin(plugin);
}

bool MainWindow::loadModelFile(const QString &filePath) {
	// helpers
	auto endsWith = [](const std::string &fullString, const std::string &ending) {
		return
			(fullString.length() >= ending.length()+1)
			&&
			(0 == fullString.compare(fullString.length()-ending.length(), ending.length(), ending));
	};
	auto fileNameToPluginName = [&](const QString &filePath) {
		if (endsWith(Q2S(filePath), ".tflite"))
			return "tf-lite";
		else
			return (const char*)nullptr;
	};

	// file name -> plugin name
	auto pluginName = fileNameToPluginName(filePath);
	if (pluginName == nullptr)
		return Util::warningOk(this, CSTR("Couldn't find a plugin to open the file '" << Q2S(filePath) << "'"));

	// load the plugin
	plugin = PluginManager::loadPlugin(pluginName);
	if (!plugin)
		FAIL("failed to load the plugin '" << pluginName << "'")
	pluginInterface.reset(PluginManager::getInterface(plugin)());

	// load the model
	if (pluginInterface->open(Q2S(filePath)))
		PRINT("loaded the model '" << Q2S(filePath) << "' successfully")
	else
		FAIL("failed to load the model '" << Q2S(filePath) << "'")
	if (pluginInterface->numModels() != 1)
		FAIL("multi-model files aren't supported yet")
	model = pluginInterface->getModel(0);

	svgWidget.load(SvgGraphics::generateModelSvg(model));

	return true; // success
}

