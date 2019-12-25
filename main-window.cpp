
#include "main-window.h"
#include "nn-model-viewer.h"
#include "plugin-interface.h"
#include "plugin-manager.h"
#include "model-functions.h"

#include "svg-graphics-generator.h"
#include "util.h"
#include "misc.h"
#include "nn-types.h"
#include "image.h"

#include <QEvent>
#include <QWheelEvent>
#include <QDebug>
#include <QSvgRenderer>
#include <QPushButton>
#include <QFontMetrics>
//#include <QMargins>
#include <QDesktopWidget>
#include <QApplication>
#include <QFileDialog>
#include <QPixmap>

#include <assert.h>

#include <memory>

#if defined(USE_PERFTOOLS)
#include <gperftools/malloc_extension.h>
#endif


MainWindow::MainWindow()
: mainSplitter(this)
,   svgScrollArea(&mainSplitter)
,     svgWidget(&mainSplitter)
,   rhsWidget(&mainSplitter)
,      rhsLayout(&rhsWidget)
,      sourceWidget("Source Data", &rhsWidget)
,        sourceLayout(&sourceWidget)
,        sourceDetails(&sourceWidget)
,          sourceDetailsLayout(&sourceDetails)
,          sourceImageFileName(&sourceDetails)
,          sourceImageSize(&sourceDetails)
,          sourceFiller(&sourceDetails)
,          computeButton("Compute", &sourceDetails)
,        sourceImage(&sourceWidget)
,      detailsStack(&rhsWidget)
,        noDetails("Details", &detailsStack)
,        operatorDetails(&detailsStack)
,          operatorDetailsLayout(&operatorDetails)
,          operatorTypeLabel("Operator Type", &operatorDetails)
,          operatorTypeValue(&operatorDetails)
,          operatorOptionsLabel("Options", &operatorDetails)
,          operatorInputsLabel("Inputs", &operatorDetails)
,          operatorOutputsLabel("Outputs", &operatorDetails)
,          operatorComplexityLabel("Complexity", &operatorDetails)
,          operatorComplexityValue(&operatorDetails)
,        tensorDetails(&detailsStack)
,      blankRhsLabel("Select some operator", &rhsWidget)
, menuBar(this)
, statusBar(this)
#if defined(USE_PERFTOOLS)
,   memoryUseLabel(&statusBar)
,   memoryUseTimer(&statusBar)
#endif
, plugin(nullptr)
{
	// window size and position
	if (true) { // set it to center on the screen until we will have persistent app options
		QDesktopWidget *desktop = QApplication::desktop();
		resize(desktop->width()*3/4, desktop->height()*3/4); // initialize our window to be 3/4 of the size of the desktop
		move((desktop->width() - width())/2, (desktop->height() - height())/2);
	}

	// set up widgets
	setCentralWidget(&mainSplitter);
	mainSplitter.addWidget(&svgScrollArea);
	  svgScrollArea.setWidget(&svgWidget);
	mainSplitter.addWidget(&rhsWidget);

	rhsLayout.addWidget(&sourceWidget);
	  sourceLayout.addWidget(&sourceDetails);
	    sourceDetailsLayout.addWidget(&sourceImageFileName);
	    sourceDetailsLayout.addWidget(&sourceImageSize);
	    sourceDetailsLayout.addWidget(&sourceFiller);
	    sourceDetailsLayout.addWidget(&computeButton);
	  sourceLayout.addWidget(&sourceImage);
	rhsLayout.addWidget(&detailsStack);
	rhsLayout.addWidget(&blankRhsLabel);
	detailsStack.addWidget(&noDetails);
	detailsStack.addWidget(&operatorDetails);
		operatorDetails.setLayout(&operatorDetailsLayout);
	detailsStack.addWidget(&tensorDetails);

	svgScrollArea.setWidgetResizable(true);
	svgScrollArea.setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
	svgScrollArea.setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOn);

	setMenuBar(&menuBar);
	setStatusBar(&statusBar);
#if defined(USE_PERFTOOLS)
	statusBar.addWidget(&memoryUseLabel);
#endif

	// tooltips
	operatorTypeLabel.setToolTip("Operator type: what kind of operation does it perform");

	// size policies
	svgScrollArea       .setSizeAdjustPolicy(QAbstractScrollArea::AdjustToContents);
	sourceImageFileName .setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
	sourceImageSize     .setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
	//sourceFiller .setSizePolicy(QSizePolicy::Fixed, QSizePolicy::MinimumExpanding);
	sourceImage         .setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
	detailsStack        .setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);

	// widget options and flags
	sourceWidget.hide(); // hidden by default
	noDetails.setEnabled(false); // always grayed out

	// fonts
	for (auto widget : {&operatorTypeLabel, &operatorOptionsLabel, &operatorInputsLabel, &operatorOutputsLabel, &operatorComplexityLabel})
		widget->setStyleSheet("font-weight: bold;");

	// connect signals
	connect(&svgWidget, &ZoomableSvgWidget::mousePressOccurred, [this](QPointF pt) {
		if (model) {
			auto searchResult = findObjectAtThePoint(pt);
			if (searchResult.operatorId != -1)
				showOperatorDetails((PluginInterface::OperatorId)searchResult.operatorId);
			else if (searchResult.tensorId != -1)
				showTensorDetails((PluginInterface::TensorId)searchResult.tensorId);
			else {
				// no object was found: ignore the signal
			}
		}
	});
	connect(&computeButton, &QAbstractButton::pressed, [this]() {
		// allocate tensors array
		if (!tensorData) {
			tensorData.reset(new std::vector<std::shared_ptr<const float>>);
			tensorData->resize(model->numTensors());
		}
		// find the model input
		auto modelInputs = model->getInputs();
		if (modelInputs.size() != 1) {
			Util::warningOk(this, CSTR("We only support models with a single input, the current model has " << modelInputs.size() << " inputs"));
			return;
		}
		// resize the source image
		if (!(*tensorData.get())[modelInputs[0]]) {
			TensorShape requiredShape = tensorGetLastDims(model->getTensorShape(modelInputs[0]), 3);
			auto &sharedPtrInput = (*tensorData.get())[modelInputs[0]];
			if (sourceTensorShape != requiredShape)
				sharedPtrInput.reset(Image::resizeImage(sourceTensorData.get(), sourceTensorShape, requiredShape));
			else
				sharedPtrInput = sourceTensorData;
		}
	});

	// monitor memory use
#if defined(USE_PERFTOOLS)
	connect(&memoryUseTimer, &QTimer::timeout, [this]() {
		size_t inuseBytes = 0;
		(void)MallocExtension::instance()->GetNumericProperty("generic.current_allocated_bytes", &inuseBytes);
		memoryUseLabel.setText(QString("Memory use: %1 bytes").arg(S2Q(Util::formatUIntHumanReadable(inuseBytes))));
	});
	memoryUseTimer.start(1000);
#endif

	// add menus
	auto fileMenu = menuBar.addMenu(tr("&File"));
	fileMenu->addAction("Open Image", [this]() {
		QString fileName = QFileDialog::getOpenFileName(this,
			"Open image file", "",
			"Image (*.png);;All Files (*)"
		);
		if (fileName != "")
			openImageFile(fileName);
	});
	fileMenu->addAction("Open NN file", []() {
		PRINT("Open NN")
	});
	fileMenu->addAction("Close Image", [this]() {
		closeImage();
	});
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

	// render the model as an SVG image
	svgWidget.load(SvgGraphics::generateModelSvg(model, {modelIndexes.allOperatorBoxes, modelIndexes.allTensorLabelBoxes}));

	// set window title
	setWindowTitle(QString("NN Insight: %1 (%2)").arg(filePath).arg(S2Q(Util::formatFlops(ModelFunctions::computeModelFlops(model)))));

	return true; // success
}

/// private methods

MainWindow::AnyObject MainWindow::findObjectAtThePoint(const QPointF &pt) {
	// XXX ad hoc algorithm until we find some good geoindexing implementation

	// operator?
	for (PluginInterface::OperatorId oid = 0, oide = modelIndexes.allOperatorBoxes.size(); oid < oide; oid++)
		if (modelIndexes.allOperatorBoxes[oid].contains(pt))
			return {(int)oid,-1};

	// tensor label?
	for (PluginInterface::TensorId tid = 0, tide = modelIndexes.allTensorLabelBoxes.size(); tid < tide; tid++)
		if (modelIndexes.allTensorLabelBoxes[tid].contains(pt))
			return {-1,(int)tid};

	return {-1,-1}; // not found
}

void MainWindow::showOperatorDetails(PluginInterface::OperatorId operatorId) {
	removeTableIfAny();
	// switch to the details page, set title
	detailsStack.setCurrentIndex(/*page#1*/1);
	operatorDetails.setTitle(QString("Operator#%1").arg(operatorId));

	// clear items
	while (operatorDetailsLayout.count() > 0)
		operatorDetailsLayout.removeItem(operatorDetailsLayout.itemAt(0));
	tempDetailWidgets.clear();

	// helper
	auto addTensorLines = [this](auto &tensors, unsigned &row) {
		for (auto t : tensors) {
			row++;
			// tensor number
			auto label = new QLabel(QString("tensor#%1:").arg(t), &operatorDetails);
			label->setToolTip("Tensor number");
			label->setAlignment(Qt::AlignRight);
			tempDetailWidgets.push_back(std::unique_ptr<QWidget>(label));
			operatorDetailsLayout.addWidget(label,         row,   0/*column*/);
			// tensor name
			label = new QLabel(S2Q(model->getTensorName(t)), &operatorDetails);
			label->setToolTip("Tensor name");
			label->setAlignment(Qt::AlignLeft);
			tempDetailWidgets.push_back(std::unique_ptr<QWidget>(label));
			operatorDetailsLayout.addWidget(label,         row,   1/*column*/);
			// tensor shape
			auto describeShape = [](const TensorShape &shape) {
				auto flatSize = tensorFlatSize(shape);
				return STR(shape <<
				         " (" <<
				             Util::formatUIntHumanReadable(flatSize) << " floats, " <<
				             Util::formatUIntHumanReadable(flatSize*sizeof(float)) << " bytes" <<
				          ")"
				);
			};
			label = new QLabel(S2Q(describeShape(model->getTensorShape(t))), &operatorDetails);
			label->setToolTip("Tensor shape and data size");
			label->setAlignment(Qt::AlignLeft);
			tempDetailWidgets.push_back(std::unique_ptr<QWidget>(label));
			operatorDetailsLayout.addWidget(label,         row,   2/*column*/);
			// has buffer? is variable?
			bool isInput = Util::isValueIn(model->getInputs(), t);
			bool isOutput = Util::isValueIn(model->getOutputs(), t);
			auto hasStaticData = model->getTensorHasData(t);
			auto isVariable = model->getTensorIsVariableFlag(t);
			label = new QLabel(QString("<%1>").arg(
				isInput ? "input"
				: isOutput ? "output"
				: hasStaticData ? "static tensor"
				: isVariable ? "variable" : "computed"),
				&operatorDetails);
			label->setToolTip("Tensor type");
			label->setAlignment(Qt::AlignCenter);
			label->setStyleSheet("font: italic");
			tempDetailWidgets.push_back(std::unique_ptr<QWidget>(label));
			operatorDetailsLayout.addWidget(label,         row,   3/*column*/);
			// button
			if (hasStaticData || (tensorData && (*tensorData.get())[t])) {
				auto button = new QPushButton("➞", &operatorDetails);
				button->setContentsMargins(0,0,0,0);
				button->setStyleSheet("color: blue;");
				button->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
				//button->setMaximumSize(QFontMetrics(button->font()).size(Qt::TextSingleLine, button->text()).grownBy(QMargins(4,0,4,0)));
				button->setMaximumSize(QFontMetrics(button->font()).size(Qt::TextSingleLine, button->text())+QSize(8,0));
				button->setToolTip("Show the tensor data as a table");
				tempDetailWidgets.push_back(std::unique_ptr<QWidget>(button));
				operatorDetailsLayout.addWidget(button,         row,   4/*column*/);
				connect(button, &QAbstractButton::pressed, [this,t,hasStaticData]() {
					removeTableIfAny();
					// show table
					auto tableShape = model->getTensorShape(t);
					switch (tensorNumMultiDims(tableShape)) {
					case 0:
						PRINT("WARNING tensor with all ones encountered, it is meaningless in the NN models context")
						break;
					case 1:
						// TODO DataTable1D
						break;
					default: {
						dataTable.reset(new DataTable2D(tableShape,
							hasStaticData ? model->getTensorData(t) : (*tensorData.get())[t].get(),
							&rhsWidget));
						rhsLayout.addWidget(dataTable.get());
						blankRhsLabel.hide();
						break;
					}}
				});
			}
		}
	};

	// read operator inputs/outputs from the model
	std::vector<PluginInterface::TensorId> inputs, outputs;
	model->getOperatorIo(operatorId, inputs, outputs);

	// add items
	unsigned row = 0;
	operatorDetailsLayout.addWidget(&operatorTypeLabel,          row,   0/*column*/);
	operatorDetailsLayout.addWidget(&operatorTypeValue,          row,   1/*column*/);
	row++;
	operatorDetailsLayout.addWidget(&operatorOptionsLabel,       row,   0/*column*/);
	{
		std::unique_ptr<PluginInterface::OperatorOptionsList> opts(model->getOperatorOptions(operatorId));
		for (auto &opt : *opts) {
			row++;
			// option name
			auto label = new QLabel(S2Q(STR(opt.name)), &operatorDetails);
			label->setToolTip("Option name");
			label->setAlignment(Qt::AlignRight);
			tempDetailWidgets.push_back(std::unique_ptr<QWidget>(label));
			operatorDetailsLayout.addWidget(label,               row,   0/*column*/);
			// option type
			label = new QLabel(S2Q(STR("<" << opt.value.type << ">")), &operatorDetails);
			label->setToolTip("Option type");
			label->setAlignment(Qt::AlignLeft);
			label->setStyleSheet("font: italic");
			tempDetailWidgets.push_back(std::unique_ptr<QWidget>(label));
			operatorDetailsLayout.addWidget(label,               row,   1/*column*/);
			// option value
			label = new QLabel(S2Q(STR(opt.value)), &operatorDetails);
			label->setToolTip("Option value");
			label->setAlignment(Qt::AlignLeft);
			tempDetailWidgets.push_back(std::unique_ptr<QWidget>(label));
			operatorDetailsLayout.addWidget(label,               row,   2/*column*/);
		}
		if (opts->empty()) {
			row++;
			auto label = new QLabel("-none-", &operatorDetails);
			label->setAlignment(Qt::AlignRight);
			tempDetailWidgets.push_back(std::unique_ptr<QWidget>(label));
			operatorDetailsLayout.addWidget(label,               row,   0/*column*/);
		}
	}
	row++;
	operatorDetailsLayout.addWidget(&operatorInputsLabel,        row,   0/*column*/);
	addTensorLines(inputs, row);
	row++;
	operatorDetailsLayout.addWidget(&operatorOutputsLabel,       row,   0/*column*/);
	addTensorLines(outputs, row);
	row++;
	operatorDetailsLayout.addWidget(&operatorComplexityLabel,    row,   0/*column*/);
	operatorDetailsLayout.addWidget(&operatorComplexityValue,    row,   1/*column*/);

	// set texts
	operatorTypeValue.setText(S2Q(STR(model->getOperatorKind(operatorId))));
	operatorComplexityValue.setText(S2Q(Util::formatFlops(ModelFunctions::computeOperatorFlops(model, operatorId))));
}

void MainWindow::showTensorDetails(PluginInterface::TensorId tensorId) {
	removeTableIfAny();
	detailsStack.setCurrentIndex(/*page#*/2);
	tensorDetails.setTitle(QString("Tensor#%1: %2").arg(tensorId).arg(S2Q(model->getTensorName(tensorId))));
}

void MainWindow::removeTableIfAny() {
	if (dataTable.get()) {
		rhsLayout.removeWidget(dataTable.get());
		dataTable.reset(nullptr);
		blankRhsLabel.show();
	}
}

void MainWindow::openImageFile(const QString &imageFileName) {
	// enable widgets, show image
	sourceWidget.show();
	sourceImage.setPixmap(QPixmap(imageFileName));
	// read the image as tensor
	sourceTensorData.reset(Image::readPngImageFile(Q2S(imageFileName), sourceTensorShape));
	// set info on the screen
	sourceImageFileName.setText(QString("File name: %1").arg(imageFileName));
	sourceImageSize.setText(QString("Image size: %1").arg(S2Q(STR(sourceTensorShape))));
}

void MainWindow::closeImage() {
	sourceWidget.hide();
	sourceImage.setPixmap(QPixmap());
	sourceTensorData = nullptr;
	tensorData.reset(nullptr);
}

