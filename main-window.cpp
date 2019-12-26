
#include "main-window.h"
#include "nn-model-viewer.h"
#include "plugin-interface.h"
#include "plugin-manager.h"
#include "model-functions.h"

#include "svg-graphics-generator.h"
#include "util.h"
#include "misc.h"
#include "nn-types.h"
#include "nn-operators.h"
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

#include <map>
#include <memory>

#if defined(USE_PERFTOOLS)
#include <gperftools/malloc_extension.h>
#endif

/// local enums and values

enum ConvolutionEffect {
	ConvolutionEffect_None,
	ConvolutionEffect_Blur_3x3,
	ConvolutionEffect_Blur_5x5,
	ConvolutionEffect_Gaussian_3x3,
	ConvolutionEffect_Motion_3x3,
	ConvolutionEffect_Sharpen_3x3
};
#define THR 1./13.
#define S01 1./16.
#define S02 2./16.
#define S04 4./16.
#define TTT 1./3.
static const std::map<ConvolutionEffect, std::tuple<TensorShape,std::vector<float>>> convolutionEffects = {
	{ConvolutionEffect_None, {{},{}}},
	{ConvolutionEffect_Blur_3x3, {{3,3,3,3}, {
		0.0,0.0,0.0, 0.2,0.0,0.0, 0.0,0.0,0.0,
		0.2,0.0,0.0, 0.2,0.0,0.0, 0.2,0.0,0.0,
		0.0,0.0,0.0, 0.2,0.0,0.0, 0.0,0.0,0.0,

		0.0,0.0,0.0, 0.0,0.2,0.0, 0.0,0.0,0.0,
		0.0,0.2,0.0, 0.0,0.2,0.0, 0.0,0.2,0.0,
		0.0,0.0,0.0, 0.0,0.2,0.0, 0.0,0.0,0.0,

		0.0,0.0,0.0, 0.0,0.0,0.2, 0.0,0.0,0.0,
		0.0,0.0,0.2, 0.0,0.0,0.2, 0.0,0.0,0.2,
		0.0,0.0,0.0, 0.0,0.0,0.2, 0.0,0.0,0.0
	}}},
	{ConvolutionEffect_Blur_5x5, {{3,5,5,3}, {
		0.0,0.0,0.0, 0.0,0.0,0.0, THR,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
		0.0,0.0,0.0, THR,0.0,0.0, THR,0.0,0.0, THR,0.0,0.0, 0.0,0.0,0.0,
		THR,0.0,0.0, THR,0.0,0.0, THR,0.0,0.0, THR,0.0,0.0, THR,0.0,0.0,
		0.0,0.0,0.0, THR,0.0,0.0, THR,0.0,0.0, THR,0.0,0.0, 0.0,0.0,0.0,
		0.0,0.0,0.0, 0.0,0.0,0.0, THR,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,

		0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,THR,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
		0.0,0.0,0.0, 0.0,THR,0.0, 0.0,THR,0.0, 0.0,THR,0.0, 0.0,0.0,0.0,
		0.0,THR,0.0, 0.0,THR,0.0, 0.0,THR,0.0, 0.0,THR,0.0, 0.0,THR,0.0,
		0.0,0.0,0.0, 0.0,THR,0.0, 0.0,THR,0.0, 0.0,THR,0.0, 0.0,0.0,0.0,
		0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,THR,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,

		0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,THR, 0.0,0.0,0.0, 0.0,0.0,0.0,
		0.0,0.0,0.0, 0.0,0.0,THR, 0.0,0.0,THR, 0.0,0.0,THR, 0.0,0.0,0.0,
		0.0,0.0,THR, 0.0,0.0,THR, 0.0,0.0,THR, 0.0,0.0,THR, 0.0,0.0,THR,
		0.0,0.0,0.0, 0.0,0.0,THR, 0.0,0.0,THR, 0.0,0.0,THR, 0.0,0.0,0.0,
		0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,THR, 0.0,0.0,0.0, 0.0,0.0,0.0
	}}},
	{ConvolutionEffect_Gaussian_3x3, {{3,3,3,3}, {
		S01,0.0,0.0, S02,0.0,0.0, S01,0.0,0.0,
		S02,0.0,0.0, S04,0.0,0.0, S02,0.0,0.0,
		S01,0.0,0.0, S02,0.0,0.0, S01,0.0,0.0,

		0.0,S01,0.0, 0.0,S02,0.0, 0.0,S01,0.0,
		0.0,S02,0.0, 0.0,S04,0.0, 0.0,S02,0.0,
		0.0,S01,0.0, 0.0,S02,0.0, 0.0,S01,0.0,

		0.0,0.0,S01, 0.0,0.0,S02, 0.0,0.0,S01,
		0.0,0.0,S02, 0.0,0.0,S04, 0.0,0.0,S02,
		0.0,0.0,S01, 0.0,0.0,S02, 0.0,0.0,S01
	}}},
	{ConvolutionEffect_Motion_3x3, {{3,3,3,3}, {
		TTT,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
		0.0,0.0,0.0, TTT,0.0,0.0, 0.0,0.0,0.0,
		0.0,0.0,0.0, 0.0,0.0,0.0, TTT,0.0,0.0,

		0.0,TTT,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
		0.0,0.0,0.0, 0.0,TTT,0.0, 0.0,0.0,0.0,
		0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,TTT,0.0,

		0.0,0.0,TTT, 0.0,0.0,0.0, 0.0,0.0,0.0,
		0.0,0.0,0.0, 0.0,0.0,TTT, 0.0,0.0,0.0,
		0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,TTT
	}}},
	{ConvolutionEffect_Sharpen_3x3, {{3,3,3,3}, {
		-1.,0.0,0.0, -1.,0.0,0.0, -1.,0.0,0.0,
		-1.,0.0,0.0, 9.0,0.0,0.0, -1.,0.0,0.0,
		-1.,0.0,0.0, -1.,0.0,0.0, -1.,0.0,0.0,

		0.0,-1.,0.0, 0.0,-1.,0.0, 0.0,-1.,0.0,
		0.0,-1.,0.0, 0.0,9.0,0.0, 0.0,-1.,0.0,
		0.0,-1.,0.0, 0.0,-1.,0.0, 0.0,-1.,0.0,

		0.0,0.0,-1., 0.0,0.0,-1., 0.0,0.0,-1.,
		0.0,0.0,-1., 0.0,0.0,9.0, 0.0,0.0,-1.,
		0.0,0.0,-1., 0.0,0.0,-1., 0.0,0.0,-1.
	}}}
};
#undef THR
#undef S01
#undef S02
#undef S04
#undef TTT

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
,          sourceImageFileSize(&sourceDetails)
,          sourceImageSize(&sourceDetails)
,          sourceApplyEffectsWidget("Apply Effects", &sourceDetails)
,            sourceApplyEffectsLayout(&sourceApplyEffectsWidget)
,            sourceEffectFlipHorizontallyLabel("Flip horizontally", &sourceApplyEffectsWidget)
,            sourceEffectFlipHorizontallyCheckBox(&sourceApplyEffectsWidget)
,            sourceEffectFlipVerticallyLabel("Flip vertically", &sourceApplyEffectsWidget)
,            sourceEffectFlipVerticallyCheckBox(&sourceApplyEffectsWidget)
,            sourceEffectMakeGrayscaleLabel("Make grayscale", &sourceApplyEffectsWidget)
,            sourceEffectMakeGrayscaleCheckBox(&sourceApplyEffectsWidget)
,            sourceEffectConvolutionLabel("Convolution", &sourceApplyEffectsWidget)
,            sourceEffectConvolutionParamsWidget(&sourceApplyEffectsWidget)
,              sourceEffectConvolutionParamsLayout(&sourceEffectConvolutionParamsWidget)
,              sourceEffectConvolutionTypeComboBox(&sourceEffectConvolutionParamsWidget)
,              sourceEffectConvolutionCountComboBox(&sourceEffectConvolutionParamsWidget)
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
	    sourceDetailsLayout.addWidget(&sourceImageFileSize);
	    sourceDetailsLayout.addWidget(&sourceImageSize);
	    sourceDetailsLayout.addWidget(&sourceApplyEffectsWidget);
	      sourceApplyEffectsLayout.addWidget(&sourceEffectFlipHorizontallyLabel,    0/*row*/, 0/*column*/);
	      sourceApplyEffectsLayout.addWidget(&sourceEffectFlipHorizontallyCheckBox, 0/*row*/, 1/*column*/);
	      sourceApplyEffectsLayout.addWidget(&sourceEffectFlipVerticallyLabel,      1/*row*/, 0/*column*/);
	      sourceApplyEffectsLayout.addWidget(&sourceEffectFlipVerticallyCheckBox,   1/*row*/, 1/*column*/);
	      sourceApplyEffectsLayout.addWidget(&sourceEffectMakeGrayscaleLabel,       2/*row*/, 0/*column*/);
	      sourceApplyEffectsLayout.addWidget(&sourceEffectMakeGrayscaleCheckBox,    2/*row*/, 1/*column*/);
	      sourceApplyEffectsLayout.addWidget(&sourceEffectConvolutionLabel,         3/*row*/, 0/*column*/);
	      sourceApplyEffectsLayout.addWidget(&sourceEffectConvolutionParamsWidget,  3/*row*/, 1/*column*/);
	        sourceEffectConvolutionParamsLayout.addWidget(&sourceEffectConvolutionTypeComboBox);
	        sourceEffectConvolutionParamsLayout.addWidget(&sourceEffectConvolutionCountComboBox);
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

	for (auto w : {&sourceEffectFlipHorizontallyLabel, &sourceEffectFlipVerticallyLabel, &sourceEffectMakeGrayscaleLabel, &sourceEffectConvolutionLabel})
		w->setAlignment(Qt::AlignRight);

	// tooltips
	sourceImageFileName                 .setToolTip("File name of the input image");
	sourceImageFileSize                 .setToolTip("File size of the input image");
	sourceImageSize                     .setToolTip("Input image size");
	sourceApplyEffectsWidget            .setToolTip("Apply effects to the image");
	sourceEffectFlipHorizontallyLabel   .setToolTip("Flip the image horizontally");
	sourceEffectFlipHorizontallyCheckBox.setToolTip("Flip the image horizontally");
	sourceEffectFlipVerticallyLabel     .setToolTip("Flip the image vertically");
	sourceEffectFlipVerticallyCheckBox  .setToolTip("Flip the image vertically");
	sourceEffectMakeGrayscaleLabel      .setToolTip("Make the image grayscale");
	sourceEffectMakeGrayscaleCheckBox   .setToolTip("Make the image grayscale");
	sourceEffectConvolutionLabel        .setToolTip("Apply convolution to the image");
	sourceEffectConvolutionTypeComboBox .setToolTip("Convolution type to apply to the image");
	sourceEffectConvolutionCountComboBox.setToolTip("How many times to apply the convolution");
	computeButton                       .setToolTip("Perform neural network computation for the currently selected image as input");
	sourceImage                         .setToolTip("Image currently used as a NN input");
	operatorTypeLabel                   .setToolTip("Operator type: what kind of operation does it perform");
	operatorComplexityValue             .setToolTip("Complexity of the currntly selected NN in FLOPS");

	// size policies
	svgScrollArea            .setSizeAdjustPolicy(QAbstractScrollArea::AdjustToContents);
	sourceImageFileName      .setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
	sourceImageFileSize      .setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
	sourceApplyEffectsWidget .setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::Maximum);
	for (QWidget *w : {&sourceEffectConvolutionParamsWidget, (QWidget*)&sourceEffectConvolutionTypeComboBox, (QWidget*)&sourceEffectConvolutionCountComboBox})
		w->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
	sourceImageSize          .setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
	//sourceFiller .setSizePolicy(QSizePolicy::Fixed, QSizePolicy::MinimumExpanding);
	sourceImage              .setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
	detailsStack             .setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);

	// margins and spacing
	sourceEffectConvolutionParamsLayout.setContentsMargins(0,0,0,0);
	sourceEffectConvolutionParamsLayout.setSpacing(0);

	// widget options and flags
	sourceWidget.hide(); // hidden by default
	noDetails.setEnabled(false); // always grayed out

	// fill lists
	sourceEffectConvolutionTypeComboBox.addItem("None",          ConvolutionEffect_None);
	sourceEffectConvolutionTypeComboBox.addItem("Blur (3x3)",    ConvolutionEffect_Blur_3x3);
	sourceEffectConvolutionTypeComboBox.addItem("Blur (5x5)",    ConvolutionEffect_Blur_5x5);
	sourceEffectConvolutionTypeComboBox.addItem("Gauss (3x3)",   ConvolutionEffect_Gaussian_3x3);
	sourceEffectConvolutionTypeComboBox.addItem("Motion (3x3)",  ConvolutionEffect_Motion_3x3);
	sourceEffectConvolutionTypeComboBox.addItem("Sharpen (3x3)", ConvolutionEffect_Sharpen_3x3);
	for (unsigned c = 1; c <= 20; c++)
		sourceEffectConvolutionCountComboBox.addItem(QString("x%1").arg(c), c);

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
	connect(&sourceEffectFlipHorizontallyCheckBox, &QCheckBox::stateChanged, [this](int) {
		effectsChanged();
	});
	connect(&sourceEffectFlipVerticallyCheckBox, &QCheckBox::stateChanged, [this](int) {
		effectsChanged();
	});
	connect(&sourceEffectMakeGrayscaleCheckBox, &QCheckBox::stateChanged, [this](int) {
		effectsChanged();
	});
	connect(&sourceEffectConvolutionTypeComboBox, QOverload<int>::of(&QComboBox::activated), [this](int) {
		effectsChanged();
	});
	connect(&sourceEffectConvolutionCountComboBox, QOverload<int>::of(&QComboBox::activated), [this](int) {
		if (sourceEffectConvolutionTypeComboBox.currentIndex() != 0)
			effectsChanged();
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
			TensorShape requiredShape = model->getTensorShape(modelInputs[0]);
			if (requiredShape.size() == 4) {
				if (requiredShape[0] != 1) {
					Util::warningOk(this, CSTR("Model's required shape " << requiredShape << " has 4 elements but doesn't begin with B=1,"
					                           " don't know how to adjust the image for it"));
					return;
				}
				requiredShape = tensorGetLastDims(requiredShape, 3);
			} else if (requiredShape.size() == 3) {
				if (requiredShape[0] == 1) { // assume [B=1,H,W], remove B and add C=1 for monochrome image
					requiredShape = tensorGetLastDims(requiredShape, 2);
					requiredShape.push_back(1);
				} else { // see if the shape is image-like
					if (requiredShape[2]!=1 && requiredShape[2]!=3) {
						Util::warningOk(this, CSTR("Model's required shape " << requiredShape << " has 3 elements but have C=1 or C=3,"
						                           " it doesn't look limke it describes an image,"
					                                   " don't know how to adjust the image for it"));
						return;
					}
				}
			}

			auto &sharedPtrInput = (*tensorData.get())[modelInputs[0]];
			if (sourceTensorShape != requiredShape)
				sharedPtrInput.reset(Image::resizeImage(sourceTensorDataAsUsed.get(), sourceTensorShape, requiredShape));
			else
				sharedPtrInput = sourceTensorDataAsUsed;
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
	fileMenu->addAction("Take screenshot as input", [this]() {
		openImagePixmap(Util::getScreenshot(true/*hideOurWindows*/), "screenshot");
	});
	fileMenu->addAction("Close Image", [this]() {
		clearImageData();
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
	// clear the previous image data if any
	clearImageData();
	// read the image as tensor
	sourceTensorDataAsLoaded.reset(Image::readPngImageFile(Q2S(imageFileName), sourceTensorShape));
	sourceTensorDataAsUsed = sourceTensorDataAsLoaded;
	// enable widgets, show image
	sourceWidget.show();
	updateSourceImageOnScreen();
	// set info on the screen
	sourceImageFileName.setText(QString("File name: %1").arg(imageFileName));
	sourceImageFileSize.setText(QString("File size: %1 bytes").arg(S2Q(Util::formatUIntHumanReadable(Util::getFileSize(imageFileName)))));
	sourceImageSize.setText(QString("Image size: %1").arg(S2Q(STR(sourceTensorShape))));
}

void MainWindow::openImagePixmap(const QPixmap &imagePixmap, const char *sourceName) {
	// clear the previous image data if any
	clearImageData();
	// read the image as tensor
	sourceTensorDataAsLoaded.reset(Image::readPixmap(imagePixmap, sourceTensorShape));
	{ // TMP: scale down a huge screenshot 1/6
		TensorShape sourceTensorShapeNew = {sourceTensorShape[0]/6, sourceTensorShape[1]/6, sourceTensorShape[2]};
		sourceTensorDataAsLoaded.reset(Image::resizeImage(sourceTensorDataAsLoaded.get(), sourceTensorShape, sourceTensorShapeNew));
		sourceTensorShape = sourceTensorShapeNew;
	}
	if (!sourceTensorDataAsLoaded) {
		Util::warningOk(this, CSTR("Unable to take a screenshot"));
		return;
	}
	sourceTensorDataAsUsed = sourceTensorDataAsLoaded;
	// enable widgets, show image
	sourceWidget.show();
	updateSourceImageOnScreen();
	// set info on the screen
	sourceImageFileName.setText(QString("File name: n/a: %1").arg(sourceName));
	sourceImageFileSize.setText(QString("File size: n/a: %1").arg(sourceName));
	sourceImageSize.setText(QString("Image size: %1").arg(S2Q(STR(sourceTensorShape))));
}

void MainWindow::clearImageData() {
	sourceWidget.hide();
	sourceImage.setPixmap(QPixmap());
	sourceTensorDataAsLoaded = nullptr;
	sourceTensorDataAsUsed = nullptr;
	sourceTensorShape = TensorShape();
	tensorData.reset(nullptr);
}

void MainWindow::effectsChanged() {
	// all available effects that can be applied
	bool flipHorizontally = sourceEffectFlipHorizontallyCheckBox.isChecked();
	bool flipVertically   = sourceEffectFlipVerticallyCheckBox.isChecked();
	bool makeGrayscale    = sourceEffectMakeGrayscaleCheckBox.isChecked();
	auto convolution      = convolutionEffects.find((ConvolutionEffect)sourceEffectConvolutionTypeComboBox.currentData().toUInt())->second;

	// any effects to apply?
	if (flipHorizontally || flipVertically || makeGrayscale || !std::get<1>(convolution).empty()) {
		sourceTensorDataAsUsed.reset(applyEffects(sourceTensorDataAsLoaded.get(), sourceTensorShape,
			flipHorizontally, flipVertically, makeGrayscale, convolution,sourceEffectConvolutionCountComboBox.currentData().toUInt()));
	} else {
		sourceTensorDataAsUsed = sourceTensorDataAsLoaded;
	}

	updateSourceImageOnScreen();
}

float* MainWindow::applyEffects(const float *image, const TensorShape &shape,
	bool flipHorizontally, bool flipVertically, bool makeGrayscale,
	const std::tuple<TensorShape,std::vector<float>> &convolution, unsigned convolutionCount) const
{
	assert(shape.size()==3);
	assert(flipHorizontally || flipVertically || makeGrayscale || !std::get<1>(convolution).empty());

	unsigned idx = 0; // idx=0 is "image", idx can be 0,1,2
	std::unique_ptr<float> withEffects[2]; // idx=1 and idx=2 are allocatable "images"

	auto idxNext = [](unsigned idx) {
		return (idx+1)<3 ? idx+1 : 1;
	};
	auto src = [&](unsigned idx) {
		if (idx==0)
			return image;
		else
			return (const float*)withEffects[idx-1].get();
	};
	auto dst = [&](unsigned idx) {
		auto &we = withEffects[idxNext(idx)-1];
		if (!we)
			we.reset(new float[tensorFlatSize(shape)]);
		return we.get();
	};

	if (flipHorizontally) {
		Image::flipHorizontally(shape, src(idx), dst(idx));
		idx = idxNext(idx);
	}
	if (flipVertically) {
		Image::flipVertically(shape, src(idx), dst(idx));
		idx = idxNext(idx);
	}
	if (makeGrayscale) {
		Image::makeGrayscale(shape, src(idx), dst(idx));
		idx = idxNext(idx);
	}
	if (!std::get<1>(convolution).empty()) {
		TensorShape shapeWithBatch = shape;
		shapeWithBatch.insert(shapeWithBatch.begin(), 1/*batch*/);
		auto clip = [](float *a, size_t sz) {
			for (auto ae = a+sz; a<ae; a++)
				if (*a < 0.)
					*a = 0.;
				else if (*a > 255.)
					*a = 255.;
		};
		const static float bias[3] = {0,0,0};
		for (unsigned i = 1; i <= convolutionCount; i++) {
			float *d = dst(idx);
			NnOperators::Conv2D(
				shapeWithBatch, src(idx),
				std::get<0>(convolution), std::get<1>(convolution).data(),
				{3}, bias, // no bias
				shapeWithBatch, d,
				std::get<0>(convolution)[2]/2, std::get<0>(convolution)[1]/2, // padding, paddings not matching kernel size work but cause image shifts
				1,1, // strides
				1,1  // dilation factors
			);
			clip(d, tensorFlatSize(shapeWithBatch)); // we have to clip the result because otherwise some values are out of range 0..255.
			idx = idxNext(idx);
		}
	}

	return withEffects[idx-1].release();
}

void MainWindow::updateSourceImageOnScreen() {
	sourceImage.setPixmap(Image::toQPixmap(sourceTensorDataAsUsed.get(), sourceTensorShape));
}
