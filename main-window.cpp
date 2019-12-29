
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
#include "compute.h"

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
#include <QTime>
#include <QClipboard>
#include <QMimeData>

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
,      sourceWidget(tr("Source Data"), &rhsWidget)
,        sourceLayout(&sourceWidget)
,        sourceDetails(&sourceWidget)
,          sourceDetailsLayout(&sourceDetails)
,          sourceImageFileNameLabel(tr("File name:"), &sourceDetails)
,          sourceImageFileNameText(&sourceDetails)
,          sourceImageFileSizeLabel(tr("File size:"), &sourceDetails)
,          sourceImageFileSizeText(&sourceDetails)
,          sourceImageSizeLabel(tr("Image size:"), &sourceDetails)
,          sourceImageSizeText(&sourceDetails)
,          outputInterpretationSummaryLineEdit(&sourceDetails)
,          scaleImageWidget(&sourceDetails)
,            scaleImageLayout(&scaleImageWidget)
,            spacerScaleWidget(&scaleImageWidget)
,            scaleImageLabel(tr("Scale image:"), &scaleImageWidget)
,            scaleImageSpinBoxes(&scaleImageWidget)
,          sourceApplyEffectsWidget(tr("Apply Effects"), &sourceDetails)
,            sourceApplyEffectsLayout(&sourceApplyEffectsWidget)
,            sourceEffectFlipHorizontallyLabel(tr("Flip horizontally"), &sourceApplyEffectsWidget)
,            sourceEffectFlipHorizontallyCheckBox(&sourceApplyEffectsWidget)
,            sourceEffectFlipVerticallyLabel(tr("Flip vertically"), &sourceApplyEffectsWidget)
,            sourceEffectFlipVerticallyCheckBox(&sourceApplyEffectsWidget)
,            sourceEffectMakeGrayscaleLabel(tr("Make grayscale"), &sourceApplyEffectsWidget)
,            sourceEffectMakeGrayscaleCheckBox(&sourceApplyEffectsWidget)
,            sourceEffectConvolutionLabel(tr("Convolution"), &sourceApplyEffectsWidget)
,            sourceEffectConvolutionParamsWidget(&sourceApplyEffectsWidget)
,              sourceEffectConvolutionParamsLayout(&sourceEffectConvolutionParamsWidget)
,              sourceEffectConvolutionTypeComboBox(&sourceEffectConvolutionParamsWidget)
,              sourceEffectConvolutionCountComboBox(&sourceEffectConvolutionParamsWidget)
,          computeWidget(&sourceDetails)
,            computeLayout(&computeWidget)
,            computeButton(tr("Compute"), &computeWidget)
,            computeRegionComboBox(&computeWidget)
,          computeByWidget(&sourceDetails)
,            computeByLayout(&computeByWidget)
,            inputNormalizationLabel(tr("Normalization"), &computeByWidget)
,            inputNormalizationRangeComboBox(&computeByWidget)
,            spacer1Widget(&computeByWidget)
,            computationTimeLabel(QString(tr("Computed in %1")).arg(tr(": n/a")), &computeByWidget)
,            spacer2Widget(&computeByWidget)
,            outputInterpretationLabel(tr("Interpret as"), &computeByWidget)
,            outputInterpretationKindComboBox(&computeByWidget)
,            spacer3Widget(&computeByWidget)
,            clearComputationResults(tr("Clear"), &computeByWidget)
,        sourceImageScrollArea(&sourceWidget)
,          sourceImage(&sourceWidget)
,      nnDetailsStack(&rhsWidget)
,        noDetails(tr("Neural Network Details"), &nnDetailsStack)
,        nnOperatorDetails(&nnDetailsStack)
,          nnOperatorDetailsLayout(&nnOperatorDetails)
,          nnOperatorTypeLabel(tr("Operator Type"), &nnOperatorDetails)
,          nnOperatorTypeValue(&nnOperatorDetails)
,          nnOperatorOptionsLabel(tr("Options"), &nnOperatorDetails)
,          nnOperatorInputsLabel(tr("Inputs"), &nnOperatorDetails)
,          nnOperatorOutputsLabel(tr("Outputs"), &nnOperatorDetails)
,          nnOperatorComplexityLabel(tr("Complexity"), &nnOperatorDetails)
,          nnOperatorComplexityValue(&nnOperatorDetails)
,        nnTensorDetails(&nnDetailsStack)
,      blankRhsLabel(tr("Select some operator"), &rhsWidget)
, menuBar(this)
, statusBar(this)
#if defined(USE_PERFTOOLS)
,   memoryUseLabel(&statusBar)
,   memoryUseTimer(&statusBar)
#endif
, plugin(nullptr)
, scaleImageWidthPct(0)
, scaleImageHeightPct(0)
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
	    sourceDetailsLayout.addWidget(&sourceImageFileNameLabel, 0/*row*/, 0/*col*/, 1/*rowSpan*/, 1/*columnSpan*/);
	    sourceDetailsLayout.addWidget(&sourceImageFileNameText,  0/*row*/, 1/*col*/, 1/*rowSpan*/, 1/*columnSpan*/);
	    sourceDetailsLayout.addWidget(&sourceImageFileSizeLabel, 1/*row*/, 0/*col*/, 1/*rowSpan*/, 1/*columnSpan*/);
	    sourceDetailsLayout.addWidget(&sourceImageFileSizeText,  1/*row*/, 1/*col*/, 1/*rowSpan*/, 1/*columnSpan*/);
	    sourceDetailsLayout.addWidget(&sourceImageSizeLabel,     2/*row*/, 0/*col*/, 1/*rowSpan*/, 1/*columnSpan*/);
	    sourceDetailsLayout.addWidget(&sourceImageSizeText,      2/*row*/, 1/*col*/, 1/*rowSpan*/, 1/*columnSpan*/);
	    sourceDetailsLayout.addWidget(&outputInterpretationSummaryLineEdit, 0/*row*/, 2/*col*/, 2/*rowSpan*/, 2/*columnSpan*/);
	    sourceDetailsLayout.addWidget(&scaleImageWidget,         2/*row*/, 2/*col*/, 1/*rowSpan*/, 2/*columnSpan*/);
	      scaleImageLayout.addWidget(&spacerScaleWidget);
	      scaleImageLayout.addWidget(&scaleImageLabel);
	      scaleImageLayout.addWidget(&scaleImageSpinBoxes);
	    sourceDetailsLayout.addWidget(&sourceApplyEffectsWidget, 3/*row*/, 0/*col*/, 1/*rowSpan*/, 4/*columnSpan*/);
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
	    sourceDetailsLayout.addWidget(&computeWidget,            4/*row*/, 0/*col*/, 1/*rowSpan*/, 4/*columnSpan*/);
	      computeLayout.addWidget(&computeButton);
	      computeLayout.addWidget(&computeRegionComboBox);
	    sourceDetailsLayout.addWidget(&computeByWidget,          5/*row*/, 0/*col*/, 1/*rowSpan*/, 4/*columnSpan*/);
	      computeByLayout.addWidget(&inputNormalizationLabel);
	      computeByLayout.addWidget(&inputNormalizationRangeComboBox);
	      computeByLayout.addWidget(&inputNormalizationColorOrderComboBox);
	      computeByLayout.addWidget(&spacer1Widget);
	      computeByLayout.addWidget(&computationTimeLabel);
	      computeByLayout.addWidget(&spacer2Widget);
	      computeByLayout.addWidget(&outputInterpretationLabel);
	      computeByLayout.addWidget(&outputInterpretationKindComboBox);
	      computeByLayout.addWidget(&spacer3Widget);
	      computeByLayout.addWidget(&clearComputationResults);
	  sourceLayout.addWidget(&sourceImageScrollArea);
	    sourceImageScrollArea.setWidget(&sourceImage);
	rhsLayout.addWidget(&nnDetailsStack);
	rhsLayout.addWidget(&blankRhsLabel);
	nnDetailsStack.addWidget(&noDetails);
	nnDetailsStack.addWidget(&nnOperatorDetails);
		nnOperatorDetails.setLayout(&nnOperatorDetailsLayout);
	nnDetailsStack.addWidget(&nnTensorDetails);

	svgScrollArea.setWidgetResizable(true);
	svgScrollArea.setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
	svgScrollArea.setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOn);

	sourceImageScrollArea.setWidgetResizable(true);
	sourceImageScrollArea.setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
	sourceImageScrollArea.setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);

	setMenuBar(&menuBar);
	setStatusBar(&statusBar);
#if defined(USE_PERFTOOLS)
	statusBar.addWidget(&memoryUseLabel);
#endif

	// alignment
	for (auto w : {&sourceImageFileNameLabel, &sourceImageFileSizeLabel, &sourceImageSizeLabel, &scaleImageLabel})
		w->setAlignment(Qt::AlignRight|Qt::AlignVCenter);
	for (auto w : {&sourceImageFileNameText, &sourceImageFileSizeText, &sourceImageSizeText})
		w->setAlignment(Qt::AlignLeft|Qt::AlignVCenter);
	outputInterpretationSummaryLineEdit.setAlignment(Qt::AlignRight);
	for (auto w : {&sourceEffectFlipHorizontallyLabel, &sourceEffectFlipVerticallyLabel, &sourceEffectMakeGrayscaleLabel, &sourceEffectConvolutionLabel})
		w->setAlignment(Qt::AlignRight|Qt::AlignVCenter);
	for (auto w : {&inputNormalizationLabel, &computationTimeLabel})
		w->setAlignment(Qt::AlignRight|Qt::AlignVCenter);

	{ // double the font size in the summary
		QFont font = outputInterpretationSummaryLineEdit.font();
		font.setPointSize(2*font.pointSize());
		font.setBold(true);
		outputInterpretationSummaryLineEdit.setFont(font);
	}

	// tooltips
	sourceImageFileNameLabel            .setToolTip(tr("File name of the input image"));
	sourceImageFileNameText             .setToolTip(tr("File name of the input image"));
	sourceImageFileSizeLabel            .setToolTip(tr("File size of the input image"));
	sourceImageFileSizeText             .setToolTip(tr("File size of the input image"));
	sourceImageSizeLabel                .setToolTip(tr("Input image size"));
	sourceImageSizeText                 .setToolTip(tr("Input image size"));
	for (QWidget *w : {(QWidget*)&scaleImageLabel,(QWidget*)&scaleImageSpinBoxes})
		w->                          setToolTip(tr("Scale the image to fit the screen, or to select its area for NN computation"));
	sourceApplyEffectsWidget            .setToolTip(tr("Apply effects to the image"));
	sourceEffectFlipHorizontallyLabel   .setToolTip(tr("Flip the image horizontally"));
	sourceEffectFlipHorizontallyCheckBox.setToolTip(tr("Flip the image horizontally"));
	sourceEffectFlipVerticallyLabel     .setToolTip(tr("Flip the image vertically"));
	sourceEffectFlipVerticallyCheckBox  .setToolTip(tr("Flip the image vertically"));
	sourceEffectMakeGrayscaleLabel      .setToolTip(tr("Make the image grayscale"));
	sourceEffectMakeGrayscaleCheckBox   .setToolTip(tr("Make the image grayscale"));
	sourceEffectConvolutionLabel        .setToolTip(tr("Apply convolution to the image"));
	sourceEffectConvolutionTypeComboBox .setToolTip(tr("Convolution type to apply to the image"));
	sourceEffectConvolutionCountComboBox.setToolTip(tr("How many times to apply the convolution"));
	computeButton                       .setToolTip(tr("Perform neural network computation for the currently selected image as input"));
	computeRegionComboBox               .setToolTip(tr("Choose what region of the image to compute on: the visible area or the whole image"));
	inputNormalizationLabel             .setToolTip(tr("Specify how does this NN expect its input data be normalized"));
	inputNormalizationRangeComboBox     .setToolTip(tr("Specify what value range does this NN expect its input data be normalized to"));
	inputNormalizationColorOrderComboBox.setToolTip(tr("Specify what color order does this NN expect its input data be supplied in"));
	computationTimeLabel                .setToolTip(tr("Show how long did the the NN computation take"));
	for (QWidget *w : {(QWidget*)&outputInterpretationLabel,(QWidget*)&outputInterpretationKindComboBox})
		w->                          setToolTip(tr("How to interpret the computation result?"));
	clearComputationResults             .setToolTip(tr("Clear computation results"));
	sourceImage                         .setToolTip(tr("Image currently used as NN input"));
	nnOperatorTypeLabel                 .setToolTip(tr("Operator type: what kind of operation does it perform"));
	nnOperatorComplexityValue           .setToolTip(tr("Complexity of the currntly selected NN in FLOPS"));

	// size policies
	svgScrollArea                        .setSizeAdjustPolicy(QAbstractScrollArea::AdjustToContents);
	sourceWidget                         .setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
	sourceImageFileNameLabel             .setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Maximum);
	sourceImageFileNameText              .setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Maximum);
	sourceImageFileSizeLabel             .setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Maximum);
	sourceImageFileSizeText              .setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Maximum);
	sourceImageSizeLabel                 .setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Maximum);
	sourceImageSizeText                  .setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Maximum);
	scaleImageWidget                     .setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Maximum);
	spacerScaleWidget                    .setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Minimum);
	scaleImageLabel                      .setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
	scaleImageSpinBoxes                  .setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
	sourceApplyEffectsWidget             .setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::Maximum);
	for (QWidget *w : {&sourceEffectConvolutionParamsWidget, (QWidget*)&sourceEffectConvolutionTypeComboBox, (QWidget*)&sourceEffectConvolutionCountComboBox})
		w->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
	computeWidget                        .setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Maximum);
	computeButton                        .setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Maximum);
	computeRegionComboBox                .setSizePolicy(QSizePolicy::Fixed,   QSizePolicy::Maximum);
	inputNormalizationLabel              .setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum); //The sizeHint() is a maximum
	inputNormalizationRangeComboBox      .setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
	inputNormalizationColorOrderComboBox .setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
	spacer1Widget                        .setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Maximum);
	computationTimeLabel                 .setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
	spacer2Widget                        .setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Maximum);
	outputInterpretationLabel            .setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
	outputInterpretationKindComboBox     .setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
	spacer3Widget                        .setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Maximum);
	sourceImageScrollArea                .setSizePolicy(QSizePolicy::Fixed,   QSizePolicy::Fixed);
	nnDetailsStack                       .setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);

	// margins and spacing
	for (QLayout *l : {&scaleImageLayout, &sourceEffectConvolutionParamsLayout, &computeByLayout})
		l->setContentsMargins(0,0,0,0);
	for (QLayout *l : {&sourceEffectConvolutionParamsLayout, &computeByLayout})
		l->setSpacing(0);

	for (auto w : {&spacer1Widget, &spacer2Widget, &spacer3Widget})
		w->setMinimumWidth(10);

	// widget options and flags
	sourceWidget.hide(); // hidden by default
	noDetails.setEnabled(false); // always grayed out
	sourceEffectConvolutionCountComboBox.setEnabled(false); // is only enabled when some convoulution is chosen

	// widget states
	updateResultInterpretationSummary(false/*enable*/, tr("n/a"), tr("n/a"));

	// fill lists
	sourceEffectConvolutionTypeComboBox.addItem(tr("None"),          ConvolutionEffect_None);
	sourceEffectConvolutionTypeComboBox.addItem(tr("Blur (3x3)"),    ConvolutionEffect_Blur_3x3);
	sourceEffectConvolutionTypeComboBox.addItem(tr("Blur (5x5)"),    ConvolutionEffect_Blur_5x5);
	sourceEffectConvolutionTypeComboBox.addItem(tr("Gauss (3x3)"),   ConvolutionEffect_Gaussian_3x3);
	sourceEffectConvolutionTypeComboBox.addItem(tr("Motion (3x3)"),  ConvolutionEffect_Motion_3x3);
	sourceEffectConvolutionTypeComboBox.addItem(tr("Sharpen (3x3)"), ConvolutionEffect_Sharpen_3x3);
	for (unsigned c = 1; c <= 20; c++)
		sourceEffectConvolutionCountComboBox.addItem(QString("x%1").arg(c), c);

	computeRegionComboBox.addItem("on the visible region");
	computeRegionComboBox.addItem("on the whole image");

	inputNormalizationRangeComboBox.addItem("0..1",         InputNormalizationRange_0_1); // default
	inputNormalizationRangeComboBox.addItem("0..255",       InputNormalizationRange_0_255);
	inputNormalizationRangeComboBox.addItem("0..128",       InputNormalizationRange_0_128);
	inputNormalizationRangeComboBox.addItem("0..64",        InputNormalizationRange_0_64);
	inputNormalizationRangeComboBox.addItem("0..32",        InputNormalizationRange_0_32);
	inputNormalizationRangeComboBox.addItem("0..16",        InputNormalizationRange_0_16);
	inputNormalizationRangeComboBox.addItem("0..8",         InputNormalizationRange_0_8);
	inputNormalizationRangeComboBox.addItem("-1..1",        InputNormalizationRange_M1_P1);
	inputNormalizationRangeComboBox.addItem("ImageNet",     InputNormalizationRange_ImageNet);
	//
	inputNormalizationColorOrderComboBox.addItem("RGB",     InputNormalizationColorOrder_RGB); // default
	inputNormalizationColorOrderComboBox.addItem("BGR",     InputNormalizationColorOrder_BGR);
	//
	outputInterpretationKindComboBox.addItem("ImageNet-1001",   ConvolutionEffect_None);
	outputInterpretationKindComboBox.addItem("Yes/No",          ConvolutionEffect_None);
	outputInterpretationKindComboBox.addItem("No/Yes",          ConvolutionEffect_None);

	// fonts
	for (auto widget : {&nnOperatorTypeLabel, &nnOperatorOptionsLabel, &nnOperatorInputsLabel, &nnOperatorOutputsLabel, &nnOperatorComplexityLabel})
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
	connect(&scaleImageSpinBoxes, &ScaleImageWidget::scalingFactorChanged, [this](unsigned widthFactor, unsigned heightFactor) {
		assert(scaleImageWidthPct!=0 && scaleImageHeightPct!=0); // scaling percentages are initially set when the image is open/pasted/etc

		// accept percentages set by the user
		scaleImageWidthPct = widthFactor;
		scaleImageHeightPct = heightFactor;

		// update the image on screen accordingly
		updateSourceImageOnScreen();
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
	connect(&sourceEffectConvolutionTypeComboBox, QOverload<int>::of(&QComboBox::activated), [this](int index) {
		effectsChanged();
		sourceEffectConvolutionCountComboBox.setEnabled(index>0);
	});
	connect(&sourceEffectConvolutionCountComboBox, QOverload<int>::of(&QComboBox::activated), [this](int) {
		if (sourceEffectConvolutionTypeComboBox.currentIndex() != 0)
			effectsChanged();
	});
	connect(&computeButton, &QAbstractButton::pressed, [this]() {
		QTime timer;
		timer.start();

		// computation arguments
		bool doVisibleRegion = computeRegionComboBox.currentIndex()==0;
		std::array<unsigned,4> imageRegion = doVisibleRegion ? getVisibleImageRegion() : std::array<unsigned,4>{0,0, sourceTensorShape[1]-1,sourceTensorShape[0]-1};
		InputNormalization inputNormalization = {
			(InputNormalizationRange)inputNormalizationRangeComboBox.currentData().toUInt(),
			(InputNormalizationColorOrder)inputNormalizationColorOrderComboBox.currentData().toUInt()
		};

		bool succ = Compute::compute(model, imageRegion,inputNormalization, sourceTensorDataAsUsed,sourceTensorShape, tensorData, [this](const std::string &msg) {
			Util::warningOk(this, S2Q(msg));
		}, [](PluginInterface::TensorId tid) {
			//PRINT("Tensor DONE: tid=" << tid)
		});

		if (succ) {
			// interpret results: 1001
			auto result = (*tensorData)[model->getOutputs()[0]].get();
			typedef std::tuple<unsigned/*order num*/,float/*likelihood*/> Likelihood;
			std::vector<Likelihood> likelihoods;
			for (unsigned i = 0; i < 1001; i++)
				likelihoods.push_back({i,result[i]});
			std::sort(likelihoods.begin(), likelihoods.end(), [](const Likelihood &a, const Likelihood &b) {return std::get<1>(a) > std::get<1>(b);});
			// report it to the user
			auto labels = Util::readListFromFile(":/nn-labels/imagenet-labels.txt");
			assert(labels.size() == 1001);
			std::ostringstream ss;
			for (unsigned i = 0; i<10; i++)
				ss << (i>0 ? "\n" : "") << "• " << Q2S(labels[std::get<0>(likelihoods[i])]) << " = " << std::get<1>(likelihoods[i]);
			updateResultInterpretationSummary(
				true/*enable*/,
				QString("%1 (%2)").arg(labels[std::get<0>(likelihoods[0])]).arg(std::get<1>(likelihoods[0])),
				S2Q(ss.str())
			);
		} else
			PRINT("WARNING computation didn't succeed")

		computationTimeLabel.setText(QString("Computed in %1").arg(QString("%1 ms").arg(S2Q(Util::formatUIntHumanReadable(timer.elapsed())))));
	});
	connect(&computeRegionComboBox, QOverload<int>::of(&QComboBox::activated), [this](int) {
		clearComputedTensorData();
	});
	connect(&inputNormalizationRangeComboBox, QOverload<int>::of(&QComboBox::activated), [this](int) {
		inputNormalizationChanged();
	});
	connect(&inputNormalizationColorOrderComboBox, QOverload<int>::of(&QComboBox::activated), [this](int) {
		inputNormalizationChanged();
	});
	connect(&clearComputationResults, &QAbstractButton::pressed, [this]() {
		clearComputedTensorData();
	});

	// monitor memory use
#if defined(USE_PERFTOOLS)
	connect(&memoryUseTimer, &QTimer::timeout, [this]() {
		size_t inuseBytes = 0;
		(void)MallocExtension::instance()->GetNumericProperty("generic.current_allocated_bytes", &inuseBytes);
		memoryUseLabel.setText(QString(tr("Memory use: %1 bytes")).arg(S2Q(Util::formatUIntHumanReadable(inuseBytes))));
	});
	memoryUseTimer.start(1000);
#endif

	// add menus
	auto fileMenu = menuBar.addMenu(tr("&File"));
	fileMenu->addAction(tr("Open Image"), [this]() {
		QString fileName = QFileDialog::getOpenFileName(this,
			tr("Open image file"), "",
			tr("Image (*.png);;All Files (*)")
		);
		if (!fileName.isEmpty())
			openImageFile(fileName);
	})->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_O));
	fileMenu->addAction(tr("Open Neural Network File"), []() {
	})->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_N)); // non-standard because this is our custom operation
	fileMenu->addAction(tr("Take Screenshot"), [this]() {
		openImagePixmap(Util::getScreenshot(true/*hideOurWindows*/), tr("screenshot"));
	})->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_R)); // non-standard
	fileMenu->addAction(tr("Copy Image"), [this]() {
		if (sourceTensorDataAsUsed)
			QApplication::clipboard()->setPixmap(Image::toQPixmap(sourceTensorDataAsUsed.get(), sourceTensorShape), QClipboard::Clipboard);
		else
			Util::warningOk(this, QString(tr("Can't copy the image: no image in currently open")));
	})->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_C));
	fileMenu->addAction(tr("Save Image As"), [this]() {
		QString fileName = QFileDialog::getSaveFileName(this,
			tr("Save image as file"), ""
		);
		if (!fileName.isEmpty()) { // save the visible region, same as NN computation normally sees
			std::array<unsigned,4> imageRegion = getVisibleImageRegion();
			Image::writePngImageFile( // for simplicity - extract the region whether this is needed or not
				std::unique_ptr<float>(Image::regionOfImage(sourceTensorDataAsUsed.get(), sourceTensorShape, imageRegion)).get(),
				{imageRegion[3]-imageRegion[1]+1, imageRegion[2]-imageRegion[0]+1, sourceTensorShape[2]},
				Q2S(fileName.endsWith(".png") ? fileName : fileName+".png")
			);
		}
	})->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_S));
	fileMenu->addAction(tr("Paste Image"), [this]() {
		const QClipboard *clipboard = QApplication::clipboard();
		const QMimeData *mimeData = clipboard->mimeData();

		if (mimeData->hasImage()) {
			auto pixmap = qvariant_cast<QPixmap>(mimeData->imageData());
			if (pixmap.height() != 0)
				openImagePixmap(pixmap, tr("paste from clipboard"));
			else // see https://bugs.freebsd.org/bugzilla/show_bug.cgi?id=242932
				Util::warningOk(this, QString(tr("No image to paste, clipboard contains an empty image")));
		} else {
			auto formats = mimeData->formats();
			if (!formats.empty())
				Util::warningOk(this, QString(tr("%1:\n• %2"))
					.arg(tr("No image to paste, clipboard can be interpreted as"))
					.arg(mimeData->formats().join("\n• ")));
			else
				Util::warningOk(this, QString(tr("No image to paste, clipboard s empty")));
		}
	})->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_V));
	fileMenu->addAction(tr("Close Image"), [this]() {
		clearInputImageDisplay();
		clearEffects();
		clearComputedTensorData(); // closing image invalidates computation results
	})->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_W)); // like "close tab" in chrome
	fileMenu->addAction(tr("Quit"), []() {
		QApplication::quit();
	})->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_Q));
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
		return Util::warningOk(this, QString("%1 '%2'").arg(tr("Couldn't find a plugin to open the file")).arg(filePath));

	// load the plugin
	plugin = PluginManager::loadPlugin(pluginName);
	if (!plugin)
		FAIL(Q2S(QString("%1 '%2'").arg(tr("failed to load the plugin")).arg(pluginName)))
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
	nnDetailsStack.setCurrentIndex(/*page#1*/1);
	nnOperatorDetails.setTitle(QString("Operator#%1").arg(operatorId));

	// clear items
	while (nnOperatorDetailsLayout.count() > 0)
		nnOperatorDetailsLayout.removeItem(nnOperatorDetailsLayout.itemAt(0));
	tempDetailWidgets.clear();

	// helper
	auto addTensorLines = [this](auto &tensors, unsigned &row) {
		for (auto t : tensors) {
			row++;
			// tensor number
			auto label = new QLabel(QString(tr("tensor#%1:")).arg(t), &nnOperatorDetails);
			label->setToolTip(tr("Tensor number"));
			label->setAlignment(Qt::AlignRight|Qt::AlignVCenter);
			tempDetailWidgets.push_back(std::unique_ptr<QWidget>(label));
			nnOperatorDetailsLayout.addWidget(label,         row,   0/*column*/);
			// tensor name
			label = new QLabel(S2Q(model->getTensorName(t)), &nnOperatorDetails);
			label->setToolTip(tr("Tensor name"));
			label->setAlignment(Qt::AlignLeft|Qt::AlignVCenter);
			tempDetailWidgets.push_back(std::unique_ptr<QWidget>(label));
			nnOperatorDetailsLayout.addWidget(label,         row,   1/*column*/);
			// tensor shape
			auto describeShape = [](const TensorShape &shape) {
				auto flatSize = tensorFlatSize(shape);
				return STR(shape <<
				         " (" <<
				             Util::formatUIntHumanReadable(flatSize) << " " << Q2S(tr("floats")) << ", " <<
				             Util::formatUIntHumanReadable(flatSize*sizeof(float)) << " " << Q2S(tr("bytes")) <<
				          ")"
				);
			};
			label = new QLabel(S2Q(describeShape(model->getTensorShape(t))), &nnOperatorDetails);
			label->setToolTip(tr("Tensor shape and data size"));
			label->setAlignment(Qt::AlignLeft|Qt::AlignVCenter);
			tempDetailWidgets.push_back(std::unique_ptr<QWidget>(label));
			nnOperatorDetailsLayout.addWidget(label,         row,   2/*column*/);
			// has buffer? is variable?
			bool isInput = Util::isValueIn(model->getInputs(), t);
			bool isOutput = Util::isValueIn(model->getOutputs(), t);
			auto hasStaticData = model->getTensorHasData(t);
			auto isVariable = model->getTensorIsVariableFlag(t);
			label = new QLabel(QString("<%1>").arg(
				isInput ? tr("input")
				: isOutput ? tr("output")
				: hasStaticData ? tr("static tensor")
				: isVariable ? tr("variable") : tr("computed")),
				&nnOperatorDetails);
			label->setToolTip(tr("Tensor type"));
			label->setAlignment(Qt::AlignCenter|Qt::AlignVCenter);
			label->setStyleSheet("font: italic");
			tempDetailWidgets.push_back(std::unique_ptr<QWidget>(label));
			nnOperatorDetailsLayout.addWidget(label,         row,   3/*column*/);
			// button
			if (hasStaticData || (tensorData && (*tensorData.get())[t])) {
				auto button = new QPushButton("➞", &nnOperatorDetails);
				button->setContentsMargins(0,0,0,0);
				button->setStyleSheet("color: blue;");
				button->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
				//button->setMaximumSize(QFontMetrics(button->font()).size(Qt::TextSingleLine, button->text()).grownBy(QMargins(4,0,4,0)));
				button->setMaximumSize(QFontMetrics(button->font()).size(Qt::TextSingleLine, button->text())+QSize(8,0));
				button->setToolTip(tr("Show the tensor data as a table"));
				tempDetailWidgets.push_back(std::unique_ptr<QWidget>(button));
				nnOperatorDetailsLayout.addWidget(button,         row,   4/*column*/);
				connect(button, &QAbstractButton::pressed, [this,t,hasStaticData]() {
					removeTableIfAny();
					// show table
					auto tableShape = model->getTensorShape(t);
					switch (tensorNumMultiDims(tableShape)) {
					case 0:
						Util::warningOk(this, "WARNING tensor shape with all ones encountered, this is meaningless in the NN models context");
						break;
					case 1:
						Util::warningOk(this, "WARNING DataTable1D isn't yet implemented (TODO)");
						break;
					default: {
						nnDataTable.reset(new DataTable2D(tableShape,
							hasStaticData ? model->getTensorData(t) : (*tensorData.get())[t].get(),
							&rhsWidget));
						rhsLayout.addWidget(nnDataTable.get());
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
	nnOperatorDetailsLayout.addWidget(&nnOperatorTypeLabel,          row,   0/*column*/);
	nnOperatorDetailsLayout.addWidget(&nnOperatorTypeValue,          row,   1/*column*/);
	row++;
	nnOperatorDetailsLayout.addWidget(&nnOperatorOptionsLabel,       row,   0/*column*/);
	{
		std::unique_ptr<PluginInterface::OperatorOptionsList> opts(model->getOperatorOptions(operatorId));
		for (auto &opt : *opts) {
			row++;
			// option name
			auto label = new QLabel(S2Q(STR(opt.name)), &nnOperatorDetails);
			label->setToolTip("Option name");
			label->setAlignment(Qt::AlignRight|Qt::AlignVCenter);
			tempDetailWidgets.push_back(std::unique_ptr<QWidget>(label));
			nnOperatorDetailsLayout.addWidget(label,               row,   0/*column*/);
			// option type
			label = new QLabel(S2Q(STR("<" << opt.value.type << ">")), &nnOperatorDetails);
			label->setToolTip(tr("Option type"));
			label->setAlignment(Qt::AlignLeft|Qt::AlignVCenter);
			label->setStyleSheet("font: italic");
			tempDetailWidgets.push_back(std::unique_ptr<QWidget>(label));
			nnOperatorDetailsLayout.addWidget(label,               row,   1/*column*/);
			// option value
			label = new QLabel(S2Q(STR(opt.value)), &nnOperatorDetails);
			label->setToolTip(tr("Option value"));
			label->setAlignment(Qt::AlignLeft|Qt::AlignVCenter);
			tempDetailWidgets.push_back(std::unique_ptr<QWidget>(label));
			nnOperatorDetailsLayout.addWidget(label,               row,   2/*column*/);
		}
		if (opts->empty()) {
			row++;
			auto label = new QLabel("-none-", &nnOperatorDetails);
			label->setAlignment(Qt::AlignRight|Qt::AlignVCenter);
			tempDetailWidgets.push_back(std::unique_ptr<QWidget>(label));
			nnOperatorDetailsLayout.addWidget(label,               row,   0/*column*/);
		}
	}
	row++;
	nnOperatorDetailsLayout.addWidget(&nnOperatorInputsLabel,        row,   0/*column*/);
	addTensorLines(inputs, row);
	row++;
	nnOperatorDetailsLayout.addWidget(&nnOperatorOutputsLabel,       row,   0/*column*/);
	addTensorLines(outputs, row);
	row++;
	nnOperatorDetailsLayout.addWidget(&nnOperatorComplexityLabel,    row,   0/*column*/);
	nnOperatorDetailsLayout.addWidget(&nnOperatorComplexityValue,    row,   1/*column*/);

	// set texts
	nnOperatorTypeValue.setText(S2Q(STR(model->getOperatorKind(operatorId))));
	nnOperatorComplexityValue.setText(S2Q(Util::formatFlops(ModelFunctions::computeOperatorFlops(model, operatorId))));
}

void MainWindow::showTensorDetails(PluginInterface::TensorId tensorId) {
	removeTableIfAny();
	nnDetailsStack.setCurrentIndex(/*page#*/2);
	nnTensorDetails.setTitle(QString("Tensor#%1: %2").arg(tensorId).arg(S2Q(model->getTensorName(tensorId))));
}

void MainWindow::removeTableIfAny() {
	if (nnDataTable.get()) {
		rhsLayout.removeWidget(nnDataTable.get());
		nnDataTable.reset(nullptr);
		blankRhsLabel.show();
	}
}

void MainWindow::openImageFile(const QString &imageFileName) {
	// clear the previous image data if any
	clearInputImageDisplay();
	clearEffects();
	clearComputedTensorData(); // opening image invalidates computation results
	// read the image as tensor
	sourceTensorDataAsLoaded.reset(Image::readPngImageFile(Q2S(imageFileName), sourceTensorShape));
	sourceTensorDataAsUsed = sourceTensorDataAsLoaded;
	// enable widgets, show image
	sourceWidget.show();
	updateSourceImageOnScreen();
	// set info on the screen
	sourceImageFileNameText.setText(imageFileName);
	sourceImageFileSizeText.setText(QString("%1 bytes").arg(S2Q(Util::formatUIntHumanReadable(Util::getFileSize(imageFileName)))));
	sourceImageSizeText.setText(S2Q(STR(sourceTensorShape)));
	// focus
	computeButton.setFocus();
}

void MainWindow::openImagePixmap(const QPixmap &imagePixmap, const QString &sourceName) {
	// clear the previous image data if any
	clearInputImageDisplay();
	clearEffects();
	clearComputedTensorData(); // opening image invalidates computation results
	// read the image as tensor
	sourceTensorDataAsLoaded.reset(Image::readPixmap(imagePixmap, sourceTensorShape));
	if (0) { // TMP: scale down a huge screenshot 1/6
		TensorShape sourceTensorShapeNew = {sourceTensorShape[0]/6, sourceTensorShape[1]/6, sourceTensorShape[2]};
		sourceTensorDataAsLoaded.reset(Image::resizeImage(sourceTensorDataAsLoaded.get(), sourceTensorShape, sourceTensorShapeNew));
		sourceTensorShape = sourceTensorShapeNew;
	}
	if (!sourceTensorDataAsLoaded) {
		Util::warningOk(this, QString(tr("Unable to accept %1")).arg(sourceName));
		return;
	}
	sourceTensorDataAsUsed = sourceTensorDataAsLoaded;
	// enable widgets, show image
	sourceWidget.show();
	updateSourceImageOnScreen();
	// set info on the screen
	sourceImageFileNameText.setText(QString("{%1}").arg(sourceName));
	sourceImageFileSizeText.setText(QString("{%1}").arg(sourceName));
	sourceImageSizeText.setText(S2Q(STR(sourceTensorShape)));
	// focus
	computeButton.setFocus();
}

void MainWindow::clearInputImageDisplay() {
	sourceWidget.hide();
	sourceImage.setPixmap(QPixmap());
	sourceTensorDataAsLoaded = nullptr;
	sourceTensorDataAsUsed = nullptr;
	sourceTensorShape = TensorShape();
	tensorData.reset(nullptr);
	scaleImageWidthPct = 0;
	scaleImageHeightPct = 0;
}

void MainWindow::clearComputedTensorData() {
	// clear table-like display of data about to be invalidated
	removeTableIfAny();
	// clear tensor data
	tensorData.reset(nullptr);
	// clear result interpretation
	updateResultInterpretationSummary(false/*enable*/, tr("n/a"), tr("n/a"));
}

void MainWindow::effectsChanged() {
	clearComputedTensorData(); // effects change invalidates computation results

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

void MainWindow::inputNormalizationChanged() {
	clearComputedTensorData(); // input normalization change invalidates computation results
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

void MainWindow::clearEffects() {
	sourceEffectFlipHorizontallyCheckBox.setChecked(false);
	sourceEffectFlipVerticallyCheckBox  .setChecked(false);
	sourceEffectMakeGrayscaleCheckBox   .setChecked(false);
	sourceEffectConvolutionTypeComboBox .setCurrentIndex(0);
	sourceEffectConvolutionCountComboBox.setCurrentIndex(0);
	sourceEffectConvolutionCountComboBox.setEnabled(false);
}

void MainWindow::updateSourceImageOnScreen() {
	{ // fix image size to the size of details to its left, so that it would nicely align to them
		auto height = sourceDetails.height();
		sourceImageScrollArea.setMinimumSize(QSize(height,height));
		sourceImageScrollArea.setMaximumSize(QSize(height,height));
	}

	// decide on the scaling percentage
	if (scaleImageWidthPct == 0)  {
		// compute the percentage
		assert(sourceTensorShape.size() == 3);
		auto screenSize = sourceImageScrollArea.minimumSize();
		float scaleFactorWidth  = (float)screenSize.width()/(float)sourceTensorShape[1];
		float scaleFactorHeight = (float)screenSize.height()/(float)sourceTensorShape[0];
		float scaleFactor = std::min(scaleFactorWidth, scaleFactorHeight);
		if (scaleFactor < 0.01) // we only scale down to 1%
			scaleFactor = 0.01;
		if (scaleFactor > ScaleImageWidget::maxValue) // do not exceed a max value set by ScaleImageWidget
			scaleFactor = ScaleImageWidget::maxValue;
		scaleImageWidthPct  = scaleFactor*100;
		scaleImageHeightPct = scaleFactor*100;

		// set the same percentage in the scaling widget
		scaleImageSpinBoxes.setFactor(scaleImageWidthPct);
	}

	// generate and set the pixmap
	QPixmap pixmap;
	if (scaleImageWidthPct != 100 || scaleImageHeightPct != 100) {
		assert(sourceTensorShape.size() == 3);
		TensorShape resizedShape = {
			sourceTensorShape[0]*scaleImageHeightPct/100,
			sourceTensorShape[1]*scaleImageWidthPct/100,
			sourceTensorShape[2]
		};
		std::unique_ptr<float> resizedImage(Image::resizeImage(sourceTensorDataAsUsed.get(), sourceTensorShape, resizedShape));
		pixmap = Image::toQPixmap(resizedImage.get(), resizedShape);
	} else
		pixmap = Image::toQPixmap(sourceTensorDataAsUsed.get(), sourceTensorShape);
	sourceImage.setPixmap(pixmap);
	sourceImage.resize(pixmap.width(), pixmap.height());
}

void MainWindow::updateResultInterpretationSummary(bool enable, const QString &oneLine, const QString &details) {
	outputInterpretationSummaryLineEdit.setVisible(enable);
	outputInterpretationSummaryLineEdit.setText(oneLine);
	outputInterpretationSummaryLineEdit.setToolTip(QString(tr("Result interpretation:\n%1")).arg(details));
}

std::array<unsigned,4> MainWindow::getVisibleImageRegion() const {
	auto visibleRegion = sourceImage.visibleRegion().boundingRect();
	auto size = sourceImage.size();
	assert(visibleRegion.left() >= 0);
	assert(visibleRegion.right() <= size.width());
	assert(visibleRegion.top() >= 0);
	assert(visibleRegion.bottom() <= size.height());

	// region relative to the resized image normalized to 0..1
	float region[4] = {
		(float)visibleRegion.left()/(float)size.width(),
		(float)visibleRegion.top()/(float)size.height(),
		(float)(visibleRegion.right()+1)/(float)size.width(),
		(float)(visibleRegion.bottom()+1)/(float)size.height()
	};

	// scale it to the original image
	auto sourceWidth = sourceTensorShape[1];
	auto sourceHeight = sourceTensorShape[0];
	return std::array<unsigned,4>{
		(unsigned)(region[0]*sourceWidth),
		(unsigned)(region[1]*sourceHeight),
		(unsigned)(region[2]*sourceWidth-1),
		(unsigned)(region[3]*sourceHeight-1)
	};
}
