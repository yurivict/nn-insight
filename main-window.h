// Copyright (C) 2020 by Yuri Victorovich. All rights reserved.

#pragma once

#include <QMainWindow>
#include <QSplitter>
#include <QScrollArea>
#include <QStackedWidget>
#include <QGroupBox>
#include "nn-widget.h"
#include "no-nn-is-open-widget.h"
#include "operators-list-widget.h"
#include "scale-image-widget.h"
#include "data-table-2d.h"
#include <QLabel>
#include <QPushButton>
#include <QCheckBox>
#include <QComboBox>
#include <QLineEdit>
#include <QVBoxLayout>
#include <QGridLayout>
#include <QMenuBar>
#include <QStatusBar>
#include <QPixmap>
#include <QRectF>
#if defined(USE_PERFTOOLS)
#include <QTimer>
#endif

#include "plugin-manager.h"
#include "plugin-interface.h"
#include "nn-types.h"

#include <vector>
#include <array>
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
	// Lhs
	QScrollArea                        svgScrollArea;
	NnWidget                             nnWidget;
	// Rhs
	QWidget                            rhsWidget;
	QVBoxLayout                          rhsLayout;
	// Rhs/Source
	QGroupBox                            sourceWidget; // optionally visible
	QHBoxLayout                            sourceLayout;
	QWidget                                sourceDetails;
	QGridLayout                              sourceDetailsLayout;
	QLabel                                   sourceImageFileNameLabel;
	QLabel                                   sourceImageFileNameText;
	QLabel                                   sourceImageFileSizeLabel;
	QLabel                                   sourceImageFileSizeText;
	QLabel                                   sourceImageSizeLabel;
	QLabel                                   sourceImageSizeText;
	QLabel                                   sourceImageCurrentRegionLabel;
	QLabel                                   sourceImageCurrentRegionText;
	QLabel                                   outputInterpretationSummaryLineEdit;
	QWidget                                  scaleImageWidget;
	QHBoxLayout                                scaleImageLayout;
	QWidget                                    spacerScaleWidget;
	QLabel                                     scaleImageLabel;
	ScaleImageWidget                           scaleImageSpinBoxes;
	// Rhs/Source/Effects
	QGroupBox                                sourceApplyEffectsWidget;
	QGridLayout                                sourceApplyEffectsLayout;
	QLabel                                     sourceEffectFlipHorizontallyLabel;
	QCheckBox                                  sourceEffectFlipHorizontallyCheckBox;
	QLabel                                     sourceEffectFlipVerticallyLabel;
	QCheckBox                                  sourceEffectFlipVerticallyCheckBox;
	QLabel                                     sourceEffectMakeGrayscaleLabel;
	QCheckBox                                  sourceEffectMakeGrayscaleCheckBox;
	QLabel                                     sourceEffectConvolutionLabel;
	QWidget                                    sourceEffectConvolutionParamsWidget;
	QHBoxLayout                                  sourceEffectConvolutionParamsLayout;
	QComboBox                                    sourceEffectConvolutionTypeComboBox;
	QComboBox                                    sourceEffectConvolutionCountComboBox;
	// Rhs/Source/Compute
	QWidget                                  computeWidget;
	QHBoxLayout                                computeLayout;
	QPushButton                                computeButton;
	QComboBox                                  computeRegionComboBox;
	QWidget                                  computeByWidget;
	QHBoxLayout                                computeByLayout;
	QLabel                                     inputNormalizationLabel;
	QComboBox                                  inputNormalizationRangeComboBox;
	QComboBox                                  inputNormalizationColorOrderComboBox;
	QWidget                                    spacer1Widget;
	QLabel                                     computationTimeLabel;
	QWidget                                    spacer2Widget;
	QLabel                                     outputInterpretationLabel;
	QComboBox                                  outputInterpretationKindComboBox;
	QWidget                                    spacer3Widget;
	QPushButton                                clearComputationResults;
	QStackedWidget                         sourceImageStack;
	QScrollArea                              sourceImageScrollArea;
	QLabel                                     sourceImage;       // index#0
	std::unique_ptr<QLabel>                  interpretationImage; // index#1: only exists when the interpretation image is possible
	// Rhs/NN details
	QStackedWidget                       nnDetailsStack;
	QGroupBox                              nnNetworkDetails;   // page#0: network
	QGridLayout                              nnNetworkDetailsLayout;
	QLabel                                   nnNetworkDescriptionLabel;
	QLabel                                   nnNetworkDescriptionText;
	QLabel                                   nnNetworkComplexityLabel;
	QLabel                                   nnNetworkComplexityText;
	QLabel                                   nnNetworkFileSizeLabel;
	QLabel                                   nnNetworkFileSizeText;
	QLabel                                   nnNetworkNumberInsOutsLabel;
	QLabel                                   nnNetworkNumberInsOutsText;
	QLabel                                   nnNetworkNumberOperatorsLabel;
	QLabel                                   nnNetworkNumberOperatorsText;
	QLabel                                   nnNetworkStaticDataLabel;
	QLabel                                   nnNetworkStaticDataText;
	QLabel                                   nnNetworkOperatorsListLabel;
	OperatorsListWidget                      nnNetworkOperatorsListWidget;
	QGroupBox                              nnOperatorDetails; // page#1: operator
	QGridLayout                              nnOperatorDetailsLayout;
	QLabel                                   nnOperatorTypeLabel;
	QLabel                                   nnOperatorTypeValue;
	QLabel                                   nnOperatorOptionsLabel;
	QLabel                                   nnOperatorInputsLabel;
	QLabel                                   nnOperatorOutputsLabel;
	QLabel                                   nnOperatorComplexityLabel;
	QLabel                                   nnOperatorComplexityValue;
	QLabel                                   nnOperatorStaticDataLabel;
	QLabel                                   nnOperatorStaticDataValue;
	QLabel                                   nnOperatorDataRatioLabel;
	QLabel                                   nnOperatorDataRatioValue;
	QWidget                                  nnOperatorDetailsSpacer;
	QGroupBox                              nnTensorDetails;   // page#2: tensor
	int                                      nnCurrentTensorId;
	QGridLayout                              nnTensorDetailsLayout;
	QLabel                                   nnTensorKindLabel;
	QLabel                                   nnTensorKindValue;
	QLabel                                   nnTensorShapeLabel;
	QLabel                                   nnTensorShapeValue;
	QLabel                                   nnTensorTypeLabel;
	QLabel                                   nnTensorTypeValue;
	QLabel                                   nnTensorDataPlaceholder;
	QLabel                                   nnTensorDataPlaceholder1DnotImplemented;
	std::unique_ptr<DataTable2D>             nnTensorData2D;
	QGroupBox                            noNnIsOpenGroupBox; // optionally visible
	QVBoxLayout                            noNnIsOpenLayout;
	NoNnIsOpenWidget                       noNnIsOpenWidget;

	QMenuBar                         menuBar;
	QStatusBar                       statusBar;
#if defined(USE_PERFTOOLS)
	QLabel                           memoryUseLabel;
	QTimer                           memoryUseTimer;
#endif

	const PluginManager::Plugin*                   plugin;    // plugin in use for the model
	std::unique_ptr<PluginInterface>               pluginInterface; // the file is opened through this handle
	std::unique_ptr<const PluginInterface::Model>  model;     // the model from the file that is currently open

	// data associated with a specific input data (image) currently loaded by the user (static tensors from the model aren't here)
	TensorShape                      sourceTensorShape;
	std::shared_ptr<float>           sourceTensorDataAsLoaded; // original image that was loaded by the user
	std::shared_ptr<float>           sourceTensorDataAsUsed;   // image that is used as an input of NN, might be different if effects are applied
	std::unique_ptr<std::vector<std::shared_ptr<const float>>>   tensorData; // tensors corresponding to the currently used image, shared because reshape/input often shared

	std::vector<std::unique_ptr<QWidget>>   tempDetailWidgets;

	unsigned                         scaleImageWidthPct;    // percentage to scale the image to show on the screen
	unsigned                         scaleImageHeightPct;
	int                              self; // to prevent signals from programmatically changed values

private: // types
	enum HowLong {Temporary, Permanent};

private: // private methods
	bool haveImageOpen() const;
	void showNetworkDetails();
	void showOperatorDetails(PluginInterface::OperatorId operatorId);
	void showTensorDetails(PluginInterface::TensorId tensorId);
	void showTensorDetails(PluginInterface::TensorId tensorId, const char *label);
	void showInputDetails(PluginInterface::TensorId tensorId);
	void showOutputDetails(PluginInterface::TensorId tensorId);
	void removeTableIfAny();
	void openImageFile(const QString &imageFileName);
	void openImagePixmap(const QPixmap &imagePixmap, const QString &sourceName);
	void clearInputImageDisplay();
	void clearComputedTensorData(HowLong howLong);
	void effectsChanged();
	void inputNormalizationChanged();
	void inputParamsChanged();
	float* applyEffects(const float *image, const TensorShape &shape,
		bool flipHorizontally, bool flipVertically, bool makeGrayscale,
		const std::tuple<TensorShape,std::vector<float>> &convolution, unsigned convolutionCount) const;
	void clearEffects();
	void updateNetworkDetailsPage();
	void updateSourceImageOnScreen();
	void updateCurrentRegionText();
	void updateResultInterpretation();
	void updateResultInterpretationSummaryText(bool enable, const QString &oneLine, const QString &details);
	std::array<unsigned,4> getVisibleImageRegion() const;
	void updateSectionWidgetsVisibility();
	void onOpenNeuralNetworkFileUserIntent();
	void closeNeuralNetwork();
	static QLabel* makeTextSelectable(QLabel *label);
	void showNnTensorData2D();
	void clearNnTensorData2D();
};

