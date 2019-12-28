#pragma once

#include <QMainWindow>
#include <QSplitter>
#include <QScrollArea>
#include <QStackedWidget>
#include <QGroupBox>
#include "ZoomableSvgWidget.h"
#include "DataTable2D.h"
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
	QLabel                                   outputInterpretationSummaryLineEdit;
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
	QPushButton                              computeButton;
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
	QScrollArea                            sourceImageScrollArea;
	QLabel                                   sourceImage;
	QStackedWidget                       detailsStack;
	QGroupBox                              noDetails;       // page#0
	QGroupBox                              operatorDetails; // page#1
	QGridLayout                              operatorDetailsLayout;
	QLabel                                   operatorTypeLabel;
	QLabel                                   operatorTypeValue;
	QLabel                                   operatorOptionsLabel;
	QLabel                                   operatorInputsLabel;
	QLabel                                   operatorOutputsLabel;
	QLabel                                   operatorComplexityLabel;
	QLabel                                   operatorComplexityValue;
	QGroupBox                              tensorDetails;   // page#2
	QLabel                               blankRhsLabel; // leftover label
	std::unique_ptr<DataTable2D>         dataTable;

	QMenuBar                       menuBar;
	QStatusBar                     statusBar;
#if defined(USE_PERFTOOLS)
	QLabel                         memoryUseLabel;
	QTimer                         memoryUseTimer;
#endif

	const PluginManager::Plugin*     plugin;    // plugin in use for the model
	std::unique_ptr<PluginInterface> pluginInterface; // the file is opened through this handle
	const PluginInterface::Model*    model;     // the model from the file that is currently open

	// data associated with a specific input data (image) currently loaded by the user (static tensors from the model aren't here)
	TensorShape                      sourceTensorShape;
	std::shared_ptr<float>           sourceTensorDataAsLoaded; // original image that was loaded by the user
	std::shared_ptr<float>           sourceTensorDataAsUsed;   // image that is used as an input of NN, might be different if effects are applied
	std::unique_ptr<std::vector<std::shared_ptr<const float>>>   tensorData; // tensors corresponding to the currently used image, shared because reshape/input often shared

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
	void openImageFile(const QString &imageFileName);
	void openImagePixmap(const QPixmap &imagePixmap, const QString &sourceName);
	void clearInputImageDisplay();
	void clearComputedTensorData();
	void effectsChanged();
	void inputNormalizationChanged();
	float* applyEffects(const float *image, const TensorShape &shape,
		bool flipHorizontally, bool flipVertically, bool makeGrayscale,
		const std::tuple<TensorShape,std::vector<float>> &convolution, unsigned convolutionCount) const;
	void clearEffects();
	void updateSourceImageOnScreen();
	void updateResultInterpretationSummary(bool enable, const QString &oneLine, const QString &details);
};

