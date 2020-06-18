// Copyright (C) 2020 by Yuri Victorovich. All rights reserved.

#include "misc.h"
#include "rng.h"
#include "training.h"
#include "training-widget.h"
#include "util.h"
#include "3rdparty/flowlayout/flowlayout.h"

#include <QDateTime>
#include <QDialog>
#include <QDialogButtonBox>
#include <QLineEdit>
#include <QDoubleValidator>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QLabel>
#include <QSettings>
#include <QSpinBox>
#include <QTextEdit>

#include <array>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include <exprtk.hpp>


/// local types


/// Training dataset type widgets

class DataSet_FunctionApproximationByFormulaWidget : public QWidget {
private: // types
	typedef float T;
	typedef exprtk::symbol_table<T> symbol_table_t;
	typedef exprtk::expression<T>     expression_t;
	typedef exprtk::parser<T>             parser_t;

	class ArgDescriptionWidget : public QWidget {
		std::string          argName;
		QHBoxLayout          layout;
		QLabel               minLabel;
		QLineEdit            minEdit;
		QLabel               maxLabel;
		QLineEdit            maxEdit;
	public:
		ArgDescriptionWidget(
			const std::string &argName_,
			float minVal, float maxVal, std::function<void(float)> onMinValChange, std::function<void(float)> onMaxValChange,
			QWidget *parent)
		: QWidget(parent)
		, argName(argName_)
		, layout(this)
		, minLabel(QString(tr("Arg %1 min:")).arg(S2Q(argName)))
		, minEdit(this)
		, maxLabel(tr("max:"))
		, maxEdit(this)
		{
			// add widgets to layouts
			layout.addWidget(&minLabel);
			layout.addWidget(&minEdit);
			layout.addWidget(&maxLabel);
			layout.addWidget(&maxEdit);

			// policies
			for (auto w : {(QWidget*)&minLabel,(QWidget*)&minEdit,(QWidget*)&maxLabel,(QWidget*)&maxEdit})
				w->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);

			// alignment
			minLabel.setAlignment(Qt::AlignRight|Qt::AlignVCenter);
			maxLabel.setAlignment(Qt::AlignRight|Qt::AlignVCenter);

			// validators
			minEdit.setValidator(new QDoubleValidator(this));
			maxEdit.setValidator(new QDoubleValidator(this));

			// values
			minEdit.setText(S2Q(STR(minVal)));
			maxEdit.setText(S2Q(STR(maxVal)));

			// margins
			minEdit.setContentsMargins(0,0,0,0);
			maxEdit.setContentsMargins(0,0,0,0);
			setContentsMargins(0,0,0,0);
			layout.setContentsMargins(0,0,0,0);

			// signals
			connect(&minEdit, &QLineEdit::textChanged, [onMinValChange](const QString &text) {
				onMinValChange(text.toFloat());
			});
			connect(&maxEdit, &QLineEdit::textChanged, [onMaxValChange](const QString &text) {
				onMaxValChange(text.toFloat());
			});
		}
		~ArgDescriptionWidget() {
		}
	};

private: // data
	QGridLayout                  layout;
	QLabel                       formulaLabel;
	QLineEdit                    formulaEdit;
	QLabel                       formulaErrorExcl;
	QLabel                       argumentCountLabel;
	QSpinBox                     argumentCountSpinBox;
	QWidget                      argDescriptionsWidget;
	FlowLayout                     argDescriptionsLayout;
	std::vector<std::unique_ptr<ArgDescriptionWidget>> argDescriptionWidgets;

	symbol_table_t           symbolTable;
	std::vector<std::string> symbolNames;
	std::vector<T>           symbolValues;
	expression_t             expression;

	std::vector<std::array<T,2>> argumentRanges;

public:
	DataSet_FunctionApproximationByFormulaWidget(QWidget *parent)
	: QWidget(parent)
	, layout(this)
	, formulaLabel(tr("Formula"), this)
	, formulaEdit(this)
	, formulaErrorExcl("‼", this)
	, argumentCountLabel(tr("Num. Args"), this)
	, argumentCountSpinBox(this)
	, argDescriptionsWidget(this)
	,   argDescriptionsLayout(&argDescriptionsWidget)
	{
		// add widgets to layouts
		layout.addWidget(&formulaLabel,          0,   0/*column*/);
		layout.addWidget(&formulaEdit,           0,   1/*column*/);
		layout.addWidget(&formulaErrorExcl,      0,   2/*column*/);
		layout.addWidget(&argumentCountLabel,    0,   3/*column*/);
		layout.addWidget(&argumentCountSpinBox,  0,   4/*column*/);
		layout.addWidget(&argDescriptionsWidget, 1,   0/*column*/, 1/*rowSpan*/, 5/*columnSpan*/);

		// spacing
		argDescriptionsLayout.setSpacing(0);
		argDescriptionsLayout.setContentsMargins(0,0,0,0);

		// alignment
		formulaLabel      .setAlignment(Qt::AlignRight|Qt::AlignVCenter);
		argumentCountLabel.setAlignment(Qt::AlignRight|Qt::AlignVCenter);

		// widget states
		formulaEdit.setText(appSettings.value("TrainingWidget.DataSet_FunctionApproximationByFormulaWidget.formula", QVariant("sin(a*b)+tan(a-b)*cos(a-b)")).toString());
		formulaErrorExcl.setStyleSheet("QLabel {color: red;}");
		argumentCountSpinBox.setRange(1,32); // limited by the alphabet
		argumentCountSpinBox.setValue(appSettings.value("TrainingWidget.DataSet_FunctionApproximationByFormulaWidget.numArgs", QVariant(2)).toUInt());

		// signals
		connect(&formulaEdit, &QLineEdit::textChanged, [this](const QString &text) {
			checkFormula(text);
		});
		connect(&argumentCountSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), [this](int i) {
			updateNumArguments();
			checkFormula();
		});

		// update
		updateNumArguments();
		checkFormula();
	}
	~DataSet_FunctionApproximationByFormulaWidget() {
		appSettings.setValue("TrainingWidget.DataSet_FunctionApproximationByFormulaWidget.formula", formulaEdit.text());
		appSettings.setValue("TrainingWidget.DataSet_FunctionApproximationByFormulaWidget.numArgs", argumentCountSpinBox.value());
	}

public: // iface
	std::array<std::vector<float>,2> getData(bool validation) {
		static std::random_device rd;
		static std::mt19937 gen(rd());
		// set arguments
		for (unsigned a = 0, ae = symbolValues.size(); a < ae; a++)
			symbolValues[a] = std::uniform_real_distribution<float>(argumentRanges[a][0], argumentRanges[a][1])(Rng::generator);
		// compute the function value
		return {symbolValues, {expression.value()}};
	}

private:
	void checkFormula() {
		checkFormula(formulaEdit.text());
	}
	void checkFormula(const QString &text) {
		parser_t parser;
		if (!parser.compile(Q2S(text), expression))
			formulaErrorExcl.show();
		else
			formulaErrorExcl.hide();
	}
	void updateNumArguments() {
		unsigned numArgs = argumentCountSpinBox.value();

		symbolTable = symbol_table_t();
		expression = expression_t();
		symbolNames   .resize(numArgs);
		symbolValues  .resize(numArgs);
		argumentRanges.resize(numArgs);

		char argName = 'a';
		for (unsigned i = 0; i < numArgs; i++, argName++) {
			auto argNameStr = STR(argName);
			symbolNames[i] = argNameStr;
			symbolTable.add_variable(argNameStr.c_str(), symbolValues[i]);
		}
		expression.register_symbol_table(symbolTable);

		// update argument widgets
		argDescriptionWidgets.clear();
		argDescriptionWidgets.resize(numArgs);
		for (unsigned i = 0; i < numArgs; i++, argName++) {
			auto const &name = symbolNames[i];
			auto &range = argumentRanges[i];
			range = {
				appSettings.value(QString("TrainingWidget.DataSet_FunctionApproximationByFormulaWidget.arg-%1.min").arg(S2Q(name)), QVariant(0.)).toFloat(),
				appSettings.value(QString("TrainingWidget.DataSet_FunctionApproximationByFormulaWidget.arg-%1.max").arg(S2Q(name)), QVariant(1.)).toFloat()
			};
			argDescriptionWidgets[i].reset(new ArgDescriptionWidget(symbolNames[i], range[0], range[1],
				[&name,&range](float newMinVal) {
					range[0] = newMinVal;
					appSettings.setValue(QString("TrainingWidget.DataSet_FunctionApproximationByFormulaWidget.arg-%1.min").arg(S2Q(name)), newMinVal);
				},
				[&name,&range](float newMaxVal) {
					range[1] = newMaxVal;
					appSettings.setValue(QString("TrainingWidget.DataSet_FunctionApproximationByFormulaWidget.arg-%1.max").arg(S2Q(name)), newMaxVal);
				},
				&argDescriptionsWidget
			));
			argDescriptionsLayout.addWidget(argDescriptionWidgets[i].get());
			argDescriptionWidgets[i]->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
		}
	}
};

class DataSet_FunctionApproximationFromTabulatedData : public QWidget {
	QGridLayout        layout;
	QLabel             xxxLabel;
public:
	DataSet_FunctionApproximationFromTabulatedData(QWidget *parent)
	: QWidget(parent)
	, layout(this)
	, xxxLabel(tr("FunctionApproximationFromTabulatedData"), this)
	{
		// add widgets to layouts
		layout.addWidget(&xxxLabel,     0,   0/*column*/);
	}
};

class DataSet_ImageLabeling : public QWidget {
	QGridLayout        layout;
	QLabel             xxxLabel;
public:
	DataSet_ImageLabeling(QWidget *parent)
	: QWidget(parent)
	, layout(this)
	, xxxLabel(tr("ImageLabeling"), this)
	{
		// add widgets to layouts
		layout.addWidget(&xxxLabel,     0,   0/*column*/);
	}
};

/// Training thread

class TrainingThread : public QThread { // wrapper thread for Training::runTrainingLoop
	Q_OBJECT

	PluginInterface::Model                               *model;
	float                                                 modelPendingTrainingDerivativesCoefficient;
	unsigned                                              batchSize;
	float                                                 learningRate;
	unsigned                                              maxBatches;
	bool                                                 *stopFlag;
	std::function<std::array<std::vector<float>,2>(bool)> getData;

public:
	TrainingThread(QObject *parent
		, PluginInterface::Model *model_
		, float modelPendingTrainingDerivativesCoefficient_
		, unsigned batchSize_, float learningRate_, unsigned maxBatches_, bool *stopFlag_
		, std::function<std::array<std::vector<float>,2>(bool)> getData_)
	: QThread(parent)
	, model(model_)
	, modelPendingTrainingDerivativesCoefficient(modelPendingTrainingDerivativesCoefficient_)
	, batchSize(batchSize_)
	, learningRate(learningRate_)
	, maxBatches(maxBatches_)
	, stopFlag(stopFlag_)
	, getData(getData_)
	{ }

	void run() override {
		Training::runTrainingLoop(
			model, modelPendingTrainingDerivativesCoefficient,
			batchSize, learningRate, maxBatches,
			stopFlag,
			getData,
			[this](unsigned batchNo, float avgLoss) {
				emit batchDone(batchNo, avgLoss); // channel batchDone through the signal that crosses threads
			}
		);
	}
signals:
	void batchDone(unsigned batchNo, float avgLoss);
};

/// TrainingWidget methods

TrainingWidget::TrainingWidget(QWidget *parent, QWidget *topLevelWidget, PluginInterface::Model *model, float modelPendingTrainingDerivativesCoefficient)
: QWidget(parent)
, layout(this)
, trainingTypeLabel("Type of Training", this)
, trainingTypeComboBox(this)
, dataSetGroupBox(tr("Training Data"), this)
,   dataSetLayout(&dataSetGroupBox)
, parametersGroupBox(tr("Training Parameters"), this)
,   parametersLayout(&parametersGroupBox)
,   paramBatchSizeLabel(tr("Batch Size"), &parametersGroupBox)
,   paramBatchSizeSpinBox(&parametersGroupBox)
,   paramLearningRateLabel(tr("Learning Rate"), &parametersGroupBox)
,   paramLearningRateSpinBox(&parametersGroupBox)
,   paramMaxBatchesLabel(tr("Max Batches"), &parametersGroupBox)
,   paramMaxBatchesSpinBox(&parametersGroupBox)
, verifyDerivativesButton(tr("Verify Derivatives"), this)
, trainButton(tr("Start Training"), this)
, trainingStats(this)
, trainingType((TrainingType)appSettings.value("TrainingWidget.trainingType", QVariant(TrainingType_FunctionApproximationByFormula)).toUInt())
, threadStopFlag(false)
{
	// helpers
	auto enableOtherWidgets = [this](bool enable) {
		for (auto w : {(QWidget*)&trainingTypeLabel,(QWidget*)&trainingTypeComboBox,(QWidget*)&dataSetGroupBox,(QWidget*)&parametersGroupBox,(QWidget*)&verifyDerivativesButton})
			w->setEnabled(enable);
	};
	auto updateLearningRateSpinBoxStep = [this]() {
		auto val = paramLearningRateSpinBox.value();
		if (val <= 0.015)
			paramLearningRateSpinBox.setSingleStep(0.0001);
		else if (val <= 0.15)
			paramLearningRateSpinBox.setSingleStep(0.001);
		else
			paramLearningRateSpinBox.setSingleStep(0.01);
	};

	// add widgets to layouts
	layout.addWidget(&trainingTypeLabel,       0,   0/*column*/);
	layout.addWidget(&trainingTypeComboBox,    0,   1/*column*/);
	layout.addWidget(&dataSetGroupBox,         1,   0/*column*/,  1/*rowSpan*/, 2/*columnSpan*/);
	layout.addWidget(&parametersGroupBox,      2,   0/*column*/,  1/*rowSpan*/, 2/*columnSpan*/);
	  parametersLayout.addWidget(&paramBatchSizeLabel);
	  parametersLayout.addWidget(&paramBatchSizeSpinBox);
	  parametersLayout.addWidget(&paramLearningRateLabel);
	  parametersLayout.addWidget(&paramLearningRateSpinBox);
	  parametersLayout.addWidget(&paramMaxBatchesLabel);
	  parametersLayout.addWidget(&paramMaxBatchesSpinBox);
	layout.addWidget(&verifyDerivativesButton, 3,   0/*column*/,  1/*rowSpan*/, 2/*columnSpan*/);
	layout.addWidget(&trainButton,             4,   0/*column*/,  1/*rowSpan*/, 2/*columnSpan*/);
	layout.addWidget(&trainingStats,           5,   0/*column*/,  1/*rowSpan*/, 2/*columnSpan*/);

	// fill comboboxes
	trainingTypeComboBox.addItem(tr("Function approximation (by formula)"), TrainingType_FunctionApproximationByFormula);
	trainingTypeComboBox.addItem(tr("Function approximation (from tabulated data)"), TrainingType_FunctionApproximationFromTabulatedData);
	trainingTypeComboBox.addItem(tr("Image labeling"), TrainingType_ImageLabeling);

	// alignment
	for (auto l : {&trainingTypeLabel,&paramBatchSizeLabel,&paramLearningRateLabel,&paramMaxBatchesLabel})
		l->setAlignment(Qt::AlignRight|Qt::AlignVCenter);

	// validation
	paramBatchSizeSpinBox   .setRange(1, std::numeric_limits<int>::max());
	paramLearningRateSpinBox.setRange(0, 0.5);
	paramLearningRateSpinBox.setDecimals(4);
	paramMaxBatchesSpinBox  .setRange(1, std::numeric_limits<int>::max());

	// initialize values
	paramBatchSizeSpinBox   .setValue(appSettings.value("TrainingWidget.Options.BatchSize", 100).toUInt());
	paramLearningRateSpinBox.setValue(appSettings.value("TrainingWidget.Options.LearningRate", 0.01).toDouble());
	paramMaxBatchesSpinBox  .setValue(appSettings.value("TrainingWidget.Options.MaxBatches", 1000).toUInt());


	// widget states
	trainingTypeComboBox.setCurrentIndex(trainingTypeComboBox.findData(trainingType));
	updateLearningRateSpinBoxStep();
	trainingStats.hide();

	// tooltips
	for (auto l : {(QWidget*)&trainingTypeLabel,(QWidget*)&trainingTypeComboBox})
		l->setToolTip(tr("Choose the training type to perform"));
	for (auto l : {(QWidget*)&paramBatchSizeLabel,(QWidget*)&paramBatchSizeSpinBox})
		l->setToolTip(tr("Choose the batch size for the training process. Batch size is how many training iterations to perform between weight updates."));
	for (auto l : {(QWidget*)&paramLearningRateLabel,(QWidget*)&paramLearningRateSpinBox})
		l->setToolTip(tr("Choose the learning rate for the training process. Learning rate is a coefficient telling how much to update model weights after each batch."));
	for (auto l : {(QWidget*)&paramMaxBatchesLabel,(QWidget*)&paramMaxBatchesSpinBox})
		l->setToolTip(tr("Choose the maximum number of batches before the training stops."));

	// process signals
	connect(&trainingTypeComboBox, QOverload<int>::of(&QComboBox::activated), [this](int index) {
		trainingType = (TrainingType)trainingTypeComboBox.itemData(index).toUInt();
		updateTrainingType();
	});
	connect(&paramLearningRateSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged), [updateLearningRateSpinBoxStep](double) {
		updateLearningRateSpinBoxStep();
	});
	connect(&verifyDerivativesButton, &QAbstractButton::pressed, [this,model,modelPendingTrainingDerivativesCoefficient,topLevelWidget]() {
		auto msg = Training::verifyDerivatives(
			model, modelPendingTrainingDerivativesCoefficient,
			10/*numVerifications*/, 10/*numPoints*/, 0.001/*delta*/, 0.05/*tolerance*/,
			[this](bool validation) -> std::array<std::vector<float>,2> {
				return getData(validation);
			}
		);
		QDialog          dlg(this);
		dlg.setWindowTitle(tr("Derivatives Verification Report"));
		QVBoxLayout      dbgLayout(&dlg);
		QTextEdit        dlgText(&dlg);
		QDialogButtonBox dlgButtonBox(QDialogButtonBox::Ok, &dlg);
		dbgLayout.addWidget(&dlgText);
		dbgLayout.addWidget(&dlgButtonBox);
		dlgText.setReadOnly(true);
		dlgText.setText(S2Q(msg));
		connect(&dlgButtonBox, SIGNAL(accepted()), &dlg, SLOT(accept()));
		Util::centerWidgetAtOtherWidget(&dlg, topLevelWidget, 0.75/*fraction*/);
		dlg.exec();
	});
	connect(&trainButton, &QAbstractButton::pressed, [this,model,modelPendingTrainingDerivativesCoefficient,enableOtherWidgets]() {
		if (!trainingThread) { // start training
			// disable other widgets for the ducration of training
			enableOtherWidgets(false);
			// start the training thread
			threadStopFlag = false;
			auto tmStart = QDateTime::currentMSecsSinceEpoch();
			auto updateStats = [this,tmStart](unsigned batchNo, float avgLoss) {
				auto tmNow = QDateTime::currentMSecsSinceEpoch();
				trainingStats.show();
				trainingStats.setText(QString(tr("Training stats: done %1 batches in %2 milliseconds, loss=%3")).arg(batchNo).arg(tmNow-tmStart).arg(avgLoss));
			};
			updateStats(0,0);
			trainingThread.reset(new TrainingThread(this,
				model, modelPendingTrainingDerivativesCoefficient,
				paramBatchSizeSpinBox.value(), paramLearningRateSpinBox.value(), paramMaxBatchesSpinBox.value(),
				&threadStopFlag,
				[this](bool validation) -> std::array<std::vector<float>,2> { // data is pulled through a synchronous callback from the training thread
					return getData(validation);
				}
			));
			connect((TrainingThread*)trainingThread.get(), &TrainingThread::batchDone, QThread::currentThread(), [updateStats](unsigned batchNo, float avgLoss) {
				updateStats(batchNo, avgLoss);
			}, Qt::QueuedConnection);
			connect(trainingThread.get(), &TrainingThread::finished, QThread::currentThread(), [this,enableOtherWidgets]() {
				// delete the thread object
				trainingThread.reset(nullptr);
				// restore widgets to their normal state
				trainButton.setText(tr("Start Training"));
				enableOtherWidgets(true);
				trainingStats.hide();
			}, Qt::QueuedConnection);
			trainButton.setText(tr("Stop Training"));
			trainingThread->start();
		} else { // stop training
			threadStopFlag = true;
		}
	});

	// set data set widget initially
	updateTrainingType();
}

TrainingWidget::~TrainingWidget() {
	appSettings.setValue("TrainingWidget.trainingType", trainingType);
	appSettings.setValue("TrainingWidget.Options.BatchSize", paramBatchSizeSpinBox.value());
	appSettings.setValue("TrainingWidget.Options.LearningRate", paramLearningRateSpinBox.value());
	appSettings.setValue("TrainingWidget.Options.MaxBatches", paramMaxBatchesSpinBox.value());
}

// privates

void TrainingWidget::updateTrainingType() {
	switch ((TrainingType)trainingTypeComboBox.itemData(trainingTypeComboBox.currentIndex()).toUInt()) {
	case TrainingType_FunctionApproximationByFormula:
		dataSetWidget.reset(new DataSet_FunctionApproximationByFormulaWidget(&dataSetGroupBox));
		break;
	case TrainingType_FunctionApproximationFromTabulatedData:
		dataSetWidget.reset(new DataSet_FunctionApproximationFromTabulatedData(&dataSetGroupBox));
		break;
	case TrainingType_ImageLabeling:
		dataSetWidget.reset(new DataSet_ImageLabeling(&dataSetGroupBox));
		break;
	}
	dataSetLayout.addWidget(dataSetWidget.get());
}

std::array<std::vector<float>,2> TrainingWidget::getData(bool validation) const {
	switch (trainingType) {
	case TrainingType_FunctionApproximationByFormula:
		return ((DataSet_FunctionApproximationByFormulaWidget*)dataSetWidget.get())->getData(validation);
	default:
		assert(false);
		return {};
	}
}

#include "training-widget.moc" // because Q_OBJECT is in some local classes
