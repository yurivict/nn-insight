
#include "operators-list-widget.h"
#include "model-functions.h"
#include "misc.h"
#include "util.h"

#include <QHeaderView>

/// local types

enum OperatorsListColumns {
	OperatorsListColumns_No = 0,
	OperatorsListColumns_Kind,
	OperatorsListColumns_InsOuts,
	OperatorsListColumns_Complexity,
	OperatorsListColumns_StaticData,
	OperatorsListColumns_DataRatio,
	OperatorsListColumns_Count_ // pseudo-element = count of items
};

class OperatorsListModel : public QAbstractTableModel {
	const PluginInterface::Model *model;

public:
	OperatorsListModel(const PluginInterface::Model *model_, QObject *parent)
	: QAbstractTableModel(parent)
	, model(model_)
	{
	}

private: // QAbstractTableModel interface implementation
	int rowCount(const QModelIndex &parent = QModelIndex()) const override {
		return model->numOperators();
	}
	int columnCount(const QModelIndex &parent = QModelIndex()) const override {
		return OperatorsListColumns_Count_;
	}
	QVariant headerData(int section, Qt::Orientation orientation, int role) const override {
		static const QString headerLabels[OperatorsListColumns_Count_] = {
			tr("No"),
			tr("Kind"),
			tr("Ins/Outs"),
			tr("Complexity"),
			tr("Static Data"),
			tr("Data Ratio")
		};
		switch (orientation) {
		case Qt::Horizontal:
			switch (role) {
				case Qt::DisplayRole: // text
					return headerLabels[section];
				default:
					return QVariant();
			}
		default:
			return QVariant();
		}
	}
	QVariant data(const QModelIndex &index, int role) const override {
		switch (role) {
		case Qt::DisplayRole: // text
			switch ((OperatorsListColumns)index.column()) {
			case OperatorsListColumns_No: {
				return QVariant(index.row()+1);
			} case OperatorsListColumns_Kind: {
				return QVariant(S2Q(STR(model->getOperatorKind((PluginInterface::OperatorId)index.row()))));
			} case OperatorsListColumns_InsOuts: {
				std::vector<PluginInterface::TensorId> inputs, outputs;
				model->getOperatorIo(index.row(), inputs, outputs);
				return QString("%1→%2").arg(inputs.size()).arg(outputs.size());
			} case OperatorsListColumns_Complexity: {
				return S2Q(Util::formatFlops(ModelFunctions::computeOperatorFlops(model, (PluginInterface::OperatorId)index.row())));
			} case OperatorsListColumns_StaticData: {
				unsigned unused;
				return QString("%1 bytes").arg(S2Q(Util::formatUIntHumanReadable(
					ModelFunctions::sizeOfOperatorStaticData(model, (PluginInterface::OperatorId)index.row(), unused))));
			} case OperatorsListColumns_DataRatio: {
				float dataRateIncreaseOboveInput, modelInputToOut;
				return S2Q(ModelFunctions::dataRatioOfOperatorStr(model, (PluginInterface::OperatorId)index.row(),
					dataRateIncreaseOboveInput, modelInputToOut));
			} default:
				return QVariant();
			}
		case Qt::BackgroundRole: // background color
			switch ((OperatorsListColumns)index.column()) {
			case OperatorsListColumns_DataRatio: {
				float dataRateIncreaseOboveInput, modelInputToOut;
				(void)ModelFunctions::dataRatioOfOperatorStr(model, (PluginInterface::OperatorId)index.row(),
					dataRateIncreaseOboveInput, modelInputToOut);
				return dataRateIncreaseOboveInput<=1
					? QVariant()
					: modelInputToOut<2
						? QVariant(QColor(255,105,180)) // pink color
						: QVariant(QColor(Qt::red));
			} default:
				return QVariant();
			}
		default:
			return QVariant();
		}
	}
};

/// constructor

OperatorsListWidget::OperatorsListWidget(QWidget *parent)
: QTableView(parent)
, self(false)
{
	// selection behavior
	setSelectionBehavior(QAbstractItemView::SelectRows);
	setSelectionMode(QAbstractItemView::SingleSelection);

	//setWordWrap(true); // not sure
}

/// overridden
void OperatorsListWidget::selectionChanged(const QItemSelection &selected, const QItemSelection &deselected) {
	assert(selected.indexes().size()==0 || selected.indexes().size()==OperatorsListColumns_Count_);

	// emit signal
	if (!self && selected.indexes().size() == OperatorsListColumns_Count_)
		emit operatorSelected((PluginInterface::OperatorId)selected.indexes()[0].row());

	// pass
	QTableView::selectionChanged(selected, deselected);
}

/// interface

void OperatorsListWidget::setNnModel(const PluginInterface::Model *model) {
	tableModel.reset(new OperatorsListModel(model, this));
	setModel(tableModel.get());

	// set width/stretching behavior
	for (unsigned s = OperatorsListColumns_No; s < OperatorsListColumns_DataRatio; s++)
		horizontalHeader()->setSectionResizeMode(s,        QHeaderView::ResizeToContents);
	horizontalHeader()->setSectionResizeMode(OperatorsListColumns_DataRatio, QHeaderView::Stretch);
}

void OperatorsListWidget::clearNnModel() {
	tableModel.reset(nullptr);
	setModel(nullptr);
}

void OperatorsListWidget::selectOperator(PluginInterface::OperatorId operatorId) {
	self = true;
	selectRow((int)operatorId);
	self = false;
}
