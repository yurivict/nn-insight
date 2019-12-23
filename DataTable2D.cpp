
#include "DataTable2D.h"
#include "nn-types.h"
#include "misc.h"
#include "util.h"

#include <assert.h>

class DataSource {
	unsigned numRows;
	unsigned numColumns;
public:
	DataSource(unsigned numRows_, unsigned numColumns_)
	: numRows(numRows_)
	, numColumns(numColumns_)
	{
	}
	virtual ~DataSource() { }

public: // iface
	unsigned nrows() const {
		return numRows;
	}
	unsigned ncols() const {
		return numColumns;
	}
	virtual float value(unsigned r, unsigned c) const = 0;
};

class TensorSliceDataSource : public DataSource {
	const TensorShape               shape;
	const float*                    data;
	unsigned                        idxVertical;
	unsigned                        idxHorizontal;
	mutable std::vector<unsigned>   fixedIdxs;

public:
	TensorSliceDataSource(const TensorShape &shape_, unsigned idxVertical_, unsigned idxHorizontal_, const std::vector<unsigned> &fixedIdxs_, const float *data_)
	: DataSource(shape_[idxVertical_], shape_[idxHorizontal_])
	, shape(shape_)
	, data(data_)
	, idxVertical(idxVertical_)
	, idxHorizontal(idxHorizontal_)
	, fixedIdxs(fixedIdxs_)
	{
	}

	float value(unsigned r, unsigned c) const override {
		fixedIdxs[idxVertical] = r;
		fixedIdxs[idxHorizontal] = c;
		return data[offset(fixedIdxs, shape)];
	}

private:
	static unsigned offset(std::vector<unsigned> &idxs, const TensorShape &shape) {
		unsigned off = 0;
		for (unsigned i = 0, ie = shape.size(); i < ie; i++) {
			off *= shape[i];
			off += idxs[i];
		}
		return off;
	}
};

class DataModel : public QAbstractTableModel {
	std::unique_ptr<const DataSource> dataSource;
public:
	DataModel(const DataSource *dataSource_, QObject *parent)
	: QAbstractTableModel(parent)
	, dataSource(dataSource_)
	{
	}
	int rowCount(const QModelIndex &parent = QModelIndex()) const override {
		return dataSource->nrows();
	}
	int columnCount(const QModelIndex &parent = QModelIndex()) const override {
		return dataSource->ncols();
	}
	QVariant data(const QModelIndex &index, int role = Qt::DisplayRole) const override {
		if (role == Qt::DisplayRole) {
			return QVariant(dataSource->value(index.row(), index.column()));
		}
		return QVariant();
	}
};

DataTable2D::DataTable2D(const TensorShape &shape_, const float *data_, QWidget *parent)
: QWidget(parent)
, shape(shape_)
, data(data_)
, layout(this)
, headerWidget(this)
,   headerLayout(&headerWidget)
,   shapeLabel(S2Q(STR("Shape: " << shape)), &headerWidget)
,   dataRangeLabel(&headerWidget)
, tableView(this)
{
	assert(shape.size() > 1); // otherwise :DataTable1D should be used

	layout.addWidget(&headerWidget);
	  headerLayout.addWidget(&shapeLabel);
	  headerLayout.addWidget(&dataRangeLabel);
	layout.addWidget(&tableView);

	// size policies
	headerWidget.setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::Fixed);
	tableView   .setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding);

	// compute required values
	{ // data range
		auto range = Util::arrayMinMax(data, tensorFlatSize(shape));
		dataRangeLabel.setText(QString("Data Range: %1..%2").arg(std::get<0>(range)).arg(std::get<1>(range)));
	}

	// create the model
	switch (shape.size()) {
	case 2:
		tableModel.reset(new DataModel(new TensorSliceDataSource(shape, 0, 1, {0,0}, data), &tableView));
		break;
	case 3:
		tableModel.reset(new DataModel(new TensorSliceDataSource(shape, 0, 1, {0,0,1}, data), &tableView));
		break;
	case 4:
		tableModel.reset(new DataModel(new TensorSliceDataSource(shape, 1, 2, {1,0,0,1}, data), &tableView));
		break;
	default:
		assert(false);
	}
	tableView.setModel(tableModel.get());

	// tooltips
	shapeLabel    .setToolTip("Shape of tensor data that this table represents");
	dataRangeLabel.setToolTip("Range of numeric values present in the table");
}


