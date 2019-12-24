
#include "DataTable2D.h"
#include "nn-types.h"
#include "misc.h"
#include "util.h"

#include <cmath>

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

enum ColorSchemaEnum {
	COLORSCHEME_NONE,
	COLORSCHEME_BLUE_TO_RED,
	COLORSCHEME_RED_TO_BLUE,
	COLORSCHEME_GRAYSCALE_UP,
	COLORSCHEME_GRAYSCALE_DOWN,
	COLORSCHEME_GRAYSCALE_ZERO_UP,
	COLORSCHEME_GRAYSCALE_ZERO_DOWN,
	COLORSCHEME_GRAYSCALE_MIDDLE_UP,
	COLORSCHEME_GRAYSCALE_MIDDLE_DOWN
};

class BaseColorSchema {
public:
	virtual ~BaseColorSchema() { }
	virtual QColor color(float value) const = 0;
};

template<class ColorMapper, class ValueMapper>
class ColorSchema : public BaseColorSchema {
	float minValue;
	float maxValue;
public:
	ColorSchema(float minValue_, float maxValue_)
	: minValue(minValue_)
	, maxValue(maxValue_)
	{
	}
	~ColorSchema() { }

public:
	QColor color(float value) const override {
		return ColorMapper::mapColor(ValueMapper::mapValue(value, minValue, maxValue));
	}
};

class ValueMapperLinear {
public:
	static float mapValue(float value, float minValue, float maxValue) {
		return (value-minValue)/(maxValue-minValue);
	}
};
class ValueMapperFromZero {
public:
	static float mapValue(float value, float minValue, float maxValue) {
		return ValueMapperLinear::mapValue(std::abs(value), 0, std::max(std::abs(minValue), std::abs(maxValue)));
	}
};
class ValueMapperFromMiddle {
public:
	static float mapValue(float value, float minValue, float maxValue) {
		float mid = (minValue+maxValue)/2;
		return ValueMapperLinear::mapValue(value < mid ? mid - value : value, mid, maxValue);
	}
};

template<uint8_t Clr1R, uint8_t Clr1G, uint8_t Clr1B, uint8_t Clr2R, uint8_t Clr2G, uint8_t Clr2B>
class ColorMapper {
public:
	static QColor mapColor(float value) {
		return QColor(
			float(Clr1R)+value*(Clr2R-Clr1R),
			float(Clr1G)+value*(Clr2G-Clr1G),
			float(Clr1B)+value*(Clr2B-Clr1B)
		);
	}
};

class TextColorForBackground {
public:
	static QColor color(QColor clr) {
		return clr.valueF() > 0.7 ? Qt::black : Qt::white;
	}
};

static const BaseColorSchema* createColorSchema(ColorSchemaEnum colorSchema, float minValue, float maxValue) {
#define B 0   // "black"
#define W 255 // "white"
	switch (colorSchema) {
	case COLORSCHEME_NONE:
		return nullptr;
	case COLORSCHEME_BLUE_TO_RED:
		return new ColorSchema<ColorMapper<B,B,W, W,B,B>, ValueMapperLinear>(minValue, maxValue);
	case COLORSCHEME_RED_TO_BLUE:
		return new ColorSchema<ColorMapper<W,B,B, B,B,W>, ValueMapperLinear>(minValue, maxValue);
	case COLORSCHEME_GRAYSCALE_UP:
		return new ColorSchema<ColorMapper<B,B,B, W,W,W>, ValueMapperLinear>(minValue, maxValue);
	case COLORSCHEME_GRAYSCALE_DOWN:
		return new ColorSchema<ColorMapper<W,W,W, B,B,B>, ValueMapperLinear>(minValue, maxValue);
	case COLORSCHEME_GRAYSCALE_ZERO_UP:
		return new ColorSchema<ColorMapper<B,B,B, W,W,W>, ValueMapperFromZero>(minValue, maxValue);
	case COLORSCHEME_GRAYSCALE_ZERO_DOWN:
		return new ColorSchema<ColorMapper<W,W,W, B,B,B>, ValueMapperFromZero>(minValue, maxValue);
	case COLORSCHEME_GRAYSCALE_MIDDLE_UP:
		return new ColorSchema<ColorMapper<B,B,B, W,W,W>, ValueMapperFromMiddle>(minValue, maxValue);
	case COLORSCHEME_GRAYSCALE_MIDDLE_DOWN:
		return new ColorSchema<ColorMapper<W,W,W, B,B,B>, ValueMapperFromMiddle>(minValue, maxValue);
	}
#undef B
#undef W
}

class DataModel : public QAbstractTableModel {
	std::unique_ptr<const DataSource>      dataSource;
	std::unique_ptr<const BaseColorSchema> colorSchema;
public:
	DataModel(const DataSource *dataSource_, const BaseColorSchema *colorSchema_, QObject *parent)
	: QAbstractTableModel(parent)
	, dataSource(dataSource_)
	, colorSchema(colorSchema_)
	{
	}

public: // QAbstractTableModel interface implementation
	int rowCount(const QModelIndex &parent = QModelIndex()) const override {
		return dataSource->nrows();
	}
	int columnCount(const QModelIndex &parent = QModelIndex()) const override {
		return dataSource->ncols();
	}
	QVariant data(const QModelIndex &index, int role) const override {
		switch (role) {
		case Qt::DisplayRole: // text
			return QVariant(dataSource->value(index.row(), index.column()));
		case Qt::BackgroundRole: // background
			return colorSchema ? colorSchema->color(dataSource->value(index.row(), index.column())) : QVariant();
		case Qt::ForegroundRole: // text color
			return colorSchema ? TextColorForBackground::color(colorSchema->color(dataSource->value(index.row(), index.column()))) : QVariant();
		default:
			return QVariant();
		}
	}

public: // custom interface
	void setDataSource(const DataSource *dataSource_) {
		beginResetModel();
		dataSource.reset(dataSource_);
		endResetModel();
	}
	void setColorSchema(const BaseColorSchema *colorSchema_) {
		beginResetModel();
		colorSchema.reset(colorSchema_);
		endResetModel();
	}
};

DataTable2D::DataTable2D(const TensorShape &shape_, const float *data_, QWidget *parent)
: QWidget(parent)
, shape(shape_)
, data(data_)
, dimVertical(0)
, dimHorizontal(0)
, layout(this)
, headerWidget(this)
,   headerLayout(&headerWidget)
,   shapeLabel(S2Q(STR("Shape: " << shape)), &headerWidget)
,   shapeDimensionsWidget(&headerWidget)
,     shapeDimensionsLayout(&shapeDimensionsWidget)
,   dataRangeLabel(&headerWidget)
,   colorSchemaLabel("Color scheme:", &headerWidget)
,   colorSchemaComboBox(&headerWidget)
, tableView(this)
{
	assert(shape.size() > 1); // otherwise DataTable1D should be used

	layout.addWidget(&headerWidget);
	  headerLayout.addWidget(&shapeLabel);
	  headerLayout.addWidget(&shapeDimensionsWidget);
	  headerLayout.addWidget(&dataRangeLabel);
	  headerLayout.addWidget(&colorSchemaLabel);
	  headerLayout.addWidget(&colorSchemaComboBox);
	layout.addWidget(&tableView);

	{ // create comboboxes for shape dimensions
		unsigned numMultiDims = tensorNumMultiDims(shape);
		//std::vector<bool> multiDims = tensorGetMultiDims(shape);
		unsigned dim = 0;
		for (auto d : shape) {
			auto combobox = new QComboBox(&shapeDimensionsWidget);
			shapeDimensionsLayout.addWidget(combobox);
			if (d == 1) { // dimensions=1 are excluded from selection
				combobox->addItem("single 1");
				combobox->setEnabled(false);
			} else {
				combobox->addItem("--X (columns)--");
				combobox->addItem("--Y (rows)--");
				if (numMultiDims > 2) {
					for (unsigned index = 0; index < d; index++)
						combobox->addItem(QString("index: %1").arg(index+1));
				}
			}
			connect(combobox, QOverload<int>::of(&QComboBox::activated), [this,dim](int index) {
				auto swapXY = [this,dim](unsigned &dimMy, unsigned &dimOther, bool iAmY) {
					unsigned otherDim = dimOther;
					dimMy = dimOther;
					dimOther = dim;
					shapeDimensionsComboBoxes[otherDim]->setCurrentIndex(iAmY ? 1/*Y*/ : 0/*X*/);
					(static_cast<DataModel*>(tableModel.get()))->setDataSource(new TensorSliceDataSource(shape, dimVertical, dimHorizontal, mkIdxs(), data));
				};
				auto changeXYtoIndex = [this](unsigned &dimChanged, bool iAmY) {
					// choose another X/Y
					for (unsigned dim = 0; dim < shape.size(); dim++)
						if (shape[dim] > 1 && dim!=dimVertical && dim!=dimHorizontal) {
							dimChanged = dim;
							shapeDimensionsComboBoxes[dim]->setCurrentIndex(iAmY ? 1/*Y*/ : 0/*X*/);
							break;
						}
					(static_cast<DataModel*>(tableModel.get()))->setDataSource(new TensorSliceDataSource(shape, dimVertical, dimHorizontal, mkIdxs(), data));
				};
				auto changeIndexToXY = [this,dim](unsigned &dimChanged) {
					shapeDimensionsComboBoxes[dimChanged]->setCurrentIndex(2/*index=1*/);
					dimChanged = dim;
					(static_cast<DataModel*>(tableModel.get()))->setDataSource(new TensorSliceDataSource(shape, dimVertical, dimHorizontal, mkIdxs(), data));
				};
				auto changeIndexToIndex = [this]() {
					(static_cast<DataModel*>(tableModel.get()))->setDataSource(new TensorSliceDataSource(shape, dimVertical, dimHorizontal, mkIdxs(), data));
				};
				switch (index) {
				case 0: // change to X
					if (dim==dimHorizontal)
						return; // same
					else if (dim==dimVertical) // Y->X
						swapXY(dimVertical, dimHorizontal, true/*iAmY*/);
					else { // other index -> X
						changeIndexToXY(dimHorizontal);
					}
					break;
				case 1: // change to Y
					if (dim==dimVertical)
						return; // same
					else if (dim==dimHorizontal) // X->Y
						swapXY(dimHorizontal, dimVertical, false/*iAmY*/);
					else { // other index -> Y
						changeIndexToXY(dimVertical);
					}
					break;
				default: // change to index
					if (dim==dimVertical) // Y->index
						changeXYtoIndex(dimVertical, true/*iAmY*/);
					else if (dim==dimHorizontal) // X->index
						changeXYtoIndex(dimHorizontal, false/*iAmY*/);
					else // index->index
						changeIndexToIndex();
				}
			});
			shapeDimensionsComboBoxes.push_back(std::unique_ptr<QComboBox>(combobox));
			combobox->setToolTip(S2Q(STR("Directions and indexes to choose for dimension number " << (dim+1) << " of the shape " << shape)));
			dim++;
		}

		// choose the initial vertical and horizontal dimensions
		switch (numMultiDims) {
		case 0:
		case 1:
			assert(false); // 0 is invalid, 1 should be handled by the 1D class
		default:
			// in case of 2 - choose the only two that are avalable, in other cases choose two before the very last one, because the last one is usually a channel dimension
			unsigned num = 0, numMatch = (numMultiDims==2 ? 0 : numMultiDims-3), idx = 0;
			for (auto d : shape) {
				if (d > 1) {
					if (num == numMatch)
						dimVertical = idx;
					else if (num == numMatch+1) {
						dimHorizontal = idx;
						break;
					}
					num++;
				}
				idx++;
			}
			break;
		}
		for (unsigned dim = 0; dim < shapeDimensionsComboBoxes.size(); dim++) {
			if (dim == dimVertical)
				shapeDimensionsComboBoxes[dim]->setCurrentIndex(1/*Y*/);
			else if (dim == dimHorizontal)
				shapeDimensionsComboBoxes[dim]->setCurrentIndex(0/*X*/);
			else
				shapeDimensionsComboBoxes[dim]->setCurrentIndex(2/*index=1*/);
		}
	}

	// alignment
	colorSchemaLabel.setAlignment(Qt::AlignRight|Qt::AlignVCenter);

	// size policies
	headerWidget.setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::Fixed);
	tableView   .setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding);

	// data range
	auto dataRange = Util::arrayMinMax(data, tensorFlatSize(shape));
	dataRangeLabel.setText(QString("Data Range: %1..%2").arg(std::get<0>(dataRange)).arg(std::get<1>(dataRange)));

	// create the model
	tableModel.reset(new DataModel(new TensorSliceDataSource(shape, dimVertical, dimHorizontal, mkIdxs(), data), nullptr, &tableView));
	tableView.setModel(tableModel.get());

	// combobox values
	colorSchemaComboBox.addItem("None (default)",        COLORSCHEME_NONE);
	colorSchemaComboBox.addItem("Blue-to-Red",           COLORSCHEME_BLUE_TO_RED);
	colorSchemaComboBox.addItem("Red-to-Blue",           COLORSCHEME_RED_TO_BLUE);
	colorSchemaComboBox.addItem("Grayscale Up",          COLORSCHEME_GRAYSCALE_UP);
	colorSchemaComboBox.addItem("Grayscale Down",        COLORSCHEME_GRAYSCALE_DOWN);
	colorSchemaComboBox.addItem("Grayscale Zero Up",     COLORSCHEME_GRAYSCALE_ZERO_UP);
	colorSchemaComboBox.addItem("Grayscale Zero Down",   COLORSCHEME_GRAYSCALE_ZERO_DOWN);
	colorSchemaComboBox.addItem("Grayscale Middle Up",   COLORSCHEME_GRAYSCALE_MIDDLE_UP);
	colorSchemaComboBox.addItem("Grayscale Middle Down", COLORSCHEME_GRAYSCALE_MIDDLE_DOWN);

	// connect signals
	connect(&colorSchemaComboBox, QOverload<int>::of(&QComboBox::activated), [this,dataRange](int index) {
		(static_cast<DataModel*>(tableModel.get()))->setColorSchema(createColorSchema(
			(ColorSchemaEnum)colorSchemaComboBox.itemData(index).toInt(),
			std::get<0>(dataRange),
			std::get<1>(dataRange)
		));
	});

	// tooltips
	shapeLabel         .setToolTip("Shape of tensor data that this table represents");
	dataRangeLabel     .setToolTip("Range of numeric values present in the table");
	colorSchemaComboBox.setToolTip("Change the color schema of data visualization");
	tableView          .setToolTip("Tensor data values");
}

/// internals

std::vector<unsigned> DataTable2D::mkIdxs() const {
	std::vector<unsigned> idxs;
	unsigned i = 0;
	for (auto d : shape) {
		if (i==dimVertical || i==dimHorizontal || d==1)
			idxs.push_back(0);
		else
			idxs.push_back(shapeDimensionsComboBoxes[i]->currentIndex()-2);
		i++;
	}
	return idxs;
}
