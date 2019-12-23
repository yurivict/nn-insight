
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
, layout(this)
, headerWidget(this)
,   headerLayout(&headerWidget)
,   shapeLabel(S2Q(STR("Shape: " << shape)), &headerWidget)
,   dataRangeLabel(&headerWidget)
,   colorSchemaLabel("Color scheme:", &headerWidget)
,   colorSchemaComboBox(&headerWidget)
, tableView(this)
{
	assert(shape.size() > 1); // otherwise :DataTable1D should be used

	layout.addWidget(&headerWidget);
	  headerLayout.addWidget(&shapeLabel);
	  headerLayout.addWidget(&dataRangeLabel);
	  headerLayout.addWidget(&colorSchemaLabel);
	  headerLayout.addWidget(&colorSchemaComboBox);
	layout.addWidget(&tableView);

	// alignment
	colorSchemaLabel.setAlignment(Qt::AlignRight|Qt::AlignVCenter);

	// size policies
	headerWidget.setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::Fixed);
	tableView   .setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding);

	// data range
	auto dataRange = Util::arrayMinMax(data, tensorFlatSize(shape));
	dataRangeLabel.setText(QString("Data Range: %1..%2").arg(std::get<0>(dataRange)).arg(std::get<1>(dataRange)));

	// create the model
	switch (shape.size()) {
	case 2:
		tableModel.reset(new DataModel(new TensorSliceDataSource(shape, 0, 1, {0,0}, data), nullptr, &tableView));
		break;
	case 3:
		tableModel.reset(new DataModel(new TensorSliceDataSource(shape, 0, 1, {0,0,1}, data), nullptr, &tableView));
		break;
	case 4:
		tableModel.reset(new DataModel(new TensorSliceDataSource(shape, 1, 2, {1,0,0,1}, data), nullptr, &tableView));
		break;
	default:
		assert(false);
	}
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
}


