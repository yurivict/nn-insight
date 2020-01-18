// Copyright (C) 2020 by Yuri Victorovich. All rights reserved.

#include "data-table-2d.h"
#include "tensor.h"
#include "misc.h"
#include "util.h"

#include <QFontMetrics>
#include <QImage>
#include <QPixmap>
#include <QSettings>

#include <cmath>
#include <functional>
#include <sstream>
#include <tuple>

#include <assert.h>

/// local helper classes

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

public: // iface: data access
	unsigned nrows() const {
		return numRows;
	}
	unsigned ncols() const {
		return numColumns;
	}
	virtual float value(unsigned r, unsigned c) const = 0;

public: // iface: data computations
	std::tuple<float,float> computeMinMax() const {
		float dmin = std::numeric_limits<float>::max();
		float dmax = std::numeric_limits<float>::lowest();
		for (unsigned r = 0, re = nrows(); r < re; r++)
			for (unsigned c = 0, ce = ncols(); c < ce; c++) {
				auto d = value(r,c);
				if (d < dmin)
					dmin = d;
				if (d > dmax)
					dmax = d;
			}
		return std::tuple<float,float>(dmin, dmax);
	}
};

class TensorSliceDataSource : public DataSource {
	const TensorShape               shape;
	const float*                   &data; // shares data pointer with the DataTable2D instance
	unsigned                        idxVertical;
	unsigned                        idxHorizontal;
	mutable std::vector<unsigned>   fixedIdxs;

public:
	TensorSliceDataSource(const TensorShape &shape_, unsigned idxVertical_, unsigned idxHorizontal_, const std::vector<unsigned> &fixedIdxs_, const float *&data_)
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
	const DataSource* getDataSource() const {
		return dataSource.get();
	}
	void beginResetModel() {
		QAbstractTableModel::beginResetModel();
	}
	void endResetModel() {
		QAbstractTableModel::endResetModel();
	}
};

/// DataTable2D

DataTable2D::DataTable2D(const TensorShape &shape_, const float *data_, QWidget *parent)
: QWidget(parent)
, shape(shape_)
, data(data_)
, dimVertical(0)
, dimHorizontal(0)
, self(false)
, layout(this)
, headerWidget(this)
,   headerLayout(&headerWidget)
,   shapeLabel(QString(tr("Shape: %1")).arg(S2Q(STR(shape))), &headerWidget)
,   shapeDimensionsWidget(&headerWidget)
,     shapeDimensionsLayout(&shapeDimensionsWidget)
,   dataRangeLabel(&headerWidget)
,   colorSchemaLabel(tr("Color scheme:"), &headerWidget)
,   colorSchemaComboBox(&headerWidget)
, header1Widget(this)
,   header1Layout(&header1Widget)
,   filler1Widget(&header1Widget)
,   scaleBwImageLabel(tr("Scale image"), &header1Widget)
,   scaleBwImageSpinBox(&header1Widget)
,   viewDataAsBwImageCheckBox(tr("View Data as B/W Image"), &header1Widget)
, dataViewStackWidget(this)
,   tableView(&dataViewStackWidget)
,   imageViewScrollArea(&dataViewStackWidget)
,     imageView(&imageViewScrollArea)
,     imageViewInitialized(false)
{
	assert(shape.size() > 1); // otherwise DataTable1D should be used

	layout.addWidget(&headerWidget);
	  headerLayout.addWidget(&shapeLabel);
	  headerLayout.addWidget(&shapeDimensionsWidget);
	  headerLayout.addWidget(&dataRangeLabel);
	  headerLayout.addWidget(&colorSchemaLabel);
	  headerLayout.addWidget(&colorSchemaComboBox);
	layout.addWidget(&header1Widget);
	  header1Layout.addWidget(&filler1Widget);
	  header1Layout.addWidget(&scaleBwImageLabel);
	  header1Layout.addWidget(&scaleBwImageSpinBox);
	  header1Layout.addWidget(&viewDataAsBwImageCheckBox);
	layout.addWidget(&dataViewStackWidget);
	dataViewStackWidget.insertWidget(0, &tableView);
	dataViewStackWidget.insertWidget(1, &imageViewScrollArea);
	imageViewScrollArea.setWidget(&imageView);

	{ // create comboboxes for shape dimensions
		unsigned numMultiDims = Tensor::numMultiDims(shape);
		//std::vector<bool> multiDims = tensorGetMultiDims(shape);
		unsigned dim = 0;
		for (auto d : shape) {
			auto combobox = new QComboBox(&shapeDimensionsWidget);
			combobox->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum); // (H) The sizeHint() is a maximum (V) The sizeHint() is a maximum
			shapeDimensionsLayout.addWidget(combobox);
			if (d == 1) { // dimensions=1 are excluded from selection
				combobox->addItem(tr("single 1"));
				combobox->setEnabled(false);
			} else {
				combobox->addItem(tr("--X (columns)--"));
				combobox->addItem(tr("--Y (rows)--"));
				if (numMultiDims > 2) {
					for (unsigned index = 0; index < d; index++)
						combobox->addItem(QString(tr("index: %1")).arg(index+1));
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
			combobox->setToolTip(QString(tr("Directions and indexes to choose for dimension number %1 of the shape %2")).arg(dim+1).arg(S2Q(STR(shape))));
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
	headerWidget             .setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::Fixed);
	colorSchemaComboBox      .setSizePolicy(QSizePolicy::Maximum,          QSizePolicy::Fixed); // (H) The sizeHint() is a maximum
	header1Widget            .setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::Fixed);
	filler1Widget            .setSizePolicy(QSizePolicy::Minimum,          QSizePolicy::Maximum); // (H) The widget can be expanded (V) The sizeHint() is a maximum
	scaleBwImageLabel        .setSizePolicy(QSizePolicy::Maximum,          QSizePolicy::Maximum);
	scaleBwImageSpinBox      .setSizePolicy(QSizePolicy::Maximum,          QSizePolicy::Maximum);
	viewDataAsBwImageCheckBox.setSizePolicy(QSizePolicy::Maximum,          QSizePolicy::Fixed); // (H) The sizeHint() is a maximum
	tableView    .setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding);
	//imageView    .setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding);
	for (auto widget : {&shapeLabel, &dataRangeLabel, &colorSchemaLabel}) // make labels minimal
		widget->setSizePolicy(QSizePolicy::Maximum,          QSizePolicy::Maximum);
	layout.setSpacing(0);
	layout.setContentsMargins(0,0,0,0);
	headerWidget.setContentsMargins(0,0,0,0);
	headerLayout.setContentsMargins(0,0,0,0); // XXX this doesn't work for some reason: widget still has vertical margins
	header1Layout.setContentsMargins(0,0,0,0);

	// data range
	auto dataRange = Util::arrayMinMax(data, Tensor::flatSize(shape));
	dataRangeLabel.setText(QString(tr("Data Range: %1..%2")).arg(std::get<0>(dataRange)).arg(std::get<1>(dataRange)));

	// create the model
	tableModel.reset(new DataModel(new TensorSliceDataSource(shape, dimVertical, dimHorizontal, mkIdxs(), data), nullptr, &tableView));
	tableView.setModel(tableModel.get());

	// combobox values
	colorSchemaComboBox.addItem(tr("None (default)"),        COLORSCHEME_NONE);
	colorSchemaComboBox.addItem(tr("Blue-to-Red"),           COLORSCHEME_BLUE_TO_RED);
	colorSchemaComboBox.addItem(tr("Red-to-Blue"),           COLORSCHEME_RED_TO_BLUE);
	colorSchemaComboBox.addItem(tr("Grayscale Up"),          COLORSCHEME_GRAYSCALE_UP);
	colorSchemaComboBox.addItem(tr("Grayscale Down"),        COLORSCHEME_GRAYSCALE_DOWN);
	colorSchemaComboBox.addItem(tr("Grayscale Zero Up"),     COLORSCHEME_GRAYSCALE_ZERO_UP);
	colorSchemaComboBox.addItem(tr("Grayscale Zero Down"),   COLORSCHEME_GRAYSCALE_ZERO_DOWN);
	colorSchemaComboBox.addItem(tr("Grayscale Middle Up"),   COLORSCHEME_GRAYSCALE_MIDDLE_UP);
	colorSchemaComboBox.addItem(tr("Grayscale Middle Down"), COLORSCHEME_GRAYSCALE_MIDDLE_DOWN);

	// set up the spin-box
	scaleBwImageSpinBox.setSuffix(tr(" times"));
	//scaleBwImageSpinBox.lineEdit()->setReadOnly(true);

	// visibility
	scaleBwImageLabel.setVisible(false); // scale widgets are only visible when the image to scale is displayed
	scaleBwImageSpinBox.setVisible(false);

	// values
	viewDataAsBwImageCheckBox.setChecked(appSettings.value("DataTable2D.viewAsImages", true).toBool()); // view mode based on user's choice

	// set the initial mode
	setShowImageViewMode(viewDataAsBwImageCheckBox.isChecked());

	// connect signals
	connect(&colorSchemaComboBox, QOverload<int>::of(&QComboBox::activated), [this,dataRange](int index) {
		(static_cast<DataModel*>(tableModel.get()))->setColorSchema(createColorSchema(
			(ColorSchemaEnum)colorSchemaComboBox.itemData(index).toInt(),
			std::get<0>(dataRange),
			std::get<1>(dataRange)
		));
	});
	connect(&scaleBwImageSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), [this](int i) {
		if (self)
			return;
		updateBwImageView(false/*initialUpdate*/);
	});
	connect(&viewDataAsBwImageCheckBox, &QCheckBox::stateChanged, [this](int state) {
		bool showImageView = state!=0;
		// save user's choice
		appSettings.setValue("DataTable2D.viewAsImages", showImageView);
		// set the mode
		setShowImageViewMode(showImageView);

	});

	// tooltips
	shapeLabel               .setToolTip(tr("Shape of tensor data that this table represents"));
	dataRangeLabel           .setToolTip(tr("Range of numeric values present in the table"));
	colorSchemaComboBox      .setToolTip(tr("Change the color schema of data visualization"));
	viewDataAsBwImageCheckBox.setToolTip(tr("View data as image"));
	tableView                .setToolTip(tr("Tensor data values"));
}

/// interface

void DataTable2D::dataChanged(const float *data_) {
	// update the numeric table
	(static_cast<DataModel*>(tableModel.get()))->beginResetModel();
	data = data_;
	(static_cast<DataModel*>(tableModel.get()))->endResetModel();
	// update XRay-style view if it is enabled
	if (viewDataAsBwImageCheckBox.isChecked())
		updateBwImageView(false/*initialUpdate*/);
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

void DataTable2D::updateBwImageView(bool initialUpdate) {
	// helpers
	auto dataSourceToBwImage = [](const DataSource *dataSource, unsigned scaleFactor, std::tuple<float,float> &minMax) {
		std::unique_ptr<uchar> data(new uchar[dataSource->ncols()*dataSource->nrows()*scaleFactor*scaleFactor]);
		minMax = dataSource->computeMinMax();
		auto minMaxRange = std::get<1>(minMax)-std::get<0>(minMax);
		auto normalize = [minMax,minMaxRange](float d) {
			return 255.*(d-std::get<0>(minMax))/minMaxRange;
		};
		uchar *p = data.get();
		for (unsigned r = 0, re = dataSource->nrows(); r < re; r++)
			for (unsigned rptRow = 0; rptRow<scaleFactor; rptRow++)
				for (unsigned c = 0, ce = dataSource->ncols(); c < ce; c++)
					for (unsigned rptCol = 0; rptCol<scaleFactor; rptCol++)
						*p++ = normalize(dataSource->value(r,c));
		return QImage(data.get(), dataSource->ncols()*scaleFactor, dataSource->nrows()*scaleFactor, dataSource->ncols()*scaleFactor, QImage::Format_Grayscale8);
	};

	// list all combinations
	std::vector<std::vector<unsigned>> indexes; // elements are sized like the shape, with ones in places of X and Y // indexes are 0-based
	{
		std::function<void(unsigned pos, std::vector<unsigned> curr, const std::vector<unsigned> &dims, std::vector<std::vector<unsigned>> &indexes)> iterate;
		iterate = [&iterate](unsigned pos, std::vector<unsigned> curr, const std::vector<unsigned> &dims, std::vector<std::vector<unsigned>> &indexes) {
			curr.push_back(0);
			for (auto &index = *curr.rbegin(); index < dims[pos]; index++)
				if (pos+1 < dims.size())
					iterate(pos+1, curr, dims, indexes);
				else
					indexes.push_back(curr);
			curr.resize(curr.size()-1);
		};

		auto dims = shape;
		dims[dimVertical] = 1; // X and Y are set to 1 - we don't need to list them because they are on X,Y axes of pictures
		dims[dimHorizontal] = 1;
		iterate(0, {}, dims, indexes);
	}

	auto fmtIndex = [this](const std::vector<unsigned> &index) {
		std::ostringstream ss;
		ss << "[";
		unsigned n = 0;
		for (auto i : index) {
			ss << (n==0 ? "" : ",") << (n==dimVertical ? "Y" : n==dimHorizontal ? "X" : STR(i+1));
			++n;
		}
		ss << "]";
		return ss.str();
	};


	if (initialUpdate) { // decide on a scaling factor
		auto maxLabelWidth = QFontMetrics(shapeLabel.font()).size(0/*flags*/,
			QString("%1\n%2 .. %3").arg(S2Q(fmtIndex({1000,1000,1000,1000}))).arg(-1.234567891).arg(+1.234567891)).width();
		unsigned minScaleFactor = maxLabelWidth/shape[dimHorizontal];
		if (minScaleFactor == 0)
			minScaleFactor = 1;
		else
			while (minScaleFactor*shape[dimHorizontal] < maxLabelWidth)
			minScaleFactor++;
		self = true;
		scaleBwImageSpinBox.setRange(minScaleFactor, 5*minScaleFactor); // XXX sizing issues and image not being in the center issues arise when
		scaleBwImageSpinBox.setValue(minScaleFactor);                   //     images are allowed to be smaller tha labels
		self = false;
	}

	const unsigned numColumns = 16; // TODO should be based on width()/cell.width // TODO scaling coefficient initial value should also be adaptable
	const unsigned numRows = (indexes.size()+numColumns-1)/numColumns;
	imageView.setSizesAndData(numColumns/*width*/, numRows/*height*/, numColumns - (numRows*numColumns - indexes.size()), [&](unsigned x, unsigned y) {
		std::unique_ptr<DataSource> dataSource(new TensorSliceDataSource(shape, dimVertical, dimHorizontal, indexes[y*numColumns+x], data));
		std::tuple<float,float> minMax;
		QImage image = dataSourceToBwImage(dataSource.get(), scaleBwImageSpinBox.value()/*scale 1+*/, minMax);
		return std::tuple<QString,QImage>(
			QString("%1\n%2 .. %3")
				.arg(S2Q(fmtIndex(indexes[y*numColumns+x])))
				.arg(std::get<0>(minMax))
				.arg(std::get<1>(minMax)),
			image);
	});
}

void DataTable2D::setShowImageViewMode(bool showImageView) {
	// switch the view
	dataViewStackWidget.setCurrentIndex(showImageView ? 1/*imageView*/ : 0/*tableView*/);
	// update the screen
	if (showImageView) {
		if (!imageViewInitialized) {
			// create the image from the datasource that the table sees
			updateBwImageView(true/*initialUpdate*/);
			// set tooltip with explanation custom to the data
			imageView.setToolTip(QString(tr("Tensor data as a B/W image normalized to the data range of currently viewed tensor slice")));
			imageViewInitialized = true;
		}
		// disable all other widgets so that the data view can't be changed
		headerWidget.setEnabled(false);
	} else {
		headerWidget.setEnabled(true);
	}
	// visibility of scale controls
	scaleBwImageLabel.setVisible(showImageView);
	scaleBwImageSpinBox.setVisible(showImageView);
}
