// Copyright (C) 2020 by Yuri Victorovich. All rights reserved.

//
// tflite-reference-implementation.cpp contains portions of the Apache 2.0 licensed code from the TensorFlow source tree
//

#include <cstdint>
#include <vector>
#include <cmath>

#include <assert.h>

namespace tflite {

/// some common macros used in the code

#define TFLITE_DCHECK(condition) assert(condition); // (condition) ? (void)0 : TFLITE_ASSERT_FALSE
#define TFLITE_DCHECK_EQ(a,b) assert(a == b);
#define TFLITE_DCHECK_LE(a,b) assert(a <= b);
#define TFLITE_DCHECK_GE(a,b) assert(a >= b);

/// common types used in this code

typedef int8_t   int8;
typedef uint8_t  uint8;
typedef int16_t  int16;
typedef uint16_t uint16;
typedef int32_t  int32;
typedef uint32_t uint32;

enum class FusedActivationFunctionType : uint8 { kNone, kRelu6, kRelu1, kRelu };
enum class PaddingType : uint8 { kNone, kSame, kValid };

struct PaddingValues {
  int16 width;
  int16 height;
  // offset is used for calculating "remaining" padding, for example, `width`
  // is 1 and `width_offset` is 1, so padding_left is 1 while padding_right is
  // 1 + 1 = 2.
  int16 width_offset; // unused
  // Same as width_offset except it's over the height dimension.
  int16 height_offset; // unused

	PaddingValues()
	: width(0)
	, height(0)
	, width_offset(0)
	, height_offset(0)
	{ }
};

// from: tensorflow/lite/kernels/internal/types.h
class RuntimeShape : public std::vector<unsigned> {
public:
	RuntimeShape() { }
	RuntimeShape(const std::vector<unsigned> &v) : std::vector<unsigned>(v) { }
	RuntimeShape(unsigned d1) {push_back(d1);}
	RuntimeShape(unsigned d1, unsigned d2) {push_back(d1); push_back(d2);}
	RuntimeShape(unsigned d1, unsigned d2, unsigned d3) {push_back(d1); push_back(d2); push_back(d3);}
	RuntimeShape(unsigned d1, unsigned d2, unsigned d3, unsigned d4) {push_back(d1); push_back(d2); push_back(d3); push_back(d4);}
	void Resize(unsigned sz_) {resize(sz_);}
	void SetDim(unsigned dim, unsigned val) {(*this)[dim] = val;}
	unsigned Dims(unsigned n) const {return (*this)[n];}
	int DimensionsCount() const {return size();}
	int FlatSize() const {
		int sz = 1;
		for (auto i = begin(); i != end(); i++)
			sz *= *i;
		return sz;
	}
	inline const unsigned* DimsData() const { return &(*this)[0]; }
	inline static RuntimeShape ExtendedShape(int new_shape_size, const RuntimeShape& shape) {
		return RuntimeShape(new_shape_size, shape, 1);
	}
	RuntimeShape(int new_shape_size, const RuntimeShape& shape, int pad_value) {
		assert(new_shape_size >= shape.DimensionsCount());
		resize(new_shape_size);
		const int size_increase = new_shape_size - shape.DimensionsCount();
		int i = 0;
		for (; i < size_increase; ++i)
			*(begin()+i) = pad_value;
		for (auto n : shape)
			*(begin()+(i++)) = n;
	}
};

static unsigned MatchingDim(const RuntimeShape &shape1, unsigned dim1, const RuntimeShape &shape2, unsigned dim2) {
	assert(shape1[dim1] == shape2[dim2]);
	return shape1[dim1];
}

inline int MatchingFlatSize(const RuntimeShape& shape,
                            const RuntimeShape& check_shape_0) {
  TFLITE_DCHECK_EQ(shape.DimensionsCount(), check_shape_0.DimensionsCount());
  const int dims_count = shape.DimensionsCount();
  for (int i = 0; i < dims_count; ++i) {
    TFLITE_DCHECK_EQ(shape.Dims(i), check_shape_0.Dims(i));
  }
  return shape.FlatSize();
}

inline int MatchingFlatSize(const RuntimeShape& shape,
                            const RuntimeShape& check_shape_0,
                            const RuntimeShape& check_shape_1) {
  TFLITE_DCHECK_EQ(shape.DimensionsCount(), check_shape_0.DimensionsCount());
  const int dims_count = shape.DimensionsCount();
  for (int i = 0; i < dims_count; ++i) {
    TFLITE_DCHECK_EQ(shape.Dims(i), check_shape_0.Dims(i));
  }
  return MatchingFlatSize(shape, check_shape_1);
}

inline int FlatSizeSkipDim(const RuntimeShape& shape, int skip_dim) {
  const int dims_count = shape.DimensionsCount();
  TFLITE_DCHECK(skip_dim >= 0 && skip_dim < dims_count);
  const auto* dims_data = shape.DimsData();
  int flat_size = 1;
  for (int i = 0; i < dims_count; ++i) {
    flat_size *= (i == skip_dim) ? 1 : dims_data[i];
  }
  return flat_size;
}

inline int MatchingFlatSizeSkipDim(const RuntimeShape& shape, int skip_dim,
                                   const RuntimeShape& check_shape_0) {
  const int dims_count = shape.DimensionsCount();
  for (int i = 0; i < dims_count; ++i) {
    if (i != skip_dim) {
      TFLITE_DCHECK_EQ(shape.Dims(i), check_shape_0.Dims(i));
    }
  }
  return FlatSizeSkipDim(shape, skip_dim);
}

/// supporting functions

static unsigned Offset(const RuntimeShape &shape, unsigned batch, unsigned y, unsigned x, unsigned channel) {
	assert(shape.size()==4);
	return	batch*shape.Dims(1)*shape.Dims(2)*shape.Dims(3) +
		y*shape.Dims(2)*shape.Dims(3) +
		x*shape.Dims(3) +
		channel;
	//return (((batch*Dim(1))+y)*Dim(2)+x)*shape.Dim(3)+channel;
}

inline float ActivationFunctionWithMinMax(float x, float output_activation_min,
                                          float output_activation_max) {
  return std::min(std::max(x, output_activation_min), output_activation_max);
}

/// parameter structures that operators accept

// from: tensorflow/lite/kernels/internal/types.h
struct ConvParams { // <= BuiltinOptions_Conv2DOptions = 1
  PaddingType padding_type; // unused
  PaddingValues padding_values;
  // TODO(starka): This was just "stride", so check that width+height is OK.
  int16 stride_width;
  int16 stride_height;
  int16 dilation_width_factor;
  int16 dilation_height_factor;
  // uint8 inference params.
  // TODO(b/65838351): Use smaller types if appropriate.
  int32 input_offset; // unused
  int32 weights_offset; // unused
  int32 output_offset; // unused
  int32 output_multiplier; // unused
  int output_shift; // unused
  // uint8, etc, activation params.
  int32 quantized_activation_min; // unused
  int32 quantized_activation_max; // unused
  // float activation params.
  float float_activation_min;
  float float_activation_max;
};

// from: tensorflow/lite/kernels/internal/types.h
struct DepthwiseParams { // <= BuiltinOptions_DepthwiseConv2DOptions = 2
  PaddingType padding_type; // unused
  PaddingValues padding_values;
  int16 stride_width;
  int16 stride_height;
  int16 dilation_width_factor;
  int16 dilation_height_factor;
  int16 depth_multiplier;
  // uint8 inference params.
  // TODO(b/65838351): Use smaller types if appropriate.
  int32 input_offset; // unused
  int32 weights_offset; // unused
  int32 output_offset; // unused
  int32 output_multiplier; // unused
  int output_shift; // unused
  // uint8, etc, activation params.
  int32 quantized_activation_min; // unused
  int32 quantized_activation_max; // unused
  float activation_params; // unused
  float float_activation_min;
  float float_activation_max;
};

struct FullyConnectedParams { // <= BuiltinOptions_FullyConnectedOptions = 8
  // uint8 inference params.
  // TODO(b/65838351): Use smaller types if appropriate.
  int32 input_offset; // unused
  int32 weights_offset; // unused
  int32 output_offset; // unused
  int32 output_multiplier; // unused
  int output_shift; // unused
  // uint8, etc, activation params.
  int32 quantized_activation_min; // unused
  int32 quantized_activation_max; // unused
  // float activation params.
  float float_activation_min;
  float float_activation_max;
  //FullyConnectedWeightsFormat weights_format; // // DEFAULT,SHUFFLED4x16INT8
};

// from: tensorflow/lite/kernels/internal/types.h
struct PoolParams { // <= BuiltinOptions_Pool2DOptions = 5
  FusedActivationFunctionType activation; // unused
  PaddingType padding_type; // unused
  PaddingValues padding_values;
  int stride_height;
  int stride_width;
  int filter_height;
  int filter_width;
  // uint8, etc, activation params.
  int32 quantized_activation_min; // unused
  int32 quantized_activation_max; // unused
  // float activation params.
  float float_activation_min;
  float float_activation_max;
};

// from: tensorflow/lite/kernels/internal/types.htensorflow/lite/kernels/internal/types.h
// For Add, Sub, Mul ops.
struct ArithmeticParams { // <= BuiltinOptions_AddOptions = 11 / <= BuiltinOptions_MulOptions = 21 / <= BuiltinOptions_SubOptions = 28 / <= BuiltinOptions_DivOptions = 29
  // Shape dependent / common to data / op types.
  //BroadcastableOpCategory broadcast_category; // unused
  // uint8 inference params.
  int32 input1_offset; // unused
  int32 input2_offset; // unused
  int32 output_offset; // unused
  int32 output_multiplier; // unused
  int output_shift; // unused
  // Add / Sub, not Mul, uint8 inference params.
  int left_shift; // unused
  int32 input1_multiplier; // unused
  int input1_shift; // unused
  int32 input2_multiplier; // unused
  int input2_shift; // unused
  // uint8, etc, activation params.
  int32 quantized_activation_min; // unused (ony used used in int-versions)
  int32 quantized_activation_max; // unused
  // float activation params.
  float float_activation_min;
  float float_activation_max;

  // Processed output dimensions.
  // Let input "a" be the one that broadcasts in the faster-changing dimension.
  // Then, after coalescing, for shapes {a0, a1, a2, a3, a4} and
  // {b0, b1, b2, b3, b4},
  // broadcast_shape[4] = b0 = a0.
  // broadcast_shape[3] = b1; a1 = 1.
  // broadcast_shape[2] = b2 = a2.
  // broadcast_shape[1] = a3; b3 = 1.
  // broadcast_shape[0] = b4 = a4.
  int broadcast_shape[5];
};

// from: tensorflow/lite/kernels/internal/types.h
struct SoftmaxParams { // <= BuiltinOptions_SoftmaxOptions = 9
  // beta is not really used (not a Tensorflow parameter) and not implemented
  // for LogSoftmax.
  double beta;
  // uint8 inference params.  Used even when beta defaults to 1.0.
  int32 input_multiplier; // unused
  int32 input_left_shift; // unused
  // Reverse scaling is only used by LogSoftmax.
  int32 reverse_scaling_divisor; // unused
  int32 reverse_scaling_right_shift; // unused
  int diff_min; // unused
};

// from kernels/internal/types.h
struct ResizeBilinearParams {
  bool align_corners;
};

/// operator code

// from: tensorflow/lite/kernels/internal/reference/conv.h
inline void Conv(const ConvParams& params, const RuntimeShape& input_shape,
                 const float* input_data, const RuntimeShape& filter_shape,
                 const float* filter_data, const RuntimeShape& bias_shape,
                 const float* bias_data, const RuntimeShape& output_shape,
                 float* output_data, const RuntimeShape& im2col_shape,
                 float* im2col_data) {
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  //const float output_activation_min = params.float_activation_min;
  //const float output_activation_max = params.float_activation_max;
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

  (void)im2col_data;   // only used in optimized code.
  (void)im2col_shape;  // only used in optimized code.
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = MatchingDim(input_shape, 3, filter_shape, 3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);

  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
          const int in_x_origin = (out_x * stride_width) - pad_width;
          const int in_y_origin = (out_y * stride_height) - pad_height;
          float total = 0.f;
          for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
              for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
                const int in_x = in_x_origin + dilation_width_factor * filter_x;
                const int in_y =
                    in_y_origin + dilation_height_factor * filter_y;
                // If the location is outside the bounds of the input image,
                // use zero as a default value.
                if ((in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                    (in_y < input_height)) {
                  float input_value = input_data[Offset(
                      input_shape, batch, in_y, in_x, in_channel)];
                  float filter_value =
                      filter_data[Offset(filter_shape, out_channel, filter_y,
                                         filter_x, in_channel)];
                  total += (input_value * filter_value);
                }
              }
            }
          }
          float bias_value = 0.0f;
          if (bias_data) {
            bias_value = bias_data[out_channel];
          }
          output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] =
              // DISABLE ActivationFunctionWithMinMax
              //ActivationFunctionWithMinMax(total + bias_value,
              //                             output_activation_min,
              //                             output_activation_max);

	      // INSTEAD
	      total + bias_value;
        }
      }
    }
  }
}

// from: tensorflow/lite/kernels/internal/reference/depthwiseconv_float.h
inline void DepthwiseConv(
    const DepthwiseParams& params, const RuntimeShape& input_shape,
    const float* input_data, const RuntimeShape& filter_shape,
    const float* filter_data, const RuntimeShape& bias_shape,
    const float* bias_data, const RuntimeShape& output_shape,
    float* output_data) {
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int depth_multiplier = params.depth_multiplier;
  //const float output_activation_min = params.float_activation_min;
  //const float output_activation_max = params.float_activation_max;
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int output_depth = MatchingDim(filter_shape, 3, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int input_depth = input_shape.Dims(3);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  TFLITE_DCHECK_EQ(output_depth, input_depth * depth_multiplier);
  TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);

  for (int b = 0; b < batches; ++b) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int ic = 0; ic < input_depth; ++ic) {
          for (int m = 0; m < depth_multiplier; m++) {
            const int oc = m + ic * depth_multiplier;
            const int in_x_origin = (out_x * stride_width) - pad_width;
            const int in_y_origin = (out_y * stride_height) - pad_height;
            float total = 0.f;
            for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
              for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                const int in_x = in_x_origin + dilation_width_factor * filter_x;
                const int in_y =
                    in_y_origin + dilation_height_factor * filter_y;
                // If the location is outside the bounds of the input image,
                // use zero as a default value.
                if ((in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                    (in_y < input_height)) {
                  float input_value =
                      input_data[Offset(input_shape, b, in_y, in_x, ic)];
                  float filter_value = filter_data[Offset(
                      filter_shape, 0, filter_y, filter_x, oc)];
                  total += (input_value * filter_value);
                }
              }
            }
            float bias_value = 0.0f;
            if (bias_data) {
              bias_value = bias_data[oc];
            }
            output_data[Offset(output_shape, b, out_y, out_x, oc)] =
              // DISABLE ActivationFunctionWithMinMax
              //  ActivationFunctionWithMinMax(total + bias_value,
              //                               output_activation_min,
              //                               output_activation_max);

	      // INSTEAD
	      total + bias_value;
          }
        }
      }
    }
  }
}

// from: tensorflow/lite/kernels/internal/reference/fully_connected.h
inline void FullyConnected(
    const FullyConnectedParams& params, const RuntimeShape& input_shape,
    const float* input_data, const RuntimeShape& weights_shape,
    const float* weights_data, const RuntimeShape& bias_shape,
    const float* bias_data, const RuntimeShape& output_shape,
    float* output_data) {
//  const float output_activation_min = params.float_activation_min;
//  const float output_activation_max = params.float_activation_max;
  // TODO(benoitjacob): This really should be:
  //     const int batches = ArraySize(output_dims, 1);
  // but the current --variable_batch hack consists in overwriting the 3rd
  // dimension with the runtime batch size, as we don't keep track for each
  // array of which dimension is the batch dimension in it.
  const int output_dims_count = output_shape.DimensionsCount();
  const int weights_dims_count = weights_shape.DimensionsCount();
  const int batches = FlatSizeSkipDim(output_shape, output_dims_count - 1);
  const int output_depth = MatchingDim(weights_shape, weights_dims_count - 2,
                                       output_shape, output_dims_count - 1);
  const int accum_depth = weights_shape.Dims(weights_dims_count - 1);
  for (int b = 0; b < batches; ++b) {
    for (int out_c = 0; out_c < output_depth; ++out_c) {
      float total = 0.f;
      for (int d = 0; d < accum_depth; ++d) {
        total += input_data[b * accum_depth + d] *
                 weights_data[out_c * accum_depth + d];
      }
      float bias_value = 0.0f;
      if (bias_data) {
        bias_value = bias_data[out_c];
      }

      // DISABLE ActivationFunctionWithMinMax
      //output_data[out_c + output_depth * b] = ActivationFunctionWithMinMax(
      //    total + bias_value, output_activation_min, output_activation_max);

      // INSTEAD
      output_data[out_c + output_depth * b] = total + bias_value;
    }
  }
}

// from: tensorflow/lite/kernels/internal/reference/pooling.h
inline void MaxPool(const PoolParams& params, const RuntimeShape& input_shape,
                    const float* input_data, const RuntimeShape& output_shape,
                    float* output_data) {
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int depth = MatchingDim(input_shape, 3, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int stride_height = params.stride_height;
  const int stride_width = params.stride_width;
  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int channel = 0; channel < depth; ++channel) {
          const int in_x_origin =
              (out_x * stride_width) - params.padding_values.width;
          const int in_y_origin =
              (out_y * stride_height) - params.padding_values.height;
          // Compute the boundaries of the filter region clamped so as to
          // ensure that the filter window fits in the input array.
          const int filter_x_start = std::max(0, -in_x_origin);
          const int filter_x_end =
              std::min(params.filter_width, input_width - in_x_origin);
          const int filter_y_start = std::max(0, -in_y_origin);
          const int filter_y_end =
              std::min(params.filter_height, input_height - in_y_origin);
          float max = std::numeric_limits<float>::lowest();
          for (int filter_y = filter_y_start; filter_y < filter_y_end;
               ++filter_y) {
            for (int filter_x = filter_x_start; filter_x < filter_x_end;
                 ++filter_x) {
              const int in_x = in_x_origin + filter_x;
              const int in_y = in_y_origin + filter_y;
              max = std::max(
                  max,
                  input_data[Offset(input_shape, batch, in_y, in_x, channel)]);
            }
          }
          output_data[Offset(output_shape, batch, out_y, out_x, channel)] =
              // DISABLE ActivationFunctionWithMinMax
              //ActivationFunctionWithMinMax(max, params.float_activation_min,
              //                             params.float_activation_max);

	      // INSTEAD
	      max;
        }
      }
    }
  }
}

// from: tensorflow/lite/kernels/internal/reference/pooling.h
inline void AveragePool(const PoolParams& params,
                        const RuntimeShape& input_shape,
                        const float* input_data,
                        const RuntimeShape& output_shape, float* output_data) {
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int depth = MatchingDim(input_shape, 3, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int stride_height = params.stride_height;
  const int stride_width = params.stride_width;
  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int channel = 0; channel < depth; ++channel) {
          const int in_x_origin =
              (out_x * stride_width) - params.padding_values.width;
          const int in_y_origin =
              (out_y * stride_height) - params.padding_values.height;
          // Compute the boundaries of the filter region clamped so as to
          // ensure that the filter window fits in the input array.
          const int filter_x_start = std::max(0, -in_x_origin);
          const int filter_x_end =
              std::min(params.filter_width, input_width - in_x_origin);
          const int filter_y_start = std::max(0, -in_y_origin);
          const int filter_y_end =
              std::min(params.filter_height, input_height - in_y_origin);
          float total = 0.f;
          float filter_count = 0;
          for (int filter_y = filter_y_start; filter_y < filter_y_end;
               ++filter_y) {
            for (int filter_x = filter_x_start; filter_x < filter_x_end;
                 ++filter_x) {
              const int in_x = in_x_origin + filter_x;
              const int in_y = in_y_origin + filter_y;
              total +=
                  input_data[Offset(input_shape, batch, in_y, in_x, channel)];
              filter_count++;
            }
          }
          const float average = total / filter_count;
          output_data[Offset(output_shape, batch, out_y, out_x, channel)] =
              // DISABLE ActivationFunctionWithMinMax
              //ActivationFunctionWithMinMax(average, params.float_activation_min,
              //                             params.float_activation_max);

	      // INSTEAD
	      average;
        }
      }
    }
  }
}

// from: tensorflow/lite/kernels/internal/reference/reference_ops.h
inline void Add(const ArithmeticParams& params,
                const RuntimeShape& input1_shape, const float* input1_data,
                const RuntimeShape& input2_shape, const float* input2_data,
                const RuntimeShape& output_shape, float* output_data) {
  const int size = MatchingFlatSize(input1_shape, input2_shape, output_shape);
  for (int i = 0; i < size; i++) {
    auto x = input1_data[i] + input2_data[i];
    // DISABLE ActivationFunctionWithMinMax
    //output_data[i] = ActivationFunctionWithMinMax(
    //    x, params.float_activation_min, params.float_activation_max);

    // INSTEAD
    output_data[i] = x;
  }
}

// self-written: couldn't find the code in the TF tree
template <typename T>
void Add(const RuntimeShape& input1_shape, const T* input1_data,
             const T* input2_data, const RuntimeShape& output_shape,
             T* output_data) {
  const int flat_size = MatchingFlatSize(input1_shape, output_shape);

  auto one_value = input2_data[0];
  for (int i = 0; i < flat_size; i++) {
    output_data[i] = input1_data[i] + one_value;
  }
}

// self-written
inline void Mul(const ArithmeticParams& params,
                const RuntimeShape& input1_shape, const float* input1_data,
                const RuntimeShape& input2_shape, const float* input2_data,
                const RuntimeShape& output_shape, float* output_data) {
  const int size = MatchingFlatSize(input1_shape, input2_shape, output_shape);
  for (int i = 0; i < size; i++) {
    auto x = input1_data[i] * input2_data[i];
    // DISABLE ActivationFunctionWithMinMax
    //output_data[i] = ActivationFunctionWithMinMax(
    //    x, params.float_activation_min, params.float_activation_max);

    // INSTEAD
    output_data[i] = x;
  }
}

// from: tensorflow/lite/kernels/internal/reference/softmax.h
inline void Softmax(const SoftmaxParams& params,
                    const RuntimeShape& input_shape, const float* input_data,
                    const RuntimeShape& output_shape, float* output_data) {
  const int trailing_dim = input_shape.DimensionsCount() - 1;
  const int outer_size =
      MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
  const int depth =
      MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);

  for (int i = 0; i < outer_size; ++i) {
    // Find max element value which we'll use to ensure numerical stability
    // taking advantage of the following equality:
    // exp(x[i])/sum(exp(x[i])) == exp(x[i]+C)/sum(exp(x[i]+C))
    float max = std::numeric_limits<float>::lowest();
    for (int c = 0; c < depth; ++c) {
      max = std::max(max, input_data[i * depth + c]);
    }

    // Compute sum.
    float sum = 0.f;
    for (int c = 0; c < depth; ++c) {
      sum += std::exp((input_data[i * depth + c] - max) * params.beta);
    }

    // Compute result.
    for (int c = 0; c < depth; ++c) {
      output_data[i * depth + c] =
          std::exp((input_data[i * depth + c] - max) * params.beta) / sum;
    }
  }
}

// from kernels/internal/reference/reference_ops.h
template <typename T>
inline void ResizeBilinear(const tflite::ResizeBilinearParams& op_params,
                           const RuntimeShape& unextended_input_shape,
                           const T* input_data,
                           const RuntimeShape& unextended_output_size_shape,
                           const int32* output_size_data,
                           const RuntimeShape& unextended_output_shape,
                           T* output_data) {
  TFLITE_DCHECK_LE(unextended_input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_size_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_shape.DimensionsCount(), 4);
  const RuntimeShape input_shape =
      RuntimeShape::ExtendedShape(4, unextended_input_shape);
  const RuntimeShape output_size_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_size_shape);
  const RuntimeShape output_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_shape);

  int32 batches = MatchingDim(input_shape, 0, output_shape, 0);
  int32 input_height = input_shape.Dims(1);
  int32 input_width = input_shape.Dims(2);
  int32 depth = MatchingDim(input_shape, 3, output_shape, 3);

  TFLITE_DCHECK_EQ(output_size_shape.Dims(0), 1);
  TFLITE_DCHECK_EQ(output_size_shape.Dims(1), 1);
  TFLITE_DCHECK_EQ(output_size_shape.Dims(2), 1);
  TFLITE_DCHECK_EQ(output_size_shape.Dims(3), 2);
  int32 output_height = output_size_data[Offset(output_size_shape, 0, 0, 0, 0)];
  int32 output_width = output_size_data[Offset(output_size_shape, 0, 0, 0, 1)];

  float height_scale = static_cast<float>(input_height) / output_height;
  float width_scale = static_cast<float>(input_width) / output_width;
  if (op_params.align_corners && output_height > 1) {
    height_scale = static_cast<float>(input_height - 1) / (output_height - 1);
  }
  if (op_params.align_corners && output_width > 1) {
    width_scale = static_cast<float>(input_width - 1) / (output_width - 1);
  }

  for (int b = 0; b < batches; ++b) {
    for (int y = 0; y < output_height; ++y) {
      float input_y = y * height_scale;
      int32 y0 = static_cast<int32>(std::floor(input_y));
      int32 y1 = std::min(y0 + 1, input_height - 1);
      for (int x = 0; x < output_width; ++x) {
        float input_x = x * width_scale;
        int32 x0 = static_cast<int32>(std::floor(input_x));
        int32 x1 = std::min(x0 + 1, input_width - 1);
        for (int c = 0; c < depth; ++c) {
          T interpolation =
              static_cast<T>(input_data[Offset(input_shape, b, y0, x0, c)] *
                                 (1 - (input_y - y0)) * (1 - (input_x - x0)) +
                             input_data[Offset(input_shape, b, y1, x0, c)] *
                                 (input_y - y0) * (1 - (input_x - x0)) +
                             input_data[Offset(input_shape, b, y0, x1, c)] *
                                 (1 - (input_y - y0)) * (input_x - x0) +
                             input_data[Offset(input_shape, b, y1, x1, c)] *
                                 (input_y - y0) * (input_x - x0));
          output_data[Offset(output_shape, b, y, x, c)] = interpolation;
        }
      }
    }
  }
}

}

//
// exporting this functionality by wrapping it in our API
//

#include "../../nn-types.h"

namespace NnOperators {

void Conv2D(
	const TensorShape &inputShape, const float *inputData,
	const TensorShape &filterShape, const float *filterData,
	const TensorShape &biasShape, const float *biasData,
	const TensorShape &outputShape, float *outputData,
	unsigned paddingWidth, unsigned paddingHeight,
	unsigned strideWidth, unsigned strideHeight,
	unsigned dilationWidthFactor, unsigned dilationHeightFactor
) {
	tflite::ConvParams params;
	params.padding_values.width = paddingWidth;
	params.padding_values.height = paddingHeight;
	params.stride_width = strideWidth;
	params.stride_height = strideHeight;
	params.dilation_width_factor = dilationWidthFactor;
	params.dilation_height_factor = dilationHeightFactor;

	tflite::Conv(params,
		tflite::RuntimeShape(inputShape),  inputData,
		tflite::RuntimeShape(filterShape), filterData,
		tflite::RuntimeShape(biasShape),   biasData,
		tflite::RuntimeShape(outputShape), outputData,
		tflite::RuntimeShape(0),
		nullptr
	);
}

void DepthwiseConv2D(
	const TensorShape &inputShape, const float *inputData,
	const TensorShape &filterShape, const float *filterData,
	const TensorShape &biasShape, const float *biasData,
	const TensorShape &outputShape, float *outputData,
	unsigned paddingWidth, unsigned paddingHeight,
	unsigned strideWidth, unsigned strideHeight,
	unsigned dilationWidthFactor, unsigned dilationHeightFactor,
	unsigned depthMultiplier
) {
	tflite::DepthwiseParams params;
	params.padding_values.width = paddingWidth;
	params.padding_values.height = paddingHeight;
	params.stride_width = strideWidth;
	params.stride_height = strideHeight;
	params.dilation_width_factor = dilationWidthFactor;
	params.dilation_height_factor = dilationHeightFactor;
	params.depth_multiplier = depthMultiplier;

	tflite::DepthwiseConv(params,
		tflite::RuntimeShape(inputShape),  inputData,
		tflite::RuntimeShape(filterShape), filterData,
		tflite::RuntimeShape(biasShape),   biasData,
		tflite::RuntimeShape(outputShape), outputData
	);
}

void FullyConnected(
	const TensorShape &inputShape, const float *inputData,
	const TensorShape &filterShape, const float *filterData,
	const TensorShape &biasShape, const float *biasData,
	const TensorShape &outputShape, float *outputData
) {
	tflite::FullyConnectedParams params;

	tflite::FullyConnected(params,
		tflite::RuntimeShape(inputShape),  inputData,
		tflite::RuntimeShape(filterShape), filterData,
		tflite::RuntimeShape(biasShape),   biasData,
		tflite::RuntimeShape(outputShape), outputData
	);
}

void MaxPool(
	const TensorShape &inputShape, const float *inputData,
	const TensorShape &outputShape, float *outputData,
	unsigned paddingWidth, unsigned paddingHeight,
	unsigned strideWidth, unsigned strideHeight,
	unsigned filterWidth, unsigned filterHeight
) {
	tflite::PoolParams params;
	params.padding_values.width = paddingWidth;
	params.padding_values.height = paddingHeight;
	params.stride_width = strideWidth;
	params.stride_height = strideHeight;
	params.filter_width = filterWidth;
	params.filter_height = filterHeight;

	tflite::MaxPool(params,
		tflite::RuntimeShape(inputShape),  inputData,
		tflite::RuntimeShape(outputShape), outputData
	);
}

void AveragePool(
	const TensorShape &inputShape, const float *inputData,
	const TensorShape &outputShape, float *outputData,
	unsigned paddingWidth, unsigned paddingHeight,
	unsigned strideWidth, unsigned strideHeight,
	unsigned filterWidth, unsigned filterHeight
) {
	tflite::PoolParams params;
	params.padding_values.width = paddingWidth;
	params.padding_values.height = paddingHeight;
	params.stride_width = strideWidth;
	params.stride_height = strideHeight;
	params.filter_width = filterWidth;
	params.filter_height = filterHeight;

	tflite::AveragePool(params,
		tflite::RuntimeShape(inputShape),  inputData,
		tflite::RuntimeShape(outputShape), outputData
	);
}

void Add(
	const TensorShape &input1Shape, const float *input1Data,
	const TensorShape &input2Shape, const float *input2Data,
	const TensorShape &outputShape, float *outputData
) {
	tflite::ArithmeticParams params; // not used by Add

	tflite::Add(params,
		tflite::RuntimeShape(input1Shape), input1Data,
		tflite::RuntimeShape(input2Shape), input2Data,
		tflite::RuntimeShape(outputShape), outputData
	);
}

void Mul(
	const TensorShape &input1Shape, const float *input1Data,
	const TensorShape &input2Shape, const float *input2Data,
	const TensorShape &outputShape, float *outputData
) {
	tflite::ArithmeticParams params; // not used by Mul

	tflite::Mul(params,
		tflite::RuntimeShape(input1Shape), input1Data,
		tflite::RuntimeShape(input2Shape), input2Data,
		tflite::RuntimeShape(outputShape), outputData
	);
}

void Softmax(
	const TensorShape &inputShape, const float *inputData,
	const TensorShape &outputShape, float *outputData,
	float beta
) {
	tflite::SoftmaxParams params;
	params.beta = beta;

	tflite::Softmax(params,
		tflite::RuntimeShape(inputShape),  inputData,
		tflite::RuntimeShape(outputShape), outputData
	);
}

void ResizeBilinear(
	const TensorShape &inputShape, const float *inputData,
	const TensorShape &outputShape, float *outputData,
	bool alignCorners
) {
	tflite::ResizeBilinearParams params;
	params.align_corners = alignCorners;

	// not sure why the operation was defiened to need these
	tflite::RuntimeShape outputSizeDims = {1, 1, 1, 2};
	tflite::int32 outputSizeData[2] = {(tflite::int32)outputShape[1/*height*/], (tflite::int32)outputShape[2/*width*/]};

	tflite::ResizeBilinear(params,
		tflite::RuntimeShape(inputShape),  inputData,
		outputSizeDims, outputSizeData,
		tflite::RuntimeShape(outputShape), outputData
	);
}

}
