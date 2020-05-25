// Copyright (C) 2020 by Yuri Victorovich. All rights reserved.

// corresponds to https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/schema/schema.fbs 58719c1 Dec 17, 2019

// Regenerate from schema.fb 'flatc --jsonschema schema.fb', then prepend with 'var fbsSchema = ' and append the following:
#if 0
if (true) {
        var sorted = {};
        fbsSchema.definitions.tflite_BuiltinOperator.enum.forEach(function(operator) {
                sorted[operator] = true;
        });
        Object.keys(sorted).forEach(function(operator) {
                print("CASE("+operator+",Unknown)");
        });
}
#endif

// list is in the alphabetic order
CASE(ABS, Unknown)
CASE(ADD, Add)
CASE(ADD_N, Unknown)
CASE(ARG_MAX, ArgMax)
CASE(ARG_MIN, ArgMin)
CASE(AVERAGE_POOL_2D, AveragePool)
CASE(BATCH_TO_SPACE_ND, Unknown)
CASE(BIDIRECTIONAL_SEQUENCE_LSTM, Unknown)
CASE(BIDIRECTIONAL_SEQUENCE_RNN, Unknown)
CASE(CALL, Unknown)
CASE(CAST, Unknown)
CASE(CEIL, Unknown)
CASE(CONCATENATION, Concatenation)
CASE(CONCAT_EMBEDDINGS, Unknown)
CASE(CONV_2D, Conv2D)
CASE(COS, Unknown)
CASE(CUSTOM, Unknown)
CASE(DELEGATE, Unknown)
CASE(DENSIFY,Unknown)
CASE(DEPTHWISE_CONV_2D, DepthwiseConv2D)
CASE(DEPTH_TO_SPACE, Unknown)
CASE(DEQUANTIZE, Dequantize)
CASE(DIV, Div)
CASE(ELU, Unknown)
CASE(EMBEDDING_LOOKUP, Unknown)
CASE(EMBEDDING_LOOKUP_SPARSE, Unknown)
CASE(EQUAL, Unknown)
CASE(EXP, Unknown)
CASE(EXPAND_DIMS, Unknown)
CASE(FAKE_QUANT, Unknown)
CASE(FILL, Unknown)
CASE(FLOOR, Unknown)
CASE(FLOOR_DIV, Unknown)
CASE(FLOOR_MOD, Unknown)
CASE(FULLY_CONNECTED, FullyConnected)
CASE(GATHER, Unknown)
CASE(GATHER_ND, Unknown)
CASE(GREATER, Unknown)
CASE(GREATER_EQUAL, Unknown)
CASE(HARD_SWISH, HardSwish)
CASE(HASHTABLE_LOOKUP, Unknown)
CASE(IF, Unknown)
CASE(L2_NORMALIZATION, Unknown)
CASE(L2_POOL_2D, Unknown)
CASE(LEAKY_RELU, LeakyRelu)
CASE(LESS, Unknown)
CASE(LESS_EQUAL, Unknown)
CASE(LOCAL_RESPONSE_NORMALIZATION, LocalResponseNormalization)
CASE(LOG, Unknown)
CASE(LOGICAL_AND, Unknown)
CASE(LOGICAL_NOT, Unknown)
CASE(LOGICAL_OR, Unknown)
CASE(LOGISTIC, Logistic)
CASE(LOG_SOFTMAX, Unknown)
CASE(LSH_PROJECTION, Unknown)
CASE(LSTM, Unknown)
CASE(MATRIX_DIAG, Unknown)
CASE(MATRIX_SET_DIAG, Unknown)
CASE(MAXIMUM, Maximum)
CASE(MAX_POOL_2D, MaxPool)
CASE(MEAN, Mean)
CASE(MINIMUM, Minimum)
CASE(MIRROR_PAD, MirrorPad)
CASE(MUL, Mul)
CASE(NEG, Unknown)
CASE(NON_MAX_SUPPRESSION_V4, Unknown)
CASE(NON_MAX_SUPPRESSION_V5, Unknown)
CASE(NOT_EQUAL, Unknown)
CASE(ONE_HOT, Unknown)
CASE(PACK, Unknown)
CASE(PAD, Pad)
CASE(PADV2, Unknown)
CASE(POW, Unknown)
CASE(PRELU, Unknown)
CASE(QUANTIZE, Unknown)
CASE(RANGE, Unknown)
CASE(RANK, Unknown)
CASE(REDUCE_ANY, Unknown)
CASE(REDUCE_MAX, Unknown)
CASE(REDUCE_MIN, Unknown)
CASE(REDUCE_PROD, Unknown)
CASE(RELU, Relu)
CASE(RELU6, Relu6)
CASE(RELU_N1_TO_1, Unknown)
CASE(RESHAPE, Reshape)
CASE(RESIZE_BILINEAR, ResizeBilinear)
CASE(RESIZE_NEAREST_NEIGHBOR, ResizeNearestNeighbor)
CASE(REVERSE_SEQUENCE, Unknown)
CASE(REVERSE_V2, Unknown)
CASE(RNN, Unknown)
CASE(ROUND, Unknown)
CASE(RSQRT, RSqrt)
CASE(SCATTER_ND, Unknown)
CASE(SELECT, Unknown)
CASE(SELECT_V2, Unknown)
CASE(SHAPE, Unknown)
CASE(SIN, Unknown)
CASE(SKIP_GRAM, Unknown)
CASE(SLICE, Unknown)
CASE(SOFTMAX, Softmax)
CASE(SPACE_TO_BATCH_ND, Unknown)
CASE(SPACE_TO_DEPTH, Unknown)
CASE(SPARSE_TO_DENSE, Unknown)
CASE(SPLIT, Split)
CASE(SPLIT_V, Unknown)
CASE(SQRT, Unknown)
CASE(SQUARE, Unknown)
CASE(SQUARED_DIFFERENCE, SquaredDifference)
CASE(SQUEEZE, Unknown)
CASE(STRIDED_SLICE, StridedSlice)
CASE(SUB, Sub)
CASE(SUM, Unknown)
CASE(SVDF, Unknown)
CASE(TANH, Tanh)
CASE(TILE, Unknown)
CASE(TOPK_V2, Unknown)
CASE(TRANSPOSE, Transpose)
CASE(TRANSPOSE_CONV, Unknown)
CASE(UNIDIRECTIONAL_SEQUENCE_LSTM, Unknown)
CASE(UNIDIRECTIONAL_SEQUENCE_RNN, Unknown)
CASE(UNIQUE, Unknown)
CASE(UNPACK, Unknown)
CASE(WHERE, Unknown)
CASE(WHILE, Unknown)
CASE(ZEROS_LIKE, Unknown)
