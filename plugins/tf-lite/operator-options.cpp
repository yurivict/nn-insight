// corresponds to https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/schema/schema.fbs 58719c1 Dec 17, 2019

// Regenerate from schema.fbs 'flatc --jsonschema schema.fb', then prepend with 'var fbsSchema = ' and append the following:
#if 0
if (true) {
	function camelCaseToUpperCase(name) {
		return name
			.replace(/([a-z0-9])([A-Z])/g, function(match, $1, $2, offset, original) {return $1+"_"+$2;})
			.toUpperCase();
	}
	function fixupName(name) {
		return name
			.replace(/CONV2_D/, "CONV_2D")
			.replace(/POOL2_D/, "POOL_2D")
			.replace(/PAD_V2/, "PADV2")
			.replace(/LSHPROJECTION/, "LSH_PROJECTION")
			.replace(/TOP_KV2/, "TOPK_V2");
	}
	Object.keys(fbsSchema.definitions).forEach(function(name) {
		if (name.length>7 && name.substring(name.length-7)=="Options" && name!="tflite_BuiltinOptions") {
			var uname = fixupName(camelCaseToUpperCase(name.substring(7,name.length-7)));
			switch (uname) {
			case "MAXIMUM_MINIMUM":
				print("case tflite::BuiltinOperator_MINIMUM:");
				print("case tflite::BuiltinOperator_MAXIMUM: {");
				break;
			case "L2_NORM":
				print("case tflite::BuiltinOperator_L2_NORMALIZATION: {");
				break;
			case "POOL_2D":
				print("case tflite::BuiltinOperator_AVERAGE_POOL_2D:");
				print("case tflite::BuiltinOperator_MAX_POOL_2D:");
				print("case tflite::BuiltinOperator_L2_POOL_2D: {");
				break;
			case "REDUCER":
				print("case tflite::BuiltinOperator_REDUCE_PROD:");
				print("case tflite::BuiltinOperator_REDUCE_MAX:");
				print("case tflite::BuiltinOperator_REDUCE_MIN:");
				print("case tflite::BuiltinOperator_REDUCE_ANY: {");
				break;
			case "SEQUENCE_RNN":
				print("case tflite::BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_RNN: {");
				break;
			default:
				print("case tflite::BuiltinOperator_"+uname+": {");
			}
			var properties = fbsSchema.definitions[name].properties;
			if (Object.keys(properties).length > 0) {
				print("	auto oo = o->builtin_options_as_"+name.substring(7,name.length)+"();");
				Object.keys(fbsSchema.definitions[name].properties).forEach(function(opt) {
					// ignire deprecated fields
					if (opt=="new_width" || opt=="new_height")
						return; // see also https://github.com/google/flatbuffers/issues/5679
					var valueAdjustFn = "";
					switch (opt) {
					case "embedding_dim_per_channel":
					case "num_columns_per_channel":
					case "new_shape":
					case "squeeze_dims":
						valueAdjustFn = "Helpers::convertFlatbuffersIntListToStl";
						break;
					case "padding":
						valueAdjustFn = "Helpers::convertPaddingType";
						break;
					case "fused_activation_function":
						valueAdjustFn = "Helpers::convertActivationFunction";
						break;
					}
					print("	ourOpts->push_back({"
						+"PluginInterface::OperatorOption_"+opt.toUpperCase()
						+", PluginInterface::OperatorOptionValue("+valueAdjustFn+"(oo->"+opt+"()))"
					+"});");
				});
			}
			print("	break;");
			print("}");
		}
	});
}
#endif


/// ------------------- pasted generated code below -------------------



case tflite::BuiltinOperator_ABS: {
	break;
}
case tflite::BuiltinOperator_ADD_N: {
	break;
}
case tflite::BuiltinOperator_ADD: {
	auto oo = o->builtin_options_as_AddOptions();
	ourOpts->push_back({PluginInterface::OperatorOption_FUSED_ACTIVATION_FUNCTION, PluginInterface::OperatorOptionValue(Helpers::convertActivationFunction(oo->fused_activation_function()))});
	break;
}
case tflite::BuiltinOperator_ARG_MAX: {
	auto oo = o->builtin_options_as_ArgMaxOptions();
	ourOpts->push_back({PluginInterface::OperatorOption_OUTPUT_TYPE, PluginInterface::OperatorOptionValue((oo->output_type()))});
	break;
}
case tflite::BuiltinOperator_ARG_MIN: {
	auto oo = o->builtin_options_as_ArgMinOptions();
	ourOpts->push_back({PluginInterface::OperatorOption_OUTPUT_TYPE, PluginInterface::OperatorOptionValue((oo->output_type()))});
	break;
}
case tflite::BuiltinOperator_BATCH_TO_SPACE_ND: {
	break;
}
case tflite::BuiltinOperator_BIDIRECTIONAL_SEQUENCE_LSTM: {
	auto oo = o->builtin_options_as_BidirectionalSequenceLSTMOptions();
	ourOpts->push_back({PluginInterface::OperatorOption_CELL_CLIP, PluginInterface::OperatorOptionValue((oo->cell_clip()))});
	ourOpts->push_back({PluginInterface::OperatorOption_FUSED_ACTIVATION_FUNCTION, PluginInterface::OperatorOptionValue(Helpers::convertActivationFunction(oo->fused_activation_function()))});
	ourOpts->push_back({PluginInterface::OperatorOption_MERGE_OUTPUTS, PluginInterface::OperatorOptionValue((oo->merge_outputs()))});
	ourOpts->push_back({PluginInterface::OperatorOption_PROJ_CLIP, PluginInterface::OperatorOptionValue((oo->proj_clip()))});
	ourOpts->push_back({PluginInterface::OperatorOption_TIME_MAJOR, PluginInterface::OperatorOptionValue((oo->time_major()))});
	break;
}
case tflite::BuiltinOperator_BIDIRECTIONAL_SEQUENCE_RNN: {
	auto oo = o->builtin_options_as_BidirectionalSequenceRNNOptions();
	ourOpts->push_back({PluginInterface::OperatorOption_FUSED_ACTIVATION_FUNCTION, PluginInterface::OperatorOptionValue(Helpers::convertActivationFunction(oo->fused_activation_function()))});
	ourOpts->push_back({PluginInterface::OperatorOption_MERGE_OUTPUTS, PluginInterface::OperatorOptionValue((oo->merge_outputs()))});
	ourOpts->push_back({PluginInterface::OperatorOption_TIME_MAJOR, PluginInterface::OperatorOptionValue((oo->time_major()))});
	break;
}
case tflite::BuiltinOperator_CALL: {
	auto oo = o->builtin_options_as_CallOptions();
	ourOpts->push_back({PluginInterface::OperatorOption_SUBGRAPH, PluginInterface::OperatorOptionValue((oo->subgraph()))});
	break;
}
case tflite::BuiltinOperator_CAST: {
	auto oo = o->builtin_options_as_CastOptions();
	ourOpts->push_back({PluginInterface::OperatorOption_IN_DATA_TYPE, PluginInterface::OperatorOptionValue((oo->in_data_type()))});
	ourOpts->push_back({PluginInterface::OperatorOption_OUT_DATA_TYPE, PluginInterface::OperatorOptionValue((oo->out_data_type()))});
	break;
}
case tflite::BuiltinOperator_CONCAT_EMBEDDINGS: {
	auto oo = o->builtin_options_as_ConcatEmbeddingsOptions();
	ourOpts->push_back({PluginInterface::OperatorOption_EMBEDDING_DIM_PER_CHANNEL, PluginInterface::OperatorOptionValue(Helpers::convertFlatbuffersIntListToStl(oo->embedding_dim_per_channel()))});
	ourOpts->push_back({PluginInterface::OperatorOption_NUM_CHANNELS, PluginInterface::OperatorOptionValue((oo->num_channels()))});
	ourOpts->push_back({PluginInterface::OperatorOption_NUM_COLUMNS_PER_CHANNEL, PluginInterface::OperatorOptionValue(Helpers::convertFlatbuffersIntListToStl(oo->num_columns_per_channel()))});
	break;
}
case tflite::BuiltinOperator_CONCATENATION: {
	auto oo = o->builtin_options_as_ConcatenationOptions();
	ourOpts->push_back({PluginInterface::OperatorOption_AXIS, PluginInterface::OperatorOptionValue((oo->axis()))});
	ourOpts->push_back({PluginInterface::OperatorOption_FUSED_ACTIVATION_FUNCTION, PluginInterface::OperatorOptionValue(Helpers::convertActivationFunction(oo->fused_activation_function()))});
	break;
}
case tflite::BuiltinOperator_CONV_2D: {
	auto oo = o->builtin_options_as_Conv2DOptions();
	ourOpts->push_back({PluginInterface::OperatorOption_DILATION_H_FACTOR, PluginInterface::OperatorOptionValue((oo->dilation_h_factor()))});
	ourOpts->push_back({PluginInterface::OperatorOption_DILATION_W_FACTOR, PluginInterface::OperatorOptionValue((oo->dilation_w_factor()))});
	ourOpts->push_back({PluginInterface::OperatorOption_FUSED_ACTIVATION_FUNCTION, PluginInterface::OperatorOptionValue(Helpers::convertActivationFunction(oo->fused_activation_function()))});
	ourOpts->push_back({PluginInterface::OperatorOption_PADDING, PluginInterface::OperatorOptionValue(Helpers::convertPaddingType(oo->padding()))});
	ourOpts->push_back({PluginInterface::OperatorOption_STRIDE_H, PluginInterface::OperatorOptionValue((oo->stride_h()))});
	ourOpts->push_back({PluginInterface::OperatorOption_STRIDE_W, PluginInterface::OperatorOptionValue((oo->stride_w()))});
	break;
}
case tflite::BuiltinOperator_COS: {
	break;
}
case tflite::BuiltinOperator_DENSIFY: {
	break;
}
case tflite::BuiltinOperator_DEPTH_TO_SPACE: {
	auto oo = o->builtin_options_as_DepthToSpaceOptions();
	ourOpts->push_back({PluginInterface::OperatorOption_BLOCK_SIZE, PluginInterface::OperatorOptionValue((oo->block_size()))});
	break;
}
case tflite::BuiltinOperator_DEPTHWISE_CONV_2D: {
	auto oo = o->builtin_options_as_DepthwiseConv2DOptions();
	ourOpts->push_back({PluginInterface::OperatorOption_DEPTH_MULTIPLIER, PluginInterface::OperatorOptionValue((oo->depth_multiplier()))});
	ourOpts->push_back({PluginInterface::OperatorOption_DILATION_H_FACTOR, PluginInterface::OperatorOptionValue((oo->dilation_h_factor()))});
	ourOpts->push_back({PluginInterface::OperatorOption_DILATION_W_FACTOR, PluginInterface::OperatorOptionValue((oo->dilation_w_factor()))});
	ourOpts->push_back({PluginInterface::OperatorOption_FUSED_ACTIVATION_FUNCTION, PluginInterface::OperatorOptionValue(Helpers::convertActivationFunction(oo->fused_activation_function()))});
	ourOpts->push_back({PluginInterface::OperatorOption_PADDING, PluginInterface::OperatorOptionValue(Helpers::convertPaddingType(oo->padding()))});
	ourOpts->push_back({PluginInterface::OperatorOption_STRIDE_H, PluginInterface::OperatorOptionValue((oo->stride_h()))});
	ourOpts->push_back({PluginInterface::OperatorOption_STRIDE_W, PluginInterface::OperatorOptionValue((oo->stride_w()))});
	break;
}
case tflite::BuiltinOperator_DEQUANTIZE: {
	break;
}
case tflite::BuiltinOperator_DIV: {
	auto oo = o->builtin_options_as_DivOptions();
	ourOpts->push_back({PluginInterface::OperatorOption_FUSED_ACTIVATION_FUNCTION, PluginInterface::OperatorOptionValue(Helpers::convertActivationFunction(oo->fused_activation_function()))});
	break;
}
case tflite::BuiltinOperator_EMBEDDING_LOOKUP_SPARSE: {
	auto oo = o->builtin_options_as_EmbeddingLookupSparseOptions();
	ourOpts->push_back({PluginInterface::OperatorOption_COMBINER, PluginInterface::OperatorOptionValue((oo->combiner()))});
	break;
}
case tflite::BuiltinOperator_EQUAL: {
	break;
}
case tflite::BuiltinOperator_EXP: {
	break;
}
case tflite::BuiltinOperator_EXPAND_DIMS: {
	break;
}
case tflite::BuiltinOperator_FAKE_QUANT: {
	auto oo = o->builtin_options_as_FakeQuantOptions();
	ourOpts->push_back({PluginInterface::OperatorOption_MAX, PluginInterface::OperatorOptionValue((oo->max()))});
	ourOpts->push_back({PluginInterface::OperatorOption_MIN, PluginInterface::OperatorOptionValue((oo->min()))});
	ourOpts->push_back({PluginInterface::OperatorOption_NARROW_RANGE, PluginInterface::OperatorOptionValue((oo->narrow_range()))});
	ourOpts->push_back({PluginInterface::OperatorOption_NUM_BITS, PluginInterface::OperatorOptionValue((oo->num_bits()))});
	break;
}
case tflite::BuiltinOperator_FILL: {
	break;
}
case tflite::BuiltinOperator_FLOOR_DIV: {
	break;
}
case tflite::BuiltinOperator_FLOOR_MOD: {
	break;
}
case tflite::BuiltinOperator_FULLY_CONNECTED: {
	auto oo = o->builtin_options_as_FullyConnectedOptions();
	ourOpts->push_back({PluginInterface::OperatorOption_FUSED_ACTIVATION_FUNCTION, PluginInterface::OperatorOptionValue(Helpers::convertActivationFunction(oo->fused_activation_function()))});
	ourOpts->push_back({PluginInterface::OperatorOption_KEEP_NUM_DIMS, PluginInterface::OperatorOptionValue((oo->keep_num_dims()))});
	ourOpts->push_back({PluginInterface::OperatorOption_WEIGHTS_FORMAT, PluginInterface::OperatorOptionValue((oo->weights_format()))});
	break;
}
case tflite::BuiltinOperator_GATHER_ND: {
	break;
}
case tflite::BuiltinOperator_GATHER: {
	auto oo = o->builtin_options_as_GatherOptions();
	ourOpts->push_back({PluginInterface::OperatorOption_AXIS, PluginInterface::OperatorOptionValue((oo->axis()))});
	break;
}
case tflite::BuiltinOperator_GREATER_EQUAL: {
	break;
}
case tflite::BuiltinOperator_GREATER: {
	break;
}
case tflite::BuiltinOperator_HARD_SWISH: {
	break;
}
case tflite::BuiltinOperator_IF: {
	auto oo = o->builtin_options_as_IfOptions();
	ourOpts->push_back({PluginInterface::OperatorOption_ELSE_SUBGRAPH_INDEX, PluginInterface::OperatorOptionValue((oo->else_subgraph_index()))});
	ourOpts->push_back({PluginInterface::OperatorOption_THEN_SUBGRAPH_INDEX, PluginInterface::OperatorOptionValue((oo->then_subgraph_index()))});
	break;
}
case tflite::BuiltinOperator_L2_NORMALIZATION: {
	auto oo = o->builtin_options_as_L2NormOptions();
	ourOpts->push_back({PluginInterface::OperatorOption_FUSED_ACTIVATION_FUNCTION, PluginInterface::OperatorOptionValue(Helpers::convertActivationFunction(oo->fused_activation_function()))});
	break;
}
case tflite::BuiltinOperator_LSH_PROJECTION: {
	auto oo = o->builtin_options_as_LSHProjectionOptions();
	ourOpts->push_back({PluginInterface::OperatorOption_TYPE, PluginInterface::OperatorOptionValue((oo->type()))});
	break;
}
case tflite::BuiltinOperator_LSTM: {
	auto oo = o->builtin_options_as_LSTMOptions();
	ourOpts->push_back({PluginInterface::OperatorOption_CELL_CLIP, PluginInterface::OperatorOptionValue((oo->cell_clip()))});
	ourOpts->push_back({PluginInterface::OperatorOption_FUSED_ACTIVATION_FUNCTION, PluginInterface::OperatorOptionValue(Helpers::convertActivationFunction(oo->fused_activation_function()))});
	ourOpts->push_back({PluginInterface::OperatorOption_KERNEL_TYPE, PluginInterface::OperatorOptionValue((oo->kernel_type()))});
	ourOpts->push_back({PluginInterface::OperatorOption_PROJ_CLIP, PluginInterface::OperatorOptionValue((oo->proj_clip()))});
	break;
}
case tflite::BuiltinOperator_LEAKY_RELU: {
	auto oo = o->builtin_options_as_LeakyReluOptions();
	ourOpts->push_back({PluginInterface::OperatorOption_ALPHA, PluginInterface::OperatorOptionValue((oo->alpha()))});
	break;
}
case tflite::BuiltinOperator_LESS_EQUAL: {
	break;
}
case tflite::BuiltinOperator_LESS: {
	break;
}
case tflite::BuiltinOperator_LOCAL_RESPONSE_NORMALIZATION: {
	auto oo = o->builtin_options_as_LocalResponseNormalizationOptions();
	ourOpts->push_back({PluginInterface::OperatorOption_ALPHA, PluginInterface::OperatorOptionValue((oo->alpha()))});
	ourOpts->push_back({PluginInterface::OperatorOption_BETA, PluginInterface::OperatorOptionValue((oo->beta()))});
	ourOpts->push_back({PluginInterface::OperatorOption_BIAS, PluginInterface::OperatorOptionValue((oo->bias()))});
	ourOpts->push_back({PluginInterface::OperatorOption_RADIUS, PluginInterface::OperatorOptionValue((oo->radius()))});
	break;
}
case tflite::BuiltinOperator_LOG_SOFTMAX: {
	break;
}
case tflite::BuiltinOperator_LOGICAL_AND: {
	break;
}
case tflite::BuiltinOperator_LOGICAL_NOT: {
	break;
}
case tflite::BuiltinOperator_LOGICAL_OR: {
	break;
}
case tflite::BuiltinOperator_MATRIX_DIAG: {
	break;
}
case tflite::BuiltinOperator_MATRIX_SET_DIAG: {
	break;
}
case tflite::BuiltinOperator_MINIMUM:
case tflite::BuiltinOperator_MAXIMUM: {
	break;
}
case tflite::BuiltinOperator_MIRROR_PAD: {
	auto oo = o->builtin_options_as_MirrorPadOptions();
	ourOpts->push_back({PluginInterface::OperatorOption_MODE, PluginInterface::OperatorOptionValue((oo->mode()))});
	break;
}
case tflite::BuiltinOperator_MUL: {
	auto oo = o->builtin_options_as_MulOptions();
	ourOpts->push_back({PluginInterface::OperatorOption_FUSED_ACTIVATION_FUNCTION, PluginInterface::OperatorOptionValue(Helpers::convertActivationFunction(oo->fused_activation_function()))});
	break;
}
case tflite::BuiltinOperator_NEG: {
	break;
}
case tflite::BuiltinOperator_NON_MAX_SUPPRESSION_V4: {
	break;
}
case tflite::BuiltinOperator_NON_MAX_SUPPRESSION_V5: {
	break;
}
case tflite::BuiltinOperator_NOT_EQUAL: {
	break;
}
case tflite::BuiltinOperator_ONE_HOT: {
	auto oo = o->builtin_options_as_OneHotOptions();
	ourOpts->push_back({PluginInterface::OperatorOption_AXIS, PluginInterface::OperatorOptionValue((oo->axis()))});
	break;
}
case tflite::BuiltinOperator_PACK: {
	auto oo = o->builtin_options_as_PackOptions();
	ourOpts->push_back({PluginInterface::OperatorOption_AXIS, PluginInterface::OperatorOptionValue((oo->axis()))});
	ourOpts->push_back({PluginInterface::OperatorOption_VALUES_COUNT, PluginInterface::OperatorOptionValue((oo->values_count()))});
	break;
}
case tflite::BuiltinOperator_PAD: {
	break;
}
case tflite::BuiltinOperator_PADV2: {
	break;
}
case tflite::BuiltinOperator_AVERAGE_POOL_2D:
case tflite::BuiltinOperator_MAX_POOL_2D:
case tflite::BuiltinOperator_L2_POOL_2D: {
	auto oo = o->builtin_options_as_Pool2DOptions();
	ourOpts->push_back({PluginInterface::OperatorOption_FILTER_HEIGHT, PluginInterface::OperatorOptionValue((oo->filter_height()))});
	ourOpts->push_back({PluginInterface::OperatorOption_FILTER_WIDTH, PluginInterface::OperatorOptionValue((oo->filter_width()))});
	ourOpts->push_back({PluginInterface::OperatorOption_FUSED_ACTIVATION_FUNCTION, PluginInterface::OperatorOptionValue(Helpers::convertActivationFunction(oo->fused_activation_function()))});
	ourOpts->push_back({PluginInterface::OperatorOption_PADDING, PluginInterface::OperatorOptionValue(Helpers::convertPaddingType(oo->padding()))});
	ourOpts->push_back({PluginInterface::OperatorOption_STRIDE_H, PluginInterface::OperatorOptionValue((oo->stride_h()))});
	ourOpts->push_back({PluginInterface::OperatorOption_STRIDE_W, PluginInterface::OperatorOptionValue((oo->stride_w()))});
	break;
}
case tflite::BuiltinOperator_POW: {
	break;
}
case tflite::BuiltinOperator_QUANTIZE: {
	break;
}
case tflite::BuiltinOperator_RNN: {
	auto oo = o->builtin_options_as_RNNOptions();
	ourOpts->push_back({PluginInterface::OperatorOption_FUSED_ACTIVATION_FUNCTION, PluginInterface::OperatorOptionValue(Helpers::convertActivationFunction(oo->fused_activation_function()))});
	break;
}
case tflite::BuiltinOperator_RANGE: {
	break;
}
case tflite::BuiltinOperator_RANK: {
	break;
}
case tflite::BuiltinOperator_REDUCE_PROD:
case tflite::BuiltinOperator_REDUCE_MAX:
case tflite::BuiltinOperator_REDUCE_MIN:
case tflite::BuiltinOperator_REDUCE_ANY: {
	auto oo = o->builtin_options_as_ReducerOptions();
	ourOpts->push_back({PluginInterface::OperatorOption_KEEP_DIMS, PluginInterface::OperatorOptionValue((oo->keep_dims()))});
	break;
}
case tflite::BuiltinOperator_RESHAPE: {
	auto oo = o->builtin_options_as_ReshapeOptions();
	ourOpts->push_back({PluginInterface::OperatorOption_NEW_SHAPE, PluginInterface::OperatorOptionValue(Helpers::convertFlatbuffersIntListToStl(oo->new_shape()))});
	break;
}
case tflite::BuiltinOperator_RESIZE_BILINEAR: {
	auto oo = o->builtin_options_as_ResizeBilinearOptions();
	ourOpts->push_back({PluginInterface::OperatorOption_ALIGN_CORNERS, PluginInterface::OperatorOptionValue((oo->align_corners()))});
	break;
}
case tflite::BuiltinOperator_RESIZE_NEAREST_NEIGHBOR: {
	auto oo = o->builtin_options_as_ResizeNearestNeighborOptions();
	ourOpts->push_back({PluginInterface::OperatorOption_ALIGN_CORNERS, PluginInterface::OperatorOptionValue((oo->align_corners()))});
	break;
}
case tflite::BuiltinOperator_REVERSE_SEQUENCE: {
	auto oo = o->builtin_options_as_ReverseSequenceOptions();
	ourOpts->push_back({PluginInterface::OperatorOption_BATCH_DIM, PluginInterface::OperatorOptionValue((oo->batch_dim()))});
	ourOpts->push_back({PluginInterface::OperatorOption_SEQ_DIM, PluginInterface::OperatorOptionValue((oo->seq_dim()))});
	break;
}
case tflite::BuiltinOperator_REVERSE_V2: {
	break;
}
case tflite::BuiltinOperator_SVDF: {
	auto oo = o->builtin_options_as_SVDFOptions();
	ourOpts->push_back({PluginInterface::OperatorOption_FUSED_ACTIVATION_FUNCTION, PluginInterface::OperatorOptionValue(Helpers::convertActivationFunction(oo->fused_activation_function()))});
	ourOpts->push_back({PluginInterface::OperatorOption_RANK, PluginInterface::OperatorOptionValue((oo->rank()))});
	break;
}
case tflite::BuiltinOperator_SCATTER_ND: {
	break;
}
case tflite::BuiltinOperator_SELECT: {
	break;
}
case tflite::BuiltinOperator_SELECT_V2: {
	break;
}
case tflite::BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_RNN: {
	auto oo = o->builtin_options_as_SequenceRNNOptions();
	ourOpts->push_back({PluginInterface::OperatorOption_FUSED_ACTIVATION_FUNCTION, PluginInterface::OperatorOptionValue(Helpers::convertActivationFunction(oo->fused_activation_function()))});
	ourOpts->push_back({PluginInterface::OperatorOption_TIME_MAJOR, PluginInterface::OperatorOptionValue((oo->time_major()))});
	break;
}
case tflite::BuiltinOperator_SHAPE: {
	auto oo = o->builtin_options_as_ShapeOptions();
	ourOpts->push_back({PluginInterface::OperatorOption_OUT_TYPE, PluginInterface::OperatorOptionValue((oo->out_type()))});
	break;
}
case tflite::BuiltinOperator_SKIP_GRAM: {
	auto oo = o->builtin_options_as_SkipGramOptions();
	ourOpts->push_back({PluginInterface::OperatorOption_INCLUDE_ALL_NGRAMS, PluginInterface::OperatorOptionValue((oo->include_all_ngrams()))});
	ourOpts->push_back({PluginInterface::OperatorOption_MAX_SKIP_SIZE, PluginInterface::OperatorOptionValue((oo->max_skip_size()))});
	ourOpts->push_back({PluginInterface::OperatorOption_NGRAM_SIZE, PluginInterface::OperatorOptionValue((oo->ngram_size()))});
	break;
}
case tflite::BuiltinOperator_SLICE: {
	break;
}
case tflite::BuiltinOperator_SOFTMAX: {
	auto oo = o->builtin_options_as_SoftmaxOptions();
	ourOpts->push_back({PluginInterface::OperatorOption_BETA, PluginInterface::OperatorOptionValue((oo->beta()))});
	break;
}
case tflite::BuiltinOperator_SPACE_TO_BATCH_ND: {
	break;
}
case tflite::BuiltinOperator_SPACE_TO_DEPTH: {
	auto oo = o->builtin_options_as_SpaceToDepthOptions();
	ourOpts->push_back({PluginInterface::OperatorOption_BLOCK_SIZE, PluginInterface::OperatorOptionValue((oo->block_size()))});
	break;
}
case tflite::BuiltinOperator_SPARSE_TO_DENSE: {
	auto oo = o->builtin_options_as_SparseToDenseOptions();
	ourOpts->push_back({PluginInterface::OperatorOption_VALIDATE_INDICES, PluginInterface::OperatorOptionValue((oo->validate_indices()))});
	break;
}
case tflite::BuiltinOperator_SPLIT: {
	auto oo = o->builtin_options_as_SplitOptions();
	ourOpts->push_back({PluginInterface::OperatorOption_NUM_SPLITS, PluginInterface::OperatorOptionValue((oo->num_splits()))});
	break;
}
case tflite::BuiltinOperator_SPLIT_V: {
	auto oo = o->builtin_options_as_SplitVOptions();
	ourOpts->push_back({PluginInterface::OperatorOption_NUM_SPLITS, PluginInterface::OperatorOptionValue((oo->num_splits()))});
	break;
}
case tflite::BuiltinOperator_SQUARE: {
	break;
}
case tflite::BuiltinOperator_SQUARED_DIFFERENCE: {
	break;
}
case tflite::BuiltinOperator_SQUEEZE: {
	auto oo = o->builtin_options_as_SqueezeOptions();
	ourOpts->push_back({PluginInterface::OperatorOption_SQUEEZE_DIMS, PluginInterface::OperatorOptionValue(Helpers::convertFlatbuffersIntListToStl(oo->squeeze_dims()))});
	break;
}
case tflite::BuiltinOperator_STRIDED_SLICE: {
	auto oo = o->builtin_options_as_StridedSliceOptions();
	ourOpts->push_back({PluginInterface::OperatorOption_BEGIN_MASK, PluginInterface::OperatorOptionValue((oo->begin_mask()))});
	ourOpts->push_back({PluginInterface::OperatorOption_ELLIPSIS_MASK, PluginInterface::OperatorOptionValue((oo->ellipsis_mask()))});
	ourOpts->push_back({PluginInterface::OperatorOption_END_MASK, PluginInterface::OperatorOptionValue((oo->end_mask()))});
	ourOpts->push_back({PluginInterface::OperatorOption_NEW_AXIS_MASK, PluginInterface::OperatorOptionValue((oo->new_axis_mask()))});
	ourOpts->push_back({PluginInterface::OperatorOption_SHRINK_AXIS_MASK, PluginInterface::OperatorOptionValue((oo->shrink_axis_mask()))});
	break;
}
case tflite::BuiltinOperator_SUB: {
	auto oo = o->builtin_options_as_SubOptions();
	ourOpts->push_back({PluginInterface::OperatorOption_FUSED_ACTIVATION_FUNCTION, PluginInterface::OperatorOptionValue(Helpers::convertActivationFunction(oo->fused_activation_function()))});
	break;
}
case tflite::BuiltinOperator_TILE: {
	break;
}
case tflite::BuiltinOperator_TOPK_V2: {
	break;
}
case tflite::BuiltinOperator_TRANSPOSE_CONV: {
	auto oo = o->builtin_options_as_TransposeConvOptions();
	ourOpts->push_back({PluginInterface::OperatorOption_PADDING, PluginInterface::OperatorOptionValue(Helpers::convertPaddingType(oo->padding()))});
	ourOpts->push_back({PluginInterface::OperatorOption_STRIDE_H, PluginInterface::OperatorOptionValue((oo->stride_h()))});
	ourOpts->push_back({PluginInterface::OperatorOption_STRIDE_W, PluginInterface::OperatorOptionValue((oo->stride_w()))});
	break;
}
case tflite::BuiltinOperator_TRANSPOSE: {
	break;
}
case tflite::BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_LSTM: {
	auto oo = o->builtin_options_as_UnidirectionalSequenceLSTMOptions();
	ourOpts->push_back({PluginInterface::OperatorOption_CELL_CLIP, PluginInterface::OperatorOptionValue((oo->cell_clip()))});
	ourOpts->push_back({PluginInterface::OperatorOption_FUSED_ACTIVATION_FUNCTION, PluginInterface::OperatorOptionValue(Helpers::convertActivationFunction(oo->fused_activation_function()))});
	ourOpts->push_back({PluginInterface::OperatorOption_PROJ_CLIP, PluginInterface::OperatorOptionValue((oo->proj_clip()))});
	ourOpts->push_back({PluginInterface::OperatorOption_TIME_MAJOR, PluginInterface::OperatorOptionValue((oo->time_major()))});
	break;
}
case tflite::BuiltinOperator_UNIQUE: {
	auto oo = o->builtin_options_as_UniqueOptions();
	ourOpts->push_back({PluginInterface::OperatorOption_IDX_OUT_TYPE, PluginInterface::OperatorOptionValue((oo->idx_out_type()))});
	break;
}
case tflite::BuiltinOperator_UNPACK: {
	auto oo = o->builtin_options_as_UnpackOptions();
	ourOpts->push_back({PluginInterface::OperatorOption_AXIS, PluginInterface::OperatorOptionValue((oo->axis()))});
	ourOpts->push_back({PluginInterface::OperatorOption_NUM, PluginInterface::OperatorOptionValue((oo->num()))});
	break;
}
case tflite::BuiltinOperator_WHERE: {
	break;
}
case tflite::BuiltinOperator_WHILE: {
	auto oo = o->builtin_options_as_WhileOptions();
	ourOpts->push_back({PluginInterface::OperatorOption_BODY_SUBGRAPH_INDEX, PluginInterface::OperatorOptionValue((oo->body_subgraph_index()))});
	ourOpts->push_back({PluginInterface::OperatorOption_COND_SUBGRAPH_INDEX, PluginInterface::OperatorOptionValue((oo->cond_subgraph_index()))});
	break;
}
case tflite::BuiltinOperator_ZEROS_LIKE: {
	break;
}
