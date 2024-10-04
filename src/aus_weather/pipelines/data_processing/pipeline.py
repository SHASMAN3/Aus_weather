from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    preprocess_dataframe,
    process_date_column,
    impute_missing_values,
    label_encode_object_column,
    convert_to_float,
    drop_extreme_outliers,
    min_max_scale_dataframe,
)
print("data_processing module loaded")

def create_pipeline(*args, **kwargs) -> Pipeline:
    return pipeline([
        node(
            func=preprocess_dataframe,
            inputs="weather_aus_raw",
            outputs=["filtered_data", "inference_data"],
            name="preprocess_dataframe_node"
        ),

        node(
            func=process_date_column,
            inputs=dict(df="filtered_data", date_column="params:date_column"),
            outputs="processed_data",
            name="process_date_column_node"
        ),
        node(
            func=impute_missing_values,
            inputs="processed_data",
            outputs="imputed_data",
            name="impute_missing_values_node"
        ),
        node(
            func=label_encode_object_column,
            inputs="imputed_data",
            outputs=["encoded_data", "label_encoders"],
            name="label_encode_object_column_node"
        ),
        node(
            func=convert_to_float,
            inputs="encoded_data",
            outputs="float_data",
            name="convert_to_float_node"
        ),
        node(
            func=drop_extreme_outliers,
            inputs="float_data",
            outputs="outlier_free_data",
            name="drop_extreme_outliers_node"
        ),
        node(
            func=min_max_scale_dataframe,
            inputs="outlier_free_data",
            outputs="scaled_data",
            name="min_max_scale_dataframe_node"
        ),
    ])
print("dp pipeline created")
