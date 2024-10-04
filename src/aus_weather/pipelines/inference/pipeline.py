from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    process_date_column,
    impute_missing_values,
    transform_data,
    convert_to_float,
    min_max_scale_dataframe,
    split_feature_target,
    load_model,
    predict_and_update
)

def create_pipeline(*args, **kwargs) -> Pipeline:
    return pipeline([
        
        node(
            func=process_date_column,
            inputs=dict(df="inference_data", date_column="params:date_column"),
            outputs="processed_data_infer",
            name="process_date_column_infer_node"
        ),
        
        node(
            func=impute_missing_values,
            inputs="processed_data_infer",
            outputs="imputed_data_infer",
            name="impute_missing_values_infer_node"
        ),
        
        node(
            func=transform_data,
            inputs=["imputed_data_infer", "label_encoders"],
            outputs="encoded_data_infer",
            name="label_encode_object_columns_infer_node"
        ),
        
        node(
            func=convert_to_float,
            inputs="encoded_data_infer",
            outputs="float_data_infer",
            name="convert_to_float_infer_node"
        ),
        
        node(
            func=min_max_scale_dataframe,
            inputs="float_data_infer",
            outputs="scaled_data_infer",
            name="min_max_scale_dataframe_infer_node"
        ),
        
        node(
            func=split_feature_target,
            inputs=dict(df="scaled_data_infer",target_column="params:target_column"),
            outputs=["X_infer","y_infer"],
            name="split_feature_target_infer_node"
        ),
        
        node(
            func=load_model,
            inputs="logreg_model",
            outputs="loaded_model",
            name="load_model_node"
        ),
        
        node(
            func=predict_and_update,
            inputs=dict(X_infer='X_infer',df_inference='inference_data',model='loaded_model',target_column="params:target_column"),
            outputs="updated_inference_data",
            name="predict_and_update_node"
        ),
        
    ])
