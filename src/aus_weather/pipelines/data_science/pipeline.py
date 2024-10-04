from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    split_feature_target,
    perform_train_test_split,
    train_logistic_regression_model,
    evaluate_model
    )
print("data_science pipeline started")

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        
        node(
            func=split_feature_target,
            inputs=dict(df="scaled_data",target_column="params:target_column"),
            outputs=["X","y"],
            name="split_feature_target_node"
        ),
        
        node(
            func=perform_train_test_split,
            inputs=["X","y"],
            outputs=["X_train", "X_test", "y_train", "y_test"],
            name="perform_train_test_split_node"
        ),
        
        node(
            func=train_logistic_regression_model,
            inputs=["X_train", "X_test", "y_train", "y_test"],
            outputs=["y_pred_train_df","y_pred_test_df", "logreg_model"],
            name="train_logistic_regression_model_node"
        ),
        
        node(
            func=evaluate_model,
            inputs=["y_pred_train_df","y_pred_test_df", "y_train","y_test"],
            outputs="evaluation_results",
            name="evaluate_model_node"
        ),
        
    ])