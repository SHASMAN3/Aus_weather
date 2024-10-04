"""Project pipelines."""
from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline


# def register_pipelines() -> Dict[str, Pipeline]:
#     """Register the project's pipelines.

#     Returns:
#         A mapping from pipeline names to ``Pipeline`` objects.
#     """
#     pipelines = find_pipelines()
#     pipelines["__default__"] = sum(pipelines.values())
#     return pipelines


from aus_weather.pipelines import data_processing as dp

from aus_weather.pipelines import data_science as ds
# # from src.aus_weather.pipelines import inference as infer
from aus_weather.pipelines import inference as infer

# # aus_weather/pipeline_registry.py

# from aus_weather.pipelines import data_processing as dp
print("module import successfull")

# def register_pipelines():
#     data_processing_pipeline = dp.create_pipeline()
#     return {"__default__": data_processing_pipeline}


def register_pipelines() -> Dict[str, Pipeline]:
    '''Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
   '''
    data_processing_pipeline=dp.create_pipeline()
    data_science_pipeline=ds.create_pipeline()
    inference_pipeline=infer.create_pipeline()
    
    return{
        "__default__": data_processing_pipeline + data_science_pipeline + inference_pipeline,
        "dp": data_processing_pipeline,
        "ds": data_science_pipeline,
        "infer": inference_pipeline
    }
print("pipeline registed")