
import json
import yaml

import kfp
from kfp.v2 import dsl
from kfp.v2 import compiler
from kfp.components import load_component_from_file

download_data = load_component_from_file("./components/download_data.yaml")
preprocess_data = load_component_from_file("./components/preprocess_data.yaml")
train_model = load_component_from_file("./components/train.yaml")
evaluate_model = load_component_from_file("./components/evaluate_model.yaml")
register_model = load_component_from_file("./components/register_model.yaml")
deploy_model = load_component_from_file("./components/deploy_model.yaml")

#read configuration from file
with open("config.json") as json_file:
    config = json.load(json_file)
    
PIPELINE_NAME = config.get("pipeline_name")
PACKAGE_PATH = config.get("pipeline_package_path")
BUCKET_URI = config.get("staging_bucket_uri")
PIPELINE_ROOT = "{}/pipeline_root/kfp_tabular_data_regression".format(BUCKET_URI)
print(f"PIPELINE_ROOT :{PIPELINE_ROOT}")

@dsl.pipeline(
    # Default pipeline root. You can override it when submitting the pipeline.
    pipeline_root=PIPELINE_ROOT,
    # A name for the pipeline. Use to define the pipeline Context.
    name=PIPELINE_NAME,
    
)
def pipeline(project: str = "",
 region: str = "",
 service_account: str = "",
 staging_bucket_uri: str = "",
 pipeline_name: str = "",
 pipeline_package_path: str = "",
 input_data_path: str = "",
 input_data_filename: str = "",
 target_column_name: str = "",
 train_size: float = 0.8,
 test_size: float = 0.1,
 valid_size: float = 0.1,
 hypertune_container_image_uri: str = "" ,
 hypertune_machine_type: str = "",
 hypertune_machine_replica_count: int = 1 ,
 hypertune_max_trial_count: int = 1 ,
 hypertune_parallel_trial_count: int = 1 ,
 hypertune_metric : str = "" ,
 hypertune_metric_objective : str = "" ,
 hypertune_job_name: str = "" ,
 deployment_metric: str = "" ,
 deployment_metric_threshold: float = 0.8 ,
 serving_container_uri : str = "" ,
 model_name : str = "", 
 user_email : str = "",
 monitoring_job_name : str = "" ,
 predict_instance_schema_uri : str = ""
):
    
    download_data_op = download_data(input_data_path = input_data_path 
                                     , input_data_filename = input_data_filename
                                    )
    
    
    preprocess_data_op = preprocess_data(train_size = train_size
                                         , test_size = test_size
                                         , valid_size = valid_size
                                         , input_data = download_data_op.outputs["downloaded_data"])
    
    train_model_op = train_model(train_data = preprocess_data_op.outputs["train_data"])
    
    
    evaluate_model_op = evaluate_model(test_data = preprocess_data_op.outputs["test_data"]
                                       ,model = train_model_op.outputs["model"]
                                       ,target_column_name = target_column_name
                                       ,deployment_metric = deployment_metric 
                                       ,deployment_metric_threshold = deployment_metric_threshold
                                      )
    
    with dsl.Condition(evaluate_model_op.outputs["deploy_flag"] == "True"):
        
        register_model_op = register_model(serving_container_uri = serving_container_uri 
                                       , model = train_model_op.outputs["model"]
                                       , model_name = model_name
                                       , project_id = project 
                                       , region = region)
        
        #deploy only if metric value exceeds deployment threshold
        deploy_model_op = deploy_model(model_resource_name = register_model_op.outputs["model_resource_name"] 
                                   , project_id = project 
                                   , region = region)
    
compiler.Compiler().compile(
    pipeline_func=pipeline
    , package_path=PACKAGE_PATH
)
