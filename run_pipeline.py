from google.cloud import aiplatform
import yaml
import json

with open("config.json") as json_file:
    config = json.load(json_file)
    
SERVICE_ACCOUNT = config.get("service_account")
DISPLAY_NAME = config.get("pipeline_name")
PACKAGE_PATH = config.get("pipeline_package_path")
BUCKET_URI = config.get("staging_bucket_uri")
PIPELINE_ROOT = "{}/pipeline_root/kfp_tabular_data_regression".format(BUCKET_URI)
print(f"PIPELINE_ROOT :{PIPELINE_ROOT}")




job = aiplatform.PipelineJob(
    display_name=DISPLAY_NAME,
    template_path=PACKAGE_PATH,
    pipeline_root=PIPELINE_ROOT,
    parameter_values=config,
)

job.submit(service_account = SERVICE_ACCOUNT)
