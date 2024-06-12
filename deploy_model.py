from typing import NamedTuple

import kfp
from kfp.v2 import dsl
from kfp.v2.dsl import (Artifact, Dataset, Input, InputPath, Model, Output,
                        OutputPath, component , ClassificationMetrics , Metrics)

from kfp.v2 import compiler
from kfp.components import load_component_from_file

import json
import yaml
@component(
    packages_to_install=["google-cloud-aiplatform"],
    base_image="python:3.9",
    output_component_file="./components/deploy_model.yaml"
)
def deploy_model(model_resource_name: str, project_id: str, region: str) -> NamedTuple("Outputs", [("endpoint_resource_name", str)]):
    from google.cloud import aiplatform

    aiplatform.init(project=project_id, location=region)

    model = aiplatform.Model(model_resource_name)
    endpoint = model.deploy(machine_type="n1-standard-4", min_replica_count=1, max_replica_count=1)

    return (endpoint.resource_name,)
