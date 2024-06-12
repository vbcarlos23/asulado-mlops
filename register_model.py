from typing import NamedTuple

import kfp
from kfp.v2 import dsl
from kfp.v2.dsl import (Artifact, Dataset, Input, InputPath, Model, Output,
                        OutputPath, component , ClassificationMetrics , Metrics)

from kfp.v2 import compiler
from kfp.components import load_component_from_file

import json
import yaml
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
    output_component_file="./components/register_model.yaml"
)
def register_model(serving_container_uri: str, project_id: str, region: str, model_name: str, model: Input[Model]) -> NamedTuple("Outputs", [("model_resource_name", str)]):
    from google.cloud import aiplatform

    aiplatform.init(project=project_id, location=region)
    model = aiplatform.Model.upload(display_name=model_name, artifact_uri=model.uri[:-5], serving_container_image_uri=serving_container_uri)
    return (model.resource_name,)