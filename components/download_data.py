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
    packages_to_install=["pandas", "pyarrow" , "fsspec" , "gcsfs"],
    base_image="python:3.9",
    output_component_file="./components/download_data.yaml"
)
def download_data(input_data_path : str
                  , input_data_filename : str
                  , downloaded_data : Output[Dataset]):
    
    import pandas as pd
    import os
    
    print(f"input_data_path : {input_data_path}")
    print(f"input_data_filename : {input_data_filename}")
    print(f"downloaded_data : {downloaded_data}")
    print(f"downloaded_data.path : {downloaded_data.path}")
                
    url = os.path.join(input_data_path , input_data_filename)
    
    #read data from GCS location
    data = pd.read_csv(url)
    
    #write to output dataset path
    output_data_uri = downloaded_data.path + ".csv" 
    data.to_csv(output_data_uri 
                , index=False
                , encoding='utf-8-sig')