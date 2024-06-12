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
    packages_to_install=["pandas", "pyarrow", "fsspec" , "gcsfs"],
    base_image="python:3.9",
    output_component_file="./components/preprocess_data.yaml"
)
def preprocess_data(train_size : float 
                    , test_size : float
                    , valid_size : float
                    , train_data : Output[Dataset]
                    , valid_data : Output[Dataset]
                    , test_data : Output[Dataset]
                    , input_data:  Input[Dataset]
                   ):
    
    import numpy as np
    import pandas as pd
    
    print(f"train_size : {train_size}")
    print(f"test_size : {test_size}")
    print(f"valid_size : {valid_size}")
    print(f"train_data : {train_data}")
    print(f"valid_data : {valid_data}")
    print(f"test_data : {test_data}")
    print(f"input_data : {input_data}")
    
    data = pd.read_csv(input_data.path + ".csv")
    
    modelling_columns = ["season" 
                     , "yr" 
                     ,"mnth" 
                     ,"hr" 
                     ,"holiday" 
                     , "weekday" 
                     , "workingday" 
                     , "weathersit" 
                     , "temp" 
                     , "atemp" 
                     , "hum" 
                     , "windspeed" 
                     , "casual" 
                     , "registered" 
                     , "cnt"
                    ]
    
    data = data[modelling_columns]

    train_ds, valid_ds, test_ds = np.split(data.sample(frac=1, random_state=42), [int((train_size)*len(data)), int((1-test_size)*len(data))])
    train_ds.to_csv(train_data.path + ".csv" , index=False, encoding='utf-8-sig')
    valid_ds.to_csv(valid_data.path + ".csv" , index=False, encoding='utf-8-sig')
    test_ds.to_csv(test_data.path + ".csv" , index=False, encoding='utf-8-sig')
