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
    packages_to_install=["pandas", "pyarrow",  "scikit-learn==1.0" , "fsspec" , "gcsfs"]
    , base_image="python:3.9"
    , output_component_file="./components/train.yaml"
)
def train_model(
    train_data:  Input[Dataset],
    model: Output[Model], 
):
    
    print(f"train_data : {train_data}")
    print(f"model : {model}")
    
    from sklearn.ensemble import RandomForestRegressor
    import pandas as pd
    import pickle
    import sklearn

    train_ds = pd.read_csv(train_data.path+".csv")
    my_model = RandomForestRegressor()
    
    target = "cnt"
    
    x_train = train_ds.drop(columns=target, axis=1)
    y_train = train_ds[target]
    
    my_model.fit(x_train , y_train)
    model.metadata["model_name"] = "RandomForestRegressor"
    model.metadata["framework"] = "sklearn"
    model.metadata["framework_version"] = sklearn.__version__
    file_name = model.path + f".pkl"
    
    with open(file_name, 'wb') as file:  
        pickle.dump(my_model, file)
