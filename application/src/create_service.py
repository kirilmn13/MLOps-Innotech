import bentoml
import numpy as np
import pandas as pd
from bentoml.io import JSON, NumpyNdarray
from hydra import compose, initialize
from patsy import dmatrix
from pydantic import BaseModel

with initialize(version_base=None, config_path='../../config'):
    config = compose(config_name='main')
    FEATURES = config.process.features
    MODEL_NAME = config.model.name


class Employee(BaseModel):
    City: str = 'Pune'
    PaymentTier: int = 1
    Age: int = 25
    Gender: str = 'Female'
    EverBenched: str = 'No'
    ExperienceInCurrentDomain: int = 1


def add_dummy_data(df: pd.DataFrame):
    """Add dummy rows so that patsy can create features similar to the train dataset"""
    rows = {
        'City': ['Bangalore', 'New Delhi', 'Pune'],
        'Gender': ['Male', 'Female', 'Female'],
        'EverBenched': ['Yes', 'Yes', 'No'],
        'PaymentTier': [0, 0, 0],
        'Age': [0, 0, 0],
        'ExperienceInCurrentDomain': [0, 0, 0],
    }
    dummy_df = pd.DataFrame(rows)
    return pd.concat([df, dummy_df])


def rename_columns(X: pd.DataFrame):
    X.columns = X.columns.str.replace('[', '_', regex=True).str.replace(
        ']', '', regex=True
    )
    return X
# test

def transform_data(df: pd.DataFrame):
    """Transform the data"""
    dummy_df = add_dummy_data(df)
    feature_str = ' + '.join(FEATURES)
    dummy_X = dmatrix(f"{feature_str} - 1", dummy_df, return_type='dataframe')
    dummy_X = rename_columns(dummy_X)
    return dummy_X.iloc[0, :].values.reshape(1, -1)




model_ref = bentoml.xgboost.get('xgboost:latest')
_model_runner = model_ref.to_runner()
#model_runner.init_local()

# Create service with the model
service = bentoml.Service('predict_employee', runners=[_model_runner])


# @service.api(input=JSON(pydantic_model=Employee), output=NumpyNdarray())
# def predict(employee: Employee) -> np.ndarray:
#     """Transform the data then make predictions"""
#     df = pd.DataFrame(employee.dict(), index=[0])
#     df = transform_data(df)
#     result = service.run(df)[0]
#     return np.array(result)


@service.api(JSON(pydantic_model=Employee), output=NumpyNdarray())
async def predict(employee: Employee) -> np.ndarray:
    df = pd.DataFrame(employee.dict(), index=[0])
    df = transform_data(df)
    results = await _model_runner.predict_proba.async_run(df)
    predictions = np.argmax(results, axis=1)  # 0 is not fraud, 1 is fraud
    return np.array(results)
