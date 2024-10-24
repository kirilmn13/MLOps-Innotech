import os

import hydra
import mlflow
from evaluate_model import evaluate
from process import process_data
from train_model import train


@hydra.main(version_base=None, config_path='../../config', config_name='main')
def main(config):
    os.environ['MLFLOW_TRACKING_USERNAME'] = config.mlflow_user
    os.environ['MLFLOW_TRACKING_PASSWORD'] = config.mlflow_api
    os.environ['MLFLOW_TRACKING_URI'] = config.mlflow_tracking_ui
    mlflow.set_tracking_uri(config.mlflow_tracking_ui)
    process_data(config)
    train(config)
    evaluate(config)



if __name__ == '__main__':
    main()
