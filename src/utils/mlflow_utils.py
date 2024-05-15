# utils/mlflow_utils.py
"""
Helper functions for logging the training model step to MLflow.
"""

import mlflow

def start_mlflow_run(experiment_name, run_name):
    """
    Start a new MLflow run in the specified experiment.

    Args:
        experiment_name (str): The name of the experiment.
        run_name (str): The name of the run.
    """
    mlflow.set_experiment(experiment_name)
    mlflow.start_run(run_name=run_name)

def log_mlflow_params(params):
    """
    Log the parameters to the MLflow run.

    Args:
        params (dict): The parameters to log.
    """
    mlflow.log_params(params)

def log_mlflow_metrics(metrics):
    """
    Log the metrics to the MLflow run.

    Args:
        metrics (dict): The metrics to log.
    """
    mlflow.log_metrics(metrics)

def log_mlflow_model(model, artifact_path):
    """
    Log the model to the MLflow run.

    Args:
        model: The trained model object.
        artifact_path (str): The artifact path to save the model.
    """
    mlflow.sklearn.log_model(model, artifact_path)

def end_mlflow_run():
    """
    End the current MLflow run.
    """
    mlflow.end_run()
    