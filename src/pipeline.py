# pipeline.py
"""
This module provides the main pipeline for the Mushroom Classification project.
"""

import argparse
import logging
import pandas as pd
from src.data.data_loading import load_data
from src.data.data_preprocessing import delete_outliers
from src.data.data_preprocessing import drop_unwanted_features
from src.data.data_split import split_data
from src.data.data_transformation import split_features
from src.data.data_transformation import create_preprocessor
from src.models.model_training import create_training_pipeline
from src.models.model_training import train_model
from src.models.model_training import save_model
from src.models.model_evaluation import evaluate_model
from src.utils.config import MODEL_PATH
from src.utils.config import create_preprocessor
from src.utils.config import TRAIN_DATA
from src.utils.config import VAL_DATA
from src.utils.config import TEST_DATA
from src.utils.logging_config import setup_logging
from src.utils.mlflow_utils import (
    start_mlflow_run,
    log_mlflow_params,
    log_mlflow_metrics,
    log_mlflow_model,
    end_mlflow_run,
)

def main(data_path):
    """
    The main pipeline that orchestrates the entire process.

    Args:
        data_path (str): The path to the dataset file.

    Usage:
        python pipeline.py --data_path path/to/your/dataset.csv

    Example:
        python pipeline.py --data_path data/training_data.csv
    """
    try:
        # Set up logging
        setup_logging()
        logger = logging.getLogger(__name__)

        # Load the dataset
        logger.info("Loading data...")
        try:
            df = load_data(data_path)
        except FileNotFoundError as e:
            logger.error(str(e))
            logger.error("Data loading failed. Please check the dataset file path.")
            return
        except ValueError as e:
            logger.error(str(e))
            logger.error("Data loading failed. Please check the dataset file format.")
            return

        # Preprocess the data
        logger.info("Preprocessing data...")
        try:
            # Delete outliers
            df = delete_outliers(df)

            # Drop unwanted features
            features_to_drop = ['cap-shape', 'does-bruise-or-bleed', 'gill-spacing', 'gill-color',
                                'stem-height', 'stem-color', 'ring-type', 'habitat', 'season']
            df = drop_unwanted_features(df, features_to_drop)
        except ValueError as e:
            logger.error(str(e))
            logger.error("Data preprocessing failed. Please check the required columns and data types.")
            return

        # Split the data
        logger.info("Splitting data...")
        try:
            # Split the data into features and target
            X = df.drop('class', axis=1)
            y = df['class']

            # Split the data into train, validation, and test sets
            X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

            # Concatenate features and targets
            train_df = pd.concat([X_train, y_train], axis=1)
            val_df = pd.concat([X_val, y_val], axis=1)
            test_df = pd.concat([X_test, y_test], axis=1)

            # Save the split datasets
            train_df.to_csv(TRAIN_DATA, index=False)
            val_df.to_csv(VAL_DATA, index=False)
            test_df.to_csv(TEST_DATA, index=False)
        except ValueError as e:
            logger.error(str(e))
            logger.error("Data split failed.")
            return

        # Start an MLflow run
        mlflow_run = start_mlflow_run(experiment_name="mushroom_classification", run_name="pipeline_run")

        # Log MLflow parameters
        log_mlflow_params({
            "train_data_path": TRAIN_DATA,
            "val_data_path": VAL_DATA,
            "test_data_path": TEST_DATA,
        })

        # Split features into numeric and categorical
        numeric_features, categorical_features = split_features(X_train)

        # Create the preprocessor
        preprocessor = create_preprocessor(numeric_features, categorical_features)

        # Train the model
        logger.info("Training model...")
        try:
             # Create the training pipeline
            pipeline = create_training_pipeline(preprocessor)

            # Train the model
            trained_pipeline = train_model(pipeline, X_train, y_train)
        except ValueError as e:
            logger.error(str(e))
            logger.error("Model training failed. Please check the model hyperparameters.")
            end_mlflow_run(mlflow_run)
            return
        except RuntimeError as e:
            logger.error(str(e))
            logger.error("Model training failed. Please check the convergence of the model.")
            end_mlflow_run(mlflow_run)
            return

        # Log model parameters
        log_mlflow_params({
            "model_type": "GradientBoostingClassifier",
            "model_params": trained_pipeline.named_steps["classifier"].get_params(),
        })

        # Save the trained model
        logger.info("Saving trained model...")
        try:
            save_model(trained_pipeline, preprocessor, MODEL_PATH)
        except FileNotFoundError as e:
            logger.error(str(e))
            logger.error("Saving trained model failed. Please check the model save path.")
            end_mlflow_run(mlflow_run)
            return

        # Evaluate the model
        logger.info("Evaluating model...")
        try:
            eval_metrics = evaluate_model(trained_pipeline, X_val, y_val)
            logger.info("Evaluation Metrics:")
            for metric, value in eval_metrics.items():
                logger.info(f"{metric}: {value:.4f}")
                end_mlflow_run(mlflow_run)
        except ValueError as e:
            logger.error(str(e))
            logger.error("Model evaluation failed. Please check the input data.")

        # Log evaluation metrics
        log_mlflow_metrics(eval_metrics)

        # Log the trained model
        log_mlflow_model(trained_pipeline, "model")

        # End MLflow run
        end_mlflow_run()

    except Exception as e:
        logger.error("An unexpected error occurred:")
        logger.error(str(e))
        end_mlflow_run()
        return

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Mushroom Classification Pipeline')
    parser.add_argument('--data_path', type=str, default='data', help='Path to the dataset file')
    args = parser.parse_args()

    # Run the pipeline
    main(args.data_path)
