import os
import sys
import logging
import warnings
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model as AzureMLModel
from azure.identity import DefaultAzureCredential
from urllib.parse import urlparse  # Import urlparse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def eval_metrics(actual, pred):
    """
    Evaluate model performance metrics.
    """
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def register_model_in_azureml(model_name, model_path):
    """
    Register the model in AzureML using azure.ai.ml SDK.
    """
    try:
        # Setup the MLClient with DefaultAzureCredential
        ml_client = MLClient(
            credential=DefaultAzureCredential(),
            subscription_id=os.getenv("AZUREML_SUBSCRIPTION_ID"),
            resource_group_name=os.getenv("AZUREML_RESOURCE_GROUP"),
            workspace_name=os.getenv("AZUREML_WORKSPACE_NAME"),
        )

        # Create the Azure ML model object
        azure_model = AzureMLModel(
            path=model_path,  # Path to the model directory or file
            name=model_name,
            description="ElasticNet model for predicting wine quality",
            tags={"framework": "sklearn", "mlflow": "true"},
            type="custom_model"  # Specify model type, adjust as needed
        )

        # Register the model in AzureML
        registered_model = ml_client.models.create_or_update(azure_model)
        logger.info(f"Model registered in AzureML: {registered_model.name}, Version: {registered_model.version}")

    except Exception as e:
        logger.exception("Model registration in AzureML failed. Error: %s", e)
        raise

def main():
    # Suppress warnings
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    # Load processed data
    data_path = "processed_data.csv"  # Data prepared by prepare_data.py
    try:
        data = pd.read_csv(data_path)
        logger.info("Processed data loaded successfully.")
    except Exception as e:
        logger.exception("Failed to load processed data. Error: %s", e)
        sys.exit(1)

    # Split the data into training and test sets (0.75, 0.25 split)
    train, test = train_test_split(data)
    train_x = train.drop(["quality"], axis=1)  # 'quality' is the target column
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    # Model hyperparameters
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    # Set up MLflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))  # Default to local tracking URI
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "Default"))

    with mlflow.start_run():
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        try:
            lr.fit(train_x, train_y)
            logger.info("Model training completed.")
        except Exception as e:
            logger.exception("Model training failed. Error: %s", e)
            sys.exit(1)

        predicted_qualities = lr.predict(test_x)
        rmse, mae, r2 = eval_metrics(test_y, predicted_qualities)

        # Log parameters and metrics to MLflow
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        # Infer the model signature for logging
        predictions = lr.predict(train_x)
        signature = infer_signature(train_x, predictions)

        # Log the model in MLflow
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                lr, 
                "model", 
                registered_model_name="ElasticnetWineModel", 
                signature=signature
            )
            logger.info("Model registered successfully in MLflow.")
        else:
            mlflow.sklearn.log_model(lr, "model", signature=signature)
            logger.info("Model logged successfully.")

        # Register the model with AzureML if required
        if os.getenv("AZUREML_REGISTER_MODEL") == "true":
            model_path = mlflow.get_artifact_uri("model")
            try:
                register_model_in_azureml("ElasticnetWineModel", model_path)
            except Exception as e:
                logger.exception("Failed to register model in AzureML: %s", e)

if __name__ == "__main__":
    main()
