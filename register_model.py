import os
import logging
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model as AzureMLModel
from azure.identity import DefaultAzureCredential

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_latest_model_version(model_name):
    """
    Fetch the latest model version registered in MLflow.
    """
    try:
        client = MlflowClient()
        latest_versions = client.get_latest_versions(model_name, stages=["None"])
        if not latest_versions:
            logger.error(f"No registered models found for {model_name}")
            return None
        latest_version = max(latest_versions, key=lambda version: int(version.version))
        logger.info(f"Latest version of model {model_name} is {latest_version.version}")
        return latest_version
    except MlflowException as e:
        logger.exception("Failed to fetch the latest model version from MLflow. Error: %s", e)
        raise

def register_model_in_azureml(model_name, model_path):
    """
    Register the model in AzureML Model Registry using azure.ai.ml.
    """
    try:
        # Setup the MLClient with DefaultAzureCredential (requires environment authentication or managed identity)
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
            description="Registered model from MLflow",
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
    # MLflow settings
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))  # Default local tracking URI
    model_name = os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "ElasticnetWineModel")

    # Fetch the latest registered model from MLflow
    latest_version = get_latest_model_version(model_name)
    if not latest_version:
        logger.error("No models found for registration.")
        return

    # Path where MLflow saves the model artifacts
    model_path = latest_version.source  # The path to the latest model artifact

    # Register the model in AzureML if enabled
    if os.getenv("AZUREML_REGISTER_MODEL") == "true":
        try:
            register_model_in_azureml(model_name, model_path)
        except Exception as e:
            logger.error(f"Failed to register model in AzureML: {e}")

if __name__ == "__main__":
    main()
