import os
import logging
from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment, Model
from azure.ai.ml.entities import Environment, CodeConfiguration
from azure.identity import DefaultAzureCredential

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_or_update_endpoint(ml_client, endpoint_name):
    """
    Create or update an AzureML managed online endpoint.
    """
    try:
        # Define the endpoint
        endpoint = ManagedOnlineEndpoint(
            name=endpoint_name,
            auth_mode="key",  # Use key-based authentication; alternatively, use "aml_token"
            description="Endpoint for ElasticNet Wine Quality Model"
        )

        # Check if the endpoint already exists
        try:
            existing_endpoint = ml_client.online_endpoints.get(endpoint_name)
            if existing_endpoint:
                logger.info(f"Updating existing endpoint: {endpoint_name}")
                return ml_client.online_endpoints.begin_update(endpoint).result()
        except Exception:
            logger.info(f"Creating new endpoint: {endpoint_name}")
            return ml_client.online_endpoints.begin_create_or_update(endpoint).result()

    except Exception as e:
        logger.exception(f"Failed to create or update endpoint. Error: {e}")
        raise

def deploy_model_to_endpoint(ml_client, endpoint_name, model_name, model_version):
    """
    Deploy the model to the specified AzureML managed online endpoint.
    """
    try:
        # Define environment for the deployment
        environment = Environment(
            image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",  # Example base image, adjust as needed
            conda_file=os.getenv("CONDA_ENV_FILE", "environment.yml"),  # Conda environment file
            description="Environment for ElasticNet Wine Quality Model"
        )

        # Define deployment configuration
        deployment = ManagedOnlineDeployment(
            name="default",  # Default deployment name for the endpoint
            endpoint_name=endpoint_name,
            model=Model(name=model_name, version=model_version),
            environment=environment,
            code_configuration=CodeConfiguration(code="./", scoring_script="score.py"),  # Scoring script path
            instance_type="Standard_DS2_v2",  # Specify Azure compute instance type
            instance_count=1  # Number of instances
        )

        # Create or update the deployment
        logger.info(f"Deploying model {model_name} version {model_version} to endpoint {endpoint_name}")
        deployment_result = ml_client.online_deployments.begin_create_or_update(deployment).result()

        # Set traffic to the deployment
        ml_client.online_endpoints.begin_update(
            name=endpoint_name,
            traffic={"default": 100}
        ).result()

        logger.info(f"Model deployed successfully: {deployment_result.name}")
    except Exception as e:
        logger.exception(f"Failed to deploy model to endpoint. Error: {e}")
        raise

def main():
    # Setup MLClient with DefaultAzureCredential
    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=os.getenv("AZUREML_SUBSCRIPTION_ID"),
        resource_group_name=os.getenv("AZUREML_RESOURCE_GROUP"),
        workspace_name=os.getenv("AZUREML_WORKSPACE_NAME"),
    )

    # Define endpoint and model details
    endpoint_name = os.getenv("AZUREML_ENDPOINT_NAME", "elasticnet-wine-quality-endpoint")
    model_name = os.getenv("AZUREML_MODEL_NAME", "ElasticnetWineModel")
    model_version = os.getenv("AZUREML_MODEL_VERSION", "1")

    # Create or update endpoint
    create_or_update_endpoint(ml_client, endpoint_name)

    # Deploy model to endpoint
    deploy_model_to_endpoint(ml_client, endpoint_name, model_name, model_version)

if __name__ == "__main__":
    main()
