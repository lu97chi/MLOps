from azure.identity import ClientSecretCredential
from azure.ai.ml import MLClient
import os

# Set up Azure ML credentials and client
credential = ClientSecretCredential(
    tenant_id=os.getenv("AZURE_TENANT_ID"),
    client_id=os.getenv("AZURE_CLIENT_ID"),
    client_secret=os.getenv("AZURE_CLIENT_SECRET")
)

ml_client = MLClient(
    credential=credential,
    subscription_id=os.getenv("AZURE_SUBSCRIPTION_ID"),
    resource_group_name=os.getenv("AZURE_RESOURCE_GROUP"),
    workspace_name=os.getenv("AZURE_WORKSPACE_NAME")
)

# Fetch the registered model from Azure ML
model_name = "model"  # The name used in the training script
model_version = "1"  # Adjust if needed, or use None to fetch the latest
model = ml_client.models.get(name=model_name, version=model_version)

# Download the model to a local path
download_path = "./downloaded_model"
ml_client.models.download(name=model.name, version=model.version, download_path=download_path)

print(f"Model downloaded to {download_path}")
