from azure.identity import ClientSecretCredential
from azure.ai.ml import MLClient
import os

# Set up Azure ML credentials and MLClient
credential = ClientSecretCredential(
    tenant_id=os.getenv("AZURE_TENANT_ID"),
    client_id=os.getenv("AZURE_CLIENT_ID"),
    client_secret=os.getenv("AZURECLIENTSECRET")
)

ml_client = MLClient(
    credential=credential,
    subscription_id=os.getenv("AZURE_SUBSCRIPTION_ID"),
    resource_group_name=os.getenv("AZURE_RESOURCE_GROUP"),
    workspace_name=os.getenv("AZURE_WORKSPACE_NAME")
)

# Retrieve the model by name or ID
model_name = "RandomForestClassifierModel"  # Update with your actual model name
model_version = "1"  # Update with your model version if necessary
model = ml_client.models.get(name=model_name, version=model_version)

# Download the model locally
download_path = "models/"
model.download(download_path, exist_ok=True)
print(f"Model downloaded to {download_path}")
