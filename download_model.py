from azure.identity import ClientSecretCredential
from azure.ai.ml import MLClient
import os

for key, value in os.environ.items():
    print(f'{key}: {value}')
    
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

# Retrieve the latest version of the model by name
model_name = "RandomForestClassifierModel"  # Replace with your model name

# Fetch all versions of the model
models = ml_client.models.list(name=model_name)

# Sort models by version and get the latest one
latest_model = max(models, key=lambda m: int(m.version))  # Assuming version is numeric

if latest_model is None:
    raise Exception(f"No models found with name: {model_name}")

# Define the download path
download_path = "models/"

# Download the latest model using ml_client
ml_client.models.download(name=latest_model.name, version=latest_model.version, download_path=download_path)

# Construct the model path
model_path = os.path.join(download_path, "RandomForestClassifierModel.pkl")  # Adjust if the downloaded file has a specific name

# Output the model path for GitHub Actions
print(f"::set-output name=model_path::{model_path}")

print(f"Latest model downloaded to {model_path} (version: {latest_model.version})")
