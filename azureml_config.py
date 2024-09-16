import os
from azure.identity import ClientSecretCredential
from azure.ai.ml import MLClient

credential = ClientSecretCredential(
    tenant_id=os.getenv('AZURE_TENANT_ID'),
    client_id=os.getenv('AZURE_CLIENT_ID'),
    client_secret=os.getenv('AZURECLIENTSECRET')
)

ml_client = MLClient(
    credential=credential,
    subscription_id=os.getenv('AZURE_SUBSCRIPTION_ID'),
    resource_group_name=os.getenv('AZURE_RESOURCE_GROUP'),
    workspace_name=os.getenv('AZURE_WORKSPACE_NAME')
)
