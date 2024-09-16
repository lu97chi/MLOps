import os
from azure.ai.ml import command
from azure.ai.ml.entities import Environment
from azureml_config import ml_client

# Define environment
environment = Environment(
    name='training-environment',
    conda_file='environment.yml',
    image='mcr.microsoft.com/azureml/base:latest'
)

# Fetch compute and experiment names from environment variables or hard-code them
compute_name = os.getenv('COMPUTE_NAME', 'AMLOPSTEST2')  # Default to 'AmlTest'
experiment_name = os.getenv('EXPERIMENT_NAME', 'training-experiment')  # Default to 'training-experiment'

# Define environment variables to pass to the training job
env_vars = {
    "AZURE_CLIENT_ID": os.getenv("AZURE_CLIENT_ID"),
    "AZURE_TENANT_ID": os.getenv("AZURE_TENANT_ID"),
    "AZURE_CLIENT_SECRET": os.getenv("AZURE_CLIENT_SECRET"),
    "AZURE_SUBSCRIPTION_ID": os.getenv("AZURE_SUBSCRIPTION_ID"),
    "AZURE_RESOURCE_GROUP": os.getenv("AZURE_RESOURCE_GROUP"),
    "AZURE_WORKSPACE_NAME": os.getenv("AZURE_WORKSPACE_NAME"),
    "COMPUTE_NAME": compute_name,
    "EXPERIMENT_NAME": experiment_name
}

# Create command job
job = command(
    code='.',  # current directory
    command='python train.py',
    environment=environment,
    compute=compute_name,  # Use the compute target
    experiment_name=experiment_name,  # Use the experiment name
    display_name='model-training-job',  # Optional: Set a display name for clarity
    environment_variables=env_vars
)

# Submit job to Azure ML
returned_job = ml_client.jobs.create_or_update(job)
returned_job.wait_for_completion(show_output=True) 

print(f"Submitted job: {returned_job.name} with status: {returned_job.status}")
