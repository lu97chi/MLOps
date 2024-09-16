import os
from azure.ai.ml import command
from azure.ai.ml.entities import Environment
from azureml_config import ml_client
import logging

# Set up basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Log all environment variables
for key, value in os.environ.items():
    logging.info(f'{key}: {value}')

# Define environment
environment = Environment(
    name='training-environment',
    conda_file='environment.yml',
    image='mcr.microsoft.com/azureml/base:latest'
)

# Fetch compute and experiment names from environment variables or hard-code them
compute_name = os.getenv('COMPUTE_NAME', 'AMLCompute2')  # Default to 'AmlTest'
experiment_name = os.getenv('EXPERIMENT_NAME', 'training-experiment')  # Default to 'training-experiment'

# Create command job
job = command(
    code='.',  # current directory
    command='python train.py',
    environment=environment,
    compute=compute_name,  # Use the compute target
    experiment_name=experiment_name,  # Use the experiment name
    display_name='model-training-job'  # Optional: Set a display name for clarity
)

# Submit job to Azure ML
returned_job = ml_client.jobs.create_or_update(job)
print(f"Submitted job: {returned_job.name}")
