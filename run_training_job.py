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

# Create command job
job = command(
    code='.',  # current directory
    command='python train.py',
    environment=environment,
    compute='cpu-cluster',
    experiment_name='training-experiment'
)

# Submit job
returned_job = ml_client.jobs.create_or_update(job)
