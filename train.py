import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
from azure.identity import ClientSecretCredential
from azure.ai.ml import MLClient
import os
import logging

# Set up basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Log all environment variables
for key, value in os.environ.items():
    logging.info(f'{key}: {value}')

# Set up Azure ML credentials and MLClient
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

# Set the tracking URI to Azure ML workspace
mlflow.set_tracking_uri(ml_client.workspaces.get().mlflow_tracking_uri)

# Set up MLflow experiment
mlflow.set_experiment('training-experiment')

# Start an MLflow run
with mlflow.start_run() as run:
    # Load data
    df = pd.read_csv('data.csv')

    # Data preprocessing
    X = df.drop('target', axis=1)
    y = df['target']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model accuracy: {accuracy}')

    # Log metrics to MLflow
    mlflow.log_metric('accuracy', accuracy)

    # Log model using Azure ML integration with MLflow
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path='model',
        registered_model_name='RandomForestClassifierModel'  # Change the model name as needed
    )
