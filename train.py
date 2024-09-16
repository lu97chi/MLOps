import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from azure.identity import ClientSecretCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model as AzureModel
import os
import azureml.mlflow  # Importing Azure ML's MLflow integration

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
workspace = ml_client.workspaces.get(name=os.getenv("AZURE_WORKSPACE_NAME"))
mlflow.set_tracking_uri(workspace.mlflow_tracking_uri)

# Set up MLflow experiment
experiment_name = 'training-experiment'
mlflow.set_experiment(experiment_name)

# Start an MLflow run
with mlflow.start_run() as run:
    # Load data
    df = pd.read_csv('data.csv')

    # Data preprocessing
    X = df.drop('QUALITY', axis=1)  # Features
    y = df['QUALITY']  # Target variable

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model accuracy: {accuracy}')

    # Log metrics to MLflow
    mlflow.log_metric('accuracy', accuracy)

    # Prepare an input example and infer the model signature
    input_example = X_test.iloc[:1]  # Taking the first row as an example input
    signature = infer_signature(X_train, model.predict(X_train))

    # Log model using MLflow
    artifact_path = 'model'
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path=artifact_path,
        input_example=input_example,
        signature=signature
    )

    # Retrieve the model path from MLflow
    model_uri = f"runs:/{run.info.run_id}/{artifact_path}"

    # Register the model directly with Azure ML
    registered_model = AzureModel(
        path=model_uri,  # Using MLflow model URI
        name='RandomForestClassifierModel',  # Name of the model in Azure ML
        description="Random Forest model for predicting quality",
        tags={"framework": "sklearn", "accuracy": str(accuracy)}
    )
    
    ml_client.models.create_or_update(registered_model)

print("Model logged and registered successfully in Azure ML.")
