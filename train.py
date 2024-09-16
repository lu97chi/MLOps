import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

# Set up MLflow experiment (ensure this matches your Azure ML experiment name)
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

    # Log metrics and model to MLflow
    mlflow.log_metric('accuracy', accuracy)
    mlflow.sklearn.log_model(model, 'model')
