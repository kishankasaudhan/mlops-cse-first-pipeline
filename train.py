from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from mlflow import MlflowClient

# Configure MLflow
mlflow.set_tracking_uri("http://127.0.0.1:7001")
mlflow.set_experiment("iris_classification")

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.65,
    random_state=42
)

# Model parameters
n_estimators = 15
random_state = 42

model_parameters = {
    "model_type": "RandomForestClassifier",
    "n_estimators": n_estimators,
    "random_state": random_state,
    "test_size": 0.65
}

# Create the models directory
model_dir = Path("models")
model_dir.mkdir(parents=True, exist_ok=True)

model_path = model_dir / "random_forest_model.joblib"

# Start an MLflow run
with mlflow.start_run() as run:

    # Log parameters
    mlflow.log_params(model_parameters)

    # Build classifier
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state
    )

    # Train the model
    clf.fit(X_train, y_train)

    # Make predictions
    y_pred = clf.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    precision = precision_score(
        y_test, y_pred, average="macro", zero_division=0
    )
    print(f"Precision: {precision:.4f}")

    recall = recall_score(
        y_test, y_pred, average="macro", zero_division=0
    )
    print(f"Recall: {recall:.4f}")

    # Model quality gate
    # Define the minimum acceptable test accuracy
    min_test_accuracy = 0.90

    if accuracy < min_test_accuracy:
        raise ValueError(
            f"Model accuracy {accuracy:.4f} is rejected because "
            f"accuracy is below the minimum required."
        )

    print("Model passed the check.")

    # Log metrics to MLflow
    mlflow.log_metrics({
        "test_accuracy": accuracy,
        "test_precision": precision,
        "test_recall": recall
    })

    # Save the model to a file
    joblib.dump(clf, model_path)
    print(f"Model saved to {model_path}")

    # Register the model in MLflow
    mlflow.sklearn.log_model(
        sk_model=clf,
        name="iris_model",
        registered_model_name="iris_classifier",
        skops_trusted_types=[
            "sklearn.tree._tree.Tree"
        ]
    )

    # Assign the champion alias to the newly registered model
    client = MlflowClient()
    run_id = mlflow.active_run().info.run_id

    # Retrieve registered model versions
    regi_version = client.search_model_versions(
        "name='iris_classifier'"
    )

    # Find the version created by the current training run
    current_model_version = next(
        (
            version
            for version in regi_version
            if version.run_id == run_id
        ),
        None
    )

    if current_model_version is None:
        raise ValueError("No model version found for the current run.")

    client.set_registered_model_alias(
        name="iris_classifier",
        alias="champion",
        version=current_model_version.version
    )

    print("Model logged and registered successfully.")