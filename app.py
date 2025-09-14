import mlflow
import mlflow.sklearn
from sklearn.tree import DecisionTreeClassifier 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import dagshub
dagshub.init(repo_owner='Agrim912', repo_name='dagshub-demo', mlflow=True)


max_depth = 5

# Load dataset
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

mlflow.set_experiment("Iris_Classification_Experiment_using_dt")

with mlflow.start_run() as run:
    # Initialize and train model
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Evaluate model
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy}")

    # log parameters
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    # Log model and metrics to MLflow
    
    # mlflow.sklearn.log_model(model, "random_forest_model")
    mlflow.log_metric("accuracy", accuracy)

    run_id = run.info.run_id
    print(f"Run ID: {run_id}")