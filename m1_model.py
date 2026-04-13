import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Create dataset

data = pd.DataFrame({
"hours_studied":[1,2,3,4,5,6,7,8],
"attendance":[50,60,65,70,75,80,90,95],
"passed":[0,0,0,1,1,1,1,1]
})

# Split data

X = data[["hours_studied","attendance"]]
y = data["passed"]

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)

# Start MLflow run

with mlflow.start_run():
	model = RandomForestClassifier(n_estimators=100)
	model.fit(X_train, y_train)


predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# Log details
mlflow.log_param("n_estimators", 100)
mlflow.log_metric("accuracy", accuracy)
mlflow.sklearn.log_model(model, "student_pass_model")

print("Accuracy:", accuracy)

