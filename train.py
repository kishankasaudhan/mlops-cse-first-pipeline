from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from pathlib import Path

iris=load_iris()
X=iris.data
y=iris.target

X_train, X_test, y_train, y_test= train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model=RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred=model.predict(X_test)
accuracy=accuracy_score(y_test, y_pred)
print(f"Test accuracy: {accuracy:.4f}")

model_dir= Path("models")
model_dir.mkdir(exist_ok=True)
model_path=model_dir / "model.joblib"
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")