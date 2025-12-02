import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = pd.read_csv("Titan_Data_Set.csv")

data["Age"] = data["Age"].fillna(data["Age"].mean())
data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])
data["Cabin"] = data["Cabin"].fillna("Unknown")

# Feature Engineering
data["FamilySize"] = data["SibSp"] + data["Parch"] + 1
data["IsAlone"] = (data["FamilySize"] == 1).astype(int)

# Drop unused
data = data.drop(columns=["Name", "Ticket", "PassengerId", "Cabin"])

# One-Hot Encoding
data = pd.get_dummies(data, columns=["Sex", "Embarked"], drop_first=True, dtype=int)

scaler_columns = ["Age", "SibSp", "Parch", "Fare"]
scaler = StandardScaler()
data[scaler_columns] = scaler.fit_transform(data[scaler_columns])

X = data.drop("Survived", axis=1)
y = data["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Improved Random Forest model
model = RandomForestClassifier(
    n_estimators=400,
    max_depth=15,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    bootstrap=True,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
m_pred = model.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, m_pred)*100, 2), "%")
print("Classification Report:\n", classification_report(y_test, m_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, m_pred))

param = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [10, 15, 20, 25, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

Random = RandomizedSearchCV(estimator=model,param_distributions=param,cv=5,n_iter=10)

Random.fit(X_train,y_train)

best_model = Random.best_estimator_

best_pred = best_model.predict(X_test)

score = cross_val_score(best_model, X, y, cv=5)

mean_score = np.mean(score)

print("Tuned Model Accuracy:", round(accuracy_score(y_test, best_pred)*100, 2), "%")
print("Cross-Validated Accuracy:", round(mean_score*100, 2), "%")