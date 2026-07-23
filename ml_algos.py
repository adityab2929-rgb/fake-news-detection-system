#converting dataset into dataframe and scaling it 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#importing all necessary ml algorithms 
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

#importing metrics to check the accuracy of the model with respect to test data 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#For conversion into joblib files for later use
import joblib
import os

# Load final NLP feature dataset
df = pd.read_csv("reqfiles/AfterNLP.csv")

# Independent features and target
X = df.drop("target", axis=1)
y = df["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

#Standard scaling
scaler = StandardScaler()

#getting the scaled X_train and X_test
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Dictionary of multiple ML algorithms that can train on the dataset

models = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "decision_tree": DecisionTreeClassifier(random_state=42),
    "random_forest": RandomForestClassifier(random_state=42),
    "gradient_boosting": GradientBoostingClassifier(random_state=42),
    "naive_bayes": GaussianNB()
}

#Dictionary of parameters for each ML algorithm
params = {
    "logistic_regression": {
        "C": [0.01, 0.1, 1, 10],
        "solver": ["liblinear", "lbfgs"]
    },

    "decision_tree": {
        "criterion": ["gini", "entropy"],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10]
    },

    "random_forest": {
        "n_estimators": [100, 200],
        "criterion": ["gini", "entropy"],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5]
    },

    "gradient_boosting": {
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.1],
        "max_depth": [3, 5]
    },

    "naive_bayes": {
        "var_smoothing": [1e-9, 1e-8, 1e-7]
    }
}

#Appplying Grid SearchCV on each of the models with there parameters 
best_score = 0
best_model = None
best_model_name = None
best_params = None

for model_name in models:
    print(f"Training {model_name}...")

    grid = GridSearchCV(
        estimator=models[model_name],
        param_grid=params[model_name],
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    grid.fit(X_train_scaled, y_train)

    print(f"Best score for {model_name}: {grid.best_score_}")
    print(f"Best params for {model_name}: {grid.best_params_}")
    print("-" * 50)

    if grid.best_score_ > best_score:
        best_score = grid.best_score_
        best_model = grid.best_estimator_
        best_model_name = model_name
        best_params = grid.best_params_

#printing the best model,its best score and its best parameters 
print("Best Model Name:", best_model_name)
print("Best Score:", best_score)
print("Best Parameters:", best_params)
print("Best Model:", best_model)

#predicting the accuracy of the best_model using y_test
y_pred = best_model.predict(X_test_scaled)

print("Test Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

#Creating files for the standard scaler and best_model
os.makedirs("models", exist_ok=True)

joblib.dump(scaler, "models/standard_scaler.pkl")
joblib.dump(best_model, "models/best_model.pkl")
