import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    # Fill missing values for age with the median age
    data["Age"].fillna(
        np.random.randint(
            data["Age"].mean() - data["Age"].std(), 
            data["Age"].mean() + data["Age"].std()
        ),
        inplace=True
    )

    # Fill missing values for embarked with the most frequent value
    data["Embarked"].fillna(data["Embarked"].mode()[0], inplace=True)

    # Fill missing values for fare with the median fare
    data["Fare"].fillna(data["Fare"].mean(), inplace=True)

     # Create a new feature "FamilySize" by combining "SibSp" and "Parch"
    data["FamilySize"] = data["SibSp"] + data["Parch"] + 1

    data["IsAlone"] = data["FamilySize"].apply(lambda x: 1 if x == 1 else 0)

    # Convert categorical features into numerical format using one-hot encoding
    categorical_features = ["Pclass", "Sex", "Embarked"]
    one_hot_encoded_data = pd.get_dummies(data[categorical_features])

    # Combine one-hot encoded categorical features with the remaining numerical features
    numerical_features = ["Age", "SibSp", "Parch", "Fare"]
    preprocessed_data = pd.concat([data[numerical_features], one_hot_encoded_data], axis=1)

    # Normalize the numerical features using a StandardScaler
    scaler = StandardScaler()
    preprocessed_data[numerical_features] = scaler.fit_transform(preprocessed_data[numerical_features])


    return preprocessed_data

train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")

# Preprocess the training data
X = preprocess_data(train_data)
y = train_data["Survived"]

# Split the data into a training set and a validation set
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)

# Train the model
#model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=1)
model = LogisticRegression(random_state=1, max_iter = 500)
model.fit(X_train, y_train)

# Make predictions on the validation set
val_predictions = model.predict(X_val)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_val, val_predictions)
print("Accuracy of the model on the validation set:", accuracy)

# Load the current best test accuracy value from a file
accuracy_file = "model/test_accuracy.txt"
try:
    with open(accuracy_file, "r") as f:
        best_test_accuracy = float(f.read())
except FileNotFoundError:
    best_test_accuracy = 0

# Compare the current model's accuracy to the stored best accuracy
if accuracy > best_test_accuracy:
    # Save the current model
    with open("model/best_regression_model.pkl", "wb") as f:
        pickle.dump(model, f)

    # Update the stored best test accuracy value
    with open(accuracy_file, "w") as f:
        f.write(str(accuracy))

    print("The current model has a higher accuracy. It has been saved.")