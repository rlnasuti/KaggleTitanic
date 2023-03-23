import pandas as pd
import numpy as np
import pickle
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

# Load the model from the 'model/best_model.pkl' file
with open("model/best_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the test data from 'data/test.csv'
test_data = pd.read_csv("data/test.csv")

# Save the passenger IDs for the submission file
passenger_ids = test_data["PassengerId"]

# Preprocess the test data
X_test = preprocess_data(test_data)

# Get predictions for the test data
predictions = model.predict(X_test)

# Save the predictions to a file called 'submission.csv'
submission = pd.DataFrame({"PassengerId": passenger_ids, "Survived": predictions})
submission.to_csv("submission.csv", index=False)
