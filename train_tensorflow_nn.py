import re
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    # Fill missing values
    data["Age"].fillna(data["Age"].mean(), inplace=True)
    data['Cabin'].fillna("Unknown", inplace=True)
    data["Embarked"].fillna("S", inplace=True)
    
    # Convert categorical variables to numerical
    data["Sex"] = data["Sex"].map({"male": 0, "female": 1})
    data["Embarked"] = data["Embarked"].map({"S": 0, "C": 1, "Q": 2})

    # Extract Cabin Prefix from each cabin into new feature
    data['CabinPrefix'] = data['Cabin'].str.extract('([A-Za-z]+)', expand=False)

    #Create family size feature
    data["FamilySize"] = data['SibSp'] + data['Parch'] + 1

    #Create is alone feature
    data["IsAlone"] = data["FamilySize"].apply(lambda x: 1 if x == 1 else 0)

    # Map the cabin prefix to a numerical value
    prefix_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 8}
    data['CabinPrefix'] = data['CabinPrefix'].apply(lambda x: prefix_map.get(x, 0))
    
    # Feature scaling
    global features_to_scale
    features_to_scale = [
        "Age", 
        "Fare", 
        "Sex", 
        "Embarked", 
        "Pclass", 
        "SibSp", 
        "Parch", 
        "CabinPrefix",
        "FamilySize",
        "IsAlone"
    ]
        
    scaler = StandardScaler()
    data[features_to_scale] = scaler.fit_transform(data[features_to_scale])
    
    return data

# Load the data
train_data: pd.DataFrame = pd.read_csv("data/train.csv")

train_data = preprocess_data(train_data)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_data[features_to_scale], train_data["Survived"], test_size=0.2, random_state=42)

# Define the model
model: tf.keras.Sequential = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation="relu", input_shape=(len(features_to_scale),)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

# Compile the model
model.compile(loss="binary_crossentropy", optimizer="Adam", metrics=["accuracy"])

# Define the callback function to save the model with the best validation accuracy
checkpoint_path = "model/potential_best_sequentialnn_model.h5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_accuracy', verbose=1,
                             save_best_only=True, mode='max')

# Train the model
history: tf.keras.callbacks.History = model.fit(
    X_train, 
    y_train, 
    epochs=100, 
    batch_size=32, 
    validation_data=(X_val, y_val), 
    callbacks=[checkpoint]
    )

# Load the best saved model
best_model = tf.keras.models.load_model(checkpoint_path)

# Evaluate the model on the test data
test_loss, test_acc = best_model.evaluate(X_val, y_val)
print('Test accuracy:', test_acc)

# Load the current best test accuracy value from a file
accuracy_file = "model/test_accuracy.txt"
try:
    with open(accuracy_file, "r") as f:
        best_test_accuracy = float(f.read())
except FileNotFoundError:
    best_test_accuracy = 0

# Compare the current model's accuracy to the stored best accuracy
if test_acc > best_test_accuracy:
    # Save the current model
    model.save('model/best_sequentialnn_model.h5')

    # Update the stored best test accuracy value
    with open(accuracy_file, "w") as f:
        f.write(str(test_acc))

    print("The current model has a higher accuracy. It has been saved.")

    # Prepare a submission

    # Load the test data from 'data/test.csv'
    test_data = pd.read_csv("data/test.csv")

    # Save the passenger IDs for the submission file
    passenger_ids = test_data["PassengerId"]

    # Preprocess the test data
    X_test = preprocess_data(test_data)

    # Get predictions for the test data
    predictions = best_model.predict(X_test[features_to_scale].astype(np.float32)).round().astype(int).flatten()

    # Save the predictions to a file called 'submission.csv'
    submission = pd.DataFrame({"PassengerId": passenger_ids, "Survived": predictions})
    submission.to_csv("submission.csv", index=False)

