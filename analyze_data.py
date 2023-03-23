import pandas as pd

train_data = pd.read_csv("data/train.csv")
train_data['Cabin'].fillna('Unknown', inplace=True)

# Check for missing values in each column
missing_values = train_data.isnull().sum()

# Extract the cabin prefix from each cabin number
train_data['CabinPrefix'] = train_data['Cabin'].str.extract('([A-Za-z]+)', expand=False)

# Extract the cabin prefix from each cabin number
train_data['CabinPrefix'] = train_data['Cabin'].str.extract('([A-Za-z]+)', expand=False)

# Map the cabin prefix to a numerical value
prefix_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 8}
train_data['CabinPrefix'] = train_data['CabinPrefix'].apply(lambda x: prefix_map.get(x, 0))

# Drop the 'Cabin' column
train_data.drop('Cabin', axis=1, inplace=True)

# Print the first few rows of the updated dataset
# print(train_data.head())


# Count the number of occurrences of each prefix
prefix_counts = train_data['CabinPrefix'].value_counts()
print(prefix_counts)

# Print the counts for each prefix
# print(prefix_counts)
