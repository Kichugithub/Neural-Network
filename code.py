import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

# Load the training data
train_data = pd.read_csv(r'C:\Users\DeLL\Downloads\train.csv')

# Load the test data
test_data = pd.read_csv(r'C:\Users\DeLL\Downloads\test.csv')

# Combine training and test data
combined_data = pd.concat([train_data, test_data], axis=0)

# Handle missing values
combined_data.fillna(combined_data.mean(), inplace=True)

# Convert non-numeric columns to numeric 
combined_data = pd.get_dummies(combined_data)

# Split the combined data back into training and test data
train_data = combined_data[:len(train_data)]
test_data = combined_data[len(train_data):]

test_data.rename(columns={'Unnamed: 0': 'row_id'}, inplace=True)

# Align columns to ensure the same set of features in both training and test data
train_data, test_data = train_data.align(test_data, join='outer', axis=1, fill_value=0)

# the 'num_sold' column is dropped from the test data
test_data.drop('num_sold', axis=1, inplace=True)

# Split the data into features (X) and target variable (y)
X_train = train_data.drop(['num_sold'], axis=1)
y_train = train_data['num_sold']

# Model is trained
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
p = model.predict(test_data)

# Predictions rounded to the nearest integer
p = p.round()

# DataFrame with the predictions
output = pd.DataFrame({'row_id': test_data['row_id'].values, 'num_sold': p})

# Output the predictions to a CSV file
output.to_csv('pred.csv', index=False)
