# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 19:52:14 2022

@author: dcsem
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import datetime



# Read in data
features = pd.read_csv('report_2.csv', sep =';')
# Drop the empty columns
features.drop(['Unnamed: 0', 'База', 'Unnamed: 21'], axis=1, inplace=True)
# One-hot encode the months using pandas get_dummies
features = pd.get_dummies(features, columns = ['Месец'])

# Define abels
labels = np.array(features['Общ ИПЦ'])

# Remove the labels from the features
features= features.drop('Общ ИПЦ', axis = 1)

# Saving feature names
feature_list = list(features.columns)

# Converting feature names to numpy array
features = np.array(features)

# Using Skicit-learn to split data into training and testing sets
train_features, test_features, train_labels, test_labels =\
train_test_split(features, labels, test_size = 0.25, random_state = 42)

# Check for errors after splitting
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)



# Instantiate the model we are usingwith 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)

# Train the model on training data
rf.fit(train_features, train_labels)

# Use the forest's predict method on the test data
predictions = rf.predict(test_features)
# Calculate the absolute errors
errors = abs(predictions - test_labels)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'bps.')

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

# Get numerical feature importances
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];


# New random forest with only the few most important variables to check if the
# performance will be better
rf_most_important = RandomForestRegressor(n_estimators= 1000, random_state=42)

# Extract the most important features
important_indices = [feature_list.index('Жилища, вода, електроенергия, газ и други горива')\
                     , feature_list.index('Жилищно обзавеждане, стоки и услуги за домакинството и за обичайно поддържане на дома')\
                     , feature_list.index('Година')\
                    , feature_list.index('Алкохолни напитки и тютюневи изделия')\
                    , feature_list.index('Здравеопазване')\
                    , feature_list.index('Образование')\
                    , feature_list.index('Ресторанти и хотели')\
                    , feature_list.index('Разнообразни стоки и услуги')\
                    , feature_list.index('Нехранителни')\
                    , feature_list.index('Услуги')\
                    , feature_list.index('Хранителни')\
                    , feature_list.index('Обществено хранене')\
                    , feature_list.index('Хранителни продукти и безалкохолни напитки')]
    
train_important = train_features[:, important_indices]
test_important = test_features[:, important_indices]

# Train the new random forest
rf_most_important.fit(train_important, train_labels)

# Make predictions and determine the error
predictions = rf_most_important.predict(test_important)
errors = abs(predictions - test_labels)

# Display the new performance metrics
print('Mean Absolute Error:', round(np.mean(errors), 2), 'bps.')
mape = np.mean(100 * (errors / test_labels))
accuracy = 100 - mape
print('Accuracy:', round(accuracy, 2), '%.')


# Plotting the variables and their importance
import matplotlib.pyplot as plt
#%matplotlib inline
# Set the style
plt.style.use('fivethirtyeight')
# list of x locations for plotting
x_values = list(range(len(importances)))
# Make a bar chart
plt.bar(x_values, importances, orientation = 'vertical')
# Tick labels for x axis
plt.xticks(x_values, feature_list, rotation='vertical')
# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importance');

plt.savefig('Variable_Importance.png')


# Plotting the actual and predicted values

# Dates of training values
years = features[:, feature_list.index('Година')]

# List and then convert to datetime object
dates = [str(int(year)) for year in years]
dates = [datetime.datetime.strptime(date, '%Y') for date in dates]
# Dataframe with true values and dates
true_data = pd.DataFrame(data = {'date': dates, 'actual': labels})
# Dates of predictions

years = test_features[:, feature_list.index('Година')]

# Column of dates
test_dates = [str(int(year)) for year in years]
# Convert to datetime objects

test_dates = [datetime.datetime.strptime(date, '%Y') for date in test_dates]
# Dataframe with predictions and dates

predictions_data = pd.DataFrame(data = {'date': test_dates, 'prediction': predictions})
# Plot the actual values

plt.plot(true_data['date'], true_data['actual'], 'b-', label = 'actual')
# Plot the predicted values
plt.plot(predictions_data['date'], predictions_data['prediction'], 'ro', label = 'prediction')
plt.xticks(rotation = '60'); 
plt.legend()
# Graph labels
plt.xlabel('Date'); plt.ylabel('Общ ИПЦ bps'); plt.title('Actual and Predicted Values');









