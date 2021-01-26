# import modules
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the passenger data
passengers_df = pd.read_csv('passengers.csv')

# Update sex column to numerical
passengers_df['Sex'] = passengers_df['Sex'].apply(lambda sx: 1 if sx == 'female' else 0 if sx == 'male' else np.nan)

# Fill the nan values in the age column
median_age = passengers_df['Age'].median()
passengers_df['Age'] = passengers_df['Age'].fillna(median_age)

# Create a first class column
passengers_df['FirstClass'] = passengers_df['Pclass'].apply(lambda cl: 1 if cl == 1 else 0)

# Create a third class column
passengers_df['ThirdClass'] = passengers_df['Pclass'].apply(lambda cl: 1 if cl == 3 else 0)

# Select the desired features
features = passengers_df[['Sex', 'Age', 'FirstClass', 'ThirdClass', 'SibSp']]
survival = passengers_df['Survived']

# Perform train, test, split
train_features, test_features, train_survival, test_survival = train_test_split(features, survival, test_size=0.2, random_state=7)

# Scale the feature data so it has mean = 0 and standard deviation = 1
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)
features = scaler.transform(features)

# Create and train the model
model = LogisticRegression(solver='lbfgs')
model.fit(train_features, train_survival)

# Score the model on the train data
score1 = model.score(train_features, train_survival)
print('Train score: ' + str(score1))

# Score the model on the test data
score2 = model.score(test_features, test_survival)
print('Test score: ' + str(score2))

# Total Score
total_score = model.score(features, survival)
print('Total score: ' + str(total_score))

# Analyze the coefficients
print('Coefficients: ' + str(model.coef_))