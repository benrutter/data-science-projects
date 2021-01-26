def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

os.chdir(r'C:\Users\benrr\Documents\Projects\Python\Income Predictor')
income_data = pd.read_csv('income.csv', delimiter = ', ', engine='python')

income_data["sex-int"] = income_data["sex"].apply(lambda row: 0 if row == "Male" else 1)
income_data['country-int'] = income_data['native-country'].apply(lambda row: 0 if row == 'United-States' else 1)
income_data['race-int'] = income_data['race'].apply(lambda row: 0 if row == 'White' else 1)

labels = income_data[['income']]
data = income_data[['age', 'capital-gain', 'capital-loss', 'hours-per-week', 'sex-int', 'country-int', 'race-int']]

train_data, test_data, train_labels, test_labels = train_test_split(data, labels, random_state = 1)

forest = RandomForestClassifier(n_estimators = 100)
forest.fit(train_data, train_labels)
score = forest.score(test_data, test_labels)
print(score)