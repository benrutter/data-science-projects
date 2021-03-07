import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Normalizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.keras.optimizers import Adam

# importing and preprocessing dataset

dataset = pd.read_csv('life_expectancy.csv')
dataset = dataset.drop(['Country'], axis=1)

labels = dataset['Life expectancy']
features = dataset.drop(['Life expectancy'], axis=1)

del dataset

features = pd.concat(
    [features.drop(['Year', 'Status'], axis=1),
    pd.get_dummies(features['Year']),
    pd.get_dummies(features['Status'])],
    axis=1)

features_train, features_test, labels_train, labels_test = train_test_split(
    features,
    labels,
    test_size=0.2,
    random_state=7)

del features, labels

column_transformer = ColumnTransformer(
    [("only numeric", Normalizer(), features_train.columns)])
features_columns = features_train.columns
features_train = column_transformer.fit_transform(features_train)
features_test = column_transformer.transform(features_test)


# creating model

model = Sequential()
model.add(InputLayer(input_shape=(features_train.shape[1], )))
model.add(Dense(64, activation="relu"))
model.add(Dense(1))

adam = Adam(learning_rate=0.01)

model.compile(loss = 'mse', metrics=['mae'], optimizer=adam)


# training model

model.fit(features_train, labels_train, epochs=40, batch_size=10, verbose=0)
res_mse, res_mae = model.evaluate(features_test, labels_test, verbose=0)

# batch size 1 is 7.196 MAE
print(f'Mean Square Error {res_mse}')
print(f'Mean average error {res_mae}')
