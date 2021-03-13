import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow	import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, make_scorer, mean_squared_error
from sklearn.model_selection import GridSearchCV


admissions = pd.read_csv('admissions_data.csv')

admissions_x = admissions[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA', 'Research']]
admissions_y = admissions['Chance of Admit']

x_train, x_test, y_train, y_test = train_test_split(admissions_x, admissions_y, test_size=0.2, random_state=7)

x_columns = x_train.columns
column_transformer = ColumnTransformer([("only numeric", StandardScaler(), x_columns)])
x_train = column_transformer.fit_transform(x_train)
x_test = column_transformer.transform(x_test)

# creating model
def design_model():
    model = Sequential()
    model.add(InputLayer(input_shape=(x_train.shape[1], )))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(1))
    adam = Adam(learning_rate=0.01)
    model.compile(loss = 'mse', metrics=['mae'], optimizer=adam)
    return model

# hyperparameter tuning

def do_grid_search():
    batch_size = [1, 10, 32, 64]
    epochs = [10, 50, 100]
    model = KerasRegressor(build_fn=design_model)
    param_grid = dict(batch_size=batch_size, epochs=epochs)
    grid = GridSearchCV(
        estimator = model,
        param_grid=param_grid,
        scoring = make_scorer(mean_squared_error, greater_is_better=False),
        return_train_score = True
    )
    grid_result = grid.fit(x_train, y_train, verbose = 0)
    print(grid_result)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))


model = design_model()
stop = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=30)
model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=0, callbacks=[stop])
model.evaluate(x_train, y_test, verbose=1)
