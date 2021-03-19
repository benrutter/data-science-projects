from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV


# hyperparameters
BATCH_SIZE = 32
EPOCHS = 25

# image generator and iterators
train_generator = ImageDataGenerator(rescale=1.0/255, horizontal_flip=True)
test_generator = ImageDataGenerator(rescale=1.0/255)
train_iterator = train_generator.flow_from_directory(
    'Covid19-dataset/train',
    target_size=(256, 256),
    batch_size=BATCH_SIZE,
    color_mode='grayscale',
    class_mode='categorical'
)
test_iterator = test_generator.flow_from_directory(
    'Covid19-dataset/test',
    target_size=(256, 256),
    batch_size=BATCH_SIZE,
    color_mode='grayscale',
    class_mode='categorical'
)

# building model
def build_model():
    model = Sequential()
    model.add(keras.Input(shape=(256, 256, 1)))
    model.add(layers.Conv2D(15, 5, strides=2, activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Conv2D(15, 3, strides=2, activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(3, activation='softmax'))
    model.compile(
      optimizer=keras.optimizers.Adam(learning_rate=0.001),
      loss=keras.losses.CategoricalCrossentropy(),
      metrics=[keras.metrics.CategoricalAccuracy(), keras.metrics.AUC()],
    )
    return model

model = build_model()


# early stopping implementation
early_stop = EarlyStopping(monitor='val_auc', mode='min', verbose=1, patience=20)

# training / testing
model.fit(
  train_iterator,
  steps_per_epoch=train_iterator.samples/BATCH_SIZE,
  epochs=EPOCHS,
  validation_data=test_iterator,
  validation_steps=test_iterator.samples/BATCH_SIZE,
  callbacks=[early_stop],
)
