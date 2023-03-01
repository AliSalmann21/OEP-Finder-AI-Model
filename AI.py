import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import shuffle

# Load the dataset
dataset = pd.read_csv('pe_files_dataset.csv')

# Split the dataset into features and labels
X = dataset.drop(['label'], axis=1).values
y = dataset['label'].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a function to create the neural network model
def create_model(num_hidden_layers=2, num_neurons=256, learning_rate=0.001, dropout_rate=0.2):
    model = keras.Sequential()
    model.add(layers.Dense(num_neurons, activation='relu', input_shape=(X_train.shape[1],)))
    for i in range(num_hidden_layers):
        model.add(layers.Dense(num_neurons, activation='relu'))
        model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(1, activation='sigmoid'))
    optimizer = keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Use grid search to find the optimal hyperparameters
model = keras.wrappers.scikit_learn.KerasClassifier(build_fn= create_model, verbose=0)
param_grid = {'num_hidden_layers': [1, 2, 3], 'num_neurons': [128, 256, 512], 'learning_rate': [0.001, 0.01, 0.1], 'dropout_rate': [0.1, 0.2, 0.3]}
grid_search = GridSearchCV(model, param_grid=param_grid, cv=3)
grid_search.fit(X_train, y_train)
print('Best parameters:', grid_search.best_params_)

num_hidden_layers = grid_search.best_params_['num_hidden_layers']
num_neurons = grid_search.best_params_['num_neurons']
learning_rate = grid_search.best_params_['learning_rate']
dropout_rate = grid_search.best_params_['dropout_rate']
model = create_model(num_hidden_layers=num_hidden_layers, num_neurons=num_neurons, learning_rate=learning_rate, dropout_rate=dropout_rate)
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)
print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))
print('Classification report:\n', classification_report(y_test, y_pred))

num_models = 5
models = []
for i in range(num_models):
    model = create_model(num_hidden_layers=num_hidden_layers, num_neurons=num_neurons, learning_rate=learning_rate, dropout_rate=dropout_rate)
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
    models.append(model)

y_pred = np.zeros((X_test.shape[0],))
for model in models:
    y_pred += model.predict(X_test).flatten()
    y_pred = (y_pred / num_models) > 0.5
    print('Confusion matrix (ensemble):\n', confusion_matrix(y_test, y_pred))
    print('Classification report (ensemble):\n', classification_report(y_test, y_pred))

def extract_ngrams(bytecode, n=3):
 ngrams = []
 for i in range(len(bytecode) - n + 1):
    ngram = bytecode[i:i+n]
    ngrams.append(ngram)
 return ngrams

def extract_image_features(bytecode):
    # Convert bytecode to grayscale image
    image = np.zeros((32, 32))
    for i in range(1024):
        row = i // 32
        col = i % 32
        pixel_value = bytecode[i]
        image[row][col] = pixel_value
        # Resize image to 64x64
        resized_image = tf.image.resize(image, size=(64, 64)).numpy()
        # Flatten image
        flattened_image = resized_image.flatten()
        return flattened_image

def extract_features(X):
    ngram_features = np.zeros((X.shape[0], 1000))
    image_features = np.zeros((X.shape[0], 4096))
    for i in range(X.shape[0]):
        
        bytecode = X[i]
        ngrams = extract_ngrams(bytecode)
    for ngram in ngrams:
        
        ngram_features[i][hash(ngram) % 1000] += 1
        image_features[i] = extract_image_features(bytecode)
        return np.hstack((X, ngram_features, image_features))

X_train = extract_features(X_train)
X_test = extract_features(X_test)
data_augmentation = keras.Sequential([    layers.experimental.preprocessing.RandomFlip("horizontal"),    layers.experimental.preprocessing.RandomRotation(0.1),])

augmented_X_train = np.zeros((X_train.shape[0] * 2, X_train.shape[1]))
augmented_y_train = np.zeros((y_train.shape[0] * 2,))
for i in range(X_train.shape[0]):
    augmented_X_train[2*i] = X_train[i]
    augmented_y_train[2*i] = y_train[i]
    augmented_X_train[2*i+1] = data_augmentation(np.expand_dims(X_train[i], axis=0)).numpy()[0]
    augmented_y_train[2*i+1] = y_train[i]

# Train the final model on the augmented training set
num_models = 5
models = []
for i in range(num_models):
    model = create_model(num_hidden_layers=num_hidden_layers, num_neurons=num_neurons, learning_rate=learning_rate, dropout_rate=dropout_rate)
    model.fit(augmented_X_train, augmented_y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
    models.append(model)

y_pred = np.zeros((X_test.shape[0],))
for model in models:
    y_pred += model.predict(X_test).flatten()
y_pred = (y_pred / num_models) > 0.5
print('Confusion matrix (ensemble, with data augmentation):\n', confusion_matrix(y_test, y_pred))
print('Classification report (ensemble, with data augmentation):\n', classification_report(y_test, y_pred))








