import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam


# Dataset: given 2 number compute sum
N_train = 100
N_test = 10
X_train = np.random.randint(0, 100, (N_train, 2))
Y_train = np.sum(X_train, axis=-1)
X_test = np.random.randint(0, 100, (N_test, 2))
Y_test = np.sum(X_test, axis=-1)

# Model
x = Input(shape=(2))
y = Dense(units=100)(x)
y = Dense(units=1)(y)
model = Model(inputs=x, outputs=y)

# Compiling
model.compile(loss='mse', optimizer='adam')

# Fitting
model.fit(X_train, Y_train, batch_size=8, epochs=100, validation_split=0.2)

# Evaluation
print('Evaluation')
test_scores = model.evaluate(X_test, Y_test, verbose=2)
print(test_scores)