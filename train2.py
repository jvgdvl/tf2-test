import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mean_squared_error


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

# Training conf
optimizer = Adam()
loss_fn = mean_squared_error

# Dataset iteration
for epoch in range(10):
    for i in tqdm(range(len(X_train))):
        x = X_train[i]
        y = Y_train[i]
        x = tf.expand_dims(x, axis=0)
        y = tf.expand_dims(y, axis=0)

        with tf.GradientTape() as tape:
            # Forward pass
            y_pred = prediction = model(x)
            loss = loss_fn(y, y_pred)

        # Getting gradients from forward pass
        gradients = tape.gradient(loss, model.trainable_variables)

        # Backward pass
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Test
    Y_test_pred = model(X_test, Y_test)
    loss_test = loss_fn(Y_test, Y_test_pred)
    loss_test_avg = tf.reduce_mean(loss_test)
    print('Loss test: %s' % loss_test_avg.numpy())