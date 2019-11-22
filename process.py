import numpy as np
import tensorflow as tf

"""Data pre/post-processing with for loops"""

data = np.random.randint(0, 9, (3,5))
data = tf.Variable(data)
print(data)

@tf.function
def process(data):
    """Sets value over 5 to 1 and below 5 to 0"""
    for i in range(data.shape[0]):    
        for j in range(data.shape[1]):
            if data[i,j] > 5: 
                data[i,j].assign(1)
            else:
                data[i,j].assign(0)
    return data

data_processed = process(data)
print(data_processed)