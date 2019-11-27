import tensorflow_probability as tfp 
import tensorflow as tf
import numpy as np

X = np.zeros((10, 100, 1))
Y = np.zeros((10, 1))

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Input(shape=X.shape[1:]))
model.add(tfp.layers.Convolution1DFlipout(10, 2))
model.add(tf.keras.layers.Flatten())
model.add(tfp.layers.DenseFlipout(tfp.layers.MultivariateNormalTriL.params_size(Y.shape[1])))
model.add(tfp.layers.MultivariateNormalTriL(Y.shape[1]))

neglog = lambda y, p_y: -p_y.log_prob(y)
model.compile(optimizer="adam", loss=neglog, experimental_run_tf_function=False)
print(model.summary())
model.fit(X, Y, batch_size=3)
