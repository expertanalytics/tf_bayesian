import unittest

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

from tf_bayesian.models import BayesianModel
from tf_bayesian.callbacks import StdLogger
from tf_bayesian.losses import BayesianMeanSquaredError


class BayesianDense(BayesianModel):

    def __init__(
            self,
            out_nodes,
            activation=tf.keras.activations.relu,
            regularization=tf.keras.regularizers.l2(0.001)
            ):
        super(BayesianDense, self).__init__()
        self.dense1 = tfp.layers.DenseFlipout(10, activation=activation,)
        self.dense2 = tfp.layers.DenseFlipout(10, activation=activation,)
        self.out_layer = tfp.layers.DenseFlipout(out_nodes,)
        self.std_layer = tfp.layers.DenseFlipout(out_nodes,)

    def call(self, x):
        hidden = x
        for l in [self.dense1, self.dense2]:
            hidden = l(hidden)
        pred = self.out_layer(hidden)
        std = self.std_layer(hidden)
        return tf.stack([pred, std])


class TestErrorConformity(unittest.TestCase):

    def test_homoscedastic(self):

        def err(x, y):
            return np.random.normal(scale=0.1*np.abs(x.mean(1)))

        def process(x):
            y1 = 1.3 * np.sin(x[:, 0]) + 0.1 * 2 * np.sin(np.pi *
                                                          0.3 * x[:, 0]) + 0.1*np.square(x[:, 0])
            if x.shape[1] > 1:
                yn = y1
                for n in range(x.shape[1]):
                    yn = yn * (1.3**n * np.cos(n*0.01 * np.pi *
                                               x[:, n])) * x[:, n]**(-n)
                y = yn
            else:
                y = y1

            return y

        n_data = 1000
        n_features = 1
        train_partition = 0.8
        n_train = int(train_partition * n_data)
        n_test = n_data - n_train

        domain = np.linspace([-10]*n_features, [10]*n_features, n_data)
        indices = np.arange(n_data)
        train_indices = np.random.choice(
            indices, size=n_train, replace=False)
        test_indices = np.setdiff1d(indices, train_indices)
        Xtr = domain[np.random.choice(train_indices, size=n_data*10)]
        Xte = domain[np.random.choice(test_indices, size=n_test*10)]

        Ytr = process(Xtr)
        tr_err_vals = err(Xtr, Ytr)
        NoisyYtr = Ytr + tr_err_vals

        Yte = process(Xte)
        te_err_vals = err(Xte, Yte)
        NoisyYte = Yte + te_err_vals
        
        # tf.config.experimental_run_functions_eagerly(True)
        model = BayesianDense(1, activation=tf.keras.activations.elu)
        optimizer = tf.keras.optimizers.Adam() 
        loss = BayesianMeanSquaredError()
        model.compile(loss=loss, optimizer=optimizer)
        model.fit(x=Xtr, y=NoisyYtr, batch_size=100, epochs=10)
        mean_test = np.squeeze(model.predict_mean(Xte).numpy())
        std_test = np.squeeze(model.std(Xte).numpy())
        print("MEAN SHP", mean_test.shape)
        print("STD SHP", std_test.shape)
        print("appropriate vals", NoisyYte.shape)

        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(20, 15))
        ax[0, 0].scatter(Xte, Yte)
        ax[0, 1].scatter(Xte, te_err_vals)
        ax[0, 2].scatter(Xte, Yte + te_err_vals)
        
        ax[1, 0].scatter(Xte, mean_test)
        ax[1, 1].scatter(Xte, std_test)
        ax[1, 2].scatter(Xte, np.square(NoisyYte - mean_test))
        plt.show()


if __name__ == "__main__":
    unittest.main()
