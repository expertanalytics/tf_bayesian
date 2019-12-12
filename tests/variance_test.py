import unittest

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

from tf_bayesian.models import BayesianModel
from tf_bayesian.callbacks import StdLogger, MetricLogger, InferenceLogger
from tf_bayesian.metrics import mse
from tf_bayesian.losses import BayesianMeanSquaredError, EpistemicMeanSquaredError

np.random.seed(42)
pl = tfp.layers
kl = tf.keras.layers
tf.config.experimental_run_functions_eagerly(True)


class BayesianDense(BayesianModel):

    def __init__(
            self,
            out_nodes,
            n_layers=1,
            hidden_nodes=10,
            include_epistemic=True,
            activation=tf.keras.activations.relu,
            regularization=tf.keras.regularizers.l2(0.001),
            dense_layer=kl.Dense,
            dense_kwargs={}
    ):
        super(BayesianDense, self).__init__(regularization=regularization)
        self.n_layers = n_layers
        self.hidden_nodes = hidden_nodes
        self.hidden_layers = []

        for l in range(n_layers):
            setattr(
                self,
                "dense{}".format(l),
                dense_layer(
                    hidden_nodes,
                    activation=activation,
                    **dense_kwargs
                )
            )
            self.hidden_layers.append(getattr(self, "dense{}".format(l)))

        self.output_layers = []
        self.output_layers.append(dense_layer(out_nodes, **dense_kwargs))
        if include_epistemic:
            self.output_layers.append(dense_layer(out_nodes, **dense_kwargs))

    def call(self, x):
        hidden = x
        for l in self.hidden_layers:
            hidden = l(hidden)
        outputs = []
        for o in self.output_layers:
            outputs.append(o(hidden))
        return tf.stack(outputs)


def posterior_moments(model):
    layers = model.layers
    means = []
    stds = []
    for l in layers:
        k_post = l.kernel_posterior
        means.append(k_post.mean())
        stds.append(k_post.stddev())
    return means, stds


class TestTraining(unittest.TestCase):
    x = np.random.uniform(-10, 10, size=(10000, 3))
    x_test = np.random.uniform(-10, 10, size=(1000, 3))
    test_in = np.array([[0.5, 0.5, 1], [2, 2, 3]])
    epochs = 1
    batch_size = 200

    def test_posteriors(self):
        tf.keras.backend.clear_session()
        model = BayesianDense(3, include_epistemic=False,
                              dense_layer=pl.DenseFlipout)

        callbacks = [MetricLogger([mse], (self.x_test, self.x_test))]
        opt = tf.keras.optimizers.Adam()
        model.compile(loss=EpistemicMeanSquaredError(model), optimizer=opt)
        _ = model(self.test_in)

        pre_means, pre_stds = posterior_moments(model)
        history = model.fit(
            self.x, self.x, batch_size=self.batch_size, epochs=self.epochs, verbose=0, callbacks=callbacks)

        post_means, post_stds = posterior_moments(model)
        means = [pre_means, post_means]
        stds = [pre_stds, post_stds]
        moments = [means, stds]
        for moment in moments:
            for i, pre in enumerate(moment[0]):
                post = moment[1][i]
                is_equal = tf.equal(pre, post).numpy().all()
                self.assertFalse(is_equal)

    def test_weights(self):
        tf.keras.backend.clear_session()
        model = BayesianDense(3, include_epistemic=False,
                              dense_layer=pl.DenseFlipout)

        callbacks = [MetricLogger([mse], (self.x_test, self.x_test))]
        opt = tf.keras.optimizers.Adam()
        model.compile(loss=EpistemicMeanSquaredError(model), optimizer=opt)
        _ = model(self.test_in)
        pre_train_vars = [t.numpy() for t in model.trainable_variables]

        history = model.fit(
            self.x, self.x, batch_size=self.batch_size, epochs=self.epochs, verbose=0, callbacks=callbacks)
        post_train_vars = [t.numpy() for t in model.trainable_variables]

        for i, pre in enumerate(pre_train_vars):
            post = post_train_vars[i]
            is_equal = (pre == post).all()
            self.assertFalse(is_equal)


class TestModeltypes(unittest.TestCase):
    n_features = 1
    x = np.random.uniform(-10, 10, size=(10000, n_features))
    x_test = np.random.uniform(-10, 10, size=(100, n_features))
    test_in = np.array([[0.5, ]*n_features, [2, ]*n_features])
    epochs = 1
    batch_size = 200

    def test_aleatoric(self):
        self.x_test.sort()
        tf.keras.backend.clear_session()
        model = BayesianDense(
                self.n_features,
                dense_layer=kl.Dense,
                n_layers=1,
                activation=tf.keras.activations.linear
                )

        opt = tf.keras.optimizers.Adam()
        loss = BayesianMeanSquaredError()
        callbacks = [MetricLogger([mse], (self.x_test, self.x_test))]
        model.compile(loss=loss, optimizer=opt)
        history = model.fit(
            self.x, self.x, batch_size=self.batch_size, epochs=self.epochs, verbose=0, callbacks=callbacks)

        y_pred, _ = model(self.x_test).numpy()
        std = model.std(self.x_test).numpy()
        self.assertTrue((y_pred.shape == std.shape))


class TestNetworkMoments(unittest.TestCase):

    def F_test_mean(self):
        np.random.seed(42)
        x = np.random.uniform(-10, 10, size=(10000, 3))
        x_test = np.random.uniform(-10, 10, size=(1000, 3))
        test_in = np.array([[0.5, 0.5, 1], [2, 2, 3]])
        epochs = 50
        callbacks = [MetricLogger([mse], (x_test, x_test)), InferenceLogger()]
        model = BayesianDense(
            3, n_layers=2, hidden_nodes=5, regularization=None)
        opt = tf.keras.optimizers.Adam()
        loss = BayesianMeanSquaredError()

        model.compile(loss=loss, optimizer=opt)
        history = model.fit(x, x, batch_size=200, epochs=epochs,
                            shuffle=True, callbacks=callbacks)
        pred = model.predict_mean(test_in)
        std = model.std(test_in)

        """
        model = tf.keras.models.Sequential()
        model.add(tfp.layers.DenseReparameterization(3))
        model.add(tfp.layers.DenseReparameterization(3))
        opt = tf.keras.optimizers.Adam()

        model.compile(loss="mse", optimizer=opt)
        model.fit(x, x, batch_size=50, epochs=100)
        pred = model.predict(test_in)

        """
        fig, ax = plt.subplots(nrows=3, figsize=(17, 10))
        ax[0].scatter(np.arange(epochs),
                      callbacks[0].metrics_logs, label="MSE")
        ax[1].scatter(np.arange(epochs), callbacks[1].loss_logs, label="KLD")
        ax[2].scatter(np.arange(epochs), np.array(
            history.history["loss"]), label="TOT LOSS")
        for i in range(len(ax)):
            ax[i].legend()
        plt.show()
        print("PREDS", pred)
        print("STD", std)


class TestErrorConformity(unittest.TestCase):

    def F_test_homoscedastic(self):
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
