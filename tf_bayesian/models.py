import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from tf_bayesian.fitting_methods import ndarray_fit


class BayesianModel(tf.keras.Model):
    """Custom class for a bayesian neural network. Implementing a tf 2 style
    training loop. Assumes that the network predicts a tuple (y, sigma) where
    y is a vector/tensor and sigma is scalar valued for each sample.
    """

    def __init__(
            self,
            regularization=None,
            include_epistemic=True
    ):
        super(BayesianModel, self).__init__()
        if callable(regularization):
            self.do_reg = True
            self.regularization = regularization
        else:
            self.do_reg = False

        self.grads = []
        self.loss_val = None

    def fit(
            self,
            x=None,
            y=None,
            batch_size=None,
            epochs=1,
            verbose=1,
            callbacks=None,
            validation_split=0.0,
            **kwargs,
    ):
        """Invoking the training loop for the given dataset.

        Arguments:
            x: array type of samples
            y: array type of targets
            batch_size: integer
            epochs: integer
            verbose: boolean trigger for verbosity of training.
                True for verbose training, False for silent.
            callbacks: iterable of tf.keras.callbacks.Callback instances
            validation_split: deprecated, use validation callback instead.
        """
        datatypes_to_fit = {
            np.ndarray: ndarray_fit
        }
        try:
            fit_method = datatypes_to_fit[type(x)]
        except KeyError:
            raise KeyError("No support for datatype {}, please use one of {}".format(
                type(x), datatypes_to_fit.keys()))

        if validation_split != 0.0:
            raise ValueError(
                "Validation split is unsupported, use validation callback instead")

        if not x.shape[0] == y.shape[0]:
            raise ValueError("x and y must have same first dimension")

        return fit_method(
            self,
            x,
            targets=y,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            callbacks=callbacks,
            **kwargs,
        )

    def compute_grads(self, x, y):
        """Computes the loss and gradients for a batch of samples, x, and
        corresponding labels/targets y. This method is stateful and assigns
        self variables BayesianModel.loss_val and BayesianModel.gradients
        for use with a tf.optimizer object.

        Arguments:
            x: array type of samples
            y: array type of targets corresponding to x

        Returns:
            loss value tensor for the given sample-targets batch
        """
        # TODO: do conversion outside of train loop?
        if isinstance(x, np.ndarray):
            x = tf.convert_to_tensor(x, self.dtype)
        if isinstance(y, np.ndarray):
            y = tf.convert_to_tensor(y, self.dtype)
        with tf.GradientTape(persistent=False) as tape:
            tape.watch(x)
            tape.watch(y)
            tape.watch(self.trainable_variables)
            yhat = self.__call__(x)
            loss_val = self.loss(y, yhat)
            kld = tf.reduce_sum(self.losses) / \
                tf.dtypes.cast(tf.shape(x)[0], yhat.dtype)
            self.loss_val = loss_val + kld
            if self.do_reg:
                for v in self.trainable_variables:
                    self.loss_val = self.loss_val + \
                        self.regularization(v)
        self.grads = tape.gradient(self.loss_val, self.trainable_variables)
        return loss_val

    @tf.function
    def std(self, x, N=10):
        """Computes the standard deviation for a batch of samples

        Arguments:
            x: array type of a batch of different samples

        Returns:
            standard deviation of the predctions corresponding
            to the samples x
        """
        x = tf.dtypes.cast(x, self.dtype)
        x = tf.unstack(x, axis=0)
        shape_x = tf.shape(x)
        tile_shape = tf.concat([[N], tf.ones(tf.size(shape_x) - 1)], 0)
        tile_shape = tf.dtypes.cast(tile_shape, tf.int32)
        back_transpose = tf.range(2, tf.size(shape_x)+1)
        transpose_order = tf.concat([[1, 0], back_transpose], 0)

        tile_x = tf.tile(x, tile_shape)
        primary_shape = tf.stack([N, shape_x[0]])
        tile_first_shape = tf.concat([primary_shape, shape_x[1:]], 0)
        tile_x = tf.reshape(tile_x, tile_first_shape)
        tile_x_sample_first = tf.transpose(tile_x, transpose_order)
        sample_stacked = tf.dtypes.cast(tile_x_sample_first, self.dtype)
        var = tf.map_fn(
            self.estimate_variance,
            sample_stacked,
            parallel_iterations=10,
            back_prop=False,
            infer_shape=False
        )
        return tf.sqrt(var)

    @tf.function
    def predict_mean(self, x, N=10):
        """Computes the mean prediction of a batch of samples.

        Arguments:
            x: array type of a batch of different samples

        Returns:
            Tensor of the predicted means
        """
        means = []
        x = tf.dtypes.cast(x, self.dtype)
        x = tf.unstack(x, axis=0)
        shape_x = tf.shape(x)
        tile_shape = tf.concat([[N], tf.ones(tf.size(shape_x) - 1)], 0)
        tile_shape = tf.dtypes.cast(tile_shape, tf.int32)

        tile_x = tf.tile(x, tile_shape)
        sample_stacked = tf.dtypes.cast(tile_x, self.dtype)
        sample_outs = self.__call__(sample_stacked)
        y_out, *_ = tf.unstack(sample_outs, axis=0)

        primary_shape = tf.stack([N, shape_x[0]])
        out_shape = tf.concat([primary_shape, y_out.shape[1:]], 0)
        y_out = tf.reshape(y_out, out_shape)
        means = tf.reduce_mean(y_out, axis=0)
        return means

    def estimate_variance(self, x):
        """Computes an estimate of the predicted variance for a single sample x.
        The variance is computed from eq. 9 in https://arxiv.org/pdf/1703.04977.pdf

        Arguments:
            x: array type a repeated sample on which to perform prediction.

        Returns:
            Predicted variance for the repeated sample x
        """
        retval = self.__call__(x)
        yhat, log_var = tf.unstack(retval)
        var = tf.exp(log_var)
        square_mean_pred = tf.square(
            tf.reduce_mean(yhat, axis=0, keepdims=True))
        weight_var = tf.reduce_mean(
            (tf.square(yhat) - square_mean_pred), axis=0)
        return weight_var + tf.reduce_mean(var, axis=0)
