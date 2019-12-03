import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from tf_bayesian.fitting_methods import ndarray_fit


class BayesianModel(tf.keras.Model):
    def __init__(self,):
        super(BayesianModel, self).__init__()

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
        datatypes_to_fit = {
            np.ndarray: ndarray_fit
        }
        try:
            fit_method = datatypes_to_fit[type(x)]
        except KeyError:
            raise KeyError("No support for datatype {}, please use one of {}".format(
                type(x), datatypes_to_fit.keys()))

        if validation_split != 0.0:
            raise ValueError("Validation split is unsupported")

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
        # TODO: do conversion outside of train loop?
        if isinstance(x, np.ndarray):
            x = tf.convert_to_tensor(x, self.dtype)
        if isinstance(y, np.ndarray):
            y = tf.convert_to_tensor(y, self.dtype)
        with tf.GradientTape(persistent=False) as tape:
            tape.watch(x)
            tape.watch(y)
            yhat = self.__call__(x)
            loss_val = self.loss(y, yhat)
        self.loss_val = loss_val
        self.grads = tape.gradient(loss_val, self.trainable_variables)
        return loss_val




